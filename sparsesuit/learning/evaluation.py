import os

import numpy as np
import procrustes
from webdataset import WebDataset
from welford import Welford
import cv2

from sparsesuit.utils import utils, smpl_helpers, visualization
from sparsesuit.learning import models
from sparsesuit.constants import paths, sensors
import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig


class Evaluator:
    # TODO: update code to latest evaluation version with online-mode
    def __init__(self, cfg):
        eval_config = cfg.evaluation
        self.visualize = cfg.visualize

        # load training configuration of experiment
        exp_path = os.path.join(os.getcwd(), "runs/", eval_config.experiment_path)
        train_config = utils.load_config(exp_path)

        # cuda setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        # load neutral smpl model for joint position evaluation
        self.smpl_model = smpl_helpers.load_smplx(["neutral"])["neutral"]

        # setup model
        # TODO: fix this with actual config
        num_train_sens = len(train_config.training.sensors)
        ori_dim = (num_train_sens - 1) * 9
        acc_dim = (num_train_sens - 1) * 3
        pose_dim = len(train_config.dataset.pred_trgt_joints) * 9

        input_dim = ori_dim + acc_dim
        target_dim = pose_dim + acc_dim

        self.model = models.BiRNN(input_dim=input_dim, target_dim=target_dim).to(self.device)
        model_path = os.path.join(exp_path, "checkpoint.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # get indices of sensors the model was trained with in dataset
        ds_sens = train_config.dataset.sensors
        train_sens = train_config.training.sensors
        self.sens_ind = [ds_sens.index(sens) for sens in train_sens]

        # load appropriate dataset for evaluation
        if eval_config.dataset_type == "synthetic":
            ds_folder = paths.AMASS_PATH

            if cfg.debug:
                ds_folder += "_debug"

            if train_config.config == "SSP":
                ds_folder += "_SSP"

            elif train_config.config == "MVN":
                ds_folder += "_MVN"

            else:
                raise NameError("Invalid configuration. Aborting!")

            if train_config.noise:
                ds_folder += "_noisy"
            ds_folder += "_nn"

        elif train_config.dataset_type == "real":

            if train_config.config == "MVN":
                ds_folder = paths.DIP_17_NN_PATH

            else:
                raise NameError("Invalid configuration. Aborting!")

        # get test dataset paths
        ds_dir = os.path.join(paths.DATA_PATH, ds_folder)
        test_ds_path = os.path.join(ds_dir, paths.TEST_FILE)

        ds_config = utils.load_config(ds_dir).dataset
        test_ds_size = ds_config.assets.test

        # load test dataset statistics
        with np.load(ds_dir, allow_pickle=True) as data:
            stats = dict(data)
        self.pose_mean, self.pose_std = stats['pose_mean'], stats['pose_std']

        test_ds = WebDataset(test_ds_path, length=test_ds_size).decode().to_tuple('ori.npy',
                                                                                  'acc.npy',
                                                                                  'pose.npy')
        self.test_dl = DataLoader(test_ds)

    def evaluate(self):
        # set up error statistics
        stats_pos_err, stats_ang_err = Welford(), Welford()

        # iterate over test dataset
        for batch_num, (ori, acc, pose_trgt_norm) in enumerate(self.test_dl):
            print('computing metrics on asset {}'.format(batch_num))

            # load input and target
            input_vec, target_vec = utils.assemble_input_target(ori, acc, pose_trgt_norm, self.sens_ind)

            # predict SMPL pose params
            x, y = input_vec.to(self.device).float(), target_vec.to(self.device).float()
            pred_mean, _, _, _ = self.model(x)

            # undo normalization of SMPL predictions and targets
            pose_pred_norm = pred_mean[:, :, :135].cpu().detach().numpy()  # extract poses from predictions
            pose_pred = pose_pred_norm * self.pose_std + self.pose_mean
            pose_trgt = pose_trgt_norm.cpu().detach().numpy() * self.pose_std + self.pose_mean

            # expand reduced (15 joints) to full 24 smpl joints
            pose_pred_full = [smpl_helpers.smpl_reduced_to_full(p) for p in pose_pred]
            pose_trgt_full = [smpl_helpers.smpl_reduced_to_full(p) for p in pose_trgt]

            # rotate root for upright human-beings
            pose_pred_full[0][:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]
            pose_trgt_full[0][:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]

            # evaluate predictions for angular and positional error in chunks
            max_chunk_size = 500 if paths.ON_MAC else 1000
            chunks = input_vec.shape[1] // max_chunk_size + 1
            for k in range(chunks):
                pred_k = [pose_pred_full[0][k * max_chunk_size:(k + 1) * max_chunk_size]]
                targ_k = [pose_trgt_full[0][k * max_chunk_size:(k + 1) * max_chunk_size]]
                ang_err, pos_err = self.compute_metrics(pred_k, targ_k)

                # add errors to stats
                for i in range(len(ang_err)):
                    stats_ang_err.add(ang_err[i])
                    stats_pos_err.add(pos_err[i])

        # summarize errors
        avg_ang_err = np.mean(stats_ang_err.mean[sensors.ANG_EVAL_JOINTS])
        std_ang_err = np.mean(np.sqrt(stats_ang_err.var_p[sensors.ANG_EVAL_JOINTS]))

        avg_pos_err = np.mean(stats_pos_err.mean[sensors.POS_EVAL_JOINTS])
        std_pos_err = np.mean(np.sqrt(stats_pos_err.var_p[sensors.POS_EVAL_JOINTS]))

        print('Average joint angle error (deg): {:.4f} (+/- {:.3f})'.format(avg_ang_err, std_ang_err))
        print('Average joint position error (cm): {:.4f} (+/- {:.3f})'.format(avg_pos_err, std_pos_err))

    def compute_metrics(self, prediction, target, compute_positional_error=True):
        """
        Compute the metrics on the predictions. The function can handle variable sequence lengths for each pair of
        prediction-target array. The pose parameters can either be represented as rotation matrices (dof = 9) or
        quaternions (dof = 4)
        :param prediction: a list of np arrays of size (seq_length, 24*dof)
        :param target: a list of np arrays of size (seq_length, 24*dof)
        :param compute_positional_error: if set, the euclidean pose error is calculated which can take some time.
        """
        assert len(prediction) == len(target)
        dof = prediction[0].shape[1] // sensors.NUM_SMPL_JOINTS
        assert dof == 9 or dof == 4

        # because we are interested in difference per frame, flatten inputs
        pred = np.concatenate(prediction, axis=0)
        targ = np.concatenate(target, axis=0)

        pred_g = smpl_helpers.smpl_rot_to_global(pred)
        targ_g = smpl_helpers.smpl_rot_to_global(targ)

        # compute angle error for all SMPL joints
        angle_err = self.joint_angle_error(pred_g, targ_g)

        # compute positional error for all SMPL joints (optional as computationally heavy)
        if compute_positional_error:
            pos_err = self.joint_pos_error(pred, targ)
        else:
            pos_err = np.zeros(angle_err.shape)

        return angle_err, pos_err

    def joint_angle_error(self, predicted_pose_params, target_pose_params):
        """
        Computes the distance in joint angles in degrees between predicted and target joints for every given frame. Currently,
        this function can only handle input pose parameters represented as rotation matrices.

        :param predicted_pose_params: An np array of shape `(seq_length, dof)` where `dof` is 216, i.e. a 3-by-3 rotation
          matrix for each of the 24 joints.
        :param target_pose_params: An np array of the same shape as `predicted_pose_params` representing the target poses.
        :return: An np array of shape `(seq_length, 24)` containing the joint angle error in Radians for each joint.
        """
        seq_length, dof = predicted_pose_params.shape[0], predicted_pose_params.shape[1]
        assert dof == 216, 'unexpected number of degrees of freedom'
        assert target_pose_params.shape[0] == seq_length and target_pose_params.shape[
            1] == dof, 'target_pose_params must match predicted_pose_params'

        # reshape to have rotation matrices explicit
        n_joints = dof // 9
        p1 = np.reshape(predicted_pose_params, [-1, n_joints, 3, 3])
        p2 = np.reshape(target_pose_params, [-1, n_joints, 3, 3])

        # compute R1 * R2.T, if prediction and target match, this will be the identity matrix
        r1 = np.reshape(p1, [-1, 3, 3])
        r2 = np.reshape(p2, [-1, 3, 3])
        r2t = np.transpose(r2, [0, 2, 1])
        r = np.matmul(r1, r2t)

        # convert `r` to angle-axis representation and extract the angle, which is our measure of difference between
        # the predicted and target orientations
        angles = []
        for i in range(r1.shape[0]):
            aa, _ = cv2.Rodrigues(r[i])
            angles.append(utils.rad2deg(np.linalg.norm(aa)))

        return np.reshape(np.array(angles), [seq_length, n_joints])

    def joint_pos_error(self, predicted_pose_params, target_pose_params):
        # compute 3d joint positions for prediction and target
        batch_size = predicted_pose_params.shape[0]
        betas_torch = torch.zeros([batch_size, 10], dtype=torch.float32)

        pose_dim = 9  # 9 if rotation matrices, 3 if angle axis
        # add rotation matrices (identity) for missing SMPL-X joints
        padding = np.zeros([batch_size, 55 - 24, 3, 3])
        for frame_idx, frame in enumerate(padding):
            for joint_idx, _ in enumerate(frame):
                padding[frame_idx][joint_idx] = np.identity(3)
        padding_flat = np.reshape(padding, [batch_size, -1])
        pred_poses_padded = np.concatenate((predicted_pose_params[:, :24 * pose_dim], padding_flat), axis=1)
        pred_poses = np.reshape(pred_poses_padded, [batch_size, -1, 3, 3])
        targ_poses_padded = np.concatenate((target_pose_params[:, :24 * pose_dim], padding_flat), axis=1)
        targ_poses = np.reshape(targ_poses_padded, [batch_size, -1, 3, 3])

        # make predictions proper rotation matrices (removing scale information)
        u, _, v = np.linalg.svd(pred_poses)
        pred_poses_proper = u @ v

        # pred_poses_aa = rot_matrix_to_aa(np.reshape(pred_poses, [-1, 9]))
        # pred_poses_proper = np.reshape(pred_poses_aa, [batch_size, -1, 3])  # pose2rot=True

        pred_poses_torch = torch.from_numpy(pred_poses_proper).float()
        pred_verts, pred_joints, _ = smpl_helpers.my_lbs(self.smpl_model, pred_poses_torch, betas_torch, pose2rot=False)
        pred_verts_np, pred_joints_np = pred_verts.detach().numpy(), pred_joints.detach().numpy()

        targ_poses_torch = torch.from_numpy(targ_poses).float()
        targ_verts, targ_joints, _ = smpl_helpers.my_lbs(self.smpl_model, targ_poses_torch, betas_torch, pose2rot=False)
        targ_verts_np, targ_joints_np = targ_verts.detach().numpy(), targ_joints.detach().numpy()

        # select SMPL joints (first 24) from SMPL-X joints
        pred_joints_sel = pred_joints_np[:, :24]
        targ_joints_sel = targ_joints_np[:, :24]

        # rotationally align joints via Procrustes
        pred_verts_aligned = []
        pred_joints_aligned = []
        for i in range(batch_size):
            rot_i = procrustes.rotational(pred_joints_sel[i], targ_joints_sel[i]).get('t')
            pred_joints_aligned_i = pred_joints_sel[i] @ rot_i
            pred_joints_aligned.append(pred_joints_aligned_i)

            # align vertices of predicted SMPL mesh for visualization purposes
            pred_verts_aligned.append(pred_verts_np[i] @ rot_i)

        # compute euclidean distances between prediction and target joints
        mm = np.linalg.norm(np.asarray(pred_joints_aligned) - targ_joints_sel, axis=2)

        if self.visualize:
            verts = [targ_verts_np, np.asarray(pred_verts_aligned)]
            vertex_colors = ['green', 'orange']
            joints = [targ_joints_np, np.asarray(pred_joints_aligned)]
            visualization.vis_smpl(self.smpl_model,
                                     verts,
                                     vertex_colors=vertex_colors,
                                     # joints=joints,
                                     play_frames=300,
                                     playback_speed=1,
                                     )

        return mm * 100  # convert m to cm


@hydra.main(config_path="conf", config_name="evaluation")
def do_evaluation(cfg: DictConfig):
    eval = Evaluator(cfg=cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_evaluation()
