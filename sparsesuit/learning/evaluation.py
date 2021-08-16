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
from omegaconf import DictConfig, OmegaConf


class Evaluator:
    def __init__(self, cfg):
        print("Evaluation\n*******************\n")
        self.eval_config = cfg.evaluation
        self.visualize = cfg.visualize
        self.past_frames = self.eval_config.past_frames
        self.future_frames = self.eval_config.future_frames

        # load training configuration of experiment
        self.exp_path = os.path.join(os.getcwd(), "runs/", self.eval_config.experiment)
        self.train_config = utils.load_config(self.exp_path)

        # cuda setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        # load neutral smpl model for joint position evaluation
        self.smpl_model = smpl_helpers.load_smplx(["neutral"])["neutral"]

        # setup model
        self.pred_trgt_joints = self.train_config.dataset.pred_trgt_joints
        num_input_sens = len(self.train_config.training.sensors)
        ori_dim = num_input_sens * 9
        acc_dim = num_input_sens * 3
        self.pose_dim = len(self.pred_trgt_joints) * 9

        input_dim = ori_dim + acc_dim
        target_dim = self.pose_dim + acc_dim

        self.model = models.BiRNN(input_dim=input_dim, target_dim=target_dim).to(
            self.device
        )
        model_path = os.path.join(self.exp_path, "checkpoint.pt")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # load appropriate dataset for evaluation
        self.suit_config = self.train_config.training.config
        if self.eval_config.dataset == "synthetic":
            ds_folder = paths.AMASS_PATH

            if cfg.debug:
                ds_folder += "_debug"

            if self.suit_config == "SSP":
                ds_folder += "_SSP"

            elif self.suit_config == "MVN":
                ds_folder += "_MVN"

            else:
                raise NameError("Invalid configuration. Aborting!")

            if self.eval_config.noise:
                ds_folder += "_noisy"
            ds_folder += "_nn"

        elif self.eval_config.dataset == "real":

            if self.suit_config == "MVN":
                ds_folder = paths.DIP_17_NN_PATH

            else:
                raise NameError("Invalid configuration. Aborting!")

        # get test dataset paths
        ds_dir = os.path.join(paths.DATA_PATH, ds_folder)
        test_ds_path = os.path.join(ds_dir, paths.TEST_FILE)

        test_ds_config = utils.load_config(ds_dir).dataset
        test_ds_size = test_ds_config.assets.test

        # get indices of sensors the model was trained with in evaluation dataset
        test_ds_sens = test_ds_config.sensors
        train_sens = self.train_config.training.sensors
        self.sens_ind = [test_ds_sens.index(sens) for sens in train_sens]

        # load test dataset statistics
        stats_path = os.path.join(ds_dir, "stats.npz")
        with np.load(stats_path, allow_pickle=True) as data:
            stats = dict(data)
        self.pose_mean, self.pose_std = stats["pose_mean"], stats["pose_std"]

        test_ds = (
            WebDataset(test_ds_path, length=test_ds_size)
            .decode()
            .to_tuple("ori.npy", "acc.npy", "pose.npy")
        )
        self.test_dl = DataLoader(test_ds)

    def evaluate(self):
        # set up error statistics
        stats_pos_err, stats_ang_err = Welford(), Welford()

        # iterate over test dataset
        for batch_num, (ori, acc, pose_trgt_norm) in enumerate(self.test_dl):
            print("computing metrics on asset {}".format(batch_num))

            # load input and target
            input_vec, target_vec = utils.assemble_input_target(
                ori, acc, pose_trgt_norm, self.sens_ind
            )

            # predict SMPL pose params online or offline depending on past_/future_frames
            x, y = input_vec.to(self.device).float(), target_vec.to(self.device).float()
            if self.past_frames == -1 or self.future_frames == -1:
                pred_mean, _, _, _ = self.model(x)
            else:
                pred_mean = self.predict_window(x)

            # extract poses from predictions
            pose_pred_norm = pred_mean[:, :, : self.pose_dim].cpu().detach().numpy()
            # undo normalization of SMPL predictions and targets
            pose_pred = pose_pred_norm * self.pose_std + self.pose_mean
            pose_trgt = (
                pose_trgt_norm.cpu().detach().numpy() * self.pose_std + self.pose_mean
            )

            # expand reduced (15/19 joints) to full 24 smpl joints
            pose_pred_full = [
                smpl_helpers.smpl_reduced_to_full(p, self.pred_trgt_joints)
                for p in pose_pred
            ]
            pose_trgt_full = [
                smpl_helpers.smpl_reduced_to_full(p, self.pred_trgt_joints)
                for p in pose_trgt
            ]

            # rotate root for upright human-beings
            pose_pred_full[0][:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]
            pose_trgt_full[0][:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]

            # evaluate predictions for angular and positional error in chunks
            max_chunk_size = 500 if paths.ON_MAC else 1000
            chunks = input_vec.shape[1] // max_chunk_size + 1
            for k in range(chunks):
                pred_k = [
                    pose_pred_full[0][k * max_chunk_size : (k + 1) * max_chunk_size]
                ]
                targ_k = [
                    pose_trgt_full[0][k * max_chunk_size : (k + 1) * max_chunk_size]
                ]
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

        print(
            "Average joint angle error (deg): {:.4f} (+/- {:.3f})".format(
                avg_ang_err, std_ang_err
            )
        )
        print(
            "Average joint position error (cm): {:.4f} (+/- {:.3f})".format(
                avg_pos_err, std_pos_err
            )
        )

        self.write_config(avg_ang_err, std_ang_err, avg_pos_err, std_pos_err)

    def write_config(self, avg_ang_err, std_ang_err, avg_pos_err, std_pos_err):
        # load evaluation configuration
        eval_config = OmegaConf.create(self.eval_config)

        # add stats to training config
        eval_config.mean_ang_err = float(avg_ang_err)
        eval_config.std_ang_err = float(std_ang_err)
        eval_config.mean_pos_err = float(avg_pos_err)
        eval_config.std_pos_err = float(std_pos_err)

        # compile target config to dump with model checkpoint
        trgt_config = OmegaConf.create()
        trgt_config.training = self.train_config.training
        trgt_config.evaluation = eval_config
        trgt_config.dataset = self.train_config.dataset

        utils.write_config(path=self.exp_path, config=trgt_config)

    def predict_window(self, x):
        """Pass a sliding window of (past_frames + future_frames + 1) input frames and keep only (past_frames +
        1)th prediction at each step. The full prediction is the concatenated individual predictions."""
        seq_len = x.shape[1]
        preds_mean = []

        for step in range(seq_len):
            # find start and end of current window
            start_idx = max(step - self.past_frames, 0)
            end_idx = min(step + self.future_frames + 1, seq_len)

            input_window = x[:, start_idx:end_idx]
            pred_mean_window, _, _, _ = self.model(input_window)
            pred_idx = min(step, self.past_frames)
            preds_mean.append(pred_mean_window[:, pred_idx : pred_idx + 1])

        return torch.cat(preds_mean, dim=1)

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
        assert dof == 216, "unexpected number of degrees of freedom"
        assert (
            target_pose_params.shape[0] == seq_length
            and target_pose_params.shape[1] == dof
        ), "target_pose_params must match predicted_pose_params"

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
        assert (
            predicted_pose_params.shape[1] // sensors.NUM_SMPL_JOINTS == pose_dim
        ), "Dimension mismatch. Abort!"

        # add rotation matrices (identity) for missing SMPL-X joints
        padding_len = sensors.NUM_SMPLX_JOINTS - sensors.NUM_SMPL_JOINTS
        padding = np.tile(np.identity(3), [batch_size, padding_len, 1, 1])
        padding_flat = np.reshape(padding, [batch_size, -1])
        pred_poses_padded = np.concatenate(
            (predicted_pose_params[:, : 24 * pose_dim], padding_flat), axis=1
        )
        pred_poses = np.reshape(pred_poses_padded, [batch_size, -1, 3, 3])
        targ_poses_padded = np.concatenate(
            (target_pose_params[:, : 24 * pose_dim], padding_flat), axis=1
        )
        targ_poses = np.reshape(targ_poses_padded, [batch_size, -1, 3, 3])

        # make predictions proper rotation matrices (removing scale information)
        u, _, v = np.linalg.svd(pred_poses)
        pred_poses_proper = u @ v

        # pred_poses_aa = rot_matrix_to_aa(np.reshape(pred_poses, [-1, 9]))
        # pred_poses_proper = np.reshape(pred_poses_aa, [batch_size, -1, 3])  # pose2rot=True

        pred_poses_torch = torch.from_numpy(pred_poses_proper).float()
        pred_verts, pred_joints, _ = smpl_helpers.my_lbs(
            self.smpl_model, pred_poses_torch, betas_torch, pose2rot=False
        )
        pred_verts_np, pred_joints_np = (
            pred_verts.detach().numpy(),
            pred_joints.detach().numpy(),
        )

        targ_poses_torch = torch.from_numpy(targ_poses).float()
        targ_verts, targ_joints, _ = smpl_helpers.my_lbs(
            self.smpl_model, targ_poses_torch, betas_torch, pose2rot=False
        )
        targ_verts_np, targ_joints_np = (
            targ_verts.detach().numpy(),
            targ_joints.detach().numpy(),
        )

        # select SMPL joints (first 24) from SMPL-X joints
        pred_joints_sel = pred_joints_np[:, : sensors.NUM_SMPL_JOINTS]
        targ_joints_sel = targ_joints_np[:, : sensors.NUM_SMPL_JOINTS]

        # rotationally align joints via Procrustes
        pred_verts_aligned = []
        pred_joints_aligned = []
        for i in range(batch_size):
            rot_i = procrustes.rotational(pred_joints_sel[i], targ_joints_sel[i]).get(
                "t"
            )
            pred_joints_aligned_i = pred_joints_sel[i] @ rot_i
            pred_joints_aligned.append(pred_joints_aligned_i)

            # align vertices of predicted SMPL mesh for visualization purposes
            pred_verts_aligned.append(pred_verts_np[i] @ rot_i)

        # compute euclidean distances between prediction and target joints
        mm = np.linalg.norm(np.asarray(pred_joints_aligned) - targ_joints_sel, axis=2)

        if self.visualize:
            verts = [targ_verts_np, np.asarray(pred_verts_aligned)]
            vertex_colors = ["green", "orange"]
            joints = [targ_joints_np, np.asarray(pred_joints_aligned)]

            # show vertex indices of sensors used in training
            root_ids = [0, 1] if self.suit_config == "SSP" else [0]
            train_sens_ids = root_ids + [ind + len(root_ids) for ind in self.sens_ind]
            sens_verts = self.train_config.dataset.vert_ids
            train_sens_verts = [sens_verts[idx] for idx in train_sens_ids]
            sensors_vis = [
                targ_verts_np[:, train_sens_verts],
            ]

            visualization.vis_smpl(
                self.smpl_model,
                verts,
                vertex_colors=vertex_colors,
                # joints=joints,
                sensors=sensors_vis,
                play_frames=300,
                playback_speed=0.3,
                add_captions=True,
            )

        return mm * 100  # convert m to cm


@hydra.main(config_path="conf", config_name="evaluation")
def do_evaluation(cfg: DictConfig):
    eval = Evaluator(cfg=cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_evaluation()
