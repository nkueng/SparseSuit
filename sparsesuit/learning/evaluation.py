import logging
import os

import cv2
import hydra
import numpy as np
import procrustes
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from welford import Welford

from sparsesuit.constants import paths, sensors
from sparsesuit.learning import models
from sparsesuit.utils import utils, smpl_helpers


class Evaluator:
    def __init__(self, cfg):
        # load evaluation parameters
        self.eval_config = cfg.evaluation
        self.visualize = cfg.visualize
        self.past_frames = self.eval_config.past_frames
        self.future_frames = self.eval_config.future_frames
        self.eval_pos_err = self.eval_config.pos_err or cfg.visualize

        # load training configuration of experiment
        self.exp_path = os.path.join(paths.RUN_PATH, self.eval_config.experiment)
        self.train_config = utils.load_config(self.exp_path)

        # cuda setup
        use_available_gpu = torch.cuda.is_available() and cfg.gpu
        self.device = torch.device("cuda" if use_available_gpu else "cpu")

        # logger setup
        log_level = logging.DEBUG if cfg.debug else logging.INFO
        self.logger = utils.configure_logger(
            name="evaluation", log_path=self.exp_path, level=log_level
        )
        print("Evaluation\n*******************\n")
        self.logger.info("Using {} device".format(self.device))

        # load neutral smpl model for joint position evaluation
        self.smpl_model = smpl_helpers.load_smplx(
            "neutral",
            # device=self.device,
        )

        # load and setup trained model
        exp_config = self.train_config.experiment
        train_ds_path = utils.ds_path_from_config(
            exp_config.train_dataset, "evaluation", cfg.debug
        )
        if self.train_config.hyperparameters.train_on_processed:
            train_ds_path += "_nn"
        else:
            train_ds_path += "_n"
        train_ds_config = utils.load_config(train_ds_path).dataset
        self.pred_trgt_joints = train_ds_config.pred_trgt_joints
        input_sensor_names = exp_config.sensors.names
        num_input_sens = len(input_sensor_names)
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

        self.sensor_config = self.train_config.experiment.sensors.sensor_config
        # ds_dir = utils.ds_path_from_config(cfg.evaluation, cfg.debug)
        # if self.eval_config.dataset == "synthetic":
        #     ds_dir = paths.AMASS_PATH
        #
        #     if cfg.debug:
        #         ds_dir += "_debug"
        #
        #     if self.sensor_config == "SSP":
        #         ds_dir += "_SSP"
        #
        #     elif self.sensor_config == "MVN":
        #         ds_dir += "_MVN"
        #
        #     else:
        #         raise NameError("Invalid configuration. Aborting!")
        #
        #     if self.eval_config.noise:
        #         ds_dir += "_noisy"
        #     ds_dir += "_nn"

        # elif self.eval_config.dataset == "real":
        #
        #     if self.sensor_config == "MVN":
        #         ds_dir = paths.DIP_17_NN_PATH
        #
        #     elif self.sensor_config == "SSP":
        #         ds_dir = paths.RKK_STUDIO_19_NN_PATH
        #
        #     else:
        #         raise NameError("Invalid configuration. Aborting!")

        # get evaluation dataset
        ds_dir = utils.ds_path_from_config(
            cfg.evaluation.eval_dataset, "evaluation", cfg.debug
        )
        self.stats = {}
        self.pose_mean = 0
        self.pose_std = 1
        if self.train_config.hyperparameters.train_on_processed:
            ds_dir += "_nn"
        else:
            ds_dir += "_n"
            # load test dataset statistics
            if self.train_config.hyperparameters.use_stats:
                stats_path = os.path.join(ds_dir, "stats.npz")
                with np.load(stats_path, allow_pickle=True) as data:
                    self.stats = dict(data)
                self.pose_mean, self.pose_std = (
                    self.stats["pose_mean"],
                    self.stats["pose_std"],
                )
        test_ds_config = utils.load_config(ds_dir).dataset
        test_ds_size = test_ds_config.normalized_assets.test
        self.ds_fps = test_ds_config.fps

        # get indices of sensors the model was trained with
        test_ds_sens = test_ds_config.normalized_sensors
        train_sens = self.train_config.experiment.sensors.names
        self.sens_ind = [test_ds_sens.index(sens) for sens in train_sens]
        # sens_ind_mask = [sens in train_sens for sens in test_ds_sens]
        # sens_ind_alt = [
        #     item.index() for keep, item in zip(sens_ind_mask, test_ds_sens) if keep
        # ]

        test_ds_path = os.path.join(ds_dir, "test")
        test_ds = utils.BigDataset(test_ds_path, test_ds_size)
        self.test_dl = DataLoader(test_ds, num_workers=4, pin_memory=True)

    def evaluate(self):
        # set up error statistics
        stats_pos_err = Welford()
        stats_ang_err = Welford()
        stats_loss = Welford()
        stats_jerk = Welford()
        stats_ang_per_asset = {}

        # container to store predicted poses
        predicted_poses = {}

        # iterate over test dataset
        with torch.no_grad():
            for batch_num, (ori, acc, pose_trgt, filename) in enumerate(self.test_dl):
                # get ang err statistics for this asset
                stats_ang_err_asset = Welford()

                self.logger.info(
                    "Computing metrics for asset {}: {} with {} frames.".format(
                        batch_num, filename[0], ori.shape[1]
                    )
                )

                # load input and target
                input_vec, target_vec = utils.assemble_input_target(
                    ori, acc, pose_trgt, self.sens_ind, self.stats
                )

                # predict SMPL pose params online or offline depending on past_/future_frames
                x, y = (
                    input_vec.to(self.device).float(),
                    target_vec.to(self.device).float(),
                )
                if self.past_frames == -1 or self.future_frames == -1:
                    pred_mean, pred_std, _, _ = self.model(x)
                else:
                    pred_mean, pred_std = self.predict_window(x)

                # compute loss
                loss = F.gaussian_nll_loss(
                    pred_mean, y, pred_std, full=True, reduction="sum"
                ) / (y.shape[0] * y.shape[1])
                stats_loss.add(utils.copy2cpu(loss))

                self.logger.debug("Loss: {:.2f}".format(loss))

                # extract poses from predictions
                pose_pred = utils.copy2cpu(pred_mean[:, :, : self.pose_dim])

                # undo normalization of SMPL predictions and targets
                pose_pred = pose_pred * self.pose_std + self.pose_mean
                # pose_trgt = (
                #         utils.copy2cpu(pose_trgt) * self.pose_std + self.pose_mean
                # )

                # expand reduced (15/19 joints) to full 24 smpl joints
                pose_pred_full = [
                    smpl_helpers.smpl_reduced_to_full(p, self.pred_trgt_joints)
                    for p in pose_pred
                ]
                pose_trgt_full = [
                    smpl_helpers.smpl_reduced_to_full(p, self.pred_trgt_joints)
                    for p in utils.copy2cpu(pose_trgt)
                ]

                # keep track of predicted poses for saving later
                predicted_poses[filename[0]] = pose_pred

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
                    ang_err, pos_err, jerk = self.compute_metrics(
                        pred_k, targ_k, self.eval_pos_err
                    )

                    # add errors to stats
                    for ang_err_i, pos_err_i, jerk_i in zip(ang_err, pos_err, jerk):
                        stats_ang_err.add(ang_err_i)
                        stats_pos_err.add(pos_err_i)
                        stats_jerk.add(jerk_i)
                        stats_ang_err_asset.add(ang_err_i)

                # keep track of angular errors per joint and per asset
                stats_ang_per_asset[filename[0]] = stats_ang_err_asset.mean

        # save predicted poses with model
        file_name = "predictions.npz"
        if self.eval_config.eval_dataset.source != "AMASS":
            file_name = "real_predictions.npz"
        poses_filename = os.path.join(self.exp_path, file_name)
        with open(poses_filename, "wb") as fout:
            np.savez_compressed(fout, **predicted_poses)

        # save per joint and asset ang err stats to local file
        err_stats_filename = os.path.join(self.exp_path, "error_stats.npz")
        with open(err_stats_filename, "wb") as fout:
            np.savez_compressed(fout, **stats_ang_per_asset)

        # summarize errors
        metrics = {
            # our metrics
            "avg_total_ang_err": round(
                float(np.mean(stats_ang_err.mean[sensors.ANG_EVAL_JOINTS])), 2
            ),
            "std_total_ang_err": round(
                float(np.mean(np.sqrt(stats_ang_err.var_p[sensors.ANG_EVAL_JOINTS]))), 2
            ),
            "avg_total_pos_err": round(
                float(np.mean(stats_pos_err.mean[sensors.POS_EVAL_JOINTS])), 2
            ),
            "std_total_pos_err": round(
                float(np.mean(np.sqrt(stats_pos_err.var_p[sensors.POS_EVAL_JOINTS]))), 2
            ),
            # TransPose
            "avg_tp_ang_err": round(
                float(np.mean(stats_ang_err.mean[sensors.TP_ANG_EVAL_JOINTS])), 2
            ),
            "std_tp_ang_err": round(
                float(
                    np.mean(np.sqrt(stats_ang_err.var_p[sensors.TP_ANG_EVAL_JOINTS]))
                ),
                2,
            ),
            "avg_tp_pos_err": round(
                float(np.mean(stats_pos_err.mean[sensors.TP_POS_EVAL_JOINTS])), 2
            ),
            "std_tp_pos_err": round(
                float(
                    np.mean(np.sqrt(stats_pos_err.var_p[sensors.TP_POS_EVAL_JOINTS]))
                ),
                2,
            ),
            # SIP & DIP
            "avg_sip_ang_err": round(
                float(np.mean(stats_ang_err.mean[sensors.SIP_ANG_EVAL_JOINTS])), 2
            ),
            "std_sip_ang_err": round(
                float(
                    np.mean(np.sqrt(stats_ang_err.var_p[sensors.SIP_ANG_EVAL_JOINTS]))
                ),
                2,
            ),
            "avg_sip_pos_err": round(
                float(np.mean(stats_pos_err.mean[sensors.SIP_POS_EVAL_JOINTS])), 2
            ),
            "std_sip_pos_err": round(
                float(
                    np.mean(np.sqrt(stats_pos_err.var_p[sensors.SIP_POS_EVAL_JOINTS]))
                ),
                2,
            ),
            # loss function
            "avg_loss": round(float(stats_loss.mean), 2),
            "std_loss": round(float(np.sqrt(stats_loss.var_p)), 2),
            # jerk
            "avg_jerk": round(float(np.mean(stats_jerk.mean) / 100), 2),
            "std_jerk": round(float(np.mean(np.sqrt(stats_jerk.var_p)) / 100), 2),
        }

        self.logger.info(
            "Average total joint angle error (deg): {:.2f} (+/- {:.2f})".format(
                metrics["avg_total_ang_err"], metrics["std_total_ang_err"]
            )
        )
        self.logger.info(
            "Average total joint position error (cm): {:.2f} (+/- {:.2f})".format(
                metrics["avg_total_pos_err"], metrics["std_total_pos_err"]
            )
        )
        self.logger.info(
            "Average SIP joint angle error (deg): {:.2f} (+/- {:.2f})".format(
                metrics["avg_sip_ang_err"], metrics["std_sip_ang_err"]
            )
        )
        self.logger.info(
            "Average SIP joint position error (cm): {:.2f} (+/- {:.2f})".format(
                metrics["avg_sip_pos_err"], metrics["std_sip_pos_err"]
            )
        )
        self.logger.info(
            "Average loss: {:.2f} (+/- {:.2f})".format(
                metrics["avg_loss"], metrics["std_loss"]
            )
        )
        self.logger.info(
            "Average jerk (100m/s^3): {:.2f} (+/- {:.2f})".format(
                metrics["avg_jerk"], metrics["std_jerk"]
            )
        )

        self.write_config(metrics)

    def write_config(self, metrics):
        # load evaluation configuration
        eval_config = OmegaConf.create(self.eval_config)

        # add error metrics to eval config
        eval_config.metrics = metrics

        # compile target config to dump with model checkpoint
        trgt_config = OmegaConf.create()
        trgt_config.experiment = self.train_config.experiment
        trgt_config.hyperparameters = self.train_config.hyperparameters
        trgt_config.evaluation = eval_config
        # trgt_config.dataset = self.train_config.experiment.train_dataset

        utils.write_config(path=self.exp_path, config=trgt_config)

    def predict_window(self, x):
        """Pass a sliding window of (past_frames + future_frames + 1) input frames and keep only (past_frames +
        1)th prediction at each step. The full prediction is the concatenated individual predictions."""
        seq_len = x.shape[1]
        preds_mean = []
        preds_std = []

        for step in range(seq_len):
            # find start and end of current window
            start_idx = max(step - self.past_frames, 0)
            end_idx = min(step + self.future_frames + 1, seq_len)

            # extract window and make prediction
            input_window = x[:, start_idx:end_idx]
            pred_mean_window, pred_std_window, _, _ = self.model(input_window)

            # find index of frame between past and future frames
            pred_idx = min(step, self.past_frames)
            preds_mean.append(pred_mean_window[:, pred_idx : pred_idx + 1])
            preds_std.append(pred_std_window[:, pred_idx : pred_idx + 1])

        return torch.cat(preds_mean, dim=1), torch.cat(preds_std, dim=1)

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

        # compute angle error for all SMPL joints based on global joint orientation
        # this way the error is not propagated along the kinematic chain
        angle_err = joint_angle_error(pred_g, targ_g)

        # compute positional error for all SMPL joints (optional as computationally heavy)
        # TODO: get visualization out of joint_pos_error
        if compute_positional_error:
            pos_err, pred_joint_pos = self.joint_pos_error(pred, targ)
            # compute jerk for all SMPL joints
            jerk_delta = 1
            jerk = utils.compute_jerk(pred_joint_pos, jerk_delta, self.ds_fps)
        else:
            pos_err = np.zeros(angle_err.shape)
            jerk = np.zeros(angle_err.shape)

        return angle_err, pos_err, jerk

    def joint_pos_error(self, predicted_pose_params, target_pose_params):
        """compute 3d joint positions for prediction and target, then evaluate euclidean distance"""
        batch_size = predicted_pose_params.shape[0]

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
        pred_poses_proper = utils.remove_scaling(pred_poses)

        pred_poses_torch = torch.from_numpy(pred_poses_proper).float()
        pred_verts, pred_joints, _ = smpl_helpers.my_lbs(
            self.smpl_model, pred_poses_torch, pose2rot=False
        )
        pred_verts_np, pred_joints_np = (
            utils.copy2cpu(pred_verts),
            utils.copy2cpu(pred_joints),
        )

        targ_poses_torch = torch.from_numpy(targ_poses).float()
        targ_verts, targ_joints, _ = smpl_helpers.my_lbs(
            self.smpl_model, targ_poses_torch, pose2rot=False
        )
        targ_verts_np, targ_joints_np = (
            utils.copy2cpu(targ_verts),
            utils.copy2cpu(targ_joints),
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
            from sparsesuit.utils import visualization

            verts = [targ_verts_np, np.asarray(pred_verts_aligned)]
            vertex_colors = ["green", "gray"]
            joints = [targ_joints_np, np.asarray(pred_joints_aligned)]

            # show vertex indices of sensors used in training
            root_ids = [0, 1] if self.sensor_config == "SSP" else [0]
            train_sens_ids = root_ids + [ind + len(root_ids) for ind in self.sens_ind]
            if self.sensor_config == "SSP":
                sens_verts = list(sensors.SENS_VERTS_SSP.values())
            else:
                sens_verts = list(sensors.SENS_VERTS_MVN.values())
            train_sens_verts = [sens_verts[idx] for idx in train_sens_ids]
            sensors_vis = [
                np.asarray(pred_verts_aligned)[:, train_sens_verts],
            ]

            visualization.vis_smpl(
                faces=self.smpl_model.faces,
                vertices=verts,
                vertex_colors=vertex_colors,
                # joints=joints,
                sensors=sensors_vis,
                play_frames=500,
                playback_speed=0.3,
                add_captions=True,
                side_by_side=False,
                fps=self.ds_fps,
            )

        return mm * 100, pred_joints_sel  # convert m to cm


def joint_angle_error(predicted_pose_params, target_pose_params):
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
        target_pose_params.shape[0] == seq_length and target_pose_params.shape[1] == dof
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


@hydra.main(config_path="conf", config_name="evaluation")
def do_evaluation(cfg: DictConfig):
    eval = Evaluator(cfg=cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_evaluation()
