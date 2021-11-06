import os

import numpy as np
import torch

from sparsesuit.learning import evaluation
from sparsesuit.constants import paths, sensors
from sparsesuit.utils import smpl_helpers, visualization, utils

smpl_model = smpl_helpers.load_smplx()


def get_data(path):
    files = os.listdir(path)
    data = []
    for file in files:
        with np.load(os.path.join(path, file)) as file_data:
            data.append(dict(file_data)["pose"])
    return data


def visualize(pred, targ):
    pred_full = smpl_helpers.extract_from_norm_ds(pred)
    targ_full = smpl_helpers.extract_from_norm_ds(targ)

    pred_verts, _, _ = smpl_helpers.my_lbs(smpl_model, pred_full)
    targ_verts, _, _ = smpl_helpers.my_lbs(smpl_model, targ_full)

    verts = [utils.copy2cpu(targ_verts), utils.copy2cpu(pred_verts)]
    vertex_colors = ["green", "gray"]

    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=verts,
        vertex_colors=vertex_colors,
        # joints=joints,
        # sensors=sensors_vis,
        play_frames=500,
        playback_speed=0.3,
        # add_captions=True,
        side_by_side=False,
        fps=100,
    )


def visualize_with_global_oris(pred, targ, pred_g, targ_g):
    batch_size = pred.shape[0]
    pose_dim = 9

    # add rotation matrices (identity) for missing SMPL-X joints
    padding_len = sensors.NUM_SMPLX_JOINTS - sensors.NUM_SMPL_JOINTS
    padding = np.tile(np.identity(3), [batch_size, padding_len, 1, 1])
    padding_flat = np.reshape(padding, [batch_size, -1])

    pred_poses_padded = np.concatenate((pred[:, : 24 * pose_dim], padding_flat), axis=1)
    pred_poses = torch.Tensor(pred_poses_padded.reshape([batch_size, -1, 3, 3]))

    targ_poses_padded = np.concatenate((targ[:, : 24 * pose_dim], padding_flat), axis=1)
    targ_poses = torch.Tensor(targ_poses_padded.reshape([batch_size, -1, 3, 3]))

    pred_verts, pred_joints, _ = smpl_helpers.my_lbs(
        smpl_model, pred_poses, pose2rot=False
    )
    targ_verts, targ_joints, _ = smpl_helpers.my_lbs(
        smpl_model, targ_poses, pose2rot=False
    )

    verts = [utils.copy2cpu(targ_verts), utils.copy2cpu(pred_verts)]
    joints = [utils.copy2cpu(targ_joints[:, :22]), utils.copy2cpu(pred_joints[:, :22])]
    poses = [
        targ_g.reshape(batch_size, -1, 3, 3)[:, :22],
        pred_g[:].reshape(batch_size, -1, 3, 3)[:, :22],
    ]
    vertex_colors = ["green", "orange"]

    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=verts,
        vertex_colors=vertex_colors,
        joints=joints,
        pose=poses,
        # sensors=sensors_vis,
        play_frames=500,
        playback_speed=0.3,
        # add_captions=True,
        side_by_side=True,
        fps=100,
    )


if __name__ == "__main__":
    # load VICON and STUDIO test sets
    vicon_path = os.path.join(paths.DATASET_PATH, "RKK_VICON/SSP_fps100_n/test")
    studio_path = os.path.join(paths.DATASET_PATH, "RKK_STUDIO/SSP_fps100_n/test")

    vicon_data = get_data(vicon_path)
    studio_data = get_data(studio_path)

    for vicon_i, studio_i in zip(vicon_data, studio_data):

        # visualize(studio_i, vicon_i)

        # expand reduced (15/19 joints) to full 24 smpl joints
        vicon_full = smpl_helpers.smpl_reduced_to_full(vicon_i, sensors.SMPL_SSP_JOINTS)
        studio_full = smpl_helpers.smpl_reduced_to_full(
            studio_i, sensors.SMPL_SSP_JOINTS
        )

        # rotate root for upright human-beings
        vicon_full[:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]
        studio_full[:, :9] = [0, 0, 1, 1, 0, 0, 0, 1, 0]

        # compute angular error for each asset in test sets
        pred_g = smpl_helpers.smpl_rot_to_global(studio_full)
        targ_g = smpl_helpers.smpl_rot_to_global(vicon_full)

        # visualize_with_global_oris(studio_full, vicon_full, pred_g, targ_g)

        # compute angle error for all SMPL joints based on global joint orientation
        # this way the error is not propagated along the kinematic chain
        angle_err = evaluation.joint_angle_error(pred_g, targ_g)
        mean_joint_ang_err = np.mean(angle_err, axis=0)
        rel_angle_err = angle_err[sensors.ANG_EVAL_JOINTS]
        mean_ang_err = np.mean(angle_err)
        print(mean_ang_err)
