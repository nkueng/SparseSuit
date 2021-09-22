"""First normalizes imu data w.r.t. root sensor and saves asset locally. Then loads normalized asset and scales for
zero mean & unit variance.
Input: pickle or npz files with N IMU orientations and accelerations as well as SMPL poses
for 24 joints
Output: normalized zero-mean unit-variance input and target vectors of N-1 IMUs ready for learning """
import logging
import os
import pickle as pkl
import shutil
import tarfile
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from smplx import lbs
from welford import Welford

from sparsesuit.constants import sensors, paths
from sparsesuit.utils import smpl_helpers, utils


class Normalizer:
    def __init__(self, cfg):
        self.config = cfg
        self.dataset_type = cfg.dataset.type
        self.sens_config = cfg.dataset.config
        self.tar = cfg.dataset.tar
        self.visualize = cfg.visualize
        self.debug = cfg.debug

        # reproducibility
        utils.make_deterministic(14)

        # choose dataset folder based on params
        if self.dataset_type == "synthetic":
            src_folder = paths.AMASS_PATH

        elif self.dataset_type == "real":
            src_folder = paths.DIP_17_PATH
            # dict mapping subject and motion types to datasets
            self.dataset_mapping = {
                "s_09": "test",
                "s_10": "test",
                "s_01/05": "validation",
                "s_03/05": "validation",
                "s_07/04": "validation",
            }

        else:
            raise NameError("Invalid dataset configuration. Aborting!")

        # adapt to sensor configuration
        if self.debug and self.dataset_type == "synthetic":
            src_folder += "_debug"

        if self.sens_config == "SSP":
            if self.dataset_type == "synthetic":
                src_folder += "_SSP"
            sens_names = sensors.SENS_NAMES_SSP
            self.pred_trgt_joints = sensors.SMPL_SSP_JOINTS

        elif self.sens_config == "MVN":
            if self.dataset_type == "synthetic":
                src_folder += "_MVN"
            sens_names = sensors.SENS_NAMES_MVN
            self.pred_trgt_joints = sensors.SMPL_DIP_JOINTS

        else:
            raise NameError("Invalid dataset configuration. Aborting!")

        sensor_ids = np.arange(0, len(sens_names))

        if "noise" in cfg.dataset:
            if cfg.dataset.noise:
                src_folder += "_noisy"

        self.src_dir = src_folder
        self.norm_dir = self.src_dir + "_n"
        self.trgt_dir = self.src_dir + "_nn"
        self.dataset_names = ["training", "validation", "test"]
        self.sens_names = sens_names
        self.sensor_ids = sensor_ids

        # logger setup
        log_level = logging.DEBUG if cfg.debug else logging.INFO
        self.logger = utils.configure_logger(
            name="normalization", log_path=self.trgt_dir, level=log_level
        )
        self.logger.info("\n\nNormalization\n*******************\n")

    def normalize_dataset(self):
        assert os.path.exists(
            self.src_dir
        ), "Source directory {} does not exist! Check spelling!".format(self.src_dir)

        if self.visualize:
            # load neutral smpl model for evaluation of normalized sensor data
            smpl_model = smpl_helpers.load_smplx("neutral")

        # set up objects for online mean and variance computation
        stats_ori = Welford()
        stats_acc = Welford()
        stats_pose = Welford()

        # determine root sensor depending on sensor config
        if self.sens_config == "SSP":
            # the SSP has no root sensor, so a virtual sensor is derived from the two hip sensors
            self.root_indx = [
                self.sens_names.index("left_pelvis"),
                self.sens_names.index("right_pelvis"),
            ]
            self.sens_names.remove("left_pelvis")
            self.sens_names.remove("right_pelvis")
        elif self.sens_config == "MVN":
            self.root_indx = self.sens_names.index("pelvis")
            self.sens_names.remove("pelvis")
        else:
            raise NameError("Invalid sensor configuration. Aborting!")

        # iterate over all subdirectories and files in given path
        for subdir, dirs, files in os.walk(self.src_dir):
            for file in files:
                # check for correct file format
                if file.startswith(".") or not (
                    file.endswith(".npz") or file.endswith(".pkl")
                ):
                    continue

                # load npz or pickle in subdirectory
                data_in = {}
                filepath = os.path.join(subdir, file)
                if filepath.endswith(".npz"):
                    with np.load(filepath, allow_pickle=True) as data:
                        data_in = dict(data)
                elif filepath.endswith(".pkl"):
                    with open(filepath, "rb") as fin:
                        data_in = pkl.load(fin, encoding="latin1")

                # extract relevant data
                try:
                    # AMASS
                    # IMU orientation matrices of sensors
                    oris = data_in["imu_ori"]
                    # IMU acceleration vectors of sensors
                    accs = data_in["imu_acc"]
                    # flattened pose vector of 24 SMPL joints (as rot mat)
                    poses = data_in["gt"]
                    # poses given in axis-angle format need transformation to rot matrix
                    pose2rot = True
                except KeyError:
                    try:
                        # DIP-IMU
                        # IMU orientation matrices of sensors
                        oris = np.array(data_in["ori"])
                        # IMU acceleration vectors of sensors
                        accs = np.array(data_in["acc"])
                        # flattened pose vector of 15 SMPL joints in rot-matrix format
                        poses = np.array(data_in["poses"])
                        pose2rot = False
                    except KeyError:
                        self.logger.info(
                            "Input data does not have the right fields. Skipping!"
                        )
                        continue
                seq_length = len(accs)

                # exit if data is empty
                if seq_length == 0:
                    continue

                subject_i = subdir.split(self.src_dir + "/")[1]
                motion_type_i = file.split(".")[0]

                # determine name of file for normalized asset
                norm_dir_name = os.path.join(self.norm_dir, subject_i)
                out_filename = os.path.join(norm_dir_name, motion_type_i)

                # exit if normalized asset already exists -> makes no sense as we need stats from all assets
                # if Path(out_filename + ".npz").exists():
                #     self.logger.info(
                #         "Skipping normalization of {}/{} as it already exists.".format(
                #             subject_i, motion_type_i
                #         )
                #     )
                #     continue

                # remove all frames with NANs
                acc_sum = np.sum(np.reshape(accs, (seq_length, -1)), axis=1)
                ori_sum = np.sum(np.reshape(oris, (seq_length, -1)), axis=1)
                nan_mask = np.isfinite(acc_sum) & np.isfinite(ori_sum)
                # nan_count = seq_length - np.sum(nan_mask)
                new_seq_length = np.sum(nan_mask)
                accs_clean = accs[nan_mask]
                oris_clean = oris[nan_mask]

                # extract only sensor data desired in input and in convention ordering
                accs_sorted = accs_clean[:, self.sensor_ids]
                oris_sorted = oris_clean[:, self.sensor_ids]

                # normalize orientation and acceleration for each frame w.r.t. root
                if self.sens_config == "SSP":
                    # get root sensor orientations
                    root_oris = np.reshape(oris_sorted[:, self.root_indx], [-1, 9])
                    # transform to angle-axis
                    root_aas = np.reshape(
                        utils.rot_matrix_to_aa(root_oris), [new_seq_length, 2, 3]
                    )
                    # compute mean
                    root_aa = np.mean(root_aas, axis=1)
                    # transform back to rotation matrix
                    root_ori = np.reshape(
                        utils.aa_to_rot_matrix(root_aa), [-1, 1, 3, 3]
                    )
                    # get inverse to normalize other sensors
                    root_ori_inv = np.transpose(root_ori, [0, 1, 3, 2])

                    # get hip sensor accelerations
                    hip_accs = accs_sorted[:, [self.root_indx]]
                    # average to get virtual sensor acceleration to normalize other sensors
                    root_acc = np.mean(hip_accs, axis=2)
                elif self.sens_config == "MVN":
                    root_ori_inv = np.transpose(
                        oris_sorted[:, [self.root_indx]], [0, 1, 3, 2]
                    )
                    root_acc = accs_sorted[:, [self.root_indx]]

                # normalize all orientations
                oris_norm = np.matmul(root_ori_inv, oris_sorted)
                # discard root orientation which is always identity
                oris_norm_wo_root = np.delete(oris_norm, self.root_indx, axis=1)
                oris_vec = np.reshape(oris_norm_wo_root, (new_seq_length, -1))

                # normalize all accelerations
                accs_norm = accs_sorted - root_acc
                accs_norm = np.matmul(root_ori_inv, accs_norm[..., np.newaxis])
                # discard root acceleration which is always zero
                accs_norm_wo_root = np.delete(accs_norm, self.root_indx, axis=1)
                accs_vec = np.reshape(accs_norm_wo_root, (new_seq_length, -1))

                # convert poses to rotation matrix format if necessary
                poses_vec = poses[nan_mask]
                if pose2rot:
                    # select target joints
                    poses_vec_sel = np.reshape(poses_vec, (new_seq_length, -1, 3))[
                        :, self.pred_trgt_joints
                    ]
                    # convert pose data from angle-axis vectors to rotation matrices
                    poses_vec_torch = torch.from_numpy(
                        np.reshape(poses_vec_sel, (-1, 3))
                    )
                    poses_vec_rot = (
                        lbs.batch_rodrigues(poses_vec_torch).detach().numpy()
                    )
                    poses_vec_sel = np.reshape(poses_vec_rot, (new_seq_length, -1))
                else:
                    # poses given in rot matrix format already, only select relevant
                    poses_vec_sel = np.reshape(poses_vec, [new_seq_length, -1, 3, 3])[
                        :, self.pred_trgt_joints
                    ]
                    poses_vec_sel = np.reshape(poses_vec_sel, [new_seq_length, -1])

                # save normalized orientations, accelerations and poses locally
                Path(norm_dir_name).mkdir(parents=True, exist_ok=True)

                # write npz file
                data_out = {
                    "orientation": oris_vec,
                    "acceleration": accs_vec,
                    "pose": poses_vec_sel,
                }
                with open(out_filename + ".npz", "wb") as fout:
                    np.savez_compressed(fout, **data_out)

                # compute statistics for this asset
                for idx in range(new_seq_length):
                    stats_ori.add(oris_vec[idx])
                    stats_acc.add(accs_vec[idx])
                    stats_pose.add(poses_vec_sel[idx])

                self.logger.info("Normalized {}/{}".format(subject_i, motion_type_i))

                if self.visualize:
                    from sparsesuit.utils import visualization

                    frames = 300
                    frame_range = range(2 * frames, 3 * frames)
                    vis_poses = np.tile(
                        np.eye(3), [frames, sensors.NUM_SMPLX_JOINTS, 1, 1]
                    )
                    vis_poses[:, self.pred_trgt_joints] = np.reshape(
                        poses_vec_sel[frame_range], [frames, -1, 3, 3]
                    )
                    # vis_poses[:, 0] = np.array([[1.20919958, 1.20919958, 1.20919958]])
                    poses_torch = torch.from_numpy(vis_poses).float()
                    # compute skin vertices from SMPL pose
                    verts, joints, _ = smpl_helpers.my_lbs(
                        model=smpl_model,
                        pose=poses_torch,
                        pose2rot=False,
                    )

                    verts_np, joints_np = (
                        utils.copy2cpu(verts),
                        utils.copy2cpu(joints)[:, : sensors.NUM_SMPL_JOINTS],
                    )

                    if self.sens_config == "SSP":
                        vertex_ids = [
                            sensors.SENS_VERTS_SSP[sensor_name]
                            for sensor_name in self.sens_names
                        ]
                    else:
                        vertex_ids = [
                            sensors.SENS_VERTS_MVN[sensor_name]
                            for sensor_name in self.sens_names
                        ]

                    vertices = [verts_np]
                    vis_sensors = [verts_np[:, vertex_ids]]
                    orientations = [np.squeeze(oris_norm_wo_root[frame_range])]
                    accelerations = [np.squeeze(accs_norm_wo_root[frame_range])]
                    visualization.vis_smpl(
                        faces=smpl_model.faces,
                        vertices=vertices,
                        sensors=vis_sensors,
                        accs=accelerations,
                        oris=orientations,
                        play_frames=frames,
                        playback_speed=0.1,
                        add_captions=False,
                    )

        # compute stats
        mean_ori = stats_ori.mean
        mean_acc = stats_acc.mean
        mean_pose = stats_pose.mean
        std_ori = np.sqrt(stats_ori.var_p)
        std_acc = np.sqrt(stats_acc.var_p)
        std_pose = np.sqrt(stats_pose.var_p)

        ori_dim = len(mean_ori)
        acc_dim = len(mean_acc)
        pose_dim = len(mean_pose)

        # iterate over all normalized assets
        # using list since it is mutable
        seq_count_train, seq_count_test, seq_count_valid = (
            [0],
            [0],
            [0],
        )

        seq_count = None
        for subdir, dirs, files in os.walk(self.norm_dir):
            for file in files:
                file_path = os.path.join(subdir, file)
                with np.load(file_path, allow_pickle=True) as data:
                    data_dict = dict(data)

                # scale data for zero mean and unit variance
                ori_scaled = (data_dict["orientation"] - mean_ori) / std_ori
                acc_scaled = (data_dict["acceleration"] - mean_acc) / std_acc
                pose_scaled = (data_dict["pose"] - mean_pose) / std_pose

                subject_i = subdir.split(self.norm_dir)[1]
                motion_type_i = file.split(".")[0]

                # save DIP-IMU different from AMASS
                if self.dataset_type == "real":
                    # DIP-IMU: distinguish training, evaluation and test set like authors did
                    # check for subjects in test set
                    dataset_name = self.dataset_mapping.get(subject_i)
                    if dataset_name is None:
                        # check for motion types in validation set
                        dataset_name = self.dataset_mapping.get(
                            subject_i + "/" + motion_type_i
                        )

                        if dataset_name is None:
                            # if subject + motion_type is not in dict, it belongs to the training dataset
                            dataset_name = "training"
                elif self.dataset_type == "synthetic":
                    # split AMASS into 96.8/3/0.2 training/validation/testing
                    train_split = 0.968
                    validation_split = 0.998
                    rand = np.random.random()
                    if rand <= train_split:
                        dataset_name = self.dataset_names[0]
                    elif rand <= validation_split:
                        dataset_name = self.dataset_names[1]
                    else:
                        dataset_name = self.dataset_names[2]

                # save in webdataset format
                file_id_i = subject_i + "_" + motion_type_i

                ori_buffer = []
                acc_buffer = []
                pose_buffer = []
                if dataset_name == "training":
                    # split sequences into chunks of N frames, discard incomplete last one
                    N = 300
                    seq_length = len(ori_scaled)
                    num_compl_seq = seq_length // N

                    ori_chunked = np.reshape(
                        ori_scaled[: N * num_compl_seq], (-1, N, ori_dim)
                    )
                    acc_chunked = np.reshape(
                        acc_scaled[: N * num_compl_seq], (-1, N, acc_dim)
                    )
                    pose_chunked = np.reshape(
                        pose_scaled[: N * num_compl_seq], (-1, N, pose_dim)
                    )

                    for i in range(num_compl_seq):
                        ori_buffer.append(ori_chunked[i])
                        acc_buffer.append(acc_chunked[i])
                        pose_buffer.append(pose_chunked[i])

                    seq_count = seq_count_train
                else:
                    # save sequence as a whole
                    ori_buffer.append(ori_scaled)
                    acc_buffer.append(acc_scaled)
                    pose_buffer.append(pose_scaled)

                    seq_count = (
                        seq_count_test if dataset_name == "test" else seq_count_valid
                    )

                for ori_i, acc_i, pose_i in zip(ori_buffer, acc_buffer, pose_buffer):
                    dir_name = os.path.join(
                        self.trgt_dir, dataset_name, str(seq_count[0])
                    )
                    Path(dir_name).mkdir(parents=True, exist_ok=True)

                    if self.tar:
                        # write npy files
                        input_filename = dir_name + "/" + file_id_i + ".ori.npy"
                        with open(input_filename, "wb") as f:
                            np.save(f, ori_i)

                        output_filename = dir_name + "/" + file_id_i + ".acc.npy"
                        with open(output_filename, "wb") as f:
                            np.save(f, acc_i)

                        output_filename = dir_name + "/" + file_id_i + ".pose.npy"
                        with open(output_filename, "wb") as f:
                            np.save(f, pose_i)
                    else:
                        # write all data to one file only
                        out_dict = {
                            "ori": ori_i,
                            "acc": acc_i,
                            "pose": pose_i,
                        }
                        output_filename = dir_name + subject_i + "_" + file
                        with open(output_filename, "wb") as fout:
                            np.savez_compressed(fout, **out_dict)

                    seq_count[0] += 1

                self.logger.info("Processed {}/{}".format(subject_i, motion_type_i))

        # save stats with dataset
        stats_dict = {
            "ori_mean": mean_ori,
            "ori_std": std_ori,
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "pose_mean": mean_pose,
            "pose_std": std_pose,
            "train_len": seq_count_train[0],
            "valid_len": seq_count_valid[0],
            "test_len": seq_count_test[0],
        }

        Path(self.trgt_dir).mkdir(parents=True, exist_ok=True)
        stats_dir = os.path.join(self.trgt_dir, "stats.npz")
        with open(stats_dir, "wb") as fout:
            np.savez_compressed(fout, **stats_dict)

        # save config with dataset
        self.write_config(seq_count_train[0], seq_count_valid[0], seq_count_test[0])

        # delete directory with normalized assets
        shutil.rmtree(self.norm_dir)

        # convert dataset directories to .tar and delete directories
        if self.tar:
            self.logger.info("Converting to .tar...")
            for dataset in self.dataset_names:
                ds_path = os.path.join(self.trgt_dir, dataset)
                with tarfile.open(ds_path + ".tar", "w") as tar:
                    tar.add(ds_path, arcname=os.path.basename(ds_path))
                # shutil.rmtree(ds_path)

    def write_config(self, train_count, valid_count, test_count):
        # load config of source dataset
        src_config = utils.load_config(self.src_dir)

        # compile info about asset counts
        asset_info = {
            "total": train_count + valid_count + test_count,
            "training": train_count,
            "validation": valid_count,
            "test": test_count,
        }

        # update source dataset config
        src_config.dataset.sensors = self.sens_names
        src_config.dataset.assets = asset_info
        src_config.dataset.pred_trgt_joints = self.pred_trgt_joints

        # dump updated config to normalized dataset
        utils.write_config(self.trgt_dir, src_config)


@hydra.main(config_path="conf", config_name="normalization")
def do_normalization(cfg: DictConfig):
    norm = Normalizer(cfg=cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_normalization()
