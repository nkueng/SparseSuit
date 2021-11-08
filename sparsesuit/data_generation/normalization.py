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
        self.ds_config = cfg.dataset
        self.ds_source = cfg.dataset.source
        self.sens_config = cfg.dataset.sensor_config
        self.tar = False
        self.visualize = cfg.visualize
        self.debug = cfg.debug
        self.skip_existing = cfg.skip_existing

        # set up objects for online mean and variance computation
        self.stats = {}
        self.stats_ori = Welford()
        self.stats_acc = Welford()
        self.stats_pose = Welford()

        # keep track of sequences for different kinds of dataset
        self.seq_count_train = 0
        self.seq_count_test = 0
        self.seq_count_valid = 0

        # reproducibility
        utils.make_deterministic(14)

        # choose dataset folder based on params
        src_folder = utils.ds_path_from_config(cfg.dataset, "normalization", self.debug)
        if self.ds_source == "AMASS":
            # src_folder = paths.AMASS_PATH
            self.dataset_mapping = paths.AMASS_MAPPING
        elif self.ds_source == "DIP_IMU":
            # src_folder = paths.DIP_17_PATH
            # dict mapping subject and motion types to datasets
            self.dataset_mapping = paths.DIP_IMU_MAPPING
        elif self.ds_source == "RKK_STUDIO":
            # src_folder = paths.RKK_STUDIO_19_PATH
            self.dataset_mapping = paths.RKK_STUDIO_MAPPING
        elif self.ds_source == "RKK_VICON":
            self.dataset_mapping = paths.RKK_VICON_MAPPING
        else:
            raise NameError("Invalid dataset configuration. Aborting!")

        # adapt to sensor configuration
        # if self.debug and self.dataset_type == "synthetic":
        #     src_folder += "_debug"
        if self.sens_config == "SSP":
            # if self.dataset_type == "synthetic":
            #     src_folder += "_SSP"
            self.sens_names = sensors.SENS_NAMES_SSP.copy()
            self.sensor_ids = np.arange(0, len(self.sens_names))
            # the SSP has no root sensor, so a virtual sensor is derived from the two hip sensors
            self.root_idx = [
                self.sens_names.index("left_pelvis"),
                self.sens_names.index("right_pelvis"),
            ]
            self.sens_names.remove("left_pelvis")
            self.sens_names.remove("right_pelvis")
            self.pred_trgt_joints = sensors.SMPL_SSP_JOINTS

        elif self.sens_config == "MVN":
            # if self.dataset_type == "synthetic":
            #     src_folder += "_MVN"
            self.sens_names = sensors.SENS_NAMES_MVN.copy()
            self.sensor_ids = np.arange(0, len(self.sens_names))
            self.root_idx = self.sens_names.index("pelvis")
            self.sens_names.remove("pelvis")
            self.pred_trgt_joints = sensors.SMPL_SSP_JOINTS

        else:
            raise NameError("Invalid dataset configuration. Aborting!")

        self.src_dir = src_folder
        norm_folder = utils.ds_path_from_config(cfg.dataset, "training", self.debug)
        self.norm_dir = norm_folder
        self.trgt_dir = norm_folder + "n"
        # clean up target directories to avoid old leftovers
        if os.path.isdir(self.norm_dir):
            shutil.rmtree(self.norm_dir)
        if os.path.isdir(self.trgt_dir):
            shutil.rmtree(self.trgt_dir)
        self.dataset_names = ["training", "validation", "test"]

        # if requested, keep specific motions away from RKK_ datasets
        self.rkk_fraction = 0
        if "rkk_fraction" in cfg.dataset:
            self.rkk_fraction = cfg.dataset.rkk_fraction

        # visualization setup
        if self.visualize:
            # load neutral smpl model for evaluation of normalized sensor data
            self.smpl_model = smpl_helpers.load_smplx("neutral")

        # logger setup
        log_level = logging.DEBUG if cfg.debug else logging.INFO
        self.logger = utils.configure_logger(
            name="normalization", log_path=self.norm_dir, level=log_level
        )
        self.logger.info("\n\nNormalization\n*******************\n")

    def normalize_asset(self, oris, accs, poses, pose2rot):
        seq_length = len(oris)
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
            root_oris = np.reshape(oris_sorted[:, self.root_idx], [-1, 9])
            # transform to angle-axis
            root_aas = np.reshape(
                utils.rot_matrix_to_aa(root_oris), [new_seq_length, 2, 3]
            )
            # compute mean
            root_aa = np.mean(root_aas, axis=1)
            # transform back to rotation matrix
            root_ori = np.reshape(utils.aa_to_rot_matrix(root_aa), [-1, 1, 3, 3])
            # get inverse to normalize other sensors
            root_ori_inv = np.transpose(root_ori, [0, 1, 3, 2])

            # get hip sensor accelerations
            hip_accs = accs_sorted[:, [self.root_idx]]
            # average to get virtual sensor acceleration to normalize other sensors
            root_acc = np.mean(hip_accs, axis=2)
        elif self.sens_config == "MVN":
            root_ori_inv = np.transpose(oris_sorted[:, [self.root_idx]], [0, 1, 3, 2])
            root_acc = accs_sorted[:, [self.root_idx]]

        # normalize all orientations
        oris_norm = np.matmul(root_ori_inv, oris_sorted)
        # discard root orientation which is always identity
        oris_norm_wo_root = np.delete(oris_norm, self.root_idx, axis=1)
        oris_vec = np.reshape(oris_norm_wo_root, (new_seq_length, -1))

        # normalize all accelerations
        accs_norm = accs_sorted - root_acc
        accs_norm = np.matmul(root_ori_inv, accs_norm[..., np.newaxis])
        # discard root acceleration which is always zero
        accs_norm_wo_root = np.delete(accs_norm, self.root_idx, axis=1)
        accs_vec = np.reshape(accs_norm_wo_root, (new_seq_length, -1))

        # convert poses to rotation matrix format if necessary
        poses_vec = poses[nan_mask]
        if pose2rot:
            # select target joints
            poses_vec_sel = np.reshape(poses_vec, [new_seq_length, -1, 3])[
                :, self.pred_trgt_joints
            ]
            # convert pose data from angle-axis vectors to rotation matrices
            poses_vec_torch = torch.from_numpy(np.reshape(poses_vec_sel, [-1, 3]))
            poses_vec_rot = lbs.batch_rodrigues(poses_vec_torch).detach().numpy()
            poses_vec_sel = np.reshape(poses_vec_rot, [new_seq_length, -1])
        else:
            # poses given in rot matrix format already, only select relevant
            poses_vec_sel = np.reshape(poses_vec, [new_seq_length, -1, 3, 3])[
                :, self.pred_trgt_joints
            ]
            poses_vec_sel = np.reshape(poses_vec_sel, [new_seq_length, -1])

        # dump normalized vectors into dict
        data_out = {
            "ori": oris_vec,
            "acc": accs_vec,
            "pose": poses_vec_sel,
        }
        return data_out

    def extract_raw_data(self, file):
        # load npz or pickle in subdirectory
        if file.endswith(".npz"):
            with np.load(file, allow_pickle=True) as data:
                data_in = dict(data)
        elif file.endswith(".pkl"):
            with open(file, "rb") as fin:
                data_in = pkl.load(fin, encoding="latin1")

        # extract relevant data
        try:
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
                # legacy convention for DIP_IMU_6
                # IMU orientation matrices of sensors
                oris = np.array(data_in["ori"])
                # IMU acceleration vectors of sensors
                accs = np.array(data_in["acc"])
                # flattened pose vector of 15 SMPL joints in rot-matrix format
                poses = np.array(data_in["poses"])
                pose2rot = False
            except KeyError:
                raise AttributeError(
                    "{} does not have the right fields. Skipping!".format(file)
                )
        seq_length = len(accs)

        # skip if data is empty
        if seq_length == 0:
            raise AttributeError("Empty sequence {}. Skipping!".format(file))

        # skip if data vectors don't have same length
        if len(accs) != len(oris) or len(accs) != len(poses):
            raise AttributeError(
                "Vectors of different lengths in {}. Skipping!".format(file)
            )

        return oris, accs, poses, pose2rot

    def visualize_norm_asset(self, norm_asset):
        from sparsesuit.utils import visualization

        poses = norm_asset["pose"]
        seq_len = len(poses)
        oris = norm_asset["ori"].reshape([seq_len, -1, 3, 3])
        accs = norm_asset["acc"].reshape([seq_len, -1, 3])

        frames = 400
        frame_range = range(0 * frames, 1 * frames)
        vis_poses = np.tile(np.eye(3), [frames, sensors.NUM_SMPLX_JOINTS, 1, 1])
        vis_poses[:, self.pred_trgt_joints] = np.reshape(
            poses[frame_range], [frames, -1, 3, 3]
        )
        # vis_poses[:, 0] = np.array([[1.20919958, 1.20919958, 1.20919958]])
        poses_torch = torch.from_numpy(vis_poses).float()
        # compute skin vertices from SMPL pose
        verts, joints, _ = smpl_helpers.my_lbs(
            model=self.smpl_model,
            pose=poses_torch,
            pose2rot=False,
        )

        verts_np, joints_np = (
            utils.copy2cpu(verts),
            utils.copy2cpu(joints)[:, : sensors.NUM_SMPL_JOINTS],
        )

        if self.sens_config == "SSP":
            vertex_ids = [
                sensors.SENS_VERTS_SSP[sensor_name] for sensor_name in self.sens_names
            ]
        else:
            vertex_ids = [
                sensors.SENS_VERTS_MVN[sensor_name] for sensor_name in self.sens_names
            ]

        vertices = [verts_np]
        vis_sensors = [verts_np[:, vertex_ids]]
        orientations = [np.squeeze(oris[frame_range])]
        accelerations = [np.squeeze(accs[frame_range])]
        visualization.vis_smpl(
            faces=self.smpl_model.faces,
            vertices=vertices,
            sensors=vis_sensors,
            accs=accelerations,
            oris=orientations,
            play_frames=frames,
            playback_speed=0.3,
            add_captions=False,
        )

    def save_asset_stats(self, asset):
        # compute statistics for this asset
        oris = asset["ori"]
        accs = asset["acc"]
        poses = asset["pose"]
        for ori_i, acc_i, pose_i in zip(oris, accs, poses):
            self.stats_ori.add(ori_i)
            self.stats_acc.add(acc_i)
            self.stats_pose.add(pose_i)

    def determine_dataset(self, subject_i, motion_type_i):
        # determine which dataset (training/validation/testing) this assets belongs to
        if len(self.dataset_mapping) == 0:
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
        else:
            # use dataset-mapping to find which dataset this assets belongs to
            dataset_name = self.dataset_mapping.get(subject_i)
            if dataset_name is None:
                # check for gait sequences in RKK_STUDIO
                # if self.ds_source == "RKK_STUDIO":
                #     if "gait" in str.lower(motion_type_i):
                #         return "test"

                # check for motion types in validation set
                dataset_name = self.dataset_mapping.get(subject_i + "/" + motion_type_i)
                if dataset_name is None:
                    if self.ds_source == "AMASS":
                        # split AMASS into 97/3 training/validation
                        train_split = 0.97
                        rand = np.random.random()
                        if rand <= train_split:
                            dataset_name = "training"
                        else:
                            dataset_name = "validation"
                    else:
                        # if subject + motion_type is not in dict, it belongs to the training dataset
                        dataset_name = "training"

        return dataset_name

    def chunk_train_assets(self, asset):
        # split sequences into chunks of chunk_len frames, discard incomplete last one
        ori = asset["ori"]
        acc = asset["acc"]
        pose = asset["pose"]

        chunk_len = self.config.train_chunk_len
        seq_length = len(ori)
        num_compl_seq = seq_length // chunk_len

        ori_ch = np.reshape(
            ori[: chunk_len * num_compl_seq],
            (-1, chunk_len, ori.shape[1]),
        )
        acc_ch = np.reshape(
            acc[: chunk_len * num_compl_seq],
            (-1, chunk_len, acc.shape[1]),
        )
        pose_ch = np.reshape(
            pose[: chunk_len * num_compl_seq],
            (-1, chunk_len, pose.shape[1]),
        )

        chunked_data = []
        for i in range(num_compl_seq):
            chunk_i = {"ori": ori_ch[i], "acc": acc_ch[i], "pose": pose_ch[i]}
            chunked_data.append(chunk_i)
        return chunked_data

    def save_asset(self, norm_data, file_name, dataset_name):
        if dataset_name == "training":
            # chunk asset up if in training dataset
            chunked_data = self.chunk_train_assets(norm_data)

            # save chunks and keep track of training asset count
            for chunk in chunked_data:
                folder_path = os.path.join(self.norm_dir, "training")
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                prefix = str(self.seq_count_train).zfill(5)
                out_name = os.path.join(folder_path, prefix + "_" + file_name)
                with open(out_name, "wb") as fout:
                    np.savez_compressed(fout, **chunk)
                self.seq_count_train += 1

        else:
            # this asset goes unchanged into valid or test dataset
            if dataset_name == "validation":
                folder_path = os.path.join(self.norm_dir, "validation")
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                prefix = str(self.seq_count_valid).zfill(5)
                out_name = os.path.join(folder_path, prefix + "_" + file_name)
                with open(out_name, "wb") as fout:
                    np.savez_compressed(fout, **norm_data)
                    self.seq_count_valid += 1

            elif dataset_name == "test":
                folder_path = os.path.join(self.norm_dir, "test")
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                prefix = str(self.seq_count_test).zfill(5)
                out_name = os.path.join(folder_path, prefix + "_" + file_name)
                with open(out_name, "wb") as fout:
                    np.savez_compressed(fout, **norm_data)
                self.seq_count_test += 1

            else:
                raise NameError("dataset_name could not be found. Aborting!")

    def compute_dataset_stats(self):
        # check if stats were previously computed
        stats_path = os.path.join(self.norm_dir, "stats.npz")
        if os.path.isfile(stats_path) and self.skip_existing:
            # load statistics
            with np.load(stats_path) as stats_data:
                self.stats = dict(stats_data)
            # skip statistics computation
            self.logger.info(
                "Statistics were previously computed. Using them for normalization."
            )
            return

        # iterate over all subdirectories and files in given path
        file_list = []
        for root, dirs, files in os.walk(self.src_dir):
            for file in files:
                # check for correct file format
                if file.startswith("."):
                    continue
                if file.endswith(".npz") or file.endswith(".pkl"):
                    file_list.append(os.path.join(root, file))

        file_list = sorted(file_list)
        # DEBUG
        # file_list = file_list[245:]
        for file in file_list:
            # DEBUG
            # file = file_list[260]

            # load and extract raw data from this file
            try:
                raw_data = self.extract_raw_data(file)
            except AttributeError:
                continue

            # skip if less than 300 frames
            if len(raw_data[0]) < 300:
                print("Sequence {} is less than 300 frames. Skipping!".format(file))
                continue

            # normalize the extracted motion asset
            norm_data = self.normalize_asset(*raw_data)

            # add this asset to the dataset statistics
            self.save_asset_stats(norm_data)

            # determine name of file for normalized asset
            subject_i = file.split("/")[-2]
            motion_type_i = file.split("/")[-1].split(".")[0]
            if self.ds_source == "AMASS":
                motion_type_i = motion_type_i.split("_poses")[0]
            file_name = "_".join([subject_i, motion_type_i]) + ".npz"

            # figure out which kind of dataset this asset belongs to
            dataset_name = self.determine_dataset(subject_i, motion_type_i)

            # discard unwanted sequences in training/validation split
            if self.rkk_fraction != 0:
                if dataset_name != "test":
                    motion_idx = motion_type_i.split("_")[0]
                    if self.rkk_fraction == int(motion_idx):
                        print(
                            "Motions similar to {} are used for evaluation only. Skipping!".format(
                                file
                            )
                        )
                        continue

            # save asset in corresponding dataset
            self.save_asset(norm_data, file_name, dataset_name)

            self.logger.info(
                "Normalized {}/{} for {}".format(subject_i, motion_type_i, dataset_name)
            )

            if self.visualize:
                self.visualize_norm_asset(norm_data)

        # compute and save stats
        self.stats["ori_mean"] = self.stats_ori.mean
        self.stats["ori_std"] = np.sqrt(self.stats_ori.var_p)
        self.stats["acc_mean"] = self.stats_acc.mean
        self.stats["acc_std"] = np.sqrt(self.stats_acc.var_p)
        self.stats["pose_mean"] = self.stats_pose.mean
        self.stats["pose_std"] = np.sqrt(self.stats_pose.var_p)

        with open(stats_path, "wb") as fout:
            np.savez_compressed(fout, **self.stats)

        # drop config file with asset count
        self.write_config(
            self.seq_count_train, self.seq_count_valid, self.seq_count_test
        )

    def make_dataset_zero_mean_unit_variance(self):
        # iterate over all normalized assets to make them zero-mean unit-variance
        file_list = []
        for subdir, dirs, files in os.walk(self.norm_dir):
            for file in files:
                if "stats" in file or not file.endswith(".npz"):
                    continue
                file_path = os.path.join(subdir, file)
                file_list.append(file_path)
        file_list = sorted(file_list)

        for file_path in file_list:
            with np.load(file_path, allow_pickle=True) as data:
                data_dict = dict(data)

            # scale data for zero mean and unit variance
            ori_scaled = data_dict["ori"] - self.stats["ori_mean"]
            ori_scaled /= self.stats["ori_std"]
            acc_scaled = data_dict["acc"] - self.stats["acc_mean"]
            acc_scaled /= self.stats["acc_std"]
            pose_scaled = data_dict["pose"] - self.stats["pose_mean"]
            pose_scaled /= self.stats["pose_std"]

            out_dict = {
                "ori": ori_scaled,
                "acc": acc_scaled,
                "pose": pose_scaled,
            }

            file_name = file_path.split("/")[-1]
            folder_path = file_path.split("/")[-3:-1]
            folder_path = os.path.join(*folder_path)
            folder_path = os.path.join(self.trgt_dir, folder_path)
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            out_path = os.path.join(folder_path, file_name)
            with open(out_path, "wb") as fout:
                np.savez_compressed(fout, **out_dict)

            self.logger.info("Processed {}".format(file_name))

        # load config from normalized dataset and save here as well
        cfg = utils.load_config(self.norm_dir)
        utils.write_config(self.trgt_dir, cfg)

        # save same stats here
        stats_path = os.path.join(self.trgt_dir, "stats.npz")
        with open(stats_path, "wb") as fout:
            np.savez_compressed(fout, **self.stats)

    def normalize_dataset(self):
        assert os.path.exists(
            self.src_dir
        ), "Source directory {} does not exist! Check configuration!".format(
            self.src_dir
        )

        self.compute_dataset_stats()

        # keeping this for legacy reasons
        # self.make_dataset_zero_mean_unit_variance()

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
        src_config.dataset.normalized_sensors = self.sens_names
        src_config.dataset.normalized_assets = asset_info
        src_config.dataset.pred_trgt_joints = self.pred_trgt_joints

        # dump updated config to normalized dataset
        utils.write_config(self.norm_dir, src_config)


@hydra.main(config_path="conf", config_name="normalization")
def do_normalization(cfg: DictConfig):
    norm = Normalizer(cfg=cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_normalization()
