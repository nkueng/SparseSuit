"""First normalizes imu data w.r.t. root sensor and saves asset locally. Then loads normalized asset and scales for
zero mean & unit variance.
Input: pickle or npz files with N IMU orientations and accelerations as well as SMPL poses
for 24 joints
Output: normalized zero-mean unit-variance input and target vectors of N-1 IMUs ready for learning """
import shutil
import sys
import tarfile

import numpy as np
import os
import pickle as pkl
from pathlib import Path
import torch
from welford import Welford
import smplx
from smplx import lbs
from sparsesuit.constants import sensors, paths
from sparsesuit.utils import visualization, smpl_helpers, utils
import hydra
from omegaconf import DictConfig, OmegaConf


class Normalizer:
    def __init__(self, cfg):
        self.config = cfg
        self.visualize = cfg["visualize"]
        self.smpl_genders = cfg["smpl_genders"]
        dataset_config = cfg["dataset"]
        self.dataset_type = dataset_config["type"]
        self.sensor_config = dataset_config["sens_config"]

        # choose dataset folder based on params
        if self.dataset_type == "synthetic":
            if self.sensor_config == "SSP":
                src_folder = paths.AMASS_19_PATH
                norm_folder = paths.AMASS_19_N_PATH
                trgt_folder = paths.AMASS_19_NN_PATH
                sensor_names = sensors.SENS_NAMES_SSP
                pred_trgt_joints = sensors.SMPL_SSP_JOINTS
            elif self.sensor_config == "MVN":
                src_folder = paths.AMASS_17_PATH
                norm_folder = paths.AMASS_17_N_PATH
                trgt_folder = paths.AMASS_17_NN_PATH
                sensor_names = sensors.SENS_NAMES_MVN
                pred_trgt_joints = sensors.SMPL_MAJOR_JOINTS
            else:
                raise NameError("Invalid dataset configuration. Aborting!")
            dataset_names = ["training", "validation", "test"]
            sensor_ids = np.arange(0, len(sensor_names))
        elif self.dataset_type == "real":
            src_folder = paths.DIP_17_PATH
            norm_folder = paths.DIP_17_N_PATH
            trgt_folder = paths.DIP_17_NN_PATH
            # dict mapping subject and motion types to datasets
            dataset_names = ["training", "validation", "test"]
            dataset_mapping = {
                "s_09": "test",
                "s_10": "test",
                "s_01/05": "validation",
                "s_03/05": "validation",
                "s_07/04": "validation",
            }
            sensor_names = sensors.SENS_NAMES_MVN
            sensor_ids = [
                sensors.SENS_VERTS_MVN.index(sensor) for sensor in sensor_names
            ]
        else:
            raise NameError("Invalid dataset configuration. Aborting!")
        self.src_dir = os.path.join(paths.DATA_PATH, src_folder)
        self.norm_dir = os.path.join(paths.DATA_PATH, norm_folder)
        self.trgt_dir = os.path.join(paths.DATA_PATH, trgt_folder)
        self.dataset_names = dataset_names
        self.dataset_mapping = dataset_mapping
        self.sensor_names = sensor_names
        self.sensor_ids = sensor_ids
        self.pred_trgt_joints = pred_trgt_joints

    def normalize_dataset(self):

        if self.visualize:
            # load neutral smpl model for evaluation of normalized sensor data
            smpl_model = smpl_helpers.load_smplx(["neutral"])["neutral"]

        # set up objects for online mean and variance computation
        stats_ori, stats_acc, stats_pose = (
            Welford(),
            Welford(),
            Welford(),
        )

        if not os.path.exists(self.src_dir):
            print("Source directory does not exist!")
            sys.exit()

        # iterate over all subdirectories and files in given path
        for subdir, dirs, files in os.walk(self.src_dir):
            for file in files:
                # check for correct file format
                if file.startswith(".") or (
                    not file.endswith(".npz") and not file.endswith(".pkl")
                ):
                    continue

                # DEBUG
                # if file.split('.')[0] == '1347_Experiment3b_subject1347_back_02_poses':
                #     stop_here = True

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
                    oris = data_in["imu_ori"]  # IMU orientation matrices of sensors
                    accs = data_in["imu_acc"]  # IMU acceleration vectors of sensors
                    poses = data_in["gt"]  # flattened pose vector of 24 SMPL joints
                    pose2rot = True  # poses given in axis-angle format and need transformation to rot matrix
                except KeyError:
                    try:
                        # IMU orientation matrices of sensors
                        oris = np.array(data_in["ori"])
                        # IMU acceleration vectors of sensors
                        accs = np.array(data_in["acc"])
                        # flattened pose vector of 15 SMPL joints
                        poses = np.array(data_in["poses"])
                        pose2rot = False
                    except KeyError:
                        print("Input data does not have the right fields. Skipping!")
                        continue
                seq_length = len(accs)

                # exit if data is empty
                if seq_length == 0:
                    continue

                subject_i = subdir.split(self.src_dir)[1]
                # subjects.append(subject_i)
                motion_type_i = file.split(".")[0]
                # motion_types.append(motion_type_i)

                # determine name of file for normalized asset
                norm_dir_name = os.path.join(self.norm_dir, subject_i)
                out_filename = os.path.join(norm_dir_name, motion_type_i)

                # exit if normalized asset already exists -> makes no sense as we need stats from all assets
                # if Path(out_filename + '.npz').exists():
                #     print('Skipping normalization of {}/{} as it already exists.'.format(subject_i, motion_type_i))
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
                if self.sensor_config == "SSP":
                    # the SSP has no root sensor, so a virtual sensor is derived from the two hip sensors
                    root_indx = [
                        self.sensor_names.index("left_hip"),
                        self.sensor_names.index("right_hip"),
                    ]

                    # get hip sensor orientations
                    hip_oris = np.reshape(oris_sorted[:, root_indx], [-1, 9])
                    # transform to angle-axis
                    hip_aas = np.reshape(
                        utils.rot_matrix_to_aa(hip_oris), [new_seq_length, 2, 3]
                    )
                    # compute mean
                    root_aa = np.mean(hip_aas, axis=1)
                    # transform back to rotation matrix
                    root_ori = np.reshape(
                        utils.aa_to_rot_matrix(root_aa), [-1, 1, 3, 3]
                    )
                    # get inverse to normalize other sensors
                    root_ori_inv = np.transpose(root_ori, [0, 1, 3, 2])

                    # get hip sensor accelerations
                    hip_accs = accs_sorted[:, [root_indx]]
                    # average to get virtual sensor acceleration to normalize other sensors
                    root_acc = np.mean(hip_accs, axis=2)
                else:
                    root_indx = self.sensor_names.index("pelvis")
                    root_ori_inv = np.transpose(
                        oris_sorted[:, [root_indx]], [0, 1, 3, 2]
                    )
                    root_acc = accs_sorted[:, [root_indx]]

                # normalize all orientations
                oris_norm = np.matmul(root_ori_inv, oris_sorted)
                oris_norm_wo_root = np.delete(
                    oris_norm, root_indx, axis=1
                )  # discard root orientation which is always identity
                oris_vec = np.reshape(oris_norm_wo_root, (new_seq_length, -1))

                # normalize all accelerations
                accs_norm = accs_sorted - root_acc
                accs_norm = np.matmul(root_ori_inv, accs_norm[..., np.newaxis])
                accs_norm_wo_root = np.delete(
                    accs_norm, root_indx, axis=1
                )  # discard root acceleration which is always zero
                accs_vec = np.reshape(accs_norm_wo_root, (new_seq_length, -1))

                # create input vectors (orientation + acc)
                # input_vec = np.concatenate((oris_vec, accs_vec), axis=1)

                # create target vectors (pose + acc) after transforming poses from angle axis to rot matrix with Rodrigues
                poses_vec = poses[nan_mask]
                if pose2rot:
                    poses_vec_sel = np.reshape(poses_vec, (new_seq_length, -1, 3))[
                        :, self.pred_trgt_joints
                    ]  # select target joints
                    poses_vec_torch = torch.from_numpy(
                        np.reshape(poses_vec_sel, (-1, 3))
                    )
                    poses_vec_rot = (
                        lbs.batch_rodrigues(poses_vec_torch).detach().numpy()
                    )
                    poses_vec = np.reshape(poses_vec_rot, (new_seq_length, -1))
                # target_vec = np.concatenate((poses_vec, accs_vec), axis=1)

                # store input and target vectors for rescaling later
                # input_data.append(input_vec)
                # target_data.append(target_vec)

                # save normalized orientations, accelerations and poses locally
                Path(norm_dir_name).mkdir(parents=True, exist_ok=True)

                # write npy files
                # input_filename = dir_name + '/' + motion_type_i + '.input.npy'
                # with open(input_filename, 'wb') as f:
                #     np.save(f, input_vec)
                #
                # output_filename = dir_name + '/' + motion_type_i + '.target.npy'
                # with open(output_filename, 'wb') as f:
                #     np.save(f, target_vec)

                # write npz file
                data_out = {
                    "orientation": oris_vec,
                    "acceleration": accs_vec,
                    "pose": poses_vec,
                }
                with open(out_filename + ".npz", "wb") as fout:
                    np.savez_compressed(fout, **data_out)

                # compute statistics for this asset
                for idx in range(new_seq_length):
                    stats_ori.add(oris_vec[idx])
                    stats_acc.add(accs_vec[idx])
                    stats_pose.add(poses_vec[idx])

                print("Normalized {}/{}".format(subject_i, motion_type_i))

                if self.visualize:
                    frames = 300
                    vis_poses = np.zeros([frames, sensors.NUM_SMPLX_JOINTS, 3])
                    vis_poses[:, self.pred_trgt_joints] = poses_vec_sel[:frames]
                    # vis_poses[:, 0] = np.array([[1.20919958, 1.20919958, 1.20919958]])
                    poses_torch = torch.from_numpy(vis_poses).float()
                    # compute skin vertices from SMPL pose
                    betas_torch = torch.zeros([frames, 10], dtype=torch.float32)
                    verts, joints, _ = smpl_helpers.my_lbs(
                        model=smpl_model,
                        pose=poses_torch,
                        betas=betas_torch,
                        pose2rot=True,
                    )

                    verts_np, joints_np = (
                        verts.detach().numpy(),
                        joints.detach().numpy()[:, : sensors.NUM_SMPL_JOINTS],
                    )
                    if self.sensor_config == "SSP":
                        vertex_ids = [
                            sensors.SENS_VERTS_SSP[sensor_name]
                            for sensor_name in self.sensor_names
                        ]
                    else:
                        vertex_ids = [
                            sensors.SENS_VERTS_MVN[sensor_name]
                            for sensor_name in self.sensor_names
                        ]

                    vertices = [verts_np]
                    vis_sensors = [verts_np[:, vertex_ids]]
                    orientations = [np.squeeze(oris_norm[:frames])]
                    accelerations = [np.squeeze(accs_norm[:frames])]
                    visualization.vis_smpl(
                        model=smpl_model,
                        vertices=vertices,
                        sensors=vis_sensors,
                        accs=accelerations,
                        oris=orientations,
                        play_frames=frames,
                        playback_speed=0.1,
                        add_captions=False,
                    )

        # print stats to console
        mean_ori = stats_ori.mean
        mean_acc = stats_acc.mean
        mean_pose = stats_pose.mean
        std_ori = np.sqrt(stats_ori.var_p)
        std_acc = np.sqrt(stats_acc.var_p)
        std_pose = np.sqrt(stats_pose.var_p)

        ori_dim = len(mean_ori)
        acc_dim = len(mean_acc)
        pose_dim = len(mean_pose)

        # print('input mean: \n{} \ninput std: \n{} \ntarget mean: \n{} \ntarget std: \n{}'.format(mean_input,
        #                                                                                          std_input,
        #                                                                                          mean_target,
        #                                                                                          std_target))

        # iterate over all normalized assets
        seq_count_train, seq_count_test, seq_count_valid = (
            [0],
            [0],
            [0],
        )  # using list since it is mutable
        seq_count = None

        for subdir, dirs, files in os.walk(self.norm_dir):
            for file in files:
                file_path = os.path.join(subdir, file)
                with np.load(file_path, allow_pickle=True) as data:
                    data_dict = dict(data)
                # input_path = os.path.join(subdir, files[0])
                # target_path = os.path.join(subdir, files[1])
                # input_data = np.load(input_path)
                # target_data = np.load(target_path)

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
                    dir_name = self.trgt_dir + dataset_name + "/" + str(seq_count[0])
                    Path(dir_name).mkdir(parents=True, exist_ok=True)
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

                    seq_count[0] += 1

                print("Processed {}/{}".format(subject_i, motion_type_i))

                # DEBUG: inspect numerical values of normalized assets
                # if motion_type_i == '75_75_01':
                #     dir_name = os.path.join(trgt_dir, 'test_asset')
                #     Path(dir_name).mkdir(parents=True, exist_ok=True)
                #     input_filename = dir_name + '/' + file_id_i + '.input.npy'
                #     with open(input_filename, 'wb') as f:
                #         np.save(f, input_scaled)
                #
                #     output_filename = dir_name + '/' + file_id_i + '.target.npy'
                #     with open(output_filename, 'wb') as f:
                #         np.save(f, target_scaled)

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

        # delete directory with normalized assets
        shutil.rmtree(self.norm_dir)

        # convert dataset directories to .tar and delete directories
        print("Converting to .tar...")
        for dataset in self.dataset_names:
            ds_path = self.trgt_dir + dataset
            with tarfile.open(ds_path + ".tar", "w") as tar:
                tar.add(ds_path, arcname=os.path.basename(ds_path))
            shutil.rmtree(ds_path)


@hydra.main(config_path="conf", config_name="normalization")
def get_config(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    global config
    config = cfg


if __name__ == "__main__":
    get_config()
    norm = Normalizer(cfg=dict(config))
    norm.normalize_dataset()
