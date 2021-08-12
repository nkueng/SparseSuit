""" A script to handle the synthesis of IMU data from 17 sensors based on the AMASS dataset of SMPL pose data. """
import os
from pathlib import Path
import pymusim
import numpy as np
import pickle as pkl
import hydra
from omegaconf import DictConfig
from sparsesuit.constants import paths, sensors
from sparsesuit.utils import smpl_helpers, visualization, utils
import torch
from normalization import Normalizer


class Synthesizer:
    def __init__(
        self,
        cfg: dict = None,
    ):
        # get params from configuration file
        self.config = cfg
        self.white_noise = cfg["white_noise"]
        self.add_noise = self.white_noise > 0
        self.sens_config = cfg["sensors"]["config"]
        self.target_fps = cfg["target_fps"]
        self.smoothing_factor = cfg["smoothing_factor"]
        self.visualize = cfg["visualize"]

        # set internal params depending on config params
        self.src_dir = os.path.join(paths.DATA_PATH, paths.AMASS_PATH)
        # choose params depending on suit and sensor configuration
        if self.sens_config == "SSP":
            target_name = paths.AMASS_19_PATH
            self.sens_names = sensors.SENS_NAMES_SSP
            self.sens_vert_ids = list(sensors.SENS_VERTS_SSP.values())
        elif self.sens_config == "MVN":
            target_name = paths.AMASS_17_PATH
            self.sens_names = sensors.SENS_NAMES_MVN
            self.sens_vert_ids = list(sensors.SENS_VERTS_MVN.values())
            if self.add_noise:
                target_name = paths.AMASS_17_NOISY_PATH
        else:
            raise NameError("Invalid sensor configuration. Aborting!")
        self.trgt_dir = os.path.join(paths.DATA_PATH, target_name)
        self.joint_ids = [sensors.SENS_JOINTS_IDS[sensor] for sensor in self.sens_names]

        # load SMPL model(s)
        self.smpl_models = smpl_helpers.load_smplx(cfg["smpl_genders"])

    def synthesize_dataset(self):
        asset_counter = 0
        for subdir, dirs, files in os.walk(self.src_dir):
            for file in files:
                if file.startswith("."):
                    continue
                if not file.endswith(".npz"):
                    continue
                # assemble path of source file
                file_path = os.path.join(subdir, file)

                # assemble target path
                curr_dir = subdir.split("/")[-1]
                dataset_name = subdir.split(self.src_dir)[1].split("/")[0]

                # DEBUG
                # if dataset_name == 'ACCAD':
                #     break

                filename = curr_dir + "_" + file
                target_dir = os.path.join(self.trgt_dir, dataset_name)
                target_path = os.path.join(target_dir, filename)

                # skip this asset, if the target_path already exists
                if os.path.exists(target_path):
                    print("Skipping {}.".format(file))
                    asset_counter += 1
                    continue

                # synthesize sensor data from this motion asset
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                print("Synthesizing {}.".format(file))
                if self.synthesize_asset(file_path, target_path):
                    asset_counter += 1

        # save configuration file for synthetic dataset
        sensor_info = {
            "config": self.sens_config,
            "type": "synthetic",
            "count": len(self.sens_names),
            "names": self.sens_names,
            "vert_ids": self.sens_vert_ids,
            "joint_ids": self.joint_ids,
            "acc_noise": self.white_noise,
        }
        ds_config = {
            "num_assets": asset_counter,
            "fps": self.target_fps,
            "smoothing_factor": self.smoothing_factor,
            "sensors": sensor_info,
        }
        utils.write_config(path=self.trgt_dir, config=ds_config)

    # Extract pose parameter from src_path, save to res_path
    def synthesize_asset(self, src_path, res_path):
        # load from pickle or npz
        data_in = {}
        if src_path.endswith(".npz"):
            with np.load(src_path, allow_pickle=True) as data:
                data_in = dict(data)
        elif src_path.endswith(".pkl"):
            with open(src_path, "rb") as fin:
                data_in = pkl.load(fin, encoding="latin1")

        # if data_in has no key for the framerate, skip the asset
        try:
            fps_ori = data_in["mocap_framerate"]
        except KeyError:
            try:
                fps_ori = data_in["frame_rate"]
            except KeyError:
                return False

        # early exit for sequences of less than 150 frames
        if data_in["poses"].shape[0] < 150:
            return False

        data_out = {}
        # In case the original frame rates (eg 40FPS) are different from target rates (60FPS)
        if (fps_ori % self.target_fps) == 0:
            data_out["gt"] = self.interpolation_integer(data_in["poses"], fps_ori)
        else:
            data_out["gt"] = self.interpolation(data_in["poses"], fps_ori)

        # skip if asset contains less than 300 frames (after synthesis)
        frames_after = data_out["gt"].shape[0] - 2 * self.smoothing_factor
        if frames_after < 300:
            return False

        # limit sequences to 11000 frames (empirical)
        # if frames_after > 11000:
        #     data_out['gt'] = data_out['gt'][:11000]

        # simulate IMU data for given SMPL mesh and poses
        gender = str(data_in["gender"])
        betas = np.array(data_in["betas"][:10])
        data_out["imu_ori"], data_out["imu_acc"] = self.compute_imu_data(
            gender,
            betas,
            data_out["gt"],
            self.target_fps,
        )

        # trim N pose sequences at beginning and end depending on smoothing factor N
        # and store only 24 SMPL joints
        data_out["gt"] = data_out["gt"][
            self.smoothing_factor : -self.smoothing_factor,
            : sensors.NUM_SMPL_JOINTS * 3,
        ]

        # # convert joint rotations to rotation matrices as prediction targets
        # for fdx in range(0, len(data_out['gt'])):  # for each frame
        #     pose_tmp = []  # np.zeros(0)
        #     for jdx in SMPL_IDS:  # for the selected joints
        #         tmp = data_out['gt'][fdx][jdx * 3:(jdx + 1) * 3]  # extract this joint's orientation
        #         tmp = cv2.Rodrigues(tmp)[0].flatten().tolist()  # axis-angle to rotation matrix
        #         pose_tmp = pose_tmp + tmp  # concatenate rotation matrices
        #
        #     data_out['gt'][fdx] = []
        #     data_out['gt'][fdx] = pose_tmp

        with open(res_path, "wb") as fout:
            np.savez_compressed(fout, **data_out)

        print(len(data_out["imu_acc"]))
        return True

    # Get orientation and acceleration from list of 4x4 matrices and vertices
    def get_ori_accel(self, A, vertices_IMU, frame_rate):
        # extract IMU orientations from transformation matrices (in global frame)
        # TODO: make orientations noisy
        oris = []
        for idx in self.joint_ids:
            oris.append(np.expand_dims(A[:, idx, :3, :3], axis=1))
        orientation = np.concatenate(oris, axis=1)

        # compute accelerations from subsequent frames and add noise with IMUsim
        acceleration = []

        # IMU sim for noise
        sensor_opt = pymusim.SensorOptions()
        sensor_opt.set_gravity_axis(-1)  # disable additive gravity
        if self.add_noise:
            sensor_opt.set_white_noise(self.white_noise)
        sensor = pymusim.BaseSensor(sensor_opt)

        time_interval = 1.0 / frame_rate * self.smoothing_factor
        total_number_frames = len(A)
        for idx in range(
            self.smoothing_factor, total_number_frames - self.smoothing_factor
        ):
            vertex_0 = vertices_IMU[idx - self.smoothing_factor].astype(float)  # 6*3
            vertex_1 = vertices_IMU[idx].astype(float)
            vertex_2 = vertices_IMU[idx + self.smoothing_factor].astype(float)
            accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (
                time_interval * time_interval
            )
            accel_tmp_noisy = np.array(sensor.transform_measurement(accel_tmp))

            acceleration.append(accel_tmp_noisy)

        return orientation[self.smoothing_factor : -self.smoothing_factor], np.asarray(
            acceleration
        )

    def compute_imu_data(self, gender, betas, poses, frame_rate):

        betas[:] = 0

        # use cuda if available
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("Using {} device".format(device))
        dtype = torch.float32

        # extract poses of all joints (used for orientation of IMUs/body segments)
        # TODO: change dtype to float64 if necessary for precision
        batch_size = len(poses)
        max_chunk_size = 3000
        num_chunks = batch_size // max_chunk_size + 1
        vertices, joints, rel_tfs = [], [], []
        for k in range(num_chunks):
            poses_k = poses[k * max_chunk_size : (k + 1) * max_chunk_size]
            chunk_size = len(poses_k)
            betas_torch = torch.zeros([chunk_size, betas.shape[0]], dtype=dtype)
            # padding pose data for compatibility with number of smplx joints (55) required by lbs
            zero_padding = np.zeros(
                (
                    chunk_size,
                    (sensors.NUM_SMPLX_JOINTS - sensors.NUM_SMPL_JOINTS) * 3,
                )
            )
            poses_padded = np.concatenate(
                (poses_k[:, : sensors.NUM_SMPL_JOINTS * 3], zero_padding),
                axis=1,
            )
            poses_torch = torch.from_numpy(poses_padded).float()

            vertices_k, joints_k, rel_tfs_k = smpl_helpers.my_lbs(
                self.smpl_models[gender], poses_torch, betas_torch
            )

            vertices.append(vertices_k.detach().numpy())
            joints.append(joints_k.detach().numpy())
            rel_tfs.append(rel_tfs_k.detach().numpy())

        vertices = np.concatenate(vertices, axis=0)
        joints = np.concatenate(joints, axis=0)
        rel_tfs = np.concatenate(rel_tfs, axis=0)

        # extract 3d positions of vertices where IMUs are placed (used for acceleration of IMUs)
        imu_vertices = vertices[:, self.sens_vert_ids]

        orientation, acceleration = self.get_ori_accel(
            rel_tfs, imu_vertices, frame_rate
        )

        if self.visualize:
            vis_verts = [vertices[self.smoothing_factor : -self.smoothing_factor]]
            vis_joints = [
                joints[
                    self.smoothing_factor : -self.smoothing_factor,
                    : sensors.NUM_SMPL_JOINTS,
                ]
            ]
            vis_sensors = [imu_vertices[self.smoothing_factor : -self.smoothing_factor]]
            visualization.vis_smpl(
                model=self.smpl_models[gender],
                # vertices=[np.expand_dims(model.v_template.detach().numpy(), axis=0)],  # vis ori of smpl's v_template
                vertices=vis_verts,
                joints=vis_joints,
                sensors=vis_sensors,
                accs=[acceleration],
                oris=[orientation],
                play_frames=300,
                playback_speed=0.1,
                add_captions=False,
            )

        return orientation, acceleration

    @staticmethod
    def findNearest(t, t_list):
        list_tmp = np.array(t_list) - t
        list_tmp = np.abs(list_tmp)
        index = np.argsort(list_tmp)[:2]
        return index

    # Turn MoCap data into 60FPS
    def interpolation_integer(self, poses_ori, fps):
        poses = []
        n_tmp = int(fps / self.target_fps)
        poses_ori = poses_ori[::n_tmp]

        for t in poses_ori:
            poses.append(t)

        return np.asarray(poses)

    def interpolation(self, poses_ori, fps):
        poses = []
        total_time = len(poses_ori) / fps
        times_ori = np.arange(0, total_time, 1.0 / fps)
        times = np.arange(0, total_time, 1.0 / self.target_fps)

        for t in times:
            index = self.findNearest(t, times_ori)
            if max(index) >= poses_ori.shape[0]:
                break
            a = poses_ori[index[0]]
            t_a = times_ori[index[0]]
            b = poses_ori[index[1]]
            t_b = times_ori[index[1]]

            if t_a == t:
                tmp_pose = a
            elif t_b == t:
                tmp_pose = b
            else:
                tmp_pose = a + (b - a) * ((t_b - t) / (t_b - t_a))
            poses.append(tmp_pose)

        return np.asarray(poses)


@hydra.main(config_path="conf", config_name="synthesis")
def do_synthesis(cfg: DictConfig):
    config = dict(cfg)
    syn = Synthesizer(cfg=config)
    syn.synthesize_dataset()

    config["dataset_type"] = "synthetic"
    norm = Normalizer(cfg=config)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_synthesis()
