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
import torch.nn.functional as F
from normalization import Normalizer


class Synthesizer:
    def __init__(self, cfg):
        self.config = cfg
        
        # synthesis parameters
        self.sens_config = cfg.sensors.config
        self.acc_delta = cfg.sensors.acc_delta
        self.acc_noise = cfg.sensors.acc_noise
        self.add_noise = self.acc_noise > 0
        self.fps = cfg.fps
        
        # run parameters
        self.visualize = cfg.visualize
        self.skip_existing = cfg.skip_existing
        self.debug = cfg.debug

        # set internal params depending on config params
        self.src_dir = os.path.join(paths.DATA_PATH, paths.AMASS_PATH)

        if self.debug:
            self.src_dir += "_debug"

        # choose params depending on suit and sensor configuration
        if self.sens_config == "SSP":
            target_name = self.src_dir + "_SSP"
            self.sens_names = sensors.SENS_NAMES_SSP
            self.sens_vert_ids = list(sensors.SENS_VERTS_SSP.values())

        elif self.sens_config == "MVN":
            target_name = self.src_dir + "_MVN"
            self.sens_names = sensors.SENS_NAMES_MVN
            self.sens_vert_ids = list(sensors.SENS_VERTS_MVN.values())

        else:
            raise NameError("Invalid sensor configuration. Aborting!")

        if self.add_noise:
            target_name += "_noisy"

        self.trgt_dir = os.path.join(paths.DATA_PATH, target_name)
        self.joint_ids = [sensors.SENS_JOINTS_IDS[sensor] for sensor in self.sens_names]

        # load SMPL model(s)
        self.smpl_models = smpl_helpers.load_smplx(cfg["smpl_genders"])

        self.asset_counter = 0

    def synthesize_dataset(self):
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
                if self.skip_existing and os.path.exists(target_path):
                    print("Skipping existing {}.".format(file))
                    self.asset_counter += 1
                    continue

                # synthesize sensor data from this motion asset
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                print("Synthesizing {}.".format(file))
                if self.synthesize_asset(file_path, target_path):
                    self.asset_counter += 1

        self.write_config()
        
    def write_config(self):
        # save configuration file for synthetic dataset
        dataset_info = {
            "num_assets": self.asset_counter,
            "config": self.sens_config,
            "type": "synthetic",
            "fps": self.fps,
            "acc_noise": self.acc_noise,
            "acc_delta": self.acc_delta,
            "count": len(self.sens_names),
            "sensors": self.sens_names,
            "vert_ids": self.sens_vert_ids,
            "joint_ids": self.joint_ids,
        }
        ds_config = {
            "dataset": dataset_info,
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
                print("No framerate specified. Skipping!")
                return False

        # early exit for sequences of less than 300 frames
        if data_in["poses"].shape[0] < 300:
            print("Fewer than 300 frames. Skipping!")
            return False

        data_out = {}
        # In case the original frame rates (eg 40FPS) are different from target rates (60FPS)
        if (fps_ori % self.fps) == 0:
            data_out["gt"] = self.interpolation_integer(data_in["poses"], fps_ori)
        else:
            data_out["gt"] = self.interpolation(data_in["poses"], fps_ori)

        # skip if asset contains less than 300 frames (after synthesis)
        frames_after = data_out["gt"].shape[0] - 2 * self.acc_delta
        if frames_after < 300:
            print("Fewer than 300 frames after synthesis. Skipping!")
            return False

        # limit sequences to 11000 frames (empirical)
        # if frames_after > 11000:
        #     data_out['gt'] = data_out['gt'][:11000]

        # simulate IMU data for given SMPL mesh and poses
        gender = utils.str2gender(str(data_in["gender"]))
        if gender is None:
            print("Gender could not be derived. Skipping!")
            return False
        data_out["imu_ori"], data_out["imu_acc"] = self.compute_imu_data(
            gender,
            data_out["gt"],
        )

        # trim N pose sequences at beginning and end depending on smoothing factor N
        # and store only 24 SMPL joints
        data_out["gt"] = data_out["gt"][
            self.acc_delta : -self.acc_delta,
            : sensors.NUM_SMPL_JOINTS * 3,
        ]

        with open(res_path, "wb") as fout:
            np.savez_compressed(fout, **data_out)

        print("Synthesized {} frames.".format(len(data_out["imu_acc"])))
        return True

    # Get orientation and acceleration from list of 4x4 matrices and vertices
    def get_ori_accel(self, A, vertices_IMU):
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
            sensor_opt.set_white_noise(self.acc_noise)
        sensor = pymusim.BaseSensor(sensor_opt)

        time_interval = 1.0 / self.fps * self.acc_delta
        total_number_frames = len(A)
        for idx in range(self.acc_delta, total_number_frames - self.acc_delta):
            vertex_0 = vertices_IMU[idx - self.acc_delta].astype(float)  # 6*3
            vertex_1 = vertices_IMU[idx].astype(float)
            vertex_2 = vertices_IMU[idx + self.acc_delta].astype(float)
            accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (
                time_interval * time_interval
            )
            accel_tmp_noisy = np.array(sensor.transform_measurement(accel_tmp))

            acceleration.append(accel_tmp_noisy)

        return orientation[self.acc_delta : -self.acc_delta], np.asarray(acceleration)

    def compute_imu_data(self, gender, poses):
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
            poses_k = poses[
                k * max_chunk_size : (k + 1) * max_chunk_size,
                : sensors.NUM_SMPL_JOINTS * 3,
            ]
            chunk_size = len(poses_k)
            betas_torch = torch.zeros([chunk_size, 10], dtype=dtype)
            # padding pose data for compatibility with number of smplx joints (55) required by lbs
            padding_len = sensors.NUM_SMPLX_JOINTS - sensors.NUM_SMPL_JOINTS
            poses_torch = F.pad(torch.from_numpy(poses_k), [0, padding_len * 3])

            vertices_k, joints_k, rel_tfs_k = smpl_helpers.my_lbs(
                self.smpl_models[gender], poses_torch.float(), betas_torch
            )

            vertices.append(vertices_k.detach().numpy())
            joints.append(joints_k.detach().numpy())
            rel_tfs.append(rel_tfs_k.detach().numpy())

        vertices = np.concatenate(vertices, axis=0)
        joints = np.concatenate(joints, axis=0)
        rel_tfs = np.concatenate(rel_tfs, axis=0)

        # extract 3d positions of vertices where IMUs are placed (used for acceleration of IMUs)
        imu_vertices = vertices[:, self.sens_vert_ids]

        orientation, acceleration = self.get_ori_accel(rel_tfs, imu_vertices)

        if self.visualize:
            vis_verts = [vertices[self.acc_delta : -self.acc_delta]]
            vis_joints = [
                joints[
                    self.acc_delta : -self.acc_delta,
                    : sensors.NUM_SMPL_JOINTS,
                ]
            ]
            vis_sensors = [imu_vertices[self.acc_delta : -self.acc_delta]]
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
        n_tmp = int(fps / self.fps)
        poses_ori = poses_ori[::n_tmp]

        for t in poses_ori:
            poses.append(t)

        return np.asarray(poses)

    def interpolation(self, poses_ori, fps):
        poses = []
        total_time = len(poses_ori) / fps
        times_ori = np.arange(0, total_time, 1.0 / fps)
        times = np.arange(0, total_time, 1.0 / self.fps)

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
    syn = Synthesizer(cfg=cfg)
    syn.synthesize_dataset()

    norm = Normalizer(cfg=cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_synthesis()
