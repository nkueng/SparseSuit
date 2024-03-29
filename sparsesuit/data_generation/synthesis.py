""" A script to handle the synthesis of IMU data from 17 sensors based on the AMASS dataset of SMPL pose data. """
import logging
import os
import pickle as pkl
import time
from pathlib import Path

import hydra
import submitit
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from normalization import Normalizer
from sparsesuit.constants import paths, sensors
from sparsesuit.utils import smpl_helpers, utils

syn_config_name = "synthesis_cluster" if paths.ON_CLUSTER else "synthesis"


class Synthesizer:
    def __init__(self, cfg):
        self.cfg = cfg

        # synthesis parameters
        self.ds_config = cfg.dataset
        self.sens_config = cfg.dataset.sensor_config
        synthesis_params = cfg.dataset.synthesis
        self.acc_delta = synthesis_params.acc_delta
        self.acc_noise = synthesis_params.acc_noise
        self.gyro_noise = synthesis_params.gyro_noise
        self.acc_saturation = synthesis_params.acc_saturation
        self.fps = cfg.dataset.fps

        # run parameters
        self.visualize = cfg.visualize
        self.debug = cfg.debug
        self.skip_existing = False if cfg.debug else cfg.skip_existing

        # evaluate config params
        src_ds = self.ds_config.source
        self.src_dir = os.path.join(paths.SOURCE_PATH, src_ds)

        if self.debug:
            self.src_dir += "_debug"

        # figure out suit and sensor configuration
        if self.sens_config == "SSP":
            # target_name = self.src_dir + "_SSP"
            self.sens_names = sensors.SENS_NAMES_SSP
            self.sens_vert_ids = list(sensors.SENS_VERTS_SSP.values())

        elif self.sens_config == "MVN":
            # target_name = self.src_dir + "_MVN"
            self.sens_names = sensors.SENS_NAMES_MVN
            self.sens_vert_ids = list(sensors.SENS_VERTS_MVN.values())

        else:
            raise NameError("Invalid sensor configuration. Aborting!")

        assert os.path.exists(self.src_dir)

        self.trgt_dir = utils.ds_path_from_config(cfg.dataset, "synthesis", self.debug)
        self.joint_ids = [sensors.SENS_JOINTS_IDS[sensor] for sensor in self.sens_names]

        # logger setup
        log_level = logging.DEBUG if cfg.debug else logging.INFO
        self.logger = utils.configure_logger(
            name="synthesis", log_path=self.trgt_dir, level=log_level
        )
        self.logger.info("Synthesis\n*******************\n")
        self.logger.info("Source data: {}".format(self.src_dir))

        # use gpu if desired and cuda is available
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.gpu else "cpu"
        )
        self.logger.info("Using {} device".format(self.device))

        # load SMPL model(s) on CPU (move to GPU later)
        self.smpl_models = smpl_helpers.load_smplx_genders(
            cfg.dataset.smpl_genders,
        )

        self.asset_counter = 0

    def synthesize_dataset(self):
        t0 = time.perf_counter()
        file_list = []
        for subdir, dirs, files in os.walk(self.src_dir):
            for file in files:
                if file.startswith("."):
                    continue
                if not file.endswith(".npz"):
                    continue
                if "SSP" in subdir:
                    continue
                if "MVN" in subdir:
                    continue
                file_list.append(os.path.join(subdir, file))
        file_list = sorted(file_list)

        # iterate over all files
        for file in file_list:

            # assemble path of source file
            # asset = file.split("/")[-1]

            # file_path = os.path.join(subdir, asset)

            # assemble target path
            # curr_dir = file.split(asset)[0].split("/")[-2]

            filename = file.split("/")[-2:]
            if "RKK" in self.ds_config.source:
                dataset_name = filename[0]
                filename = filename[1]
            else:
                filename = os.path.join(*filename).replace(" ", "").replace("/", "_")
                dataset_name = file.split("/")[-3]

            # DEBUG
            # if dataset_name == 'ACCAD':
            #     break

            target_dir = os.path.join(self.trgt_dir, dataset_name)
            target_path = os.path.join(target_dir, filename)

            # skip this asset, if the target_path already exists
            if self.skip_existing and os.path.exists(target_path):
                self.logger.info("Skipping existing {}.".format(filename))
                self.asset_counter += 1
                continue

            # synthesize sensor data from this motion asset
            Path(target_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info("Synthesizing {}.".format(filename))
            if self.synthesize_asset(file, target_path):
                self.asset_counter += 1

        self.write_config()
        self.logger.info(
            "Total synthesis runtime: {}s".format(time.perf_counter() - t0)
        )

    def write_config(self):
        # save configuration file for synthetic dataset
        dataset_info = {
            **self.ds_config,
            "sequences": self.asset_counter,
            # "sensor_config": self.sens_config,
            # "type": "synthetic",
            # "fps": self.fps,
            # "acc_noise": self.acc_noise,
            # "acc_delta": self.acc_delta,
            "sensor_count": len(self.sens_names),
            "sensor_names": self.sens_names,
            "sensor_vertices": self.sens_vert_ids,
            # "joint_ids": self.joint_ids,
        }
        ds_config = {
            "dataset": dataset_info,
        }
        utils.write_config(path=self.trgt_dir, config=ds_config)

    def synthesize_asset(self, src_path, res_path):
        """
        Extract pose parameter from src_path, synthesize asset, then save to res_path.
        """
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
                if "RKK" in self.ds_config.source:
                    fps_ori = self.fps
                else:
                    self.logger.info("No framerate specified. Skipping!")
                    return False

        # early exit for sequences of less than 300 frames
        pose_key = "poses"
        if "RKK" in self.ds_config.source:
            pose_key = "gt"
            data_in["gender"] = "neutral"
            data_in["betas"] = np.zeros([1, 10])
        if data_in[pose_key].shape[0] < 300:
            self.logger.info("Fewer than 300 frames. Skipping!")
            return False

        data_out = {}
        # In case the original frame rates (eg 40FPS) are different from target rates (60FPS)
        if (fps_ori % self.fps) == 0:
            data_out["gt"] = utils.interpolation_integer(
                data_in[pose_key], fps_ori, self.fps
            )
        else:
            data_out["gt"] = utils.interpolation(data_in[pose_key], fps_ori, self.fps)

        # skip if asset contains less than 300 frames (after synthesis)
        frames_after = data_out["gt"].shape[0] - 2 * self.acc_delta
        if frames_after < 300:
            self.logger.info("Fewer than 300 frames after synthesis. Skipping!")
            return False

        # simulate IMU data for given SMPL mesh and poses
        gender = utils.str2gender(str(data_in["gender"]))
        if gender is None:
            self.logger.info("Gender could not be derived. Skipping!")
            return False
        data_out["imu_ori"], data_out["imu_acc"] = self.compute_imu_data(
            gender, data_out["gt"], data_in["betas"][:10]
        )

        # trim N pose sequences at beginning and end depending on smoothing factor N
        # and store only 24 SMPL joints
        data_out["gt"] = data_out["gt"][
            self.acc_delta : -self.acc_delta,
            : sensors.NUM_SMPL_JOINTS * 3,
        ]

        with open(res_path, "wb") as fout:
            np.savez_compressed(fout, **data_out)

        self.logger.info("Synthesized {} frames.".format(len(data_out["imu_acc"])))
        return True

    # Get orientation and acceleration from list of 4x4 matrices and vertices
    def get_ori_accel(self, rel_tfs, vertices_IMU):
        # extract IMU orientations from transformation matrices (in global frame)
        oris = []
        for idx in self.joint_ids:
            oris.append(np.expand_dims(rel_tfs[:, idx, :3, :3], axis=1))
        orientation = np.concatenate(oris, axis=1)

        # compute accelerations from subsequent frames
        acceleration = []

        time_interval = self.acc_delta / self.fps
        total_number_frames = len(rel_tfs)
        for idx in range(self.acc_delta, total_number_frames - self.acc_delta):
            vertex_0 = vertices_IMU[idx - self.acc_delta].astype(float)  # 6*3
            vertex_1 = vertices_IMU[idx].astype(float)
            vertex_2 = vertices_IMU[idx + self.acc_delta].astype(float)
            accel_tmp = (vertex_2 + vertex_0 - 2 * vertex_1) / (
                time_interval * time_interval
            )

            acceleration.append(accel_tmp)

        acceleration = np.array(acceleration)

        # add noise to acc with IMUsim
        if self.acc_noise > 0.0:
            import pymusim

            sensor_opt = pymusim.SensorOptions()
            sensor_opt.set_gravity_axis(-1)  # disable additive gravity
            sensor_opt.set_white_noise(self.acc_noise)
            sensor = pymusim.BaseSensor(sensor_opt)

            acceleration = torch.Tensor(acceleration)
            noise_vec = np.array(
                sensor.transform_measurement(np.zeros([acceleration.size // 3, 3]))
            )
            acceleration += noise_vec.reshape(acceleration.shape)

            # accel_vec = acceleration.reshape([-1, 3])
            # accel_noisy = np.array(sensor.transform_measurement(accel_vec))
            # acceleration = accel_noisy.reshape([-1, len(self.joint_ids), 3])

        # clip acc at sensor saturation value
        if self.acc_saturation is not None:
            # convert from G to m/s²
            max_norm = self.acc_saturation * 9.81
            clip_coef = max_norm / (np.linalg.norm(acceleration, axis=2) + 1e-6)
            need_clip = clip_coef < 1
            if need_clip.any():
                acceleration[need_clip] *= np.expand_dims(clip_coef[need_clip], axis=1)

        # make orientations noisy
        if self.gyro_noise > 0.0:
            import pymusim

            sensor_opt = pymusim.SensorOptions()
            sensor_opt.set_gravity_axis(-1)  # disable additive gravity
            sensor_opt.set_white_noise(self.gyro_noise)
            sensor = pymusim.BaseSensor(sensor_opt)

            # orientation = torch.Tensor(orientation)
            # t0 = time.perf_counter()
            ori_mat = orientation.reshape([-1, 3, 3])
            # ori_mat = utils.copy2cpu(orientation.reshape([-1, 3, 3]))
            # t1 = time.perf_counter()
            noise_rot_aa = np.array(
                sensor.transform_measurement(np.zeros([len(ori_mat), 3]))
            )
            # t2 = time.perf_counter()
            noise_rot_mat = utils.aa_to_rot_matrix(noise_rot_aa).reshape([-1, 3, 3])
            # t3 = time.perf_counter()
            ori_mat_noisy = np.einsum("ijk,ikl->ijl", noise_rot_mat, ori_mat)
            # t4 = time.perf_counter()
            orientation = ori_mat_noisy.reshape(orientation.shape)
            # print(
            #     "Intervals: 1: {}, 2: {}, 3: {}, 4: {}. Total: {}".format(
            #         t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0
            #     )
            # )

            # t0 = time.perf_counter()
            # ori_mat = orientation.reshape([-1, 9])
            # t1 = time.perf_counter()
            # ori_aa = utils.rot_matrix_to_aa(ori_mat)
            # t2 = time.perf_counter()
            # ori_noisy = np.array(sensor.transform_measurement(ori_aa))
            # t3 = time.perf_counter()
            # oir_noisy_mat = utils.aa_to_rot_matrix(ori_noisy)
            # t4 = time.perf_counter()
            # orientation = oir_noisy_mat.reshape([-1, len(self.joint_ids), 3, 3])
            # print(
            #     "Intervals: 1: {}, 2: {}, 3: {}, 4: {}. Total: {}".format(
            #         t1 - t0, t2 - t1, t3 - t2, t4 - t3, t4 - t0
            #     )
            # )

        return orientation[self.acc_delta : -self.acc_delta], acceleration

    def compute_imu_data(self, gender, poses, betas=None):

        # extract poses of all joints (used for orientation of IMUs/body segments)
        batch_size = len(poses)
        max_chunk_size = (
            self.cfg.gpu_chunks
            if self.device == torch.device("cuda")
            else self.cfg.cpu_chunks
        )
        num_chunks = batch_size // (max_chunk_size + 1) + 1
        vertices, joints, rel_tfs = [], [], []
        for k in range(num_chunks):
            # extract chunk of poses
            poses_k = poses[k * max_chunk_size : (k + 1) * max_chunk_size]

            # padding pose data for compatibility with number of smplx joints (55) required by lbs
            poses_torch = smpl_helpers.extract_from_smplh(
                poses_k, list(sensors.SMPL_JOINT_IDS.values())
            )

            # preparing betas
            betas_k = None
            if betas is not None:
                betas_k = torch.tile(torch.Tensor(betas), [len(poses_k), 1])

            # load relevant smpl model onto GPU if desired
            smpl_model = self.smpl_models[gender].to(self.device)

            # do lbs for given poses
            vertices_k, joints_k, rel_tfs_k = smpl_helpers.my_lbs(
                smpl_model, poses_torch, betas_k
            )

            vertices.append(utils.copy2cpu(vertices_k))
            joints.append(utils.copy2cpu(joints_k))
            rel_tfs.append(utils.copy2cpu(rel_tfs_k))

        vertices = np.concatenate(vertices, axis=0)
        joints = np.concatenate(joints, axis=0)
        rel_tfs = np.concatenate(rel_tfs, axis=0)

        # extract 3d positions of vertices where IMUs are placed (used for acceleration of IMUs)
        imu_vertices = vertices[:, self.sens_vert_ids]

        orientation, acceleration = self.get_ori_accel(rel_tfs, imu_vertices)

        if self.visualize:
            from sparsesuit.utils import visualization

            vis_verts = [vertices[self.acc_delta : -self.acc_delta]]
            vis_joints = [
                joints[
                    self.acc_delta : -self.acc_delta,
                    : sensors.NUM_SMPL_JOINTS,
                ]
            ]
            vis_sensors = [imu_vertices[self.acc_delta : -self.acc_delta]]
            visualization.vis_smpl(
                faces=self.smpl_models[gender].faces,
                # vertices=[np.expand_dims(model.v_template.detach().numpy(), axis=0)],  # vis ori of smpl's v_template
                vertices=vis_verts,
                # joints=vis_joints,
                sensors=vis_sensors,
                accs=[acceleration],
                oris=[orientation],
                play_frames=300,
                playback_speed=0.1,
                add_captions=False,
            )

        return orientation, acceleration


@hydra.main(config_path="conf", config_name=syn_config_name)
def do_synthesis(cfg: DictConfig):
    if paths.ON_CLUSTER:
        submitit.JobEnvironment()

    syn = Synthesizer(cfg=cfg)
    syn.synthesize_dataset()

    # load default config for normalization
    norm_cfg_path = os.path.join(
        utils.get_project_folder(), "data_generation/conf/normalization.yaml"
    )
    norm_cfg = OmegaConf.load(norm_cfg_path)
    # adapt default config to synthesis configuration
    norm_cfg.dataset = cfg.dataset
    norm_cfg.debug = cfg.debug
    # run normalization with adapted configuration
    norm = Normalizer(cfg=norm_cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_synthesis()
