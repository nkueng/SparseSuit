import os
import random

import hydra
import numpy as np
import pickle as pkl
import torch
import trimesh
from body_visualizer.mesh import sphere
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools import vis_tools
from omegaconf import DictConfig
from trimesh import creation

from sparsesuit.constants import paths, sensors
from sparsesuit.utils import utils, smpl_helpers


@hydra.main(config_path="conf", config_name="rendering")
def render_from_config(config: DictConfig):
    cfg = DictConfig(config.rendering)

    # GPU setup
    utils.make_deterministic(14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load SMPL model
    smpl_model = smpl_helpers.load_smplx("neutral", device=device, model_type="smplx")
    smpl_faces = utils.copy2cpu(smpl_model.faces)

    # decide on pose
    if cfg.use_relaxed_pose:
        poses_padded = smpl_helpers.generate_relaxed_pose()
        cfg.frame_range = [0]

    elif cfg.name == "leg_raise":
        poses_padded = create_leg_raise_sequence()
        cfg.frame_range = [0]

    # load poses from requested dataset
    elif "dataset" in cfg:
        ds_path = utils.ds_path_from_config(cfg.dataset, "training")

        filelist = []
        for root, dirs, files in os.walk(os.path.join(ds_path, "test")):
            for file in files:
                if file.endswith(".npz"):
                    filelist.append(os.path.join(root, file))

        filelist = sorted(filelist)
        poses_padded = []
        for file in filelist:
            poses = np.load(file)["pose"]
            poses_padded.append(smpl_helpers.extract_from_norm_ds(poses))

        # load models and their predictions
        if "models" in cfg:
            predictions = {}
            sens = {}
            for model in cfg.models:
                # load model predictions
                model_path = os.path.join(paths.RUN_PATH, model)
                files = os.listdir(model_path)
                pred_file = [file for file in files if "prediction" in file][0]
                with np.load(os.path.join(model_path, pred_file)) as data:
                    data_in = dict(data)
                predictions[model] = list(data_in.values())
                # load model config
                exp_config = utils.load_config(model_path).experiment.sensors
                sens[model] = exp_config.names
                if exp_config.sensor_config == "SSP":
                    sens[model].append("left_pelvis")
                    sens[model].append("right_pelvis")
                elif exp_config.sensor_config == "MVN":
                    sens[model].append("pelvis")

            if "VICON" in ds_path:
                # add RKK_STUDIO poses to evaluation
                studio_path = ds_path.replace("VICON", "STUDIO")
                studio_files = []
                for root, dirs, files in os.walk(os.path.join(studio_path, "test")):
                    for file in files:
                        if file.endswith(".npz"):
                            studio_files.append(os.path.join(root, file))

                studio_files = sorted(studio_files)
                studio_poses = []
                for file in studio_files:
                    poses = np.load(file)["pose"]
                    studio_poses.append(np.expand_dims(poses, axis=0))

                predictions["rkk_studio"] = studio_poses
                cfg.models.append("rkk_studio")
                sens["rkk_studio"] = sensors.SENS_NAMES_SSP

    else:
        # load pose data from one of the datasets
        if cfg.source == "AMASS":
            # load motion asset in standard form
            src_dir = paths.AMASS_PATH
            filelist = []
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".npz"):
                        filelist.append(os.path.join(root, file))
            motion_data = np.load(filelist[cfg.asset_idx])
            # extract poses from motion asset
            poses_padded = smpl_helpers.extract_from_smplh(
                motion_data["poses"], list(sensors.SMPL_JOINT_IDS.values())
            )
            # set root orientation to zero
            poses_padded[:, :3] = 0

        elif cfg.source == "DIP_IMU_17":
            # load real motion assets in MVN configuration
            src_dir = paths.DIP_17_PATH
            filelist = []
            for root, dirs, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".pkl"):
                        filelist.append(os.path.join(root, file))
            with open(filelist[cfg.asset_idx], "rb") as fin:
                data_in = pkl.load(fin, encoding="latin1")
            # extract poses from motion asset
            poses_padded = smpl_helpers.extract_from_smplh(
                data_in["gt"], list(sensors.SMPL_JOINT_IDS.values())
            )

        elif cfg.source == "DIP_IMU_6_NN":
            # load real motion assets in DIP configuration
            src_dir = os.path.join(paths.DIP_6_NN_PATH, "imu_own_training.npz")
            with np.load(src_dir, allow_pickle=True) as data:
                data_in = dict(data)
            # extract poses from motion asset
            poses = data_in["smpl_pose"][cfg.asset_idx]
            seq_len = len(poses)
            poses_padded = np.tile(
                np.identity(3), [seq_len, sensors.NUM_SMPLX_JOINTS, 1, 1]
            )
            poses_padded[:, sensors.SMPL_DIP_JOINTS] = poses.reshape(
                [seq_len, -1, 3, 3]
            )
            poses_padded = torch.Tensor(
                utils.rot_matrix_to_aa(poses_padded.reshape([seq_len, -1]))
            )

        elif cfg.use_default_pose:
            poses_padded = torch.zeros([1, sensors.NUM_SMPLX_JOINTS * 3])
        else:
            raise NameError("Desired source of motion asset is unclear")

        if cfg.frame_idx != -1:
            assert cfg.frame_idx <= len(poses_padded), "invalid frame index, aborting!"
            last_frame = cfg.frame_idx + cfg.render_frames
            poses_padded = poses_padded[cfg.frame_idx : last_frame]

    if cfg.make_gif:
        # render every frame for smooth gifs
        cfg.render_frames *= cfg.render_every_n_frame
        cfg.render_every_n_frame = 1

    if "dataset" in cfg:
        if "models" in cfg:
            # render selected frames for all models
            for model in cfg.models:
                pred_i = predictions[model]
                for asset_idx, frame_idx in cfg.assets:
                    last_frame = (
                        frame_idx + cfg.render_every_n_frame * cfg.render_frames
                    )
                    cfg.frame_range = list(
                        range(frame_idx, last_frame, cfg.render_every_n_frame)
                    )
                    poses_i = pred_i[asset_idx][0][
                        frame_idx : last_frame : cfg.render_every_n_frame
                    ]
                    poses_padded_i = smpl_helpers.extract_from_norm_ds(poses_i)
                    poses_gt = poses_padded[asset_idx][
                        frame_idx : last_frame : cfg.render_every_n_frame
                    ]
                    cfg_i = cfg.copy()
                    cfg_i.name = cfg.name + "/" + str(model) + "/" + str(asset_idx)
                    cfg_i.sensors = sens[model]
                    render_poses(poses_padded_i, smpl_model, cfg_i, poses_gt=poses_gt)

            # render selected frames for ground truth
            for asset_idx, frame_idx in cfg.assets:
                last_frame = frame_idx + cfg.render_every_n_frame * cfg.render_frames
                cfg.frame_range = list(
                    range(frame_idx, last_frame, cfg.render_every_n_frame)
                )
                poses_padded_i = poses_padded[asset_idx][
                    frame_idx : last_frame : cfg.render_every_n_frame
                ]
                cfg_i = cfg.copy()
                cfg_i.name = cfg.name + "/target/" + str(asset_idx)
                cfg_i.show_sensors = False
                render_poses(poses_padded_i, smpl_model, cfg_i)

        else:
            for i, poses_padded_i in enumerate(poses_padded):
                cfg_i = cfg.copy()
                folder_name = filelist[i].split("test/")[1].split(".npz")[0]
                folder_name = cfg.name + "/" + str(i) + "_" + folder_name
                cfg_i.name = folder_name
                cfg_i.frame_range = list(
                    range(0, len(poses_padded_i), cfg.render_every_n_frame)
                )
                poses_padded_i = poses_padded_i[cfg_i.frame_range]
                render_poses(poses_padded_i, smpl_model, cfg_i)

    elif cfg.source == "DIP_IMU_17":
        # make sure real acc and ori signal are rendered
        cfg.frame_range = list(
            range(cfg.frame_idx, last_frame, cfg.render_every_n_frame)
        )

        acc = data_in["imu_acc"][cfg.frame_range]
        ori = data_in["imu_ori"][cfg.frame_range]
        render_poses(poses_padded, smpl_model, cfg, acc=acc, ori=ori)

    elif cfg.source == "DIP_IMU_6_NN":
        # make sure real acc and ori signal are rendered
        seq_len = len(poses_padded)
        cfg.frame_range = list(
            range(cfg.frame_idx, last_frame, cfg.render_every_n_frame)
        )

        acc = data_in["acceleration"][cfg.asset_idx][cfg.frame_range].reshape(
            [seq_len, -1, 3]
        )
        ori = data_in["orientation"][cfg.asset_idx][cfg.frame_range].reshape(
            [seq_len, -1, 3, 3]
        )
        render_poses(poses_padded, smpl_model, cfg, acc=acc, ori=ori)
    else:
        render_poses(poses_padded, smpl_model, cfg)


def render_poses(poses_padded, smpl_model, cfg, poses_gt=None, acc=None, ori=None):
    # load mesh viewer
    mv = MeshViewer(
        cfg.render_width,
        cfg.render_height,
        use_offscreen=not cfg.show_rendering,
        # camera_translation=[0, 0.3, 2],
        camera_translation=cfg.camera_translation,
        camera_angle=cfg.camera_angle,
        orthographic=cfg.orthographic_camera,
    )

    # rotate poses to show from behind
    if cfg.show_from_behind:
        poses_padded[:, 1] = np.pi
    if cfg.show_sideways:
        poses_padded[:, 1] = np.pi / 2

    images = []

    # iterate over desired frames in sequence
    seq_len = len(poses_padded)
    for idx_i in range(seq_len):
        if cfg.show_accelerations and (
            idx_i < cfg.acc_delta or seq_len - cfg.acc_delta <= idx_i
        ):
            continue

        print("rendering frame {}".format(cfg.frame_range[idx_i]))

        pose_i = poses_padded[[idx_i]]

        # lbs
        vertices_i, joints_i, rel_tfs_i = smpl_helpers.my_lbs(smpl_model, pose_i)
        vertices_i = utils.copy2cpu(vertices_i[0])
        joints_i = utils.copy2cpu(joints_i[0])
        rel_tfs_i = utils.copy2cpu(rel_tfs_i[0])

        if (cfg.show_vertex_distance or cfg.show_gt) and poses_gt is not None:
            gt_i = poses_gt[[idx_i]]
            vertices_gt, _, _ = smpl_helpers.my_lbs(smpl_model, gt_i)
            vertices_gt = utils.copy2cpu(vertices_gt[0])

        if cfg.show_accelerations:
            pose_before = poses_padded[[idx_i - cfg.acc_delta]]
            pose_after = poses_padded[[idx_i + cfg.acc_delta]]
            sens_ids = list(sensors.SENS_VERTS_SSP.values())
            vertices_before, _, _ = smpl_helpers.my_lbs(smpl_model, pose_before)
            vertices_after, _, _ = smpl_helpers.my_lbs(smpl_model, pose_after)
            vert_0 = utils.copy2cpu(vertices_before[0, sens_ids])
            vert_1 = vertices_i[sens_ids]
            vert_2 = utils.copy2cpu(vertices_after[0, sens_ids])
            time_interval = cfg.acc_delta / 60
            accs = (vert_0 - 2 * vert_1 + vert_2) / (time_interval ** 2)

        # compute meshes
        meshes = []

        # body mesh
        body_color = vis_tools.colors["grey"].copy()
        body_color.append(1 - cfg.body_transparency)
        if cfg.show_vertex_distance and poses_gt is not None:
            # interpolate between grey and red according to vertex distance between pred and gt
            body_color = np.ones(vertices_i.shape) * 0.7
            vert_diff = np.linalg.norm(vertices_i - vertices_gt, axis=1) * 3
            color_diff = np.array(vis_tools.colors["red"]) - np.array(
                vis_tools.colors["grey"]
            )
            body_color += np.minimum(
                vert_diff.reshape([-1, 1]), 1.0
            ) @ color_diff.reshape([1, 3])
        body_mesh = trimesh.Trimesh(
            vertices=vertices_i,
            faces=smpl_model.faces,
            vertex_colors=body_color,
        )
        if cfg.show_body:
            meshes.append(body_mesh)

        # ground truth body mesh
        if cfg.show_gt and poses_gt is not None:
            body_color = vis_tools.colors["green"].copy()
            body_color.append(0.4)
            gt_mesh = trimesh.Trimesh(
                vertices=vertices_gt,
                faces=smpl_model.faces,
                vertex_colors=body_color,
            )
            meshes.append(gt_mesh)

        # joint meshes
        if cfg.show_joints:
            rig_joints = joints_i[sensors.SMPLX_RIG_JOINTS]
            body_joints = rig_joints[:-2]
            lhand_joint = np.mean(rig_joints[[-4, -2]], axis=0)
            rhand_joint = np.mean(rig_joints[[-3, -1]], axis=0)
            all_joints = np.vstack(
                [
                    body_joints,
                    lhand_joint,
                    rhand_joint,
                ]
            )
            vis_joints = sphere.points_to_spheres(
                all_joints, radius=0.015, point_color=vis_tools.colors["red"]
            )
            meshes.append(vis_joints)

        # rig mesh
        if cfg.show_skeleton and cfg.show_joints:
            for joint_ind, parent_ind in enumerate(sensors.SMPL_PARENTS):
                if parent_ind == -1:
                    continue
                # define endpoints of cylinder as joint and parent joint coords
                segment = all_joints[[joint_ind, parent_ind]]
                cyl = creation.cylinder(radius=0.005, segment=segment)
                cyl.visual.vertex_colors = vis_tools.colors["red"]
                meshes.append(cyl)

        # sensor meshes
        if cfg.show_sensors or cfg.show_sensor_orientations:
            sensor_vertices = (
                sensors.SENS_VERTS_SSP
                if cfg.sensor_config == "SSP"
                else sensors.SENS_VERTS_MVN
            )
            chosen_sensors = (
                sensor_vertices.keys() if cfg.sensors == "all" else cfg.sensors
            )
            for sensor, vert_ind in sensor_vertices.items():
                unused = False
                if sensor not in chosen_sensors:
                    unused = True
                    if not cfg.show_unused_sensors:
                        continue

                ori_ind = sensors.SENS_JOINTS_IDS[sensor]
                # vert_norm = body_mesh.vertex_normals[vert_ind]
                # rot_axis = np.cross(vert_norm, [0, 1, 0])
                # rot_angle = np.arcsin(np.linalg.norm(rot_axis))
                # sensor_orientation = np.reshape(
                #     utils.aa_to_rot_matrix(rot_axis * rot_angle), [3, 3]
                # )
                tf = np.eye(4)
                tf[:3, :3] = rel_tfs_i[ori_ind, :3, :3]
                if ori is not None:
                    ori_ind = list(sensor_vertices.keys()).index(sensor)
                    if cfg.name == "dip_imu_6_nn":
                        ori_ind = cfg.sensors.index(sensor)
                    tf[:3, :3] = ori[idx_i, ori_ind]
                # tf[:3, :3] = sensor_orientation.transpose()
                tf[:3, 3] = vertices_i[vert_ind]

                if cfg.show_sensor_orientations:
                    if cfg.name == "leg_raise" and sensor == "right_knee":
                        ax = creation.axis(
                            transform=tf,
                            origin_size=0.021,
                            origin_color=[0, 0, 0],
                            axis_radius=0.015,
                            axis_length=0.15,
                        )
                    else:
                        ax = creation.axis(
                            transform=tf,
                            origin_size=0.007,
                            origin_color=[0, 0, 0],
                            axis_radius=0.005,
                            axis_length=0.05,
                        )

                    meshes.append(ax)
                else:
                    box = creation.box(
                        # extents=[0.05, 0.01, 0.03],
                        extents=[0.03, 0.03, 0.03],
                        transform=tf,
                    )
                    color = vis_tools.colors["orange"].copy()
                    if unused:
                        color = vis_tools.colors["black"].copy()
                        color.append(0.5)
                    box.visual.vertex_colors = color
                    meshes.append(box)

        # meshes for "mocap markers"
        if cfg.show_markers:
            vis_markers = sphere.points_to_spheres(
                vertices_i[sensors.MOCAP_MARKERS],
                radius=0.01,
                point_color=vis_tools.colors["brown"],
            )
            meshes.append(vis_markers)

        # meshes for acceleration cylinders
        if cfg.show_accelerations or acc is not None:
            if acc is not None:
                accs = acc[idx_i]
            if cfg.sensor_config == "SSP":
                vert_ids = list(sensors.SENS_VERTS_SSP.values())
            else:
                vert_ids = list(sensors.SENS_VERTS_MVN.values())
            if cfg.name == "dip_imu_6_nn":
                vert_ids = [sensors.SENS_VERTS_MVN[sensor] for sensor in cfg.sensors]

            for i, acc_i in enumerate(accs):
                if np.any(np.isnan(acc_i)):
                    continue
                cyl_orig = vertices_i[vert_ids[i]]
                cyl_end = cyl_orig + acc_i / 50
                segment = np.array([cyl_orig, cyl_end])
                cyl = creation.cylinder(radius=0.005, segment=segment)
                cyl.visual.vertex_colors = vis_tools.colors["orange"]
                meshes.append(cyl)

        # render meshes
        if cfg.show_rendering:
            mv.viewer.render_lock.acquire()
        mv.set_static_meshes(meshes, smooth=cfg.smooth)

        if cfg.show_rendering:
            mv.viewer.render_lock.release()
            input("press enter to continue")
        else:
            # save file
            body_image = mv.render(render_wireframe=False)

            if cfg.make_gif:
                # store frames for gif
                images.append(body_image)

            else:
                filename = cfg.name
                if "/" in filename:
                    filename = filename.split("/")[1]
                filename = os.path.join(
                    paths.DOC_PATH,
                    "images",
                    cfg.name,
                    str(cfg.frame_range[idx_i]) + "_" + filename + ".png",
                )
                print(filename)
                im_arr = np.expand_dims(body_image, axis=(0, 1, 2))
                vis_tools.imagearray2file(im_arr, filename)

    if cfg.make_gif:
        filename = os.path.join(paths.DOC_PATH, "gifs", cfg.name + ".gif")
        print(filename)
        im_arr = np.stack(images)
        im_arr = np.expand_dims(im_arr, axis=(0, 1))
        vis_tools.imagearray2file(im_arr, filename, fps=cfg.gif_fps)


def create_leg_raise_sequence():
    joint_ind = sensors.SMPL_JOINT_IDS
    def_pose = torch.zeros([sensors.NUM_SMPLX_JOINTS * 3])
    def_pose[1] = np.pi / 6
    def_pose[joint_ind["right_shoulder"] * 3 + 2] = np.pi / 3
    def_pose[joint_ind["left_shoulder"] * 3 + 2] = -np.pi / 3
    def_pose[joint_ind["right_elbow"] * 3 + 2] = np.pi / 9
    def_pose[joint_ind["left_elbow"] * 3 + 2] = -np.pi / 9

    num_frames = 60
    poses = torch.zeros([2 * num_frames, sensors.NUM_SMPLX_JOINTS * 3])
    for i in range(num_frames):
        rand_i = i + 2 * (random.random() - 0.5)
        def_pose[joint_ind["right_hip"] * 3] = -rand_i / num_frames * np.pi / 2
        def_pose[joint_ind["right_knee"] * 3] = rand_i / num_frames * np.pi / 2
        # add noise to make look more realistic
        def_pose[3 : sensors.NUM_SMPL_JOINTS] += (
            2 * torch.rand(sensors.NUM_SMPL_JOINTS - 3) - 1
        ) / 400
        poses[i] = def_pose
        poses[2 * num_frames - 1 - i] = def_pose
    return poses


if __name__ == "__main__":
    render_from_config()
