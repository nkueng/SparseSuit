import os

import hydra
import numpy as np
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
def render_from_config(cfg: DictConfig):
    cfg = cfg.rendering

    # GPU setup
    utils.make_deterministic(14)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load SMPL model
    smpl_model = smpl_helpers.load_smplx("neutral", device=device, model_type="smplx")
    smpl_faces = utils.copy2cpu(smpl_model.faces)

    # decide on pose
    if cfg.use_default_pose:
        poses_padded = torch.zeros([1, sensors.NUM_SMPLX_JOINTS * 3])
    else:
        # load pose data from one of the datasets
        if cfg.source == "AMASS":
            # load motion asset in standard form
            src_dir = os.path.join(paths.DATA_PATH, "Synthetic/AMASS")
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
        elif cfg.source == "AMASS_SSP_nn":
            # load motion asset in normalized form
            src_dir = os.path.join(paths.DATA_PATH, "Synthetic/AMASS_SSP_nn")
            filelist = []
            for root, dirs, files in os.walk(os.path.join(src_dir, "test")):
                for file in files:
                    if file.endswith(".npz"):
                        filelist.append(os.path.join(root, file))
            motions = []
            for file in filelist:
                motion_data = np.load(file)
                motions.append(motion_data["pose"])

            # load dataset stats
            stats_path = os.path.join(src_dir, "stats.npz")
            with np.load(stats_path, allow_pickle=True) as data:
                stats = dict(data)
            pose_mean, pose_std = stats["pose_mean"], stats["pose_std"]
            poses_padded = []
            for motion in motions:
                pose_trgt = motion * pose_std + pose_mean
                poses_padded.append(smpl_helpers.extract_from_norm_ds(pose_trgt))

            # load models
            if "models" in cfg:
                predictions = {}
                sens = {}
                for model in cfg.models:
                    # load model predictions
                    model_path = os.path.join(
                        utils.get_project_folder(), "learning/runs", model
                    )
                    with np.load(os.path.join(model_path, "predictions.npz")) as data:
                        data_in = dict(data)
                    predictions[model] = list(data_in.values())
                    # load model config
                    sens[model] = utils.load_config(model_path).experiment.sensors
        else:
            raise NameError("Desired source of motion asset is unclear")

        if cfg.frame_idx != -1:
            assert cfg.frame_idx <= len(poses_padded), "invalid frame index, aborting!"
            poses_padded = poses_padded[[cfg.frame_idx]]

    if cfg.source == "AMASS_SSP_nn":
        if "models" in cfg:
            # render selected frames for all models
            for model in cfg.models:
                pred_i = predictions[model]
                for asset_idx, frame_idx in cfg.assets:
                    last_frame = (
                        frame_idx + cfg.render_every_n_frame * cfg.render_frames
                    )
                    poses_i = pred_i[asset_idx][0][frame_idx:last_frame]
                    poses_padded_i = smpl_helpers.extract_from_norm_ds(poses_i)
                    cfg_i = cfg.copy()
                    cfg_i.name = cfg.name + "/" + str(model) + "/" + str(asset_idx)
                    cfg_i.sensors = sens[model]
                    render_poses(poses_padded_i, smpl_model, cfg_i)

            # render selected frames for groundtruth
            for asset_idx, frame_idx in cfg.assets:
                last_frame = frame_idx + cfg.render_every_n_frame * cfg.render_frames
                poses_padded_i = poses_padded[asset_idx][frame_idx:last_frame]
                cfg_i = cfg.copy()
                cfg_i.name = cfg.name + "/target/" + str(asset_idx)
                cfg_i.show_sensors = False
                render_poses(poses_padded_i, smpl_model, cfg_i)
        else:
            for i, poses_padded_i in enumerate(poses_padded):
                cfg_i = cfg.copy()
                folder_name = (
                    filelist[i].split("test/")[1].split(".npz")[0].split("/")[1]
                )
                folder_name = cfg.name + "/" + str(i) + "_" + folder_name
                cfg_i.name = folder_name
                render_poses(poses_padded_i, smpl_model, cfg_i)
    else:
        render_poses(poses_padded, smpl_model, cfg)


def render_poses(poses_padded, smpl_model, cfg):
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

    # iterate over desired frames in sequence
    seq_len = len(poses_padded)
    num_iterations = seq_len // cfg.render_every_n_frame
    for i in range(max(num_iterations, 1)):
        if cfg.show_accelerations and (i == 0 or i == num_iterations - 1):
            continue

        idx_i = i * cfg.render_every_n_frame
        print("rendering frame {}".format(idx_i))
        pose_i = poses_padded[[idx_i]]

        # lbs
        vertices_i, joints_i, rel_tfs_i = smpl_helpers.my_lbs(smpl_model, pose_i)
        vertices_i = utils.copy2cpu(vertices_i[0])
        joints_i = utils.copy2cpu(joints_i[0])
        rel_tfs_i = utils.copy2cpu(rel_tfs_i[0])

        if cfg.show_accelerations:
            pose_before = poses_padded[[(i - 1) * cfg.render_every_n_frame]]
            pose_after = poses_padded[[(i + 1) * cfg.render_every_n_frame]]
            sens_ids = list(sensors.SENS_VERTS_SSP.values())
            vertices_before, _, _ = smpl_helpers.my_lbs(smpl_model, pose_before)
            vertices_after, _, _ = smpl_helpers.my_lbs(smpl_model, pose_after)
            vert_0 = utils.copy2cpu(vertices_before[0, sens_ids])
            vert_1 = vertices_i[sens_ids]
            vert_2 = utils.copy2cpu(vertices_after[0, sens_ids])
            time_interval = cfg.render_every_n_frame / 60
            accs = (vert_0 - 2 * vert_1 + vert_2) / (time_interval ** 2)

        # compute meshes
        meshes = []

        # body mesh
        body_color = vis_tools.colors["grey"].copy()
        body_color.append(1 - cfg.body_transparency)
        body_mesh = trimesh.Trimesh(
            vertices=vertices_i,
            faces=smpl_model.faces,
            vertex_colors=body_color,
        )
        if cfg.show_body:
            meshes.append(body_mesh)

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
            chosen_sensors = (
                sensors.SENS_VERTS_SSP.keys() if cfg.sensors == "all" else cfg.sensors
            )
            for sensor, vert_ind in sensors.SENS_VERTS_SSP.items():
                if (
                    sensor not in ["left_pelvis", "right_pelvis"]
                    and sensor not in chosen_sensors
                ):
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
                # tf[:3, :3] = sensor_orientation.transpose()
                tf[:3, 3] = vertices_i[vert_ind]

                if cfg.show_sensor_orientations:
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
                    box.visual.vertex_colors = vis_tools.colors["orange"]
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
        if cfg.show_accelerations:
            for i, acc in enumerate(accs):
                cyl_orig = vert_1[i]
                cyl_end = cyl_orig + acc / 30
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
            filename = os.path.join(os.getcwd(), "files", cfg.name, str(idx_i) + ".png")
            im_arr = np.expand_dims(body_image, axis=(0, 1, 2))
            vis_tools.imagearray2file(im_arr, filename)


if __name__ == "__main__":
    render_from_config()
