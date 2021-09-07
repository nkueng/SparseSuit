import os
import random

import numpy as np
import torch
import trimesh
from pyrender import Mesh
from trimesh import creation
from body_visualizer.mesh import sphere, cube
from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools import vis_tools
from human_body_prior.tools import omni_tools

from sparsesuit.constants import paths, sensors
from sparsesuit.utils import smpl_helpers, utils

utils.make_deterministic(14)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load SMPL model
num_betas = 10
smpl_model = smpl_helpers.load_smplx("male", device=device, model_type="smplx")

faces = omni_tools.copy2cpu(smpl_model.faces)

# load motion asset
accad_dir_name = os.path.join(paths.DATA_PATH, "Synthetic/AMASS/ACCAD/")
filelist = []
for root, dirs, files in os.walk(accad_dir_name):
    if files == []:
        continue
    for file in files:
        filelist.append(os.path.join(root, file))

asset_ind = 4  # 2 used for introduction figures
pose_data = np.load(filelist[asset_ind])

# set up renderer
w, h = 1000, 1000
show_rendering = True
show_from_behind = False
show_body = True
body_transparency = 0.6  # [0,1.0]
show_joints = True
show_skeleton = True
show_sensors = False
show_sensor_orientation = True
show_markers = False
use_default_pose = False

mv = MeshViewer(
    w,
    h,
    use_offscreen=not show_rendering,
    # camera_translation=[0, 0.3, 2],
    camera_translation=[0, 0.65, 2],
)

# extract pose from asset
poses_padded = smpl_helpers.extract_from_smplh(
    pose_data["poses"], list(sensors.SMPL_JOINT_IDS.values())
)
# fix root orientation
poses_padded[:, :3] = 0

render_every_frame = 30
for i in range(len(poses_padded) // render_every_frame):
    frame_i = i * render_every_frame
    print("rendering frame {}".format(frame_i))

    pose_i = poses_padded[[frame_i]]
    if use_default_pose:
        pose_i = torch.zeros(pose_i.shape)
    if show_from_behind:
        pose_i[1] = np.pi
    # compute SMPL pose
    vertices_i, joints_i, rel_tfs_i = smpl_helpers.my_lbs(smpl_model, pose_i)

    # vertices_i = smpl_poses.vertices[0]
    vertices_i = utils.copy2cpu(vertices_i[0])
    # joints_i = smpl_poses.joints[0, :55]
    joints_i = utils.copy2cpu(joints_i[0])
    rel_tfs_i = utils.copy2cpu(rel_tfs_i[0])

    # container for all meshes to be added to scene
    meshes = []

    # add body
    vis_vertices = vertices_i
    body_color = vis_tools.colors["grey"].copy()
    body_color.append(1 - body_transparency)
    body_mesh = trimesh.Trimesh(
        vertices=vis_vertices,
        faces=faces,
        vertex_colors=body_color,
    )
    if show_body:
        meshes.append(body_mesh)

    # add joints
    if show_joints:
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

    # add rig (cylinders connecting joints)
    if show_skeleton and show_joints:
        vis_bones = []
        for joint_ind, parent_ind in enumerate(sensors.SMPL_PARENTS):
            if parent_ind == -1:
                continue
            # define endpoints of cylinder as joint and parent joint coords
            segment = all_joints[[joint_ind, parent_ind]]
            cyl = creation.cylinder(radius=0.005, segment=segment)
            cyl.visual.vertex_colors = vis_tools.colors["red"]
            meshes.append(cyl)

    # add sensors
    if show_sensors:
        sens_verts = vis_vertices[list(sensors.SENS_VERTS_SSP.values())]
        vis_sensors = cube.points_to_cubes(
            sens_verts, radius=0.015, point_color=vis_tools.colors["orange"]
        )
        meshes.append(vis_sensors)

    # add sensor orientations
    if show_sensor_orientation:
        chosen_sensors = ["right_wrist", "right_elbow", "right_shoulder"]
        for sensor, vert_ind in sensors.SENS_VERTS_SSP.items():
            if sensor not in chosen_sensors:
                continue
            ori_ind = sensors.SENS_JOINTS_IDS[sensor]
            tf = np.eye(4)
            tf[:3, :3] = rel_tfs_i[ori_ind, :3, :3]
            tf[:3, 3] = vis_vertices[vert_ind]
            box = creation.box(
                extents=[0.05, 0.01, 0.03],
                transform=tf,
            )
            ax = creation.axis(
                transform=tf,
                origin_size=0.007,
                origin_color=[0, 0, 0],
                axis_radius=0.005,
                axis_length=0.05,
            )

            meshes.append(ax)
            meshes.append(box)

    # add "mocap markers"
    if show_markers:
        vis_markers = sphere.points_to_spheres(
            vis_vertices[sensors.MOCAP_MARKERS],
            radius=0.01,
            point_color=vis_tools.colors["brown"],
        )
        meshes.append(vis_markers)

    if show_rendering:
        mv.viewer.render_lock.acquire()
    mv.set_static_meshes(meshes, smooth=False)
    if show_rendering:
        mv.viewer.render_lock.release()

    if mv.use_offscreen:
        body_image = mv.render(render_wireframe=False)
        filename = os.path.join(os.getcwd(), "renderings", str(frame_i) + ".png")
        im_arr = np.expand_dims(body_image, axis=(0, 1, 2))
        vis_tools.imagearray2file(im_arr, filename)
    else:
        input("press enter to continue")
