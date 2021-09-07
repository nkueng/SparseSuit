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
# TODO: count number of files in directory and choose by index
amass_npz_fname = os.path.join(
    paths.DATA_PATH, "Synthetic/AMASS/ACCAD/Female1General_c3d/A6 - lift box_poses.npz"
)
pose_data = np.load(amass_npz_fname)

# set up renderer
w, h = 1000, 1000
show_rendering = False
show_from_behind = False
show_joints = False
show_skeleton = False
show_sensors = False
show_markers = True
use_default_pose = False

mv = MeshViewer(w, h, use_offscreen=not show_rendering, camera_translation=[0, 0.3, 2])

# pose from asset
poses_padded = smpl_helpers.extract_from_smplh(
    pose_data["poses"], sensors.SMPL_SSP_JOINTS
)

# default pose
if use_default_pose:
    poses_padded = torch.zeros(poses_padded.shape)

if show_from_behind:
    poses_padded[:, 1] = np.pi

# lbs: pose -> smpl model with skin (vertices) and joints
# vertices, joints, _ = smpl_helpers.my_lbs(smpl_model, poses_padded)

render_every_frame = 30
for i in range(len(poses_padded) // render_every_frame):
    frame_i = i * render_every_frame
    # vertices_i = vertices[frame_i]
    # joints_i = joints[frame_i]
    print("rendering frame {}".format(frame_i))

    # compute SMPL pose alternatively
    # body_pose = poses_padded[[frame_i], 3:72].to(device)
    body_pose = poses_padded[[frame_i], 3:66].to(device)
    smpl_poses = smpl_model(body_pose=body_pose, return_verts=True)
    vertices_i = smpl_poses.vertices[0]
    joints_i = smpl_poses.joints[0, :55]

    # container for all meshes to be added to scene
    meshes = []

    # add body
    vis_vertices = omni_tools.copy2cpu(vertices_i)
    body_color = vis_tools.colors["grey"].copy()
    body_color.append(0.7)
    body_mesh = trimesh.Trimesh(
        vertices=vis_vertices,
        faces=faces,
        vertex_colors=body_color,
    )
    meshes.append(body_mesh)

    # add joints
    if show_joints:
        rig_joints = omni_tools.copy2cpu(joints_i[sensors.SMPLX_RIG_JOINTS])
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
            sens_verts, radius=0.015, point_color=vis_tools.colors["grey-blue"]
        )
        meshes.append(vis_sensors)

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
    mv.set_static_meshes(meshes)
    if show_rendering:
        mv.viewer.render_lock.release()

    if mv.use_offscreen:
        body_image = mv.render(render_wireframe=False)
        filename = os.path.join(os.getcwd(), "renderings", str(frame_i) + ".png")
        im_arr = np.expand_dims(body_image, axis=(0, 1, 2))
        vis_tools.imagearray2file(im_arr, filename)
    else:
        break
