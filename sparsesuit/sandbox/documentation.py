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
smpl_model = smpl_helpers.load_smplx("neutral").to(device)

faces = omni_tools.copy2cpu(smpl_model.faces)

# load motion asset
amass_npz_fname = os.path.join(
    paths.DATA_PATH, "Synthetic/AMASS/ACCAD/Female1General_c3d/A6 - lift box_poses.npz"
)
pose_data = np.load(amass_npz_fname)

# set up renderer
w, h = 1000, 1000
show_rendering = True
show_behind = False
mv = MeshViewer(w, h, use_offscreen=not show_rendering)

# pose from asset
poses_padded = smpl_helpers.extract_from_smplh(
    pose_data["poses"], sensors.SMPL_SSP_JOINTS
)

# default pose
poses_padded = torch.zeros(poses_padded.shape)

if show_behind:
    poses_padded[:, 1] = np.pi

# lbs: pose -> smpl model with skin (vertices) and joints
vertices, joints, _ = smpl_helpers.my_lbs(smpl_model, poses_padded)

render_every_frame = 30
for i in range(len(poses_padded) // render_every_frame):
    frame_i = i * render_every_frame
    print("rendering frame {}".format(frame_i))

    # add body
    vis_vertices = omni_tools.copy2cpu(vertices[frame_i])
    body_color = vis_tools.colors["grey"].copy()
    body_color.append(0.7)
    body_mesh = trimesh.Trimesh(
        vertices=vis_vertices,
        faces=faces,
        vertex_colors=body_color,
    )

    # add joints
    rig_joints = omni_tools.copy2cpu(joints[frame_i, sensors.SMPLX_RIG_JOINTS])
    body_joints = rig_joints[:-2]
    lhand_joint = np.mean(rig_joints[[-4, -2]], axis=0)
    rhand_joint = np.mean(rig_joints[[-3, -1]], axis=0)
    all_joints = np.vstack([body_joints, lhand_joint, rhand_joint])
    vis_joints = sphere.points_to_spheres(
        all_joints, radius=0.015, point_color=vis_tools.colors["red"]
    )

    # add rig (cylinders connecting joints)
    vis_bones = []
    for joint_ind, parent_ind in enumerate(sensors.SMPL_PARENTS):
        if parent_ind == -1:
            continue
        # define endpoints of cylinder as joint and parent joint coords
        segment = all_joints[[joint_ind, parent_ind]]
        cyl = creation.cylinder(radius=0.005, segment=segment)
        cyl.visual.vertex_colors = vis_tools.colors["red"]
        vis_bones.append(cyl)

    # add sensors
    sens_verts = vis_vertices[list(sensors.SENS_VERTS_SSP.values())]
    vis_sensors = cube.points_to_cubes(
        sens_verts, radius=0.015, point_color=vis_tools.colors["grey-blue"]
    )

    # add "mocap markers" (sample N random vertices)
    num_markers = 100
    marker_ind = [
        3856,
        4204,
        4532,
        4595,
        4448,
        3215,
        3440,
        3507,
        3631,
        3701,
        3757,
        5879,
        5894,
        8590,
        8580,
        6491,
        6337,
        6353,
        6224,
        7272,
        6920,
        7023,
        6022,
        3309,
        3556,
        5985,
        4452,
        3398,
        7140,
        3546,
        5993,
        6086,
        3470,
        4019,
        6737,
        6646,
        564,
        798,
        2197,
        6862,
        5614,
        8387,
        6630,
        5450,
        7188,
        2474,
        6380,
        6499,
        6854,
        8237,
        3268,
    ]
    # marker_ind = random.sample(range(len(vis_vertices)), num_markers)
    vis_markers = sphere.points_to_spheres(
        vis_vertices[marker_ind], radius=0.01, point_color=vis_tools.colors["brown"]
    )

    mv.set_static_meshes(
        [
            body_mesh,
            # *vis_bones,
            # vis_joints,
            # vis_sensors,
            vis_markers,
        ]
    )

    if mv.use_offscreen:
        body_image = mv.render(render_wireframe=False)
        filename = os.path.join(os.getcwd(), "renderings", str(frame_i) + ".png")
        im_arr = np.expand_dims(body_image, axis=(0, 1, 2))
        vis_tools.imagearray2file(im_arr, filename)
    else:
        break
