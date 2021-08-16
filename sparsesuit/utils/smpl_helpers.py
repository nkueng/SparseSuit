import numpy as np
import smplx
from smplx import lbs
import os

import torch

from sparsesuit.constants import paths, sensors


def load_smplx(genders: list = None) -> dict:
    models = {}
    for gender in genders:
        models[gender] = smplx.create(
            model_path=os.path.join(paths.DATA_PATH, paths.SMPL_PATH),
            model_type="smplx",
            gender=gender,
            ext="npz",
        )
    return models


def my_lbs(
    model: smplx.SMPLX, pose: torch.Tensor, betas: torch.Tensor, pose2rot: bool = True
):
    """Performs Linear Blend Skinning with the given shape and pose parameters

    Parameters
    ----------
    betas : torch.tensor BxNB
        The tensor of shape parameters
    pose : torch.tensor Bx(J + 1) * 3
        The pose parameters in axis-angle format
    v_template torch.tensor BxVx3
        The template mesh that will be deformed
    shapedirs : torch.tensor 1xNB
        The tensor of PCA shape displacements
    posedirs : torch.tensor Px(V * 3)
        The pose PCA coefficients
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from
        the position of the vertices
    parents: torch.tensor J
        The array that describes the kinematic tree for the model
    lbs_weights: torch.tensor N x V x (J + 1)
        The linear blend skinning weights that represent how much the
        rotation matrix of each part affects each vertex
    pose2rot: bool, optional
        Flag on whether to convert the input pose tensor to rotation
        matrices. The default value is True. If False, then the pose tensor
        should already contain rotation matrices and have a size of
        Bx(J + 1)x9
    dtype: torch.dtype, optional

    Returns
    -------
    verts: torch.tensor BxVx3
        The vertices of the mesh after applying the shape and pose
        displacements.
    joints: torch.tensor BxJx3
        The joints of the model
    """

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = model.v_template + lbs.blend_shapes(betas, model.shapedirs)

    # Get the joints
    # NxJx3 array
    joints = lbs.vertices2joints(model.J_regressor, v_shaped)

    # align root with origin
    root = joints[:, [0]]
    joints -= root
    v_shaped -= root

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = lbs.batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(pose_feature, model.posedirs).view(
            batch_size, -1, 3
        )
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(
            pose_feature.view(batch_size, -1), model.posedirs
        ).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    joints_transf, rel_tfs = lbs.batch_rigid_transform(
        rot_mats, joints, model.parents, dtype=dtype
    )

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = model.lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = model.J_regressor.shape[0]
    T = torch.matmul(W, rel_tfs.view(batch_size, num_joints, 16)).view(
        batch_size, -1, 4, 4
    )

    homogen_coord = torch.ones(
        [batch_size, v_posed.shape[1], 1], dtype=dtype, device=device
    )
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, joints_transf, rel_tfs


def smpl_reduced_to_full(smpl_reduced, target_joints):
    """
    Converts an np array that uses the reduced smpl representation (15) into the full representation (24) by filling in
    the identity rotation for the missing joints. Can handle either rotation input (dof = 9) or quaternion input
    (dof = 4).
    :param smpl_full: An np array of shape (seq_length, n_joints_reduced*dof)
    :return: An np array of shape (seq_length, 24*dof)
    """
    # TODO: make hands look nice (relaxed)
    dof = smpl_reduced.shape[1] // len(target_joints)
    assert dof == 9 or dof == 4
    seq_length = smpl_reduced.shape[0]
    smpl_full = np.zeros([seq_length, sensors.NUM_SMPL_JOINTS * dof])
    for idx in range(sensors.NUM_SMPL_JOINTS):
        if idx in target_joints:
            red_idx = target_joints.index(idx)
            smpl_full[:, idx * dof : (idx + 1) * dof] = smpl_reduced[
                :, red_idx * dof : (red_idx + 1) * dof
            ]
        else:
            if dof == 9:
                identity = np.repeat(np.eye(3, 3)[np.newaxis, ...], seq_length, axis=0)
            else:
                identity = np.concatenate(
                    [np.array([[1.0, 0.0, 0.0, 0.0]])] * seq_length, axis=0
                )
            smpl_full[:, idx * dof : (idx + 1) * dof] = np.reshape(identity, [-1, dof])
    return smpl_full


def smpl_rot_to_global(smpl_rotations_local):
    """
    Converts local smpl rotations into global rotations by "unrolling" the kinematic chain.
    :param smpl_rotations_local: np array of rotation matrices of shape (..., N, 3, 3), or (..., 216) where N
      corresponds to the amount of joints in SMPL (currently 24)
    :return: The global rotations as an np array of the same shape as the input.
    """
    in_shape = smpl_rotations_local.shape
    do_reshape = in_shape[-1] != 3
    if do_reshape:
        assert in_shape[-1] == 216
        rots = np.reshape(
            smpl_rotations_local, in_shape[:-1] + (sensors.NUM_SMPL_JOINTS, 3, 3)
        )
    else:
        rots = smpl_rotations_local

    out = np.zeros_like(rots)
    dof = rots.shape[-3]
    for j in range(dof):
        if sensors.SMPL_PARENTS[j] < 0:
            out[..., j, :, :] = rots[..., j, :, :]
        else:
            parent_rot = out[..., sensors.SMPL_PARENTS[j], :, :]
            local_rot = rots[..., j, :, :]
            out[..., j, :, :] = np.matmul(parent_rot, local_rot)

    if do_reshape:
        out = np.reshape(out, in_shape)

    return out
