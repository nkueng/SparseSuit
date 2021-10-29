"""
This script converts Rokoko Studio poses (given as .npz by fbx2npz.py) to SMPL poses by retargeting the skeletons.
"""
import os

import numpy as np

from sparsesuit.constants import paths, sensors
from Skynet.Modules import MLRetargeting

from scipy.spatial.transform import Rotation as R


def get_data(file_path):
    if os.path.isfile(file_path):
        with np.load(file_path) as file_data:
            return dict(file_data)
    else:
        return None


def retarget_motion(motion):
    # unpack input motion data
    parents_in = motion["parents"]
    skeleton_in = list(motion["skeleton"])
    positions_in = motion["positions"]
    rotations_in = motion["rotations"]

    # load retargeter with SMPL skeleton
    retargeter = MLRetargeting.SkeletonSMPL()

    # remove "root" from parents and skeleton as it is not in positions/rotations
    # parents_ = parents.copy()
    parents_in = [parent - 1 for parent in parents_in[1:]]
    # skeleton_ = skeleton.copy()
    skeleton_in.remove("Root")

    # extract relevant joints from fbx data
    fbx2smpl = sensors.FBX_2_SMPL
    joints_rel = list(fbx2smpl.keys())
    positions_rel = []
    rotations_rel = []
    parents_rel = []
    skeleton_rel = []
    for joint in joints_rel:
        idx = skeleton_in.index(joint)
        positions_rel.append(positions_in[:, idx] / 100)
        rotations_rel.append(rotations_in[:, idx])
        skeleton_rel.append(joint)
        if idx == 0:
            parents_rel.append(-1)
        else:
            parent_joint = skeleton_in[parents_in[idx]]
            parents_rel.append(skeleton_rel.index(parent_joint))

    positions_rel = np.array(positions_rel).transpose([1, 0, 2])
    rotations_rel = np.array(rotations_rel).transpose([1, 0, 2])

    # change names from fbx to smpl convention for joint matching
    skeleton_rel = [fbx2smpl[joint_name] for joint_name in skeleton_rel]

    # retarget fbx to smpl skeleton
    r_out, _, _, _, _, p_target = retargeter(
        positions_rel,
        rotations_rel,
        parents_rel,
        skeleton_rel,
        use_tiny=False,
        from_frame=1,  # clip first frame with T-pose
        max_ik_iterations=10,
        silent=True,
    )

    # change ordering to smpl convention
    seq_len = len(r_out)
    smpl_ids = [
        sensors.SMPL_JOINT_IDS[trgt_joint]
        for trgt_joint in retargeter.target_skeleton_joints
    ]
    smpl_poses = np.tile(np.zeros(3), [seq_len, sensors.NUM_SMPLX_JOINTS, 1])
    for i in range(seq_len):
        rots = R.from_quat(r_out[i])
        smpl_poses[i, smpl_ids] = rots.as_rotvec()

    return smpl_poses.reshape([seq_len, -1])


if __name__ == "__main__":
    # get a list of all .npz files in source directory
    src_dir = os.path.join(paths.SOURCE_PATH, "raw_SSP_dataset/SSP_data/Export")

    # walk over all files in directory and collect relevant paths
    npz_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".npz"):
                npz_files.append(os.path.join(root, file))

    npz_files = sorted(npz_files)

    # retarget from FBX skeleton to SMPL skeleton
    for file in npz_files:
        # DEBUG
        # file = file.split("Export")[0]
        # file += "Export/01/5 Maximal jump/take-10_DEFAULT_R15.npz"

        # skip output files
        if "smpl" in file:
            continue

        print("Converting file {}".format(file))

        # load data
        data = get_data(file)
        if data is None:
            print("Got empty file for {}. Skipping!".format(file))
            continue

        # retarget to SMPL
        smpl_poses = retarget_motion(data)

        # dump SMPL poses as npz
        out_dict = {"poses": smpl_poses}
        out_name = file.split(".npz")[0] + "_smpl.npz"
        with open(out_name, "wb") as fout:
            np.savez_compressed(fout, **out_dict)
