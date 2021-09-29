"""
Converts directories of IMU sensor data (SREC) and corresponding poses (FBX) to the format we use.
"""
import os
import numpy as np
import SrecReader
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib as mpl
import matplotlib.pyplot as plt
from bvh import Bvh

from sparsesuit.constants import paths
from sparsesuit.constants.sensors import (
    SENS_NAMES_SSP,
    SREC_2_SSP,
    SENS_VERTS_SSP,
    SENS_JOINTS_IDS,
)
from sparsesuit.utils import smpl_helpers


def get_bvh(file):
    with open(file) as f:
        bvh = Bvh(f.read())
    return bvh


def parse_bvh(files):
    for file in files:
        bvh = get_bvh(file)
    return poses


def get_srec(file):
    reader = SrecReader.SRecReader(filename=file.split(".srec")[0])
    try:
        rec = reader.ReadSrec(parse=False)
    except AssertionError:
        print("Asset {} is not valid.".format(file))
        rec = SrecReader.SRec(".")
    return rec


def parse_srec(files):
    # get sorting order for to our convention
    rec = get_srec(files[0])
    sensor_add = rec.suits[0].frames[0].addresses
    sensor_names = [SREC_2_SSP[rec.setSensorName(add)] for add in sensor_add]
    sort_ids = [sensor_names.index(sensor) for sensor in SENS_NAMES_SSP]

    # get expected orientation of sensors in straight pose
    calib_oris = smpl_helpers.get_straight_pose_oris()

    # parse all files
    accs = []
    oris = []
    for file in files:
        print(file)
        rec = get_srec(file)
        if rec.num_suits != 1:
            continue
        frames = rec.suits[0].frames
        acc_f = []
        ori_f = []
        ori_offsets = []
        for frame in frames:
            # convert acceleration from g to m/s²
            acc_local_np = np.array(frame.acceleration)[sort_ids] * 9.81
            acc_local = np.expand_dims(acc_local_np, axis=2)

            # convert orientations from quaternion to rotation matrix format
            quats = np.array(frame.quaternion)[sort_ids]
            ori_ned = R.from_quat(quats[:, [1, 2, 3, 0]])

            # convert NED to XYZ (180 deg rotation around N-axis)
            rot = R.from_rotvec([np.pi, 0, 0])
            ori_xyz = rot * ori_ned
            # rot = R.from_rotvec([0, 0, np.pi / 2])
            # ori_xyz = rot * ori_ned
            ori = ori_xyz.as_matrix()

            # transform acceleration to global frame
            acc_global = []
            for ori_i, acc_i in zip(ori, acc_local):
                acc_i_glob = ori_i @ acc_i
                # subtract gravity
                acc_i_glob[2] -= 9.81
                acc_global.append(acc_i_glob)

            acc_global = np.stack(acc_global).squeeze()

            # calibration step: Get offset from straight pose orientations of sensors and normalize by that orientation
            if len(ori_offsets) == 0:
                for i, ori_i in enumerate(ori):
                    calib_ori_i = calib_oris[SENS_JOINTS_IDS[SENS_NAMES_SSP[i]]]
                    ori_offsets.append(calib_ori_i @ ori_i.T)
                ori_offsets = np.stack(ori_offsets)

            ori_norm = np.einsum("ijk,ikl->ijl", ori_offsets, ori)

            ori_f.append(ori_norm)
            acc_f.append(acc_global)

        if VISUALIZE:
            vis_oris_accs(ori_f, acc_f)

        oris.append(ori_f)
        accs.append(acc_f)

    return np.stack(oris), np.stack(accs)


def vis_oris_accs(oris, accs):
    from sparsesuit.utils import visualization, smpl_helpers, utils

    play_frames = 1

    smpl_model = smpl_helpers.load_smplx()
    pose = smpl_helpers.generate_straight_pose()
    pose[:, :3] = torch.ones(3) * 0.5773502691896258 * 2.0943951023931957
    body_mesh, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
    verts = np.tile(utils.copy2cpu(body_mesh), [play_frames, 1, 1])
    joints = np.tile(utils.copy2cpu(joints[:, :22]), [play_frames, 1, 1])
    poses = np.tile(utils.copy2cpu(rel_tfs[:, :22, :3, :3]), [play_frames, 1, 1, 1])
    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=[verts],
        play_frames=play_frames,
        playback_speed=0.1,
        # sensors=[verts[:, list(SENS_VERTS_SSP.values())]],
        # oris=[oris],
        # accs=[accs],
        joints=[joints],
        pose=[poses],
    )


if __name__ == "__main__":
    VISUALIZE = True
    PLOT = True

    # set src directory with files
    src_dir = os.path.join(paths.DATA_PATH, "raw_SSP_dataset/SSP_data")

    # walk over all files in directory and collect relevant paths
    srec_files = []
    bvh_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".srec"):
                srec_files.append(os.path.join(root, file))

            if file.endswith(".bvh"):
                bvh_files.append(os.path.join(root, file))

    # parse SREC and extract IMU data in correct frame
    # oris, accs = parse_srec(srec_files)
    oris, accs = parse_srec([srec_files[0]])

    # plot some IMU data
    first_frame = 200
    last_frame = 300
    frame_range = range(first_frame, last_frame)
    y = accs[0, frame_range, 0]
    x = np.linspace(0, len(y), len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Real Acceleration Signals")
    plt.xlabel("Frame Number")
    plt.ylabel("Acceleration [m/s²]")
    fig.show()

    # parse BVH and extract poses in correct frame
    # poses = parse_bvh(bvh_files)

    # clean pose data
