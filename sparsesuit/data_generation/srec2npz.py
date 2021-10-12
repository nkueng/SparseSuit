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

from sparsesuit.constants import paths
from sparsesuit.constants.sensors import (
    SENS_NAMES_SSP,
    SREC_2_SSP,
    SENS_VERTS_SSP,
    SENS_JOINTS_IDS,
)
from sparsesuit.utils import smpl_helpers, utils


def get_srec(file):
    reader = SrecReader.SRecReader(filename=file.split(".srec")[0])
    try:
        rec = reader.ReadSrec(parse=False)
    except AssertionError:
        print("Asset {} is not valid.".format(file))
        rec = SrecReader.SRec(".")
    return rec


def parse_srec(file):
    # get sensor order in our convention
    rec = get_srec(file)
    if rec.num_suits != 1:
        raise AttributeError("Wrong number of suits, skipping!")
    sensor_add = rec.suits[0].frames[0].addresses
    sensor_names = [SREC_2_SSP[rec.setSensorName(add)] for add in sensor_add]
    sort_ids = [sensor_names.index(sensor) for sensor in SENS_NAMES_SSP]

    # get expected orientation of joints in straight pose
    calib_oris = smpl_helpers.get_straight_pose_oris()

    print(file)

    # parse all frames in file
    frames = rec.suits[0].frames

    # containers for calibration rotations from first frame
    ori_offsets = []
    rot_world_to_model = []

    # containers for calibrated measurements per frame
    accs = []
    oris = []
    for frame in frames:
        # convert orientations from NED to XYZ (180 deg rotation around N-axis)
        quats = np.array(frame.quaternion)[sort_ids]
        ori_ned = R.from_quat(quats[:, [1, 2, 3, 0]])
        rot_ned_to_xyz = R.from_rotvec([np.pi, 0, 0])
        ori_xyz = rot_ned_to_xyz * ori_ned
        ori_xyz_mat = ori_xyz.as_matrix()

        # convert acceleration from g to m/s² (given in sensor frame)
        acc_local_np = np.array(frame.acceleration)[sort_ids] * 9.81
        acc_local = np.expand_dims(acc_local_np, axis=2)

        # transform acceleration from local sensor to global frame
        acc_global = []
        for rot_local_to_global_i, acc_local_i in zip(ori_xyz_mat, acc_local):
            acc_glob_i = rot_local_to_global_i @ acc_local_i
            # subtract gravity from z-coordinate
            acc_glob_i[2] -= 9.81
            acc_global.append(acc_glob_i)
        acc_global = np.stack(acc_global).squeeze()

        # estimate world facing direction for calibration in first frame from sensors on back
        if len(rot_world_to_model) == 0:
            left_oris = []
            right_oris = []
            for i, sensor_name in enumerate(SENS_NAMES_SSP):
                sensor_ori = ori_xyz_mat[i]
                if "left" in sensor_name:
                    left_oris.append(sensor_ori)
                if "right" in sensor_name:
                    right_oris.append(sensor_ori)
            z_axis = np.array([0, 0, 1])
            # forward is the positive z-axis of all the sensors on the back
            forward_world = []
            for l_ori_i, r_ori_i in zip(left_oris, right_oris):
                forward_i = l_ori_i @ z_axis + r_ori_i @ z_axis
                forward_world.append(forward_i)
                # forward_world.append(forward_i)
            forward_world = np.mean(forward_world[:3], axis=0)
            rot_world_to_model = utils.rot_from_vecs(forward_world, np.array([1, 0, 0]))
            # rot_world_to_model = np.eye(3)

        # transform accelerations from global to model frame with initial facing direction
        # acc_model = (rot_world_to_model @ acc_global.T).T
        acc_model = acc_global @ rot_world_to_model.T

        # express global sensor orientations in model frame with facing direction
        ori_model = np.einsum("jk,ikl->ijl", rot_world_to_model, ori_xyz_mat)

        # get calibration offset between joint and sensor orientations from first frame in straight pose
        if len(ori_offsets) == 0:
            for i, ori_model_i in enumerate(ori_model):
                calib_ori_i = calib_oris[SENS_JOINTS_IDS[SENS_NAMES_SSP[i]]]
                ori_offsets.append(ori_model_i.T @ calib_ori_i)
            ori_offsets = np.stack(ori_offsets)

        # convert sensor orientations to joint orientations with calibration offset
        ori_norm = np.einsum("ijk,ikl->ijl", ori_model, ori_offsets)

        # debugging visualizations
        # ori_norm = ori_model
        # acc_global = np.stack(forward_world)
        # acc_global = np.insert(acc_global, 6, np.ones([1, 3]), 0)
        # acc_global[6] = np.mean(acc_global[:6:2], axis=0)

        oris.append(ori_norm)
        accs.append(acc_model)

    if VISUALIZE:
        # load corresponding smpl poses
        take = file.split(".srec")[0].split("/")[-1]
        file_path = file.split(".srec")[0].split("/")[-3:-1]
        file_path = os.path.join(*file_path)
        pose_dir = os.path.join(
            paths.DATA_PATH, "raw_SSP_dataset/SSP_data/Export", file_path
        )
        file_list = os.listdir(pose_dir)
        this_file = [file for file in file_list if "smpl" in file and take in file]
        file_name = os.path.join(pose_dir, this_file[0])
        with np.load(file_name) as data:
            pose_data = dict(data)
        poses = pose_data["poses"]
        seq_len = len(poses)
        poses_rotvec = torch.Tensor(utils.rot_matrix_to_aa(poses))
        vis_oris_accs(accs, oris, poses_rotvec)

    return np.stack(oris), np.stack(accs)


def vis_oris_accs(oris, accs, pose=None):
    from sparsesuit.utils import visualization, smpl_helpers, utils

    play_frames = 1200

    smpl_model = smpl_helpers.load_smplx()
    if pose is None:
        pose = smpl_helpers.generate_straight_pose()
        pose[:, :3] = torch.ones(3) * 0.5773502691896258 * 2.0943951023931957
        body_mesh, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
        verts = np.tile(utils.copy2cpu(body_mesh), [play_frames, 1, 1])
        joints = np.tile(utils.copy2cpu(joints[:, :22]), [play_frames, 1, 1])
        poses = np.tile(utils.copy2cpu(rel_tfs[:, :22, :3, :3]), [play_frames, 1, 1, 1])
    else:
        rot = R.from_rotvec(np.ones(3) * 0.5773502691896258 * 2.0943951023931957)
        pelvis = R.from_rotvec(pose[:, :3])
        pose[:, :3] = torch.Tensor((rot * pelvis).as_rotvec())
        verts, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
        verts = utils.copy2cpu(verts)
        joints = utils.copy2cpu(joints)
        rel_tfs = utils.copy2cpu(rel_tfs)

    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=[verts],
        play_frames=play_frames,
        playback_speed=0.3,
        sensors=[verts[:, list(SENS_VERTS_SSP.values())]],
        oris=[oris],
        accs=[accs],
        # joints=[joints],
        # pose=[poses],
    )


if __name__ == "__main__":
    VISUALIZE = False
    PLOT = False

    # set src directory with files
    src_dir = os.path.join(paths.DATA_PATH, "raw_SSP_dataset/SSP_data")

    # walk over all files in directory and collect relevant paths
    srec_files = []
    # fbx_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".srec"):
                srec_files.append(os.path.join(root, file))

            # if file.endswith(".fbx"):
            #     fbx_files.append(os.path.join(root, file))

    srec_files = sorted(srec_files)
    # fbx_files = sorted(fbx_files)

    # parse SREC and extract IMU data in correct frame
    # oris, accs = parse_srec(srec_files)
    for file in srec_files:

        # skip if previously converted
        out_name = file.split(".srec")[0] + ".npz"
        if os.path.exists(out_name):
            continue

        try:
            oris, accs = parse_srec(file)
        except AttributeError:
            continue

        # plot some IMU data
        if PLOT:
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

        # dump IMU data in same directory as source
        out_dict = {
            "acc": accs,
            "ori": oris,
        }

        # drop motion data into npz
        with open(out_name, "wb") as fout:
            np.savez_compressed(fout, **out_dict)
