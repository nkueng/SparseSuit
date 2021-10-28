import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import pickle as pkl
from pathlib import Path

import trimesh

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import signal

from body_visualizer.mesh.mesh_viewer import MeshViewer
from body_visualizer.tools import vis_tools

from sparsesuit.constants import paths
from sparsesuit.utils import smpl_helpers, utils, visualization

pio.renderers.default = "browser"


def extract_pkl_data(file_path):
    if os.path.isfile(file_path):
        with open(file_path, "rb") as fin:
            data_in = pkl.load(fin, encoding="latin1")
        return data_in["fullpose"]
    else:
        return None


def extract_npz_data(file_path):
    if os.path.isfile(file_path):
        with np.load(file_path) as file_data:
            data_in = dict(file_data)
        return data_in
    else:
        return None


def plot(joints, title):
    y = joints
    x = np.linspace(0, len(y), len(y))
    fig, ax = plt.subplots()
    ax.set_prop_cycle(color=["red", "green", "blue"])
    ax.plot(x, y)
    ax.set_title(title)
    plt.xlabel("Frame Number")
    plt.ylabel("Joint Position [m]")
    plt.legend(["x", "y", "z"])
    fig.show()


def render(verts, fps):
    # load mesh viewer
    SHOW_RENDERING = False
    mv = MeshViewer(
        1000,
        1000,
        use_offscreen=not SHOW_RENDERING,
        camera_translation=[0, 0.5, 2],
        # camera_translation=cfg.camera_translation,
        # camera_angle=cfg.camera_angle,
        # orthographic=cfg.orthographic_camera,
    )

    images = []
    for i, vertices_i in enumerate(verts):
        print("Rendering frame {}/{}".format(i, len(verts)))
        meshes = []
        body_color = np.ones(vertices_i.shape) * 0.7
        body_mesh = trimesh.Trimesh(
            vertices=vertices_i,
            faces=smpl_model.faces,
            vertex_colors=body_color,
        )
        meshes.append(body_mesh)
        mv.set_static_meshes(meshes, smooth=False)
        if not SHOW_RENDERING:
            body_image = mv.render(render_wireframe=False)
            images.append(body_image)

    print("Creating gif...")
    vicon_file_name = vicon_file.split("_stageii")[0]
    vicon_file_name = vicon_file_name.split("/")[-2:]
    folder_path = os.path.join(paths.DOC_PATH, "gifs", "Vicon_data")
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    filename = os.path.join(
        folder_path, vicon_file_name[0], vicon_file_name[1] + ".gif"
    )
    # print(filename)
    im_arr = np.stack(images)
    im_arr = np.expand_dims(im_arr, axis=(0, 1))
    vis_tools.imagearray2file(im_arr, filename, fps=fps)


def visualize(verts, fps):
    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=[verts],
        play_frames=500,
        playback_speed=1,
        # sensors=[verts[:, list(SENS_VERTS_SSP.values())]],
        # oris=[oris],
        # accs=[accs],
        # joints=[joints],
        # pose=[poses],
        fps=fps,
    )


def str2list(string):
    string = string.split(",")
    return [int(char) for char in string]


def str2int(string):
    return [int(char) for char in string if str.isdigit(char)]


def find_vicon_file(vicon_files, subject, trial):
    trial_num = trial.split("T")[1]
    name_matches = [
        file for file in vicon_files if subject in file and trial_num in file
    ]
    if len(name_matches) > 1:
        raise NameError
    if len(name_matches) == 0:
        raise NameError("Trial {} doesn't exist. Delete in yaml!".format(trial))
    return name_matches[0]


def find_studio_file(studio_files, subject, studio_key):
    take = "/take-" + str(studio_key[1]) + "_"
    take_matches = [file.split(take)[0] for file in studio_files if take in file]
    subject = str.zfill(str(subject), 2)
    subject_matches = [file for file in take_matches if subject in file]
    motion_options = [file.split(subject)[1] for file in subject_matches]
    motion_matches = [file for file in motion_options if str(studio_key[0]) in file]

    if len(motion_matches) == 0:
        raise NameError(
            "Studio file {} doesn't exist. Delete in yaml.".format(
                subject + str(studio_key)
            )
        )

    file_name_identifier = subject + motion_matches[0] + take

    name_matches = [file for file in studio_files if file_name_identifier in file]

    if len(name_matches) > 1:
        raise NameError
    return name_matches[0]


def find_sensor_file(sensor_files, subject, studio_key):
    take = "/take-" + str(studio_key[1]) + "."
    take_matches = [file.split(take)[0] for file in sensor_files if take in file]
    subject = str.zfill(str(subject), 2)
    subject_matches = [file for file in take_matches if subject in file]
    motion_options = [file.split(subject)[1] for file in subject_matches]
    motion_matches = [file for file in motion_options if str(studio_key[0]) in file]

    file_name_identifier = subject + motion_matches[0] + take

    name_matches = [file for file in sensor_files if file_name_identifier in file]

    if len(name_matches) > 1:
        raise NameError
    return name_matches[0]


def find_valid_frames(log_file):
    # open log file
    valid_frames = []
    with open(log_file, "r") as file:
        lines = file.readlines()
    for line in lines:
        if "Step 1. initial" in line:
            valid_frames.append(True)
        elif "skipping the frame" in line:
            valid_frames.append(False)
        elif "Starting mosh stageii for" in line:
            # start over
            valid_frames = []
    return valid_frames


if __name__ == "__main__":

    # set src directories
    vicon_src_dir = os.path.join(paths.SOURCE_PATH, "raw_SSP_dataset/Vicon_data/MoSh")
    studio_src_dir = os.path.join(paths.SOURCE_PATH, "raw_SSP_dataset/SSP_data/Export")
    sensor_src_dir = os.path.join(paths.SOURCE_PATH, "raw_SSP_dataset/SSP_data")

    # load vicon2srec mapping
    map_path = os.path.join(utils.get_project_folder(), "sandbox")
    vicon2srec = utils.load_config(map_path, "vicon2srec.yaml")

    # get SMPL model
    smpl_model = smpl_helpers.load_smplx()

    # walk over all vicon files in directory and collect relevant paths
    vicon_files = []
    log_files = []
    for root, dirs, files in os.walk(vicon_src_dir):
        for file in files:
            if file.endswith("_stageii.pkl"):
                vicon_files.append(os.path.join(root, file))

            if file.endswith(".log"):
                log_files.append(os.path.join(root, file))

    vicon_files = sorted(vicon_files)
    log_files = sorted(log_files)

    # walk over all studio files in directory and collect relevant paths
    studio_files = []
    for root, dirs, files in os.walk(studio_src_dir):
        for file in files:
            if file.endswith("_smpl.npz"):
                studio_files.append(os.path.join(root, file))

    studio_files = sorted(studio_files)

    # walk over all sensor files in directory and collect relevant paths
    sensor_files = []
    for root, dirs, files in os.walk(sensor_src_dir):
        for file in files:
            if "Export" in root:
                continue
            if file.endswith(".npz"):
                sensor_files.append(os.path.join(root, file))

    sensor_files = sorted(sensor_files)

    # # for every studio file, find the corresponding vicon file
    # for studio_file in studio_files:
    #     studio_file = studio_files[1]
    #     print(studio_file)
    #     studio_poses = extract_npz_data(studio_file)
    #
    #     studio_poses[:, :3] = 0
    #     pose = torch.Tensor(studio_poses)
    #     verts, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
    #     verts = utils.copy2cpu(verts)
    #     visualization.vis_smpl(
    #         faces=smpl_model.faces,
    #         vertices=[verts],
    #         play_frames=1200,
    #         playback_speed=1,
    #         # sensors=[verts[:, list(SENS_VERTS_SSP.values())]],
    #         # oris=[oris],
    #         # accs=[accs],
    #         # joints=[joints],
    #         # pose=[poses],
    #         fps=100,
    #     )
    #
    #     plot(joints)

    for subject in vicon2srec.keys():
        for trial, indices in vicon2srec[subject].items():
            indices = str2list(indices)
            # reconstruct filenames
            vicon_file = find_vicon_file(vicon_files, subject, trial)
            log_file = find_vicon_file(log_files, subject, trial)
            studio_file = find_studio_file(studio_files, str2int(subject)[0], indices)
            sensor_file = find_sensor_file(sensor_files, str2int(subject)[0], indices)

            # find valid frames in vicon file from log file
            valid_frames = find_valid_frames(log_file)

            # load pose data
            vicon_poses_orig = extract_pkl_data(vicon_file)
            vicon_name = vicon_file.split("MoSh/")[1]
            studio_poses = extract_npz_data(studio_file)["poses"]
            studio_name = studio_file.split("Export/")[1]
            sensor_data = extract_npz_data(sensor_file)["acc"]
            sensor_name = sensor_file.split("/SSP_data")[1]
            if len(vicon_poses_orig) == 0:
                print("No data in {}. Skipping!".format(vicon_name))
                continue
            if len(studio_poses) == 0:
                print("No data in {}. Skipping!".format(studio_name))
                continue

            # everything is fine, continue
            print("{} <-----> {}".format(vicon_name, sensor_name))

            # add zeros for invalid vicon frames
            valid_frames = np.array(valid_frames)
            vicon_poses = np.zeros([len(valid_frames), vicon_poses_orig.shape[1]])
            vicon_poses[valid_frames, :] = vicon_poses_orig

            # downsample vicon data from 200 to 100 fps
            valid_frames = valid_frames[::2]
            vicon_poses = vicon_poses[::2]

            # plot vicon data
            pose = torch.Tensor(vicon_poses)
            pose[:, :3] = torch.ones(3) * 0.5773502691896258 * 2.0943951023931957
            verts, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
            # verts = utils.copy2cpu(verts[:, 8588])
            vicon_joints = utils.copy2cpu(joints[:, 7, 2])
            # plot(vicon_joints, vicon_name)

            # plot studio data
            pose = torch.Tensor(studio_poses)
            pose[:, :3] = torch.ones(3) * 0.5773502691896258 * 2.0943951023931957
            verts, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
            # verts = utils.copy2cpu(verts[:, 8588])
            studio_joints = utils.copy2cpu(joints[:, 7, 2])
            # plot(studio_joints, studio_name)

            # plot sensor data
            # joints = sensor_data[:, 18]
            # plot(joints, sensor_name)

            # vicon_ind = indices[2]
            # studio_ind = indices[3]
            #
            # vicon_joints = vicon_joints[vicon_ind:]
            # studio_joints = studio_joints[studio_ind:]

            # make vectors the same length
            vicon_len = len(vicon_joints)
            studio_len = len(studio_joints)
            len_diff = abs(vicon_len - studio_len)
            if vicon_len > studio_len:
                # pad studio joints
                studio_joints = np.pad(studio_joints, [0, len_diff], "edge")
            else:
                # pad vicon joints
                vicon_joints = np.pad(vicon_joints, [0, len_diff], "edge")

            # correlate signals to find shift
            studio_joints -= np.mean(studio_joints)
            vicon_joints -= np.mean(vicon_joints)
            corr = signal.correlate(studio_joints, vicon_joints, "full")
            shift = np.argmax(corr) - len(vicon_joints) + 1
            corr /= max(corr)

            # TODO: add option to manually overwrite shift in yaml
            if len(indices) == 3:
                shift = indices[2]
            # shift studio signal
            if shift >= 0:
                studio_joints_aligned = studio_joints[shift:]
                studio_joints_aligned = np.pad(
                    studio_joints_aligned, [0, shift], "edge"
                )
            else:
                studio_joints_aligned = studio_joints[:shift]
                studio_joints_aligned = np.pad(
                    studio_joints_aligned, [abs(shift), 0], "edge"
                )

            # plot
            data = {
                "vicon": vicon_joints,
                "studio": studio_joints,
                # "corr": corr,
                "studio_aligned": studio_joints_aligned,
            }
            df = pd.DataFrame(data)
            fig = px.line(
                df,
                y=[
                    "vicon",
                    "studio",
                    # "corr",
                    "studio_aligned",
                ],
            )
            fig.show()

            continue
