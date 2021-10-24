import os

import pickle as pkl

import torch

from sparsesuit.constants import paths
from sparsesuit.utils import smpl_helpers, utils, visualization


def extract_data(file_path):
    with open(file_path, "rb") as fin:
        data_in = pkl.load(fin, encoding="latin1")
    return data_in["fullpose"]


def visualize(pose_data):
    smpl_model = smpl_helpers.load_smplx()
    pose = torch.Tensor(pose_data)
    verts, joints, rel_tfs = smpl_helpers.my_lbs(smpl_model, pose)
    verts = utils.copy2cpu(verts)
    visualization.vis_smpl(
        faces=smpl_model.faces,
        vertices=[verts],
        play_frames=500,
        playback_speed=0.5,
        # sensors=[verts[:, list(SENS_VERTS_SSP.values())]],
        # oris=[oris],
        # accs=[accs],
        # joints=[joints],
        # pose=[poses],
        fps=120,
    )


if __name__ == "__main__":

    # set src directory with files
    src_dir = os.path.join(paths.SOURCE_PATH, "raw_SSP_dataset/Vicon_data/MoSh")

    # walk over all files in directory and collect relevant paths
    pkl_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith("_stageii.pkl"):
                pkl_files.append(os.path.join(root, file))

    pkl_files = sorted(pkl_files)

    # parse SREC and extract IMU data in correct frame
    for pkl_file in pkl_files:
        # DEBUG
        # pkl_file = pkl_files[1]
        data = extract_data(pkl_file)

        visualize(data)
