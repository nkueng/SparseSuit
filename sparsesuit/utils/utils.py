"""
A collection of utility functions for all sorts of things.
"""
import logging
import os
import random
import sys

import cv2
import numpy as np
import quaternion
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset


def write_config(path, config):
    conf_file = os.path.join(path, "config.yaml")
    conf = OmegaConf.create(config)
    with open(conf_file, "wb") as f:
        OmegaConf.save(config=conf, f=f.name)


def load_config(path, file_name="config.yaml"):
    conf_file = os.path.join(path, file_name)
    return OmegaConf.load(conf_file)


def str2gender(string):
    genders = ["female", "male", "neutral"]
    if string in genders:
        return string
    else:
        for gender in genders:
            if gender in string:
                return gender

    return None


def configure_logger(name, log_path="logs/", level=logging.INFO):
    logname = os.path.join(log_path, "log.txt")
    os.makedirs(os.path.dirname(logname), exist_ok=True)
    logging.basicConfig(
        filename=logname,
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=level,
    )
    logger = logging.getLogger(name)
    # make logger output to console
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def aa_to_rot_matrix(data):
    """
    Converts the orientation data to represent angle axis as rotation matrices. `data` is expected in format
    (seq_length, n*3). Returns an array of shape (seq_length, n*9).
    """
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    # reshape to have sensor values explicit
    data_c = np.array(data, copy=True)
    seq_length, n = data_c.shape[0], data_c.shape[1] // 3
    data_r = np.reshape(data_c, [seq_length, n, 3])

    qs = quaternion.from_rotation_vector(data_r)
    rot = np.reshape(quaternion.as_rotation_matrix(qs), [seq_length, n, 9])

    return np.reshape(rot, [seq_length, 9 * n])


def rot_matrix_to_aa(data):
    """
    Converts the orientation data given in rotation matrices to angle axis representation. `data` is expected in format
    (seq_length, j*n*9). Returns an array of shape (seq_length, n*3).
    """
    seq_length, n_joints = data.shape[0], data.shape[1] // 9
    data_r = np.reshape(data, [seq_length, n_joints, 3, 3])
    data_c = np.zeros([seq_length, n_joints, 3])
    for i in range(seq_length):
        for j in range(n_joints):
            data_c[i, j] = np.ravel(cv2.Rodrigues(data_r[i, j])[0])
    return np.reshape(data_c, [seq_length, n_joints * 3])


def assemble_input_target(orientation, acceleration, pose, sens_ind):
    oris = np.reshape(orientation, [orientation.shape[0], orientation.shape[1], -1, 9])
    accs = np.reshape(
        acceleration, [acceleration.shape[0], acceleration.shape[1], -1, 3]
    )

    # extract only selected sensors
    oris_sel = oris[:, :, sens_ind]
    accs_sel = accs[:, :, sens_ind]

    # vectorize
    oris_vec = np.reshape(oris_sel, [oris_sel.shape[0], oris_sel.shape[1], -1])
    accs_vec = np.reshape(accs_sel, [accs_sel.shape[0], accs_sel.shape[1], -1])

    # convert from numpy to tensor
    input_vec = torch.cat([oris_vec, accs_vec], dim=2)
    target_vec = torch.cat([pose, accs_vec], dim=2)
    return input_vec, target_vec


def compute_jerk(data, delta: int, fps: int):
    interval = delta / fps
    res = []
    for i in range(2 * delta, len(data) - 2 * delta):
        res.append(
            (
                -0.5 * data[i - 2 * delta]
                + data[i - delta]
                - data[i + delta]
                + 0.5 * data[i + 2 * delta]
            )
            / pow(interval, 3)
        )
    return np.linalg.norm(res, axis=2)


def make_deterministic(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rad2deg(v):
    """
    Convert from radians to degrees.
    """
    return v * 180.0 / np.pi


def remove_scaling(rotations):
    """Removing scaling from 3x3 matrices by decomposition into singular values followed by composition with ones as
    singular values."""
    u, _, v = np.linalg.svd(rotations)
    return u @ v


class BigDataset(Dataset):
    def __init__(self, path, length):
        self.path = path
        self.folders = sorted(os.listdir(path))
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if 0 <= index < self.length:
            item_path = os.path.join(self.path, self.folders[index])
            filename = os.listdir(item_path)[0]
            with np.load(os.path.join(item_path, filename), allow_pickle=True) as data:
                data_in = dict(data)
            return data_in["ori"], data_in["acc"], data_in["pose"], filename
        else:
            raise IndexError("Index not in dataset. Abort!")


def copy2cpu(tensor):
    if isinstance(tensor, np.ndarray):
        return tensor
    return tensor.detach().cpu().numpy()


def get_project_folder():
    """
    Returns path to sparsesuit module folder.
    """
    return os.path.join(os.getcwd().split("sparsesuit")[0], "sparsesuit")


def rot_mat(angle=0.0, axis="x"):
    ang = angle / 180 * np.pi
    cos_ = np.cos(ang)
    sin_ = np.sin(ang)
    mat = np.eye(3)
    if axis == "x":
        mat[1, 1] = cos_
        mat[1, 2] = -sin_
        mat[2, 1] = sin_
        mat[2, 2] = cos_
    elif axis == "y":
        mat[0, 0] = cos_
        mat[0, 2] = sin_
        mat[2, 0] = -sin_
        mat[2, 2] = cos_
    elif axis == "z":
        mat[0, 0] = cos_
        mat[0, 1] = -sin_
        mat[1, 0] = sin_
        mat[1, 1] = cos_

    return mat


def rot_from_vecs(vec_init, vec_goal):
    """
    Computes the "active" rotation matrix that transforms vec_init into vec_goal.
    Can also be understood as the "passive" rotation from frame goal to frame init.
    """
    rot_axis = np.cross(
        vec_init / np.linalg.norm(vec_init), vec_goal / np.linalg.norm(vec_goal)
    )
    rot_angle = np.arcsin(np.linalg.norm(rot_axis))
    return aa_to_rot_matrix(rot_angle * rot_axis).reshape([3, 3])
