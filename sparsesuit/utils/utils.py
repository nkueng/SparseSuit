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

from sparsesuit.constants import paths


def write_config(path, config):
    conf_file = os.path.join(path, "config.yaml")
    conf = OmegaConf.create(config)
    with open(conf_file, "wb") as f:
        OmegaConf.save(config=conf, f=f.name)


def load_config(path, file_name="config.yaml"):
    conf_file = os.path.join(path, file_name)
    return OmegaConf.load(conf_file)


def ds_path_from_config(cfg, caller, debug=False):
    # if "dataset" in config:
    #     cfg = config.dataset
    # else:
    #     return None

    if caller in ["synthesis", "normalization"]:
        path = os.path.join(paths.SOURCE_PATH, cfg.source)

    elif caller in ["training", "evaluation"]:
        path = os.path.join(paths.DATASET_PATH, cfg.source)

    else:
        raise AttributeError("Invalid argument for caller. Aborting!")

    params = [cfg.sensor_config, "fps" + str(cfg.fps)]
    if "synthesis" in cfg:
        # get all parameters related to synthesis
        syn_params = cfg.synthesis
        params.append("accn" + str(syn_params.acc_noise))
        params.append("gyron" + str(syn_params.gyro_noise))
        params.append("accd" + str(syn_params.acc_delta))

    if debug:
        params.append("debug")

    # make into one string
    ds_name = "_".join(params)

    path = os.path.join(path, ds_name)
    return path


def true_in_dict(arg, dict):
    if arg in dict:
        if dict[arg]:
            return True
    else:
        return False


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


def close_logger(logger):
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()


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


def assemble_input_target(orientation, acceleration, pose, sens_ind, stats={}):
    if len(stats) != 0:
        # scale data for zero mean and unit variance
        orientation -= stats["ori_mean"]
        orientation /= stats["ori_std"]
        acceleration -= stats["acc_mean"]
        acceleration /= stats["acc_std"]
        pose -= stats["pose_mean"]
        pose /= stats["pose_std"]

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
        self.assets = sorted(os.listdir(path))
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if 0 <= index < self.length:
            item_path = os.path.join(self.path, self.assets[index])
            filename = item_path.split("/")[-1].split(".npz")[0]
            with np.load(os.path.join(item_path), allow_pickle=True) as data:
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


def interpolation(poses_ori, fps_ori, fps_target):
    """
    Interpolation of pose vectors poses_ori from original framerate fps_ori to desired framerate fps_target in the
    general case.
    """
    poses = []
    total_time = len(poses_ori) / fps_ori
    times_ori = np.arange(0, total_time, 1.0 / fps_ori)
    times = np.arange(0, total_time, 1.0 / fps_target)

    for t in times:
        index = findNearest(t, times_ori)
        if max(index) >= poses_ori.shape[0]:
            break
        a = poses_ori[index[0]]
        t_a = times_ori[index[0]]
        b = poses_ori[index[1]]
        t_b = times_ori[index[1]]

        if t_a == t:
            tmp_pose = a
        elif t_b == t:
            tmp_pose = b
        else:
            tmp_pose = a + ((t - t_a) * (b - a) / (t_b - t_a))
        poses.append(tmp_pose)

    return np.asarray(poses)


def interpolation_integer(poses_ori, fps_ori, fps_target):
    """
    Interpolation of pose vectors poses_ori from original framerate fps_ori to desired framerate fps_target in the
    special case where fps_ori is an integer multiple of fps_target.
    """
    poses = []
    n_tmp = int(fps_ori / fps_target)
    poses_ori = poses_ori[::n_tmp]

    for t in poses_ori:
        poses.append(t)

    return np.asarray(poses)


def findNearest(t, t_list):
    list_tmp = np.array(t_list) - t
    list_tmp = np.abs(list_tmp)
    index = np.argsort(list_tmp)[:2]
    return index
