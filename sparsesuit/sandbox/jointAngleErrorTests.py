"""
A script that plots the angular error of the predictions per joint and per motion type.
"""

import os

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import colorcet as cc

from sparsesuit.constants import paths, sensors


def bar_plot_2d(data_dict, title):
    # make 2d bar plot over all joints
    ang_errs = list(data_dict.values())
    y_data = np.mean(ang_errs, axis=0)[sensors.ANG_EVAL_JOINTS]
    y_avg = np.mean(y_data)
    y_data = np.insert(y_data, 0, y_avg)
    y_std = np.std(ang_errs, axis=0)[sensors.ANG_EVAL_JOINTS]
    y_avg_std = np.std(y_data)
    y_std = np.insert(y_std, 0, y_avg_std)

    xtick_labels = list(sensors.SMPL_JOINT_IDS.keys())
    xtick_labels = [xtick_labels[i] for i in sensors.ANG_EVAL_JOINTS]
    xtick_labels.insert(0, "mean")

    x_data = np.arange(0, len(xtick_labels))
    x_data[0] = -1

    values = (y_data - y_data.min()) / np.float_(y_data.max() - y_data.min())
    # colors = cc.cm.isoluminant_cgo_70_c39(values)
    colors = cm.rainbow(values)

    plt.figure(figsize=(12, 10), dpi=200)
    plt.title(title)
    plt.bar(x_data, y_data, yerr=y_std, capsize=5, color=colors)
    plt.xlabel("Joint Name")
    plt.xticks(x_data, xtick_labels, size="small", rotation=45, ha="right")
    plt.ylabel("Angular Error [$\degree$]")
    plt.gca().yaxis.grid(True)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.show()


def bar_plot_3d(data_dict, title):
    # compute the average per-joint angular error for each type of motion
    ang_errs = [[], [], [], [], []]
    for asset, err in data_dict.items():
        bucket = int(asset.split("_")[2]) - 1
        ang_errs[bucket].append(err[sensors.ANG_EVAL_JOINTS])

    mean_errs = []
    for errs in ang_errs:
        mean_errs.append(np.mean(errs, axis=0))
    mean_errs = np.array(mean_errs)

    # sort joints by average error size
    mean_joint_err = np.mean(mean_errs, axis=0)
    sort_ids_j = np.argsort(mean_joint_err)
    mean_errs = mean_errs[:, sort_ids_j]

    # add average for each motion type
    mean_motion_err = np.expand_dims(np.mean(mean_errs, axis=1), axis=1)
    mean_errs = np.concatenate((mean_motion_err, mean_errs), axis=1)

    # sort motions by average error size
    sort_ids_m = np.argsort(np.mean(mean_errs, axis=1))
    mean_errs = mean_errs[sort_ids_m[::-1]]

    # switch sidestep and sway rows for plotting
    # mean_errs[[2, 3]] = mean_errs[[3, 2]]

    # make 3d bar plot over all joints and all motions
    fig = plt.figure(figsize=(16, 10), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title)

    xtick_labels = np.array(["Walk", "Run", "Sidestep", "Sway", "Jump"])[
        sort_ids_m[::-1]
    ]
    xpos = np.arange(xtick_labels.shape[0])

    ytick_labels = np.array(list(sensors.SMPL_JOINT_IDS.keys()))[
        sensors.ANG_EVAL_JOINTS
    ][sort_ids_j]
    ytick_labels = np.insert(ytick_labels, 0, "mean")
    ypos = np.arange(ytick_labels.shape[0])
    ypos[0] = -1  # space after mean

    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)

    zpos = mean_errs.T
    zpos = zpos.ravel()

    dx = dy = 0.5
    dz = zpos

    # values = np.linspace(0.2, 1.0, xposM.ravel().shape[0])
    values = (dz - dz.min()) / np.float_(dz.max() - dz.min())

    colors = cm.rainbow(values)

    ax.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)

    ax.w_xaxis.set_ticks(xpos + dx / 2.0)
    ax.w_xaxis.set_ticklabels(xtick_labels, rotation=-15, ha="right", va="bottom")
    ax.set_xlabel("Motion Type")
    ax.xaxis.labelpad = 20

    ax.w_yaxis.set_ticks(ypos + dy / 2.0)
    ax.w_yaxis.set_ticklabels(ytick_labels, rotation=-25, ha="left")
    ax.set_ylabel("Joint Name")
    ax.yaxis.labelpad = 80

    ax.set_zlabel("Angular Error [$\degree$]")

    ax.view_init(30, -15)
    ax.set_box_aspect([5, 20, 6])
    plt.show()
    return


if __name__ == "__main__":
    # select and load an experiment to plot
    exp_name = "2111021102-SSP_baseline_finetuned"
    exp_path = os.path.join(paths.RUN_PATH, exp_name, "error_stats.npz")
    with np.load(exp_path) as data:
        data_dict = dict(data)

    bar_plot_2d(data_dict, exp_name)
    # bar_plot_3d(data_dict, exp_name)
