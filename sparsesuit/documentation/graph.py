import os
from pathlib import Path

import numpy as np
import collections
import matplotlib.pyplot as plt
from matplotlib import cm

from sparsesuit.constants import paths, sensors


def create_3d_bar_plot(values, title, error_type, folder_name):
    # compute averages for plotting
    plot_data = np.array(
        [np.mean(motion_data, axis=0) for motion_data in list(values.values())]
    )
    # sort motions by average error
    avg_motion_err = np.mean(plot_data, axis=1)
    plot_data = np.insert(plot_data, 0, avg_motion_err, axis=1)
    sort_ids = np.argsort(avg_motion_err)
    plot_data = plot_data[sort_ids[::-1]]
    motion_labels = np.array(list(values.keys()))[sort_ids[::-1]]

    # make 3d bar plot over all joints and all motions
    fig = plt.figure(figsize=(16, 10), dpi=100)
    ax = fig.add_subplot(projection="3d")
    ax.set_title(title, y=0.8)

    # axis labels and positions (motions on x-, joints on y-axis)
    xpos = np.arange(len(motion_labels))
    joint_labels = np.array(list(sensors.SMPL_JOINT_IDS.keys()))[
        sensors.ANG_EVAL_JOINTS
    ]
    joint_labels = np.insert(joint_labels, 0, "mean")
    ypos = np.arange(len(joint_labels))
    ypos[0] = -1  # space after mean
    xposM, yposM = np.meshgrid(xpos, ypos, copy=False)
    dx = dy = 0.5

    # height of bars
    zpos = plot_data.T
    zpos = zpos.ravel()
    dz = zpos

    # coloring
    values = (dz - dz.min()) / np.float_(dz.max() - dz.min())
    colors = cm.rainbow(values)

    # make plot
    ax.bar3d(xposM.ravel(), yposM.ravel(), dz * 0, dx, dy, dz, color=colors)

    # set axis labels
    ax.w_xaxis.set_ticks(xpos + dx / 2.0)
    ax.w_xaxis.set_ticklabels(motion_labels, rotation=-15, ha="right", va="bottom")
    ax.set_xlabel("Motion Type")
    ax.xaxis.labelpad = 20

    ax.w_yaxis.set_ticks(ypos + dy / 2.0)
    ax.w_yaxis.set_ticklabels(joint_labels, rotation=-25, ha="left")
    ax.set_ylabel("Joint Name")
    ax.yaxis.labelpad = 80

    if error_type == "Angular":
        ax.set_zlabel("Angular Error [$\degree$]")
        ax.set_zlim(0, 40)
    else:
        ax.set_zlabel("Positional Error [$cm$]")
        ax.set_zlim(0, 20)

    ax.view_init(30, -15)
    ax.set_box_aspect([5, 20, 6])
    folder_path = os.path.join(paths.DOC_PATH, "figures", folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(folder_path, title + "_3d.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def create_2d_bar_plot(values, title, error_type, folder_name):
    # sort by configuration name
    x_labels = list(reversed(sorted(list(values.keys()))))
    x_data = np.arange(0, len(x_labels))

    plot_values = [values[config] for config in x_labels]

    y_data = np.array([val[0] for val in plot_values])
    y_std = np.array([val[1] for val in plot_values])

    color_values = (y_data - PLOT_COLOR_MIN) / np.float_(
        PLOT_COLOR_MAX - PLOT_COLOR_MIN
    )
    # colors = cc.cm.isoluminant_cgo_70_c39(values)
    colors = cm.rainbow(color_values)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.title(title)
    plt.bar(x_data, y_data, yerr=y_std, capsize=5, color=colors)
    plt.xlabel("Sensor Configuration")
    # x_labels = [int(label) for label in x_labels]
    plt.xticks(x_data, x_labels)
    if error_type == "Angular":
        plt.ylabel("Angular Error [$\degree$]")
        y_lim = Y_LIM_ANG
    else:
        plt.ylabel("Positional Error [$cm$]")
        y_lim = Y_LIM_POS

    plt.ylim(top=y_lim)
    plt.gca().yaxis.grid(True)
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.gca().set_aspect(aspect=0.1 / 6 * len(x_labels))
    folder_path = os.path.join(paths.DOC_PATH, "figures", folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(folder_path, "_".join([title, "bar.png"]))
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def create_2d_box_plot(values, title, error_type, folder_name):
    # sort by configuration name
    x_labels = list(reversed(sorted(list(values.keys()))))
    x_data = np.arange(1, len(x_labels) + 1)

    plot_values = [values[config] for config in x_labels]

    # values = (y_data - PLOT_COLOR_MIN) / np.float_(PLOT_COLOR_MAX - PLOT_COLOR_MIN)
    # colors = cc.cm.isoluminant_cgo_70_c39(values)
    # colors = cm.rainbow(values)

    plt.figure(figsize=(8, 8), dpi=100)
    plt.title(title)
    red_cross = dict(markeredgecolor="r", marker="x")
    plt.boxplot(plot_values, flierprops=red_cross)
    plt.xlabel("Sensor Configuration")
    # x_labels = [int(label) for label in x_labels]
    plt.xticks(x_data, x_labels)
    if error_type == "Angular":
        plt.ylabel("Angular Error [$\degree$]")
        y_lim = Y_LIM_ANG
    else:
        plt.ylabel("Positional Error [$cm$]")
        y_lim = Y_LIM_POS
    plt.ylim(bottom=0)
    plt.gca().yaxis.grid(True)
    # plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    plt.gca().set_aspect(aspect=0.1)
    folder_path = os.path.join(paths.DOC_PATH, "figures", folder_name)
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    fig_path = os.path.join(folder_path, title + "_box.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.show()


def get_studio_data(error_stats):
    # collect error statistics for studio poses
    studio_err_path = os.path.join(
        paths.DATASET_PATH, "RKK_STUDIO/SSP_fps100_n/error_stats.npz"
    )
    studio_data = dict(np.load(studio_err_path))
    studio_err_stats = collections.defaultdict(dict)
    for i in range(1, 6):
        motion_i_data = []
        for asset_name, errs in studio_data.items():
            if i == int(asset_name.split("_")[2]):
                motion_i_data.append(errs[sensors.ANG_EVAL_JOINTS])
        studio_err_stats[fraction2motion[str(i)]] = np.array(motion_i_data)

    error_stats["ang_err"]["Studio"] = studio_err_stats

    return error_stats


def get_data_finetuned_on_real():
    # collect error statistics for neural networks that were finetuned and evaluated on real data
    error_stats = collections.defaultdict(dict)
    run_dir = paths.EVAL_PATH
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if "fraction" in root and "error_stats" in file:
                sensor_config = root.split("SSP_")[1].split("_fine")[0]
                motion_type = fraction2motion[root.split("fraction")[1]]
                data = np.load(os.path.join(root, file))
                data = np.array(list(dict(data).values()))[:, sensors.ANG_EVAL_JOINTS]
                error_stats[sensor_config][motion_type] = data

    # collect error statistics for studio poses
    error_stats = get_studio_data(error_stats)

    return error_stats


def get_data_pretrained_on_real():
    # collect error statistics for neural networks that were pretrained on synthetic and evaluated on real data
    error_stats = collections.defaultdict(dict)
    run_dir = paths.EVAL_PATH
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if "_real" in root and "errs" in file:
                err_type = file.replace("s.npz", "")
                error_i_stats = collections.defaultdict(dict)
                data = dict(np.load(os.path.join(root, file)))
                config_name = root.split("SSP_")[1].split("_")[0]
                for i in range(1, 6):
                    motion_i_data = []
                    for asset_name, errs in data.items():
                        if i == int(asset_name.split("_")[2]):
                            motion_i_data.append(errs[sensors.ANG_EVAL_JOINTS])
                    # studio_err_stats[fraction2motion[str(i)]] = np.array(motion_i_data)
                    error_i_stats[fraction2motion[str(i)]] = np.array(motion_i_data)
                error_stats[err_type][config_name] = error_i_stats

    # collect error statistics for studio poses
    error_stats = get_studio_data(error_stats)

    return error_stats


def get_data_trained_real():
    # collect error statistics for neural networks that were trained only on real and evaluated on real data
    error_stats = collections.defaultdict(dict)
    run_dir = paths.EVAL_PATH
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if "trainedreal" in root and "error_stats" in file:
                data = dict(np.load(os.path.join(root, file)))
                for i in range(1, 6):
                    motion_i_data = []
                    for asset_name, errs in data.items():
                        if i == int(asset_name.split("_")[2]):
                            motion_i_data.append(errs[sensors.ANG_EVAL_JOINTS])
                    # studio_err_stats[fraction2motion[str(i)]] = np.array(motion_i_data)
                    error_stats[root.split("SSP_")[1].split("_")[0]][
                        fraction2motion[str(i)]
                    ] = np.array(motion_i_data)

    # collect error statistics for studio poses
    error_stats = get_studio_data(error_stats)

    return error_stats


def get_data_pretrained_on_syn():
    # collect error statistics for neural networks that were pretrained and evaluated on synthetic data
    error_stats = collections.defaultdict(dict)
    run_dir = paths.EVAL_PATH
    motion_types = set(amass2motion.values())
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            if "syn" in root and "errs" in file:
                err_type = file.replace("s.npz", "")
                error_i_stats = collections.defaultdict(dict)
                data = dict(np.load(os.path.join(root, file)))
                config_name = root.split("SSP_")[1].split("_")[0]
                for motion in motion_types:
                    motion_i_data = []
                    for asset_name, errs in data.items():
                        if motion == amass2motion[asset_name[6:]]:
                            motion_i_data.append(errs[sensors.ANG_EVAL_JOINTS])
                    error_i_stats[motion] = np.array(motion_i_data)
                error_stats[err_type][config_name] = error_i_stats
    return error_stats


def create_summary_bar_plot(error_stats, folder_name):
    for error_type, err_values in error_stats.items():
        summary_stats = collections.defaultdict(dict)
        for sensor_config, errs in err_values.items():
            data = np.vstack(list(errs.values()))
            summary_stats[sensor_config] = (np.mean(data), np.std(data))
        title = "Mean {} Joint Error over all Motions"
        title = title.format(err_type2str[error_type])
        create_2d_bar_plot(
            summary_stats,
            title,
            err_type2str[error_type],
            folder_name,
        )


def create_summary_box_plot(error_stats, folder_name):
    for error_type, err_values in error_stats.items():
        summary_stats = collections.defaultdict(dict)
        for sensor_config, errs in err_values.items():
            data = np.vstack(list(errs.values()))
            summary_stats[sensor_config] = data.flatten()
        title = "Mean {} Joint Error over all Motions"
        title = title.format(err_type2str[error_type])
        create_2d_box_plot(
            summary_stats,
            title,
            err_type2str[error_type],
            folder_name,
        )


def create_motion_bar_plots(error_stats, folder_name):
    motion_stats = collections.defaultdict(dict)
    for error_type, err_values in error_stats.items():
        error_i_stats = collections.defaultdict(dict)
        for sensor_config, values in err_values.items():
            for motion_type, errs in values.items():
                error_i_stats[motion_type][sensor_config] = (
                    np.mean(errs),
                    np.std(errs),
                )
        motion_stats[error_type] = error_i_stats
    for error_type, err_values in motion_stats.items():
        for motion_type, values in err_values.items():
            title = "Mean {} Joint Error for {} Motions"
            title = title.format(err_type2str[error_type], infinitive2ing[motion_type])
            create_2d_bar_plot(
                values,
                title,
                err_type2str[error_type],
                folder_name,
            )


if __name__ == "__main__":

    err_type2str = {
        "ang_err": "Angular",
        "pos_err": "Positional",
    }

    infinitive2ing = {
        "Walk": "Walking",
        "Run": "Running",
        "Sidestep": "Sidestepping",
        "Sway": "Swaying",
        "Jump": "Jumping",
        "Fall": "Falling",
        "Sit": "Sitting",
        "Dance": "Dancing",
        "Throw": "Throwing",
        "Misc": "Misc",
    }

    fraction2motion = {
        "1": "Walk",
        "2": "Run",
        "3": "Sidestep",
        "4": "Sway",
        "5": "Jump",
    }

    amass2motion = {
        "KIT_11_RightTurn02": "Walk",
        "KIT_348_walking_slow01": "Walk",
        "BioMotionLab_NTroje_rub066_0006_normal_walk2": "Walk",
        "ACCAD_Female1Walking_c3d_B11-walkturnleft(135)": "Walk",
        "KIT_317_walking_slow07": "Walk",
        "ACCAD_Female1Running_c3d_C5-walktorun": "Run",
        "SFU_0017_0017_RunningOnBench002": "Run",
        "DFaust_67_50026_50026_running_on_spot": "Run",
        "ACCAD_Male2General_c3d_A2-Sway": "Sway",
        "ACCAD_Female1General_c3d_A2-Sway": "Sway",
        "BMLmovi_Subject_25_F_MoSh_Subject_25_F_13": "Sidestep",
        "BMLhandball_S08_Novice_Trial_upper_right_160": "Throw",
        "DanceDB_20151003_AndriaMichaelidou_Andria_Annoyed_v1_C3D": "Dance",
        "BioMotionLab_NTroje_rub060_0016_sitting2": "Sit",
        "TCD_handMocap_ExperimentDatabase_typing_2": "Sit",
        "KIT_3_jump_up03": "Jump",
        "BioMotionLab_NTroje_rub025_0027_jumping1": "Jump",
        "Transitions_mocap_mazen_c3d_jumpingjacks_walk": "Jump",
        "Transitions_mocap_mazen_c3d_jumpingjacks_jumpinplace": "Jump",
        "Eyes_Japan_Dataset_hamada_accident-11-falldown-hamada": "Fall",
        "BioMotionLab_NTroje_rub109_0031_rom": "Misc",  # like a warmup
        "MPI_HDM05_bk_HDM_bk_03-04_03_120": "Misc",
        "SSM_synced_20160330_03333_chicken_wings_poses": "Misc",
    }

    # plotting constants
    PLOT_COLOR_MAX = 25
    PLOT_COLOR_MIN = 10
    Y_LIM_ANG = 37
    Y_LIM_POS = 25

    # load data for plotting

    # evaluate the pretrained models on synthetic data
    # error_stats = get_data_pretrained_on_syn()
    # folder_name = "pretrained_on_syn"

    # evaluate the pretrained models on real data
    error_stats = get_data_pretrained_on_real()
    folder_name = "pretrained_on_real"

    # evaluate the finetuned models on real data
    # error_stats = get_data_finetuned_on_real()
    # folder_name = "finetuned_on_real"

    # evaluate models that were trained on real data on the same data
    # error_stats = get_data_trained_real()
    # folder_name = "trained_real"

    # create summary bar plot
    create_summary_bar_plot(error_stats, folder_name)

    # create summary box plot
    create_summary_box_plot(error_stats, folder_name)

    # create motion specific 2d bar plot
    create_motion_bar_plots(error_stats, folder_name)

    # create configuration specific 3d graphs
    for err_type, errs in error_stats.items():
        for sensor_config, values in errs.items():
            title = "{} Joint Errors for {} Sensors".format(
                err_type2str[err_type], sensor_config
            )
            create_3d_bar_plot(values, title, err_type2str[err_type], folder_name)

    pass
