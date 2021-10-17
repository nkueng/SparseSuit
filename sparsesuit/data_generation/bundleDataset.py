"""" This script bundles smpl poses with the corresponding sensor measurements and dumps them in the format ready for
normalization. """
import os

import numpy as np

from pathlib import Path
from sparsesuit.constants import paths, sensors
from sparsesuit.utils import utils


def get_data(file_path):
    if os.path.isfile(file_path):
        with np.load(file_path) as file_data:
            return dict(file_data)
    else:
        return None


smpl_folder_options = {
    "studio": "Export",
    "optical": "Vicon",  # TODO: change once the data is here
}

use_motions = [
    "gait",
    "run",
    "sidestep",
    "sway",
    # "jump",
]

if __name__ == "__main__":

    # specify source and target directories
    src_dir = os.path.join(paths.DATA_PATH, "raw_SSP_dataset/SSP_data")
    smpl_folder = smpl_folder_options["studio"]
    ignore_folder = list(smpl_folder_options.values()).copy()
    ignore_folder.remove(smpl_folder)
    ignore_folder = ignore_folder[0]
    trgt_dir = paths.RKK_STUDIO_19_PATH

    # crawl the source directory and find all relevant npz files
    sensor_files = []
    smpl_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            path = os.path.join(root, file)
            if path.endswith(".npz"):
                # ignore everything in the ignore_folder
                if ignore_folder in path:
                    continue
                # add files with "smpl" to the smpl_files
                if "smpl" in path:
                    smpl_files.append(path)
                else:
                    # ignore everything else in smpl_folder
                    if smpl_folder in path:
                        continue
                    # everything else must be a sensor file by exclusion
                    sensor_files.append(os.path.join(root, path))

    sensor_files = sorted(sensor_files)
    smpl_files = sorted(smpl_files)

    # bundle poses with sensor data and store in target directory
    success_counter = 0
    for sensor_file in sensor_files:
        # extract identifier
        file_id = sensor_file.split("/")[-3:]
        motion_type = file_id[1]
        file_id = os.path.join(*file_id).split(".npz")[0]

        # check if motion type belongs in final dataset
        occurance = [use_motion in str.lower(motion_type) for use_motion in use_motions]
        if not any(occurance):
            print(
                "Asset {} does not belong in final dataset. Skipping!".format(
                    sensor_file
                )
            )
            continue

        # find corresponding poses
        corr_smpl_files = [
            smpl_file for smpl_file in smpl_files if file_id in smpl_file
        ]
        if len(corr_smpl_files) == 2:
            # special case for names with a 1 that also catches 10
            string_with_10 = [string for string in corr_smpl_files if "10" in string]
            corr_smpl_files.remove(string_with_10[0])
        elif len(corr_smpl_files) == 0:
            print("Found no correspondence for {}. Skipping!".format(file_id))
            continue

        # get data vectors from npzs
        sensor_data = get_data(sensor_file)
        acc = sensor_data["acc"]
        ori = sensor_data["ori"]
        pose_data = get_data(corr_smpl_files[0])
        pose = pose_data["poses"]

        # make sure that vectors have same length
        if len(acc) != len(pose):
            if len(acc) > len(pose):
                acc = acc[1:]
                ori = ori[1:]
            else:
                pose = pose[1:]

        assert len(acc) == len(
            pose_data["poses"]
        ), "Cannot bundle vectors with different length, abort!"

        # save together as npz
        out_dict = {"imu_ori": ori, "imu_acc": acc, "gt": pose}
        subject, motion, take = file_id.split("/")
        out_folder = os.path.join(trgt_dir, subject)
        Path(out_folder).mkdir(parents=True, exist_ok=True)
        file_name = motion + "_" + take
        file_name = "_".join(file_name.split(" "))
        out_name = os.path.join(out_folder, file_name) + ".npz"
        with open(out_name, "wb") as fout:
            np.savez_compressed(fout, **out_dict)

        print("Bundled {}".format(file_id))
        success_counter += 1

    # TODO: write config for this dataset
    cfg = {
        "sequences": success_counter,
        "config": "SSP",
        "type": "real",
        "groundtruth": "studio",
        "fps": 100,
        "count": 19,
        "sensors": sensors.SENS_NAMES_SSP,
    }
    ds_cfg = {
        "dataset": cfg,
    }
    utils.write_config(trgt_dir, ds_cfg)
