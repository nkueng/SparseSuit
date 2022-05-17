import os
from pathlib import Path

import numpy as np
import torch

from sparsesuit.constants import paths, sensors
from sparsesuit.utils import utils
from sparsesuit.data_generation import srec2npz, studio2smpl

"""
This script aligns vicon poses with SSP poses and sensor data. 
The alignment is based on visual inspection in Maya and provided through a .yaml file.

Desired output:
    - SSP
        - IMU
        - ACC
        - pose w/o displacement (hierarchical, FBX convention)
    - VICON
        - pose w/ displacement
"""

VISUALIZE = False

source_path = "/home/nicola/Downloads/RKKmultilevelDataset/"

trgt_dir = os.path.join(
    source_path,
    "synced_ds",
)

motion_names = {
    "obstacle_in_the_way": "object in the way",
    "step_up_step_down": "step up step down",
    "jump_onto_objects": "jumping on to an object/take-1",
    "jump_onto_objects001": "jumping on to an object/take-2",
    "jump_onto_objects002": "jumping on to an object/take-3",
    "walk_upstairs_downstairs": "walk upstairs downstairs",
    "parcour": "parcour",
}

suit_names = {
    "Paul": "R68",
    "Nilas": "5O3",
    "Vittorio": "41A",
}

# load sensor data (one file per recording)
srec_folder = os.path.join(
    source_path,
    "Srec",
)
srec_files = []
for root, dirs, files in os.walk(srec_folder):
    for file in files:
        if file.endswith(".srec"):
            srec_files.append(os.path.join(root, file))
srec_files = sorted(srec_files)

# load SSP poses (one file per recording and actor)
studio_src_dir = os.path.join(source_path, "SSP_poses")
studio_files = []
for root, dirs, files in os.walk(studio_src_dir):
    for file in files:
        if file.endswith(".npz"):
            studio_files.append(os.path.join(root, file))
studio_files = sorted(studio_files)

# load vicon poses (one file per recording, actor, and repetition)
pose_path = os.path.join(
    source_path, "multileveldataset/Vicon"
)
pose_files = sorted(os.listdir(pose_path))
pose_files = [pose_file for pose_file in pose_files if "npz" in pose_file]

# load correspondance dict
clip_frames_dict = utils.load_config(pose_path, "clip_frames.yaml")["clip_frames"]

# align sensor data with poses
for profile_name, profile_values in clip_frames_dict.items():
    # # DEBUG
    # profile_name = "Paul"
    # profile_values = clip_frames_dict[profile_name]
    for motion_type, clip_frames in profile_values.items():

        sensor_file = [
            file for file in srec_files if motion_names[motion_type] in file.lower()
        ][0]
        # load sensor data
        rec = srec2npz.get_srec(sensor_file)
        # clip all recordings to the same length
        min_len = min([len(suit.frames) for suit in rec.suits])
        sensor_dict = {}
        for suit in rec.suits:
            if len(suit.frames) > min_len:
                suit.frames = suit.frames[:min_len]

            # TODO: parse every srec only once instead of 3 times 
            oris, accs = srec2npz.extract_measurements(suit)
            sensor_dict[suit.profile_name[2:]] = {
                "ori": oris,
                "acc": accs,
            }

        # load studio pose
        studio_file = [file for file in studio_files if motion_names[motion_type] in file.lower()]
        studio_file = [file for file in studio_file if suit_names[profile_name] in file]
        assert len(studio_file) == 1
        studio_pose = dict(np.load(studio_file[0]))

        # load vicon pose

        for i, clip_frames_i in enumerate(clip_frames):
            # i = 1
            # clip_frames_i = clip_frames[i]

            # get vicon poses
            pose_file = [file for file in pose_files if motion_type + "_" in file.lower()]
            pose_file = [file for file in pose_file if profile_name in file]
            pose_file = [
                file for file in pose_file if str(i + 1) in file.split(profile_name)[1]
            ][0]
            pose_clip = dict(np.load(os.path.join(pose_path, pose_file)))
            pose_len = len(pose_clip["positions"])

            # clip sensor measurements
            actual_clip_frames = np.round(np.array(clip_frames_i) / 1.2).astype(int)
            actual_clip_frames += 0
            clip_len = actual_clip_frames[1] - actual_clip_frames[0]
            if abs(clip_len - pose_len) >= 2:
                print("Length missmatch: {}({}) <==> {}({})".format(pose_file, pose_len, sensor_file, clip_len))
                # continue
            actual_clip_frames[1] = actual_clip_frames[0] + pose_len

            sensor_clip = sensor_dict[profile_name]
            ori_clip = sensor_clip["ori"][actual_clip_frames[0] : actual_clip_frames[1]]
            acc_clip = sensor_clip["acc"][actual_clip_frames[0] : actual_clip_frames[1]]

            # clip studio pose
            assert len(studio_pose["positions"]) == len(sensor_clip["ori"])
            studio_clip = studio_pose.copy()
            studio_clip["positions"] = studio_clip["positions"][actual_clip_frames[0] : actual_clip_frames[1]]
            studio_clip["rotations"] = studio_clip["rotations"][actual_clip_frames[0] : actual_clip_frames[1]]

            if VISUALIZE:
                # retarget to smpl skeleton for visualization
                pose_clip["positions"] = np.insert(
                    pose_clip["positions"], 0, [0, 0, 0], axis=1
                )
                pose_clip["rotations"] = np.insert(
                    pose_clip["rotations"], 0, [0, 0, 0, 1], axis=1
                )
                smpl_poses = studio2smpl.retarget_motion(
                    pose_clip,
                    joint_mapping=sensors.FBX_VICON_2_SMPL,
                    clip_first=False,
                )
                srec2npz.vis_oris_accs(
                    ori_clip, acc_clip, pose=torch.Tensor(smpl_poses)
                )

            # export
            out_dict_sens = {
                "ori": ori_clip,
                "acc": acc_clip,
                # "pose": pose_clip,
            }
            sens_dir = os.path.join(trgt_dir, "sensor_data")
            Path(sens_dir).mkdir(parents=True, exist_ok=True)
            out_name = os.path.join(sens_dir, pose_file)
            with open(out_name, "wb") as fout:
                np.savez_compressed(fout, **out_dict_sens)

            vicon_dir = os.path.join(trgt_dir, "vicon_poses")
            Path(vicon_dir).mkdir(parents=True, exist_ok=True)
            out_name = os.path.join(vicon_dir, pose_file)
            with open(out_name, "wb") as fout:
                np.savez_compressed(fout, **pose_clip)

            studio_dir = os.path.join(trgt_dir, "studio_poses")
            Path(studio_dir).mkdir(parents=True, exist_ok=True)
            out_name = os.path.join(studio_dir, pose_file)
            with open(out_name, "wb") as fout:
                np.savez_compressed(fout, **studio_clip)

        continue
