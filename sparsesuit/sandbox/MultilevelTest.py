import os
from pathlib import Path

import numpy as np
import torch

from sparsesuit.constants import paths, sensors
from sparsesuit.utils import utils
from sparsesuit.data_generation import srec2npz, studio2smpl

VISUALIZE = False

trgt_dir = os.path.join(
    paths.SOURCE_PATH,
    "RKKmultilevelDataset/synced_ds",
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

# load sensor data
srec_folder = os.path.join(
    paths.SOURCE_PATH,
    "RKKmultilevelDataset/Srec",
)

srec_files = []
for root, dirs, files in os.walk(srec_folder):
    for file in files:
        if file.endswith(".srec"):
            srec_files.append(os.path.join(root, file))

srec_files = sorted(srec_files)

pose_path = os.path.join(
    paths.SOURCE_PATH, "RKKmultilevelDataset/multileveldataset/Mocap_rig"
)
pose_files = sorted(os.listdir(pose_path))
pose_files = [pose_file for pose_file in pose_files if "npz" in pose_file]

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

            oris, accs = srec2npz.parse_srec(suit)
            sensor_dict[suit.profile_name[2:]] = {
                "ori": oris,
                "acc": accs,
            }

        for i, clip_frames_i in enumerate(clip_frames):
            # i = 1
            # clip_frames_i = clip_frames[i]

            # clip sensor measurements
            actual_clip_frames = np.round(np.array(clip_frames_i) / 1.2).astype(int)
            actual_clip_frames += 0
            sensor_clip = sensor_dict[profile_name]
            ori_clip = sensor_clip["ori"][actual_clip_frames[0] : actual_clip_frames[1]]
            acc_clip = sensor_clip["acc"][actual_clip_frames[0] : actual_clip_frames[1]]

            # get corresponding poses
            pose_file = [file for file in pose_files if motion_type in file.lower()]
            pose_file = [file for file in pose_file if profile_name in file]
            pose_file = [
                file for file in pose_file if str(i + 1) in file.split(profile_name)[1]
            ][0]
            pose_clip = dict(np.load(os.path.join(pose_path, pose_file)))

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
            out_dict = {
                "ori": ori_clip,
                "acc": acc_clip,
                # "pose": pose_clip,
            }
            Path(trgt_dir).mkdir(parents=True, exist_ok=True)
            out_name = os.path.join(trgt_dir, pose_file)
            with open(out_name, "wb") as fout:
                np.savez_compressed(fout, **out_dict)

        continue
