import os
import numpy as np
import torch

from sparsesuit.constants import paths, sensors
from sparsesuit.utils import utils
from sparsesuit.data_generation import srec2npz, studio2smpl

# load data
srec_file = os.path.join(
    paths.SOURCE_PATH,
    "RKKmultilevelDataset/Movement1 - object in the way/take-1-5x.srec",
)
rec = srec2npz.get_srec(srec_file)
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

pose_path = os.path.join(paths.SOURCE_PATH, "RKKmultilevelDataset/AlignmentTest")
pose_files = sorted(os.listdir(pose_path))
pose_files = [pose_file for pose_file in pose_files if "npz" in pose_file]

clip_frames_dict = utils.load_config(pose_path, "clip_frames.yaml")["clip_frames"]

# align sensor data with poses
for profile_name, profile_values in clip_frames_dict.items():
    for motion_type, clip_frames in profile_values.items():
        for i, clip_frames_i in enumerate(clip_frames):
            # clip sensor measurements
            actual_clip_frames = np.round(np.array(clip_frames_i) / 1.2).astype(int)
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

            srec2npz.vis_oris_accs(ori_clip, acc_clip, pose=torch.Tensor(smpl_poses))

        continue
