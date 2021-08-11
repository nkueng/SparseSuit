# SMPL-X vertices of virtual sensors for MVN -> ordering sets convention
SENS_VERTS_MVN = {
    "head": 8942,
    "pelvis": 5493,
    "sternum": 5528,
    "left_wrist": 4617,
    "right_wrist": 7348,
    "left_collar": 5462,
    "right_collar": 8196,
    "left_shoulder": 3952,
    "right_shoulder": 6700,
    "left_hip": 3603,
    "right_hip": 6364,
    "left_knee": 3811,
    "right_knee": 6568,
    "left_elbow": 4858,
    "right_elbow": 7559,
    "left_ankle": 5894,
    "right_ankle": 8588,
}

# SMPL-X vertices of virtual sensors for Rokoko's SSP
SENS_VERTS_SSP = {
    # "pelvis": 5493,  # just to test virtual root sensor
    "left_pelvis": 5697,
    "right_pelvis": 8391,
    "left_back": 3381,
    "right_back": 6142,
    "left_collar": 5462,
    "right_collar": 8196,
    "head": 8980,
    "left_shoulder": 3952,
    "right_shoulder": 6700,
    "left_elbow": 4858,
    "right_elbow": 7559,
    "left_wrist": 4617,
    "right_wrist": 7348,
    "left_hip": 3603,
    "right_hip": 6364,
    "left_knee": 5875,
    "right_knee": 8569,
    "left_ankle": 5894,
    "right_ankle": 8588,
}

# indices of SMPL joints corresponding to sensors in MVN and SSP
SENS_JOINTS_IDS = {
    "pelvis": 0,
    "left_pelvis": 0,
    "right_pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_back": 6,  # corresponds to spine2
    "right_back": 6,  # corresponds to spine2
    "left_ankle": 7,
    "right_ankle": 8,
    "sternum": 9,  # corresponds to spine3
    "left_foot": 10,
    "right_foot": 11,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

# the actual 24 SMPL joints and their indices
SMPL_JOINT_IDS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "spine2": 6,
    "left_ankle": 7,
    "right_ankle": 8,
    "spine3": 9,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "left_collar": 13,
    "right_collar": 14,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    "left_hand": 22,
    "right_hand": 23,
}

# IMU sensor configurations
SENS_NAMES_DIP = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "head",
    "pelvis",
]  # defined by AMASS -> convention
SENS_NAMES_MVN = list(SENS_VERTS_MVN.keys())
SENS_NAMES_SSP = list(SENS_VERTS_SSP.keys())

# indices of 15 SMPL joints used as targets for predictions in DIP (excludes pelvis, ankles, feet, wrists, and hands)
SMPL_MAJOR_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]
# indices of 19 SMPL joints used as targets for predictions in this thesis (excludes pelvis, feet, and hands)
SMPL_SSP_JOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
NUM_SMPL_JOINTS = 24
NUM_SMPLX_JOINTS = 55
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
ANG_EVAL_JOINTS = [1, 2, 16, 17]  # the 4 sensors/joints not used as input for SIP (providing accurate groundtruth)
POS_EVAL_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]  # "virtual markers" to evaluate joint positional error: hips, knees, ankles, shoulders, elbows, wrists and neck
