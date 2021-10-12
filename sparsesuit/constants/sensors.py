# SMPL-X vertices of virtual sensors for MVN -> ordering sets convention
SENS_VERTS_MVN = {
    "head": 8942,
    "pelvis": 5493,
    "sternum": 5528,
    "left_collar": 5462,
    "right_collar": 8196,
    "left_shoulder": 3952,
    "right_shoulder": 6700,
    "left_wrist": 4617,
    "right_wrist": 7348,
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

# IMU sensor configurations (defined by AMASS -> convention)
SENS_NAMES_DIP = [
    "left_elbow",
    "right_elbow",
    "left_knee",
    "right_knee",
    "head",
    "pelvis",
]
SENS_NAMES_MVN = list(SENS_VERTS_MVN.keys())
SENS_NAMES_SSP = list(SENS_VERTS_SSP.keys())

# indices of 15 SMPL joints used as targets for predictions in DIP (excludes pelvis, ankles, wrists, feet, and hands)
SMPL_DIP_JOINTS = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]

# indices of 19 SMPL joints used as targets for predictions in this project (excludes pelvis, feet, and hands)
SMPL_SSP_JOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]

# the SMPL-X joints used for the visualization of the rig
SMPLX_RIG_JOINTS = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,  # left wrist
    21,  # right wrist
    28,  # left middle finger
    43,  # right middle finger
]
NUM_SMPL_JOINTS = 24
NUM_SMPLX_JOINTS = 55

SMPL_PARENTS = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]
# the 4 sensors/joints not used as input for SIP: hips and shoulders (providing accurate groundtruth)
SIP_ANG_EVAL_JOINTS = [1, 2, 16, 17]

# "virtual markers" to evaluate joint positional error in SIP and DIP:
# hips, knees, ankles, shoulders, elbows, wrists and head
SIP_POS_EVAL_JOINTS = [1, 2, 4, 5, 7, 8, 12, 16, 17, 18, 19, 20, 21]

# the joints used for evaluation in TransPose: all DIP prediction joints
TP_POS_EVAL_JOINTS = SMPL_DIP_JOINTS  # TODO: this should include wrists and ankles
TP_ANG_EVAL_JOINTS = SMPL_DIP_JOINTS

# the joints we use for evaluation of joint errors: all our prediction joints
POS_EVAL_JOINTS = SMPL_SSP_JOINTS  # TODO: this should include hands and feet
ANG_EVAL_JOINTS = SMPL_SSP_JOINTS

# mapping from the sensor names in SREC to our convention
SREC_2_SSP = {
    "left_hip": "left_pelvis",
    "left_thigh": "left_hip",
    "left_calf": "left_knee",
    "left_foot": "left_ankle",
    "head": "head",
    "right_hip": "right_pelvis",
    "right_thigh": "right_hip",
    "right_calf": "right_knee",
    "right_foot": "right_ankle",
    "left_back": "left_back",
    "left_shoulder": "left_collar",
    "left_arm": "left_shoulder",
    "left_forearm": "left_elbow",
    "left_hand": "left_wrist",
    "right_back": "right_back",
    "right_shoulder": "right_collar",
    "right_arm": "right_shoulder",
    "right_forearm": "right_elbow",
    "right_hand": "right_wrist",
}

# mapping RKK Studio FBX joint names to our convention
FBX2SMPL = {
    "Hips": "pelvis",
    "LeftThigh": "left_hip",
    "LeftShin": "left_knee",
    "LeftFoot": "left_ankle",
    "LeftToe": "left_foot",
    "RightThigh": "right_hip",
    "RightShin": "right_knee",
    "RightFoot": "right_ankle",
    "RightToe": "right_foot",
    "Spine1": "spine1",
    "Spine2": "spine2",
    "Spine3": "spine3",
    "Spine4": "spine4",
    "LeftShoulder": "left_collar",
    "LeftArm": "left_shoulder",
    "LeftForeArm": "left_elbow",
    "LeftHand": "left_wrist",
    "LeftFinger3Metacarpal": "left_hand",
    "Neck": "neck",
    "Head": "head",
    "RightShoulder": "right_collar",
    "RightArm": "right_shoulder",
    "RightForeArm": "right_elbow",
    "RightHand": "right_wrist",
    "RightFinger3Metacarpal": "right_hand",
}

# mocap markers used for illustrative purposes
MOCAP_MARKERS = [
    5892,
    5893,
    5897,
    5898,
    8586,
    8587,
    8591,
    8592,
    8576,
    5882,
    8634,
    8846,
    3726,
    6486,
    3663,
    6424,
    3672,
    6432,
    4133,
    3996,
    6877,
    6744,
    6832,
    4088,
    4083,
    6827,
    5948,
    6827,
    5694,
    8388,
    8370,
    5676,
    5417,
    6864,
    5489,
    5523,
    8243,
    8196,
    5462,
    3353,
    1470,
    2607,
    2711,
    1575,
    707,
    2197,
    5621,
    5532,
    8339,
    5645,
    6629,
    3875,
    6821,
    7053,
    6991,
    4077,
    4315,
    4247,
    4761,
    4884,
    4608,
    4842,
    7462,
    7344,
    7620,
    7458,
]
