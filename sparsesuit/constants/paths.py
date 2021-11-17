import platform
import os

# platform check
ON_MAC = platform.system() == "Darwin"
ON_LINUX = False
ON_CLUSTER = False

# macOS dir
MAC_DATA_PATH = "/Volumes/SSD500/thesis_data"

# linux dir
LINUX_DATA_PATH = "/media/nic/ExtremeSSD/thesis_data"
CLUSTER_DATA_PATH = "/home/mjd957/thesis_data"

# root data path depending on platform; all paths should start here
if ON_MAC:
    DATA_PATH = MAC_DATA_PATH
elif platform.node() == "nic-RKK":
    DATA_PATH = LINUX_DATA_PATH
    ON_LINUX = True
else:
    DATA_PATH = CLUSTER_DATA_PATH
    ON_CLUSTER = True

# source data path
SOURCE_PATH = os.path.join(DATA_PATH, "source_data")

# training dataset paths
DATASET_PATH = os.path.join(DATA_PATH, "training_data")

AMASS_PATH = os.path.join(DATASET_PATH, "Synthetic/AMASS")
DIP_17_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_17")
DIP_17_NN_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_17_nn")
DIP_6_NN_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_6_n")
RKK_STUDIO_19_PATH = os.path.join(DATASET_PATH, "Real/RKK_19")
RKK_STUDIO_19_NN_PATH = os.path.join(DATASET_PATH, "Real/RKK_19_nn")

# smpl model data
SMPL_PATH = os.path.join(DATA_PATH, "SMPL_models")

# path to trained models
RUN_PATH = os.path.join(DATA_PATH, "trained_models")

# path to models in final evaluation
EVAL_PATH = os.path.join(DATA_PATH, "evaluation_models")

# documentation folder
DOC_PATH = os.path.join(DATA_PATH, "Documentation")

# asset-dataset mapping to guarantee comparability and good balance of test assets
DIP_IMU_MAPPING = {
    "s_09": "test",
    "s_10": "test",
    "s_01/05": "validation",
    "s_03/05": "validation",
    "s_07/04": "validation",
}

# one entire subject for test, one of each motion for validation
RKK_STUDIO_MAPPING = {
    "01": "test",
    "05/3_Sidestep_take-7": "validation",
    "02/1_Gait_take-3": "validation",
    "03/5_Jump_take-9": "validation",
    "04/2_Run_take-1": "validation",
    "02/4_Sway_take-2": "validation",
}
RKK_VICON_MAPPING = RKK_STUDIO_MAPPING
# test dataset contains all gait motions: enable code in normalization.py
# RKK_STUDIO_MAPPING = {
#     "01/3_Sidestep_take-7": "validation",
#     "02/2_Run_take-1": "validation",
#     "05/4_Sway_take-5": "validation",
# }
# use entire dataset for evaluation
# RKK_STUDIO_MAPPING = {
#     "01": "test",
#     "02": "test",
#     "03": "test",
#     "04": "test",
#     "05": "test",
# }

AMASS_MAPPING = {
    "ACCAD/Female1Running_c3d_C5-walktorun": "test",
    "SFU/0017_0017_RunningOnBench002": "test",
    "DFaust_67/50026_50026_running_on_spot": "test",
    "ACCAD/Male2General_c3d_A2-Sway": "test",
    "BMLhandball/S08_Novice_Trial_upper_right_160": "test",
    "BMLmovi/Subject_25_F_MoSh_Subject_25_F_13": "test",
    "DanceDB/20151003_AndriaMichaelidou_Andria_Annoyed_v1_C3D": "test",
    "BioMotionLab_NTroje/rub109_0031_rom": "test",
    "BioMotionLab_NTroje/rub060_0016_sitting2": "test",
    "TCD_handMocap/ExperimentDatabase_typing_2": "test",
    "KIT/11_RightTurn02": "test",
    "KIT/348_walking_slow01": "test",
    "BioMotionLab_NTroje/rub066_0006_normal_walk2": "test",
    "ACCAD/Female1Walking_c3d_B11-walkturnleft(135)": "test",
    "KIT/317_walking_slow07": "test",
    "KIT/3_jump_up03": "test",
    "BioMotionLab_NTroje/rub025_0027_jumping1": "test",
    "Transitions_mocap/mazen_c3d_jumpingjacks_walk": "test",
    "Transitions_mocap/mazen_c3d_jumpingjacks_jumpinplace": "test",
    "MPI_HDM05/bk_HDM_bk_03-04_03_120": "test",
    "Eyes_Japan_Dataset/hamada_accident-11-falldown-hamada": "test",
    "SSM_synced/20160330_03333_chicken_wings_poses": "test",
    "ACCAD/Female1General_c3d_A2-Sway": "test",
}

AMASS_MAPPING_OVERFIT = {"ACCAD": "test"}
