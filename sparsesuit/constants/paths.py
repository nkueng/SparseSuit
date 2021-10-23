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
DIP_6_NN_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_6_nn")
RKK_STUDIO_19_PATH = os.path.join(DATASET_PATH, "Real/RKK_19")
RKK_STUDIO_19_NN_PATH = os.path.join(DATASET_PATH, "Real/RKK_19_nn")

# smpl model data
SMPL_PATH = os.path.join(DATA_PATH, "SMPL_models")

# paths to trained models
RUN_PATH = os.path.join(DATA_PATH, "trained_models")

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

# RKK_STUDIO_MAPPING = {
#     "05": "test",
#     "01/3_Sidestep_take-7": "validation",
#     "03/1_Gait_take-3": "validation",
#     "04/5_Jump_take-9": "validation",
#     "02/2_Run_take-1": "validation",
#     "01/4_Sway_take-2": "validation",
# }
# test dataset contains all gait motions
# RKK_STUDIO_MAPPING = {
#     "01/3_Sidestep_take-7": "validation",
#     "02/2_Run_take-1": "validation",
#     "05/4_Sway_take-5": "validation",
# }
# use entire dataset for evaluation
RKK_STUDIO_MAPPING = {
    "01": "test",
    "02": "test",
    "03": "test",
    "04": "test",
    "05": "test",
}

AMASS_MAPPING = {
    "ACCAD/Female1Running_c3d_C5-walktorun": "test",
    "ACCAD/Male2General_c3d_A2-Sway": "test",
    "BMLhandball/S08_Novice_Trial_upper_right_160": "test",
    "BMLmovi/Subject_25_F_MoSh_Subject_25_F_13": "test",
    "DanceDB/20151003_AndriaMichaelidou_Andria_Annoyed_v1_C3D": "test",
    "BioMotionLab/NTroje_rub109_0031_rom": "test",
    "BioMotionLab/NTroje_rub060_0016_sitting2": "test",
    "TCD/handMocap_ExperimentDatabase_typing_2": "test",
    "KIT/11_RightTurn02": "test",
    "KIT/317_walking_slow07": "test",
    "KIT/3_jump_up03": "test",
    "MPI/HDM05_bk_HDM_bk_03-04_03_120": "test",
    "Eyes_Japan/Dataset_hamada_accident-11-falldown-hamada": "test",
}
