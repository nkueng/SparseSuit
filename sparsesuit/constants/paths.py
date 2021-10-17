import platform
import os

# platform check
ON_MAC = platform.system() == "Darwin"

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
else:
    DATA_PATH = CLUSTER_DATA_PATH

## dataset paths
DATASET_PATH = os.path.join(DATA_PATH, "training_data")
# synthetic data
AMASS_PATH = os.path.join(DATASET_PATH, "Synthetic/AMASS")
# real data
DIP_17_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_17")
DIP_17_NN_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_17_nn")
DIP_6_NN_PATH = os.path.join(DATASET_PATH, "Real/DIP_IMU_6_nn")
RKK_STUDIO_19_PATH = os.path.join(DATASET_PATH, "Real/RKK_19")
RKK_STUDIO_19_NN_PATH = os.path.join(DATASET_PATH, "Real/RKK_19_nn")

## smpl model data
SMPL_PATH = os.path.join(DATA_PATH, "SMPL_models")

## paths to trained models
RUN_PATH = os.path.join(DATA_PATH, "trained_models")

## documentation folder
DOC_PATH = os.path.join(DATA_PATH, "Documentation")
