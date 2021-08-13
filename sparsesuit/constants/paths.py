import platform

# platform check
ON_MAC = platform.system() == "Darwin"

# macOS dir
MAC_DATA_PATH = "/Volumes/SSD500/IMU_data/"

# linux dir
LINUX_DATA_PATH = "/media/nic/Extreme SSD/IMU_data/"

# root data path depending on platform; all paths should start here
DATA_PATH = MAC_DATA_PATH if ON_MAC else LINUX_DATA_PATH

## dataset paths
# synthetic data
AMASS_PATH = "Synthetic/AMASS"  # motion assets given as SMPL joints

# real data
DIP_17_PATH = "Real/DIP_IMU_17/"
DIP_17_NN_PATH = "Real/DIP_IMU_17_nn/"

# smpl model data
SMPL_PATH = "SMPL_models/"

# file names
# to train models
TRAIN_FILE = "training.tar"
# to evaluate models during training on cost function
VALID_FILE = "validation.tar"
# to evaluate models after training for joint angles and positions
TEST_FILE = "test.tar"
