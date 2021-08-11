import platform

# platform check
ON_MAC = platform.system() == 'Darwin'

# macOS dir
MAC_DATA_PATH = '/Volumes/SSD500/IMU_data/'

# linux dir
LINUX_DATA_PATH = '/media/nic/Extreme SSD/IMU_data/'

# root data path depending on platform; all paths should start here
DATA_PATH = MAC_DATA_PATH if ON_MAC else LINUX_DATA_PATH

## dataset paths
# synthetic data
AMASS_PATH = 'Synthetic/AMASS/'  # motion assets given as SMPL joints

AMASS_17_NOISY_PATH = 'Synthetic/AMASS_17_noisy/'  # synthesized IMU data of 17 sensors with noise
AMASS_17_NOISY_N_PATH = 'Synthetic/AMASS_17_noisy_n/'  # normalized IMU data of 17 sensors w.r.t. root sensor
AMASS_17_NOISY_NN_PATH = 'Synthetic/AMASS_17_noisy_nn/'  # zero-mean, unit-variance IMU data ready for training

AMASS_17_PATH = 'Synthetic/AMASS_17/'  # synthesized IMU data of 17 sensors without noise
AMASS_17_N_PATH = 'Synthetic/AMASS_17_n/'  # normalized IMU data of 17 sensors w.r.t. root sensor
AMASS_17_NN_PATH = 'Synthetic/AMASS_17_nn/'  # zero-mean, unit-variance IMU data ready for training

AMASS_19_PATH = 'Synthetic/AMASS_19/'  # synthesized IMU data of 19 sensors as in Rokoko's SSP with noise
AMASS_19_N_PATH = 'Synthetic/AMASS_19_n/'
AMASS_19_NN_PATH = 'Synthetic/AMASS_19_nn/'

AMASS_6_PATH = 'Synthetic/AMASS_6/'
AMASS_6_MINE_PATH = 'Synthetic/AMASS_6_mine/'
AMASS_6_NN_PATH = 'Synthetic/AMASS_6_nn/'  # open-sourced normalized version of AMASS_6
AMASS_6_NN_TEST_PATH = 'Synthetic/AMASS_6_nn_test/'  # my normalized version of AMASS_6 (to validate normalization)
AMASS_6_NN_MINE_PATH = 'Synthetic/AMASS_6_nn_mine/'  # normalized version of AMASS_6_mine

# real data
DIP_17_PATH = 'Real/DIP_IMU_17/'
DIP_17_N_PATH = 'Real/DIP_IMU_17_n/'
DIP_17_NN_PATH = 'Real/DIP_IMU_17_nn/'
DIP_6_N_PATH = 'Real/DIP_IMU_6_n/'
DIP_6_NN_PATH = 'Real/DIP_IMU_6_nn/'
DIP_6_NN_MINE_PATH = 'Real/DIP_IMU_6_nn_mine/'

# smpl model data
SMPL_PATH = 'SMPL_models/'

# file names
TRAIN_FILE = 'training.tar'  # to train models
VALID_FILE = 'validation.tar'  # to evaluate models during training on cost function
TEST_FILE = 'test.tar'  # to evaluate models after training for joint angles and positions
