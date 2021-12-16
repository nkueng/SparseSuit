import os
import numpy as np
from sparsesuit.constants import paths, sensors

"""
Compare the predictive performance for sequences that changed from training to test set: 
How much better is the prediction if the model has seen them before vs. never seen before?
"""

# get data (model where ACCAD was in training set and model where it was not)
# path_trained = os.path.join(paths.RUN_PATH, "2111171748-SSP_07_syn/pos_errs.npz")
path_trained = os.path.join(paths.RUN_PATH, "2111171748-SSP_19_syn/pos_errs.npz")
# path_untrained = os.path.join(paths.RUN_PATH, "2111202113-SSP_07_ACCAD/pos_errs.npz")
path_untrained = os.path.join(paths.RUN_PATH, "2111202118-SSP_19_ACCAD/pos_errs.npz")

data_trained = dict(np.load(path_trained))
data_untrained = dict(np.load(path_untrained))

accad_assets_trained = np.array(list(data_trained.values())[:4])
accad_assets_untrained = np.array(list(data_untrained.values())[:4])

mean_trained = np.mean(accad_assets_trained[:, sensors.POS_EVAL_JOINTS])
mean_untrained = np.mean(accad_assets_untrained[:, sensors.POS_EVAL_JOINTS])

diff = accad_assets_trained - accad_assets_untrained
