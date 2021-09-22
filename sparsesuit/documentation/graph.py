# TODO: crawl /runs directory, for each run, collect sensor number and angular error, then select min error for each sensor number and plot
import os

from sparsesuit.utils import utils
from sparsesuit.constants import paths

scores = {}
run_dir = paths.RUN_PATH
for root, dirs, files in os.walk(run_dir):
    if not files:
        continue
    if "config.yaml" not in files:
        continue
    config = utils.load_config(root)
    if "metrics" not in config.evaluation:
        continue
    sens_count = config.experiment.count
    err = config.evaluation.metrics.avg_total_ang_err
    if sens_count in scores:
        if scores[sens_count] <= err:
            continue
    scores[sens_count] = err

print(scores)
