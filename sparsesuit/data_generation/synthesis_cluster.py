""" A script to handle the synthesis of IMU data from 17 sensors based on the AMASS dataset of SMPL pose data. """
import os

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

from sparsesuit.data_generation.normalization import Normalizer
from sparsesuit.data_generation.synthesis import Synthesizer
from sparsesuit.utils import utils


@hydra.main(config_path="conf", config_name="synthesis_cluster")
def do_synthesis(cfg: DictConfig):
    # reads hydra config to run on SLURM cluster
    submitit.JobEnvironment()

    syn = Synthesizer(cfg=cfg)
    syn.synthesize_dataset()

    # load default config for normalization
    norm_cfg_path = os.path.join(
        utils.get_project_folder(), "data_generation/conf/normalization.yaml"
    )
    norm_cfg = OmegaConf.load(norm_cfg_path)
    # adapt default config to synthesis configuration
    norm_cfg.dataset = cfg.dataset
    norm_cfg.debug = cfg.debug
    # run normalization with adapted configuration
    norm = Normalizer(cfg=norm_cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_synthesis()
