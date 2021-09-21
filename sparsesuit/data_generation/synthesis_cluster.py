""" A script to handle the synthesis of IMU data from 17 sensors based on the AMASS dataset of SMPL pose data. """
import hydra
import submitit
from omegaconf import DictConfig

from sparsesuit.data_generation.normalization import Normalizer
from sparsesuit.data_generation.synthesis import Synthesizer


@hydra.main(config_path="conf", config_name="synthesis_cluster")
def do_synthesis(cfg: DictConfig):
    # reads hydra config to run on SLURM cluster
    env = submitit.JobEnvironment()

    syn = Synthesizer(cfg=cfg)
    syn.synthesize_dataset()

    norm = Normalizer(cfg=cfg)
    norm.normalize_dataset()


if __name__ == "__main__":
    do_synthesis()
