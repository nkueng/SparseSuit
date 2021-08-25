import os
import sys

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf

from sparsesuit.learning.evaluation import Evaluator
from sparsesuit.learning.training import Trainer


@hydra.main(config_path="conf", config_name="training_cluster")
def do_training(cfg: DictConfig):
    env = submitit.JobEnvironment()
    try:
        trainer = Trainer(cfg=cfg)
        trainer.train()
    except KeyboardInterrupt:
        # print("Interrupted. Deleting model_folder!")
        # shutil.rmtree(trainer.model_path)
        sys.exit()

    # evaluate trained model right away
    eval_cfg_path = os.path.join(os.getcwd(), "conf/evaluation.yaml")
    eval_cfg = OmegaConf.load(eval_cfg_path)
    eval_cfg.evaluation.experiment = trainer.experiment_name
    # keep debugging and noise flag but force without visualization
    eval_cfg.debug = cfg.debug
    eval_cfg.noise = cfg.experiment.noise
    eval_cfg.visualize = False
    eval = Evaluator(cfg=eval_cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_training()
