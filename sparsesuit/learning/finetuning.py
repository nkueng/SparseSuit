import hydra
import submitit

from sparsesuit.constants import paths
from sparsesuit.learning.training import Trainer
from sparsesuit.learning.evaluation import Evaluator
import os
import sys
from sparsesuit.utils import utils
from omegaconf import DictConfig, OmegaConf

finetune_config_name = "finetuning_cluster" if paths.ON_CLUSTER else "finetuning"


@hydra.main(config_path="conf", config_name=finetune_config_name)
def do_finetuning(cfg: DictConfig):
    if paths.ON_CLUSTER:
        submitit.JobEnvironment()

    try:
        trainer = Trainer(cfg=cfg, finetune=True)
        trainer.train()
    except KeyboardInterrupt:
        # print("Interrupted. Deleting model_folder!")
        # shutil.rmtree(trainer.model_path)
        sys.exit()

    # load default evaluation configuration
    eval_cfg_path = os.path.join(
        utils.get_project_folder(), "learning/conf/evaluation.yaml"
    )
    eval_cfg = OmegaConf.load(eval_cfg_path)
    # adapt evaluation dataset to finetuning dataset
    eval_cfg.evaluation.experiment = trainer.experiment_name
    eval_cfg.evaluation.eval_dataset = cfg.finetune_dataset
    # keep debugging and noise flag but disable visualization
    eval_cfg.debug = cfg.debug
    eval_cfg.visualize = False
    evaluator = Evaluator(cfg=eval_cfg)
    evaluator.evaluate()


if __name__ == "__main__":
    do_finetuning()
