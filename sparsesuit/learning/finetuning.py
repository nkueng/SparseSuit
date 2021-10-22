import hydra
from sparsesuit.learning.training import Trainer
from sparsesuit.learning.evaluation import Evaluator
import os
import sys
from sparsesuit.utils import utils
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="conf", config_name="finetuning")
def do_finetuning(cfg: DictConfig):
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
    eval_cfg.evaluation.eval_dataset = cfg.experiment.train_dataset
    # keep debugging and noise flag but disable visualization
    eval_cfg.debug = cfg.debug
    eval_cfg.visualize = False
    eval = Evaluator(cfg=eval_cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_finetuning()
