import os
from sparsesuit.utils import utils, smpl_helpers
from sparsesuit.learning import models
import torch
import hydra
from omegaconf import DictConfig


class Evaluator:
    def __init__(self, cfg):
        # load experiment config
        exp_path = os.path.join(os.getcwd(), "runs/", cfg.experiment_name)
        exp_conf = utils.load_config(exp_path)

        # cuda setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        # load neutral smpl model for joint position evaluation
        self.smpl_model = smpl_helpers.load_smplx(["neutral"])["neutral"]

        # setup model
        # TODO: fix this with actual config
        train_sens = exp_conf.training.training_sensors
        num_train_sens = len(train_sens)
        ori_dim = (num_train_sens - 1) * 9
        acc_dim = (num_train_sens - 1) * 3
        input_dim = ori_dim + acc_dim
        pose_dim = len(exp_conf.dataset.dataset_sensors.target_joints) * 9
        target_dim = pose_dim + acc_dim
        model = models.BiRNN(input_dim=input_dim, target_dim=target_dim).to(self.device)
        model_path = os.path.join(exp_path, "checkpoint.pt")
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()

        # get indices of sensors model was trained with in dataset
        ds_sens_names = cfg.dataset_sensors.names
        train_sens_names = cfg.training_sensors.names
        self.sens_ind = [ds_sens_names.index(sens) for sens in train_sens_names]

    def evaluate(self):
        pass


@hydra.main(config_path="conf", config_name="evaluation")
def do_evaluation(cfg: DictConfig):
    eval = Evaluator(cfg=cfg)
    eval.evaluate()


if __name__ == "__main__":
    do_evaluation()
