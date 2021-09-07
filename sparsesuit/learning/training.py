import datetime
import logging
import os
import random
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import BiRNN
from sparsesuit.constants import paths
from sparsesuit.learning.evaluation import Evaluator
from sparsesuit.utils import utils


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        # cuda setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # reproducibility
        utils.make_deterministic(14)

        # experiment setup
        train_config = cfg.experiment
        self.exp_name = "debug" if cfg.debug else train_config.name
        train_sens = train_config.sensors
        num_train_sens = train_config.count

        # hyper-parameters
        self.hyper_params = cfg.hyperparams
        self.epochs = self.hyper_params.max_epochs
        self.early_stop_tol = self.hyper_params.early_stopping_tolerance
        self.batch_size_train = self.hyper_params.batch_size
        self.train_eval_step = self.hyper_params.evaluate_every_step
        self.grad_clip_norm = self.hyper_params.grad_clip_norm
        self.init_lr = self.hyper_params.initial_learning_rate
        self.shuffle = self.hyper_params.shuffle
        self.num_workers = self.hyper_params.num_workers
        self.pin_memory = self.hyper_params.pin_memory

        # find differences between this experiment and default hyperparams
        def_path = os.path.join(os.getcwd(), "conf/hyperparams")
        hyperparams_def = utils.load_config(def_path, "default.yaml")
        hyperparams_diff = {
            k: cfg.hyperparams[k]
            for k, _ in set(hyperparams_def.items()) - set(cfg.hyperparams.items())
        }

        # tensorboard setup
        time_stamp = datetime.datetime.now().strftime("%y%m%d%H%M")

        # create folder name from time, experiment, and hyperparameter changes
        self.experiment_name = "-".join(
            [
                time_stamp,
                self.exp_name,
                # train_config.config + str(num_train_sens),
            ]
        )

        for k, v in hyperparams_diff.items():
            self.experiment_name += "-" + k + str(v)

        self.model_path = os.path.join(os.getcwd(), "runs/" + self.experiment_name)
        self.writer = SummaryWriter(self.model_path)

        # logger setup
        log_level = logging.DEBUG if cfg.debug else logging.INFO
        self.logger = utils.configure_logger(
            name="training", log_path=self.model_path, level=log_level
        )
        print("Training\n*******************\n")
        self.logger.info("Using {} device".format(self.device))
        self.logger.info(OmegaConf.to_yaml(cfg))

        # get dataset required by configuration
        ds_folder = paths.AMASS_PATH
        if cfg.debug:
            ds_folder += "_debug"
            self.epochs = 2
            self.train_eval_step = 10

        if train_config.dataset == "synthetic":

            if train_config.config == "SSP":
                ds_folder += "_SSP"

            elif train_config.config == "MVN":
                ds_folder += "_MVN"

            else:
                raise NameError("Invalid configuration. Aborting!")

            if train_config.noise:
                ds_folder += "_noisy"
            ds_folder += "_nn"

        elif train_config.dataset == "real":

            if train_config.config == "MVN":
                ds_folder = paths.DIP_17_NN_PATH

            else:
                raise NameError("Invalid configuration. Aborting!")

        # get training and validation dataset paths
        ds_dir = os.path.join(paths.DATA_PATH, ds_folder)
        self.train_ds_path = os.path.join(ds_dir, "training")
        self.valid_ds_path = os.path.join(ds_dir, "validation")

        # load config of dataset
        self.ds_config = utils.load_config(ds_dir).dataset
        self.train_ds_size = self.ds_config.assets.training
        self.valid_ds_size = self.ds_config.assets.validation
        ds_sens = self.ds_config.sensors
        num_trgt_joints = len(self.ds_config.pred_trgt_joints)

        # find indices of training sensors in dataset vectors
        self.sens_ind = [ds_sens.index(sens) for sens in train_sens]

        # derive input/output dimension of network from choice of train_sens
        num_input_sens = len(train_sens)
        ori_dim = num_input_sens * 9
        self.acc_dim = num_input_sens * 3
        self.pose_dim = num_trgt_joints * 9

        input_dim = ori_dim + self.acc_dim
        target_dim = self.pose_dim + self.acc_dim

        # init network
        self.model = BiRNN(input_dim=input_dim, target_dim=target_dim).to(self.device)
        self.logger.info(
            "Trainable parameters: {}".format(count_parameters(self.model))
        )
        # self.logger.info(self.model)

        # init loss fct and optimizer
        self.loss_fn = torch.nn.GaussianNLLLoss(reduction="sum", full=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)
        # work-around to get exponentially decaying rate at every step instead of stairs
        lr_gamma = pow(0.96, 1 / 2000)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=lr_gamma,
        )
        self.step_count = 0  # count number of training steps (x-axis in tensorboard)

        # load datasets
        train_ds = utils.BigDataset(self.train_ds_path, self.train_ds_size)
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size_train,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        valid_ds = utils.BigDataset(self.valid_ds_path, self.valid_ds_size)
        self.valid_dl = DataLoader(
            valid_ds,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def train(self):
        self.logger.info("Starting experiment: {}".format(self.experiment_name))
        # train and validate model iteratively
        best_valid_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False
        t0 = time.perf_counter()

        # iterate over epochs
        for epoch in range(self.epochs):
            self.logger.debug(
                "\nEpoch {}\n-------------------------------".format(epoch + 1)
            )

            # iterate over all sample batches in epoch
            for batch_num, (ori, acc, pose, file_id) in enumerate(self.train_dl):

                self.logger.debug("\nLoaded files {}:".format("\n".join(file_id)))

                input_vec, target_vec = utils.assemble_input_target(
                    ori, acc, pose, self.sens_ind
                )
                self.training_step(
                    epoch,
                    batch_num,
                    input_vec,
                    target_vec,
                )
                self.lr_scheduler.step()
                self.writer.add_scalar(
                    "training/learning_rate",
                    self.lr_scheduler.get_last_lr()[0],
                    self.step_count,
                )
                self.step_count += 1

                # evaluate after every train_eval_step
                if self.step_count % self.train_eval_step == 0:

                    valid_loss = self.validate()

                    # early stopping
                    if valid_loss <= best_valid_loss:
                        # save model
                        self.logger.debug("Saving model.")
                        torch.save(
                            self.model.state_dict(),
                            self.model_path + "/checkpoint.pt",
                        )
                        best_valid_loss = valid_loss
                        num_steps_wo_improvement = 0
                    else:
                        num_steps_wo_improvement += 1

                    if num_steps_wo_improvement == self.early_stop_tol:
                        stop_signal = True
                        self.logger.info(
                            "No improvement for {} steps. Stopping early!".format(
                                self.early_stop_tol
                            )
                        )
                        break

            if stop_signal:
                break

        t1 = time.perf_counter()
        duration = t1 - t0
        self.logger.info("Done in {} seconds!\n".format(duration))

        self.write_config(duration, epoch, best_valid_loss)

        self.writer.close()

    def write_config(self, duration, epoch, best_valid_loss):
        # load training configuration
        train_config = OmegaConf.create(self.cfg.experiment)

        # add stats to training config
        train_config.duration = duration
        train_config.steps = self.step_count
        train_config.epochs = epoch
        train_config.best_valid_loss = best_valid_loss

        # compile target config to dump with model checkpoint
        trgt_config = OmegaConf.create()
        trgt_config.experiment = train_config
        trgt_config.hyperparameters = self.hyper_params
        trgt_config.dataset = self.ds_config

        utils.write_config(path=self.model_path, config=trgt_config)

    def training_step(self, epoch, batch_num, x, y):
        self.model.train()

        # push CPU-loaded vectors to GPU (pin_memory -> True)
        x, y = x.to(self.device).float(), y.to(self.device).float()

        # make predictions for full batch at once (state is reset after every sample in batch)
        # predict mean and std of smpl parameters for given orientations and accelerations
        pred = self.model(x)

        # evaluate loss
        loss, loss_smpl, loss_acc = self.eval_loss(pred, y)

        # console output
        current = (batch_num + 1) * len(y)
        self.logger.debug(
            "{}. Epoch: {}. SMPL loss: {:.3f}, ACC loss: {:.3f} [{}/{}]".format(
                epoch + 1, batch_num, loss_smpl, loss_acc, current, self.train_ds_size
            )
        )

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # DEBUG: print difference in gradient norms after clipping
        # norms = []
        # for name, param in self.model.named_parameters():
        #     # print(name, float(param.grad.norm()))
        #     norms.append(float(param.grad.norm()))

        # clip gradients: only has effect for 1 and below (and if learning rate is larger than e-4)
        total_norm = 0
        if self.grad_clip_norm != 0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
        self.writer.add_scalar("training/total_norm", total_norm, self.step_count)

        # DEBUG: print difference in gradient norms after clipping
        # for i, (name, param) in enumerate(self.model.named_parameters()):
        #     # print(name, float(param.grad.norm()))
        #     print(norms[i] - float(param.grad.norm()))

        # weight update
        self.optimizer.step()

        # tensorboard logging
        self.log_loss(loss.item(), loss_smpl, loss_acc, "training")

    def eval_loss(self, pred, y):
        # evaluate smpl parameters and accelerometer predictions separately
        smpl_mean, smpl_std = (
            pred[0][..., : self.pose_dim].contiguous(),
            pred[1][..., : self.pose_dim].contiguous(),
        )
        acc_mean, acc_std = (
            pred[0][..., self.pose_dim :].contiguous(),
            pred[1][..., self.pose_dim :].contiguous(),
        )
        smpl_y, acc_y = (
            y[..., : self.pose_dim].contiguous(),
            y[..., self.pose_dim :].contiguous(),
        )

        # reduce loss to average per frame (seq_len) and samples (batch_size)
        divisor = y.shape[0] * y.shape[1]
        loss_smpl = self.loss_fn(smpl_mean, smpl_y, torch.square(smpl_std)) / divisor
        loss_acc = self.loss_fn(acc_mean, acc_y, torch.square(acc_std)) / divisor
        loss = loss_smpl + loss_acc

        return loss, loss_smpl.item(), loss_acc.item()

    def log_loss(self, loss, loss_smpl, loss_acc, mode: str):
        # tensorboard logging
        self.writer.add_scalar(mode + "/smpl_loss", loss_smpl, self.step_count)
        self.writer.add_scalar(mode + "/acc_loss", loss_acc, self.step_count)
        self.writer.add_scalar(mode + "/loss", loss, self.step_count)

    def validate(self):
        """Validate training progress of model. This checks the main training objective of the model, which is to
        reconstruct poses, thus only the smpl loss is of relevance."""
        self.model.eval()
        loss = 0
        loss_smpl = 0
        loss_acc = 0
        self.logger.debug("\nValidating:")
        with torch.no_grad():
            for sample, (ori, acc, pose, _) in enumerate(self.valid_dl):

                input_vec, target_vec = utils.assemble_input_target(
                    ori, acc, pose, self.sens_ind
                )
                x, y = (
                    input_vec.to(self.device).float(),
                    target_vec.to(self.device).float(),
                )
                pred = self.model(x)
                loss_i, loss_smpl_i, loss_acc_i = self.eval_loss(pred, y)

                # console output
                current = (sample + 1) * len(y)
                self.logger.debug(
                    "{}. SMPL loss: {:.3f}, ACC loss: {:.3f} [{}/{}]".format(
                        sample, loss_smpl_i, loss_acc_i, current, self.valid_ds_size
                    )
                )

                loss += loss_i.item()
                loss_smpl += loss_smpl_i
                loss_acc += loss_acc_i

        # tensorboard logging of averaged losses
        loss /= self.valid_ds_size
        loss_smpl /= self.valid_ds_size
        loss_acc /= self.valid_ds_size
        self.log_loss(loss, loss_smpl, loss_acc, "validation")

        # console output
        self.logger.info(
            "\nValid SMPL loss: {:.3f}, Valid ACC loss: {:.3f}\n".format(
                loss_smpl_i, loss_acc_i
            )
        )
        # only SMPL loss is used for validation
        return loss_smpl


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@hydra.main(config_path="conf", config_name="training")
def do_training(cfg: DictConfig):
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
