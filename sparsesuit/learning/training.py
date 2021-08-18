import datetime
import os
import random
import shutil
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from webdataset import WebDataset

from models import BiRNN
from sparsesuit.constants import paths
from sparsesuit.learning.evaluation import Evaluator
from sparsesuit.utils import utils

seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


class Trainer:
    def __init__(self, cfg):
        print("Training\n*******************\n")
        self.cfg = cfg

        # cuda setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(self.device))

        # experiment setup
        train_config = cfg.experiment
        self.exp_name = train_config.name
        train_sens = train_config.sensors
        num_train_sens = train_config.count

        # hyper-parameters
        hyper_params = cfg.hyperparams
        self.epochs = hyper_params.max_epochs
        self.early_stop_tol = hyper_params.early_stopping_tolerance
        self.batch_size_train = hyper_params.batch_size
        self.train_eval_step = hyper_params.evaluate_every_step
        self.grad_clip_norm = hyper_params.grad_clip_norm

        # get dataset required by configuration
        ds_folder = paths.AMASS_PATH
        if cfg.debug:
            ds_folder += "_debug"
            self.exp_name = "debug"
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
        self.train_ds_path = os.path.join(ds_dir, paths.TRAIN_FILE)
        self.valid_ds_path = os.path.join(ds_dir, paths.VALID_FILE)

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
        # print(self.model)

        # init loss fct and optimizer
        self.loss_fn = torch.nn.GaussianNLLLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # work-around to get exponentially decaying rate at every step instead of stairs
        lr_gamma = pow(0.96, 1 / 2000)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=1,
            gamma=lr_gamma,
        )
        self.step_count = 0  # count number of training steps (x-axis in tensorboard)

        # tensorboard setup
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M")
        self.experiment_name = "-".join(
            [
                time_stamp,
                self.exp_name,
                train_config.config + str(num_train_sens),
            ]
        )
        self.model_path = os.path.join(os.getcwd(), "runs/" + self.experiment_name)
        self.writer = SummaryWriter(self.model_path)

        # create datasets
        train_ds = (
            WebDataset(
                self.train_ds_path,
                shardshuffle=True,
                length=self.train_ds_size,
            )
            .decode()
            .to_tuple(
                "ori.npy",
                "acc.npy",
                "pose.npy",
            )
        )
        valid_ds = (
            WebDataset(
                self.valid_ds_path,
                shardshuffle=False,
                length=self.valid_ds_size,
            )
            .decode()
            .to_tuple(
                "ori.npy",
                "acc.npy",
                "pose.npy",
            )
        )

        # create specific dataloaders for training (batched sequences) and validation (unrolled sequences)
        self.train_dl = DataLoader(
            train_ds,
            batch_size=self.batch_size_train,
            drop_last=True,
        )
        self.valid_dl = DataLoader(valid_ds)

    def train(self):
        print("Starting experiment: {}".format(self.experiment_name))
        # train and validate model iteratively
        valid_loss = np.inf
        best_valid_loss = np.inf
        num_steps_wo_improvement = 0
        stop_signal = False
        t0 = time.perf_counter()

        # iterate over epochs
        for epoch in range(self.epochs):
            print("\nEpoch {}\n-------------------------------".format(epoch + 1))

            # iterate over all sample batches in epoch
            for batch_num, (ori, acc, pose) in enumerate(self.train_dl):
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
                        print("Saving model.")
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
                        print(
                            "No improvement for {} steps. Stopping early!".format(
                                self.early_stop_tol
                            )
                        )
                        break

            if stop_signal:
                break

        t1 = time.perf_counter()
        duration = t1 - t0
        print("Done in {} seconds!\n".format(duration))

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
        trgt_config.dataset = self.ds_config

        utils.write_config(path=self.model_path, config=trgt_config)

    def training_step(self, epoch, batch_num, x, y):
        self.model.train()

        x, y = x.to(self.device).float(), y.to(self.device).float()

        # make predictions for full batch at once (state is reset after every sample in batch)
        # predict mean and std of smpl parameters for given orientations and accelerations
        pred = self.model(x)

        # evaluate loss
        loss, loss_smpl, loss_acc = self.eval_loss(pred, y)

        # console output
        current = (batch_num + 1) * len(y)
        print(
            "{}. Epoch: {}. SMPL loss: {:.3f}, ACC loss: {:.3f} [{}/{}]".format(
                epoch + 1, batch_num, loss_smpl, loss_acc, current, self.train_ds_size
            )
        )

        # backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # # DEBUG: print gradient norms
        # for name, param in model.named_parameters():
        #     print(name, param.grad.norm())

        # clip gradients: very little effect
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        # # DEBUG: print gradient norms
        # for name, param in model.named_parameters():
        #     print(name, param.grad.norm())

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

        # compute loss for pose and acceleration reconstruction individually
        loss_smpl = self.loss_fn(smpl_mean, smpl_y, smpl_std) / torch.numel(y)
        loss_acc = self.loss_fn(acc_mean, acc_y, acc_std) / torch.numel(y)
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
        print("\nValidating:")
        with torch.no_grad():
            for sample, (ori, acc, pose) in enumerate(self.valid_dl):
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
                print(
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
        # TODO: change to "validation" before starting training for real
        self.log_loss(loss, loss_smpl, loss_acc, "testing")
        return loss_smpl


@hydra.main(config_path="conf", config_name="training")
def do_training(cfg: DictConfig):
    try:
        trainer = Trainer(cfg=cfg)
        trainer.train()
    except KeyboardInterrupt:
        print("Interrupted. Deleting model_folder!")
        shutil.rmtree(trainer.model_path)
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
