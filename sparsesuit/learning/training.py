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

        # hyper-parameters
        train_config = cfg.training
        self.exp_name = train_config.experiment_name
        self.epochs = train_config.max_epochs
        self.early_stop_tol = train_config.early_stopping_tolerance
        self.batch_size_train = train_config.batch_size
        self.train_eval_step = train_config.evaluate_every_step
        self.grad_clip_norm = train_config.grad_clip_norm

        train_sens = train_config.sensors
        num_train_sens = train_config.count

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
        acc_dim = num_input_sens * 3
        pose_dim = num_trgt_joints * 9

        input_dim = ori_dim + acc_dim
        target_dim = pose_dim + acc_dim

        # init network
        self.model = BiRNN(input_dim=input_dim, target_dim=target_dim).to(self.device)
        # print(self.model)

        # init loss fct and optimizer
        self.loss_fn = torch.nn.GaussianNLLLoss(reduction="sum")
        self.optimizer = torch.optim.Adam(self.model.parameters())
        # work-around to get exponentially decaying rate at every step instead of stairs
        lr_gamma = pow(0.96, 1 / 2000)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=self.optimizer, step_size=1, gamma=lr_gamma
        )
        self.step_count = 0  # count number of training steps (x-axis in tensorboard)

        # tensorboard setup
        time_stamp = datetime.datetime.now().strftime("%Y.%m.%d-%H:%M")
        self.experiment_name = "-".join(
            [time_stamp, self.exp_name, train_config.config + str(num_train_sens)]
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
            .to_tuple("ori.npy", "acc.npy", "pose.npy")
        )
        valid_ds = (
            WebDataset(self.valid_ds_path, length=self.valid_ds_size)
            .decode()
            .to_tuple("ori.npy", "acc.npy", "pose.npy")
        )

        # create specific dataloaders for training (batched sequences) and testing (unrolled sequences)
        self.train_dl = DataLoader(
            train_ds, batch_size=self.batch_size_train, drop_last=True
        )
        self.valid_dl = DataLoader(valid_ds)

    def train(self):
        print("Starting experiment: {}".format(self.experiment_name))
        # train and test model iteratively
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
                    batch_num,
                    input_vec,
                    target_vec,
                )
                self.lr_scheduler.step()
                self.writer.add_scalar(
                    "training/learning_rate",
                    self.lr_scheduler.get_last_lr()[0],
                )
                self.step_count += 1

                # evaluate regularly
                if self.step_count % self.train_eval_step == 0:
                    valid_loss = self.validate()

                    # early stopping
                    if valid_loss <= best_valid_loss:
                        # save model
                        print("Saving model.")
                        torch.save(
                            self.model.state_dict(), self.model_path + "/checkpoint.pt"
                        )
                        best_valid_loss = valid_loss
                        num_steps_wo_improvement = 0
                    else:
                        num_steps_wo_improvement += 1

                    if num_steps_wo_improvement == self.early_stop_tol:
                        stop_signal = True
                        print("Stopping early.")
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
        train_config = OmegaConf.create(self.cfg.training)

        # add stats to training config
        train_config.duration = duration
        train_config.steps = self.step_count
        train_config.epochs = epoch
        train_config.best_valid_loss = best_valid_loss

        # compile target config to dump with model checkpoint
        trgt_config = OmegaConf.create()
        trgt_config.training = train_config
        trgt_config.dataset = self.ds_config

        utils.write_config(path=self.model_path, config=trgt_config)

    def training_step(self, batch_num, x, y):
        self.model.train()

        x, y = x.to(self.device).float(), y.to(self.device).float()

        # make predictions for full batch at once (state is reset after every sample in batch)
        # predict mean and std of smpl parameters for given orientations and accelerations
        pred = self.model(x)

        # evaluate loss
        loss, loss_smpl, loss_acc = self.eval_loss(pred, y, batch_num)

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

    def eval_loss(self, pred, y, sample):
        # evaluate smpl parameters and accelerometer predictions separately
        smpl_mean, smpl_std = (
            pred[0][..., :135].contiguous(),
            pred[1][..., :135].contiguous(),
        )
        acc_mean, acc_std = (
            pred[0][..., 135:].contiguous(),
            pred[1][..., 135:].contiguous(),
        )
        smpl_y, acc_y = y[..., :135].contiguous(), y[..., 135:].contiguous()
        loss_smpl = self.loss_fn(smpl_mean, smpl_y, smpl_std) / torch.numel(
            y
        )  # normalize loss w.r.t. number of predictions
        loss_acc = self.loss_fn(acc_mean, acc_y, acc_std) / torch.numel(y)
        loss = loss_smpl + loss_acc

        # terminal output
        current = (sample + 1) * len(y)
        print(
            "{}. SMPL loss: {:.3f}, ACC loss: {:.3f} [{}/{}]".format(
                sample, loss_smpl, loss_acc, current, self.train_ds_size
            )
        )

        return loss, loss_smpl.item(), loss_acc.item()

    def log_loss(self, loss, loss_smpl, loss_acc, mode: str):
        # tensorboard logging
        self.writer.add_scalar(mode + "/smpl_loss", loss_smpl, self.step_count)
        self.writer.add_scalar(mode + "/acc_loss", loss_acc, self.step_count)
        self.writer.add_scalar(mode + "/loss", loss, self.step_count)

    def validate(self):
        self.model.eval()
        loss = 0
        loss_smpl = 0
        loss_acc = 0
        print("\nTesting:")
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
                loss_i, loss_smpl_i, loss_acc_i = self.eval_loss(pred, y, sample)
                loss += loss_i.item()
                loss_smpl += loss_smpl_i
                loss_acc += loss_acc_i

        # tensorboard logging
        loss /= self.valid_ds_size
        loss_smpl /= self.valid_ds_size
        loss_acc /= self.valid_ds_size
        self.log_loss(loss, loss_smpl, loss_acc, "testing")
        return loss


@hydra.main(config_path="conf", config_name="training")
def do_training(cfg: DictConfig):
    try:
        trainer = Trainer(cfg=cfg)
        trainer.train()

        # evaluate trained model right away
        eval_cfg_path = os.path.join(os.getcwd(), "conf/evaluation.yaml")
        eval_cfg = OmegaConf.load(eval_cfg_path)
        eval_cfg.evaluation.experiment = trainer.experiment_name
        eval = Evaluator(cfg=eval_cfg)
        eval.evaluate()

    except KeyboardInterrupt:
        print("Interrupted. Deleting model_folder!")
        shutil.rmtree(trainer.model_path)
        sys.exit()


if __name__ == "__main__":
    do_training()
