import glob
import os
import random
from typing import *

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from dataset import get_trn_loader
from utils import AverageMeter, accuracy
from net import wmse_loss, GloveModel



class Trainer:
    def __init__(self, hparams, model, scaler):
        self.hparams = hparams
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = scaler

        # dataloader
        self.train_loader, vocab_len = get_trn_loader(
            data_dir=hparams.dpath.strip(),
            batch_size=hparams.batch_size,
            n_words = hparams.n_words,
            window_size = hparams.window_size,
            num_workers=hparams.workers,
            pin_memory=True,
        )
        self.model_name = hparams.model
        self.model = GloveModel(vocab_len, hparams.emb_dimension)
        # convert model to DP model
        self.model = nn.DataParallel(self.model).cuda()

        # optimizer
        self.optimizer= self.configure_optimizers()

        # model-saving options
        self.version = 0
        while True:
            self.save_path = os.path.join(hparams.ckpt_path, f"version-{self.version}")
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
                break
            else:
                self.version += 1
        self.summarywriter = SummaryWriter(self.save_path)
        self.global_step = 0
        self.global_top1_loss = 1e5
        self.eval_step = hparams.eval_step
        with open(
            os.path.join(self.save_path, "hparams.yaml"), "w", encoding="utf8"
        ) as outfile:
            yaml.dump(hparams, outfile, default_flow_style=False, allow_unicode=True)

        # experiment-logging options
        self.best_result = {"version": self.version}

    def configure_optimizers(self):
        # optimizer
        optimizer = optim.Adagrad(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        return optimizer

    def save_checkpoint(self, epoch: int, train_loss: float, model: nn.Module) -> None:
        tqdm.write(
            f"train loss decreased ({self.global_top1_loss:.4f} → {train_loss:.4f}). Saving model ..."
        )
        new_path = os.path.join(
            self.save_path, f"best_model_epoch_{epoch}_acc_{train_loss:.4f}.pt"
        )

        for filename in glob.glob(os.path.join(self.save_path, "*.pt")):
            os.remove(filename)  # remove old checkpoint
        torch.save(model.state_dict(), new_path)
        self.global_top1_loss = train_loss

    def fit(self) -> dict:
        for epoch in tqdm(range(self.hparams.epoch), desc="epoch"):
            tqdm.write(f"* Learning Rate: {self.optimizer.param_groups[0]['lr']:.5f}")
            result = self._train_epoch(epoch)

            # update checkpoint
            if result["train_loss"] < self.global_top1_loss:
                self.save_checkpoint(epoch, result["train_loss"], self.model)

        self.summarywriter.close()
        return self.version

    def _train_epoch(self, epoch: int) -> dict:
        train_loss = AverageMeter()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader),
            desc="train_steps",
            total=len(self.train_loader),
        ):
            x_ij, i_idx, j_idx = map(lambda x: x.to(self.device), batch)
            self.criterion = wmse_loss(x_ij, self.hparams.X_MAX, self.hparams.ALPHA)

            if self.hparams.amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(i_idx, j_idx)
                    loss = self.criterion(outputs, torch.log(x_ij))
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(i_idx, j_idx)
                loss = self.criterion(outputs, torch.log(x_ij))
                loss.backward()
                self.optimizer.step()

            train_loss.update(loss.item())

            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    f"[DP Version {self.version} Epoch {epoch}] global step: {self.global_step}, train loss: {loss.item():.3f}"
                )

        train_loss = train_loss.avg

        # tensorboard writing
        self.summarywriter.add_scalars(
            "lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch
        )
        self.summarywriter.add_scalars(
            "loss/step", {"train": train_loss}, self.global_step
        )
        self.summarywriter.add_scalars(
            "loss/epoch", {"train": train_loss}, epoch
        )

        return {"train_loss": train_loss}