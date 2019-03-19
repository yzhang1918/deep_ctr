import numpy as np
from sklearn.metrics import roc_auc_score

import torch

import tqdm

from .base_trainer import BaseTrainer


class TikTokTrainer(BaseTrainer):

    def __init__(self, model, optimizer, train_dl, valid_dl, test_dl, criterion,
                 smooth_factor=.9, break_at=-1, save_path='./models'):
        super().__init__(model, optimizer, criterion, train_dl, save_path=save_path)
        self.valid_dl = valid_dl
        self.test_dl = test_dl
        self.smooth_factor = smooth_factor
        self.break_at = break_at

    def forward_on_batch(self, batch):
        x, labels = batch.cuda()
        preds = self.model(*x)
        loss = self.criterion(preds, labels)
        return loss

    def calc_auc(self, preds, targets):
        finish_preds = preds[:, 0]
        finish_targs = targets[:, 0]
        like_preds = preds[:, 1]
        like_targs = targets[:, 1]
        finish_auc = roc_auc_score(finish_targs, finish_preds)
        like_auc = roc_auc_score(like_targs, like_preds)
        return finish_auc, like_auc

    def train(self, i_epoch=None, scheduler=None):
        self.model.train()
        cum_loss = None
        with tqdm.tqdm(total=len(self.train_dl)) as t:
            if i_epoch is not None:
                t.set_description(f'[Epoch {i_epoch + 1:3d}]')
            for i, batch in enumerate(self.train_dl):
                if scheduler is not None:
                    scheduler.step()
                self.optimizer.zero_grad()
                x, labels = batch.cuda()
                preds = self.model(*x)
                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()

                if cum_loss is None:
                    cum_loss = loss.item()
                else:
                    cum_loss = self.smooth_factor * cum_loss + (1 - self.smooth_factor) * loss.item()

                t.set_postfix(loss=f'{cum_loss:.3f}')
                t.update()
                if 0 < self.break_at < i:
                    return

    def valid(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        with torch.no_grad():
            cum_loss = 0.
            for i, batch in tqdm.tqdm(enumerate(self.valid_dl), total=len(self.valid_dl)):
                x, labels = batch.cuda()
                preds = self.model(*x)
                loss = self.criterion(preds, labels)
                cum_loss += loss.item()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())
                if 0 < self.break_at < i:
                    break
        n = len(self.valid_dl)
        cum_loss /= n
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        finish_auc, like_auc = self.calc_auc(all_preds, all_targets)
        auc = self.criterion.finish_weight * finish_auc + (1 - self.criterion.finish_weight) * like_auc
        info = dict(loss=cum_loss, auc=auc)
        return info

    def test(self):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for i, batch in tqdm.tqdm(enumerate(self.test_dl), total=len(self.test_dl)):
                x, _ = batch.cuda()
                preds = self.model(*x)
                all_preds.append(preds.cpu().numpy())
                if 0 < self.break_at < i:
                    break
        all_preds = np.concatenate(all_preds)
        return all_preds
