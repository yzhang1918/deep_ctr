import numpy as np
from sklearn.metrics import roc_auc_score

import torch

import tqdm

from .base_trainer import BaseTrainer


class TikTokTrainer(BaseTrainer):

    def __init__(self, model, optimizer, criterion, train_dl, valid_dl, test_dl,
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
        total = self.break_at if self.break_at > 0 else len(self.train_dl)
        with tqdm.tqdm(total=total) as t:
            if i_epoch is not None:
                t.set_description(f'[Epoch {i_epoch + 1:4d}]')
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

                t.set_postfix(loss=f'{loss:.3f}')
                t.update()
                if 0 < self.break_at <= i+1:
                    return

    def valid(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        cum_loss = 0.
        total = self.break_at if self.break_at > 0 else len(self.valid_dl)
        with torch.no_grad():
            with tqdm.tqdm(total=total) as t:
                t.set_description('[Validation]')
                for i, batch in enumerate(self.valid_dl):
                    x, labels = batch.cuda()
                    preds = self.model(*x)
                    loss = self.criterion(preds, labels)
                    cum_loss += loss.item()
                    all_preds.append(preds.cpu().numpy())
                    all_targets.append(labels.cpu().numpy())
                    t.set_postfix(loss='0.000')
                    t.update()
                    if 0 < self.break_at <= i+1:
                        break
        cum_loss /= total
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)
        finish_auc, like_auc = self.calc_auc(all_preds, all_targets)
        auc = self.criterion.finish_weight * finish_auc + (1 - self.criterion.finish_weight) * like_auc
        info = dict(loss=cum_loss, auc=auc)
        return info

    def test(self):
        self.model.eval()
        all_preds = []
        total = self.break_at if self.break_at else len(self.test_dl)
        with torch.no_grad():
            with tqdm.tqdm(total=total) as t:
                t.set_description('[   Test   ]')
                for i, batch in enumerate(self.test_dl):
                    x, _ = batch.cuda()
                    preds = self.model(*x)
                    all_preds.append(preds.cpu().numpy())
                    t.set_postfix(loss='0.000')
                    t.update()
                    if 0 < self.break_at <= i+1:
                        break
        all_preds = np.concatenate(all_preds)
        return all_preds
