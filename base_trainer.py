import torch
import pathlib
import numpy as np


class BaseTrainer:

    def __init__(self, model, optimizer, criterion, train_dl, save_path='./models'):
        self.save_path = pathlib.Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl = train_dl
        self.current_iter = iter(self.train_dl)
        self.epoch = 0

    def save(self, name, with_opt=True):
        path = self.save_path / f'{name}.pth'
        state = {'model': self.model.state_dict()}
        if with_opt:
            state['opt'] = self.optimizer.state_dict()
        torch.save(state, path)

    def load(self, name, strict=True, with_opt=True):
        state = torch.load(self.save_path / f'{name}.pth',
                           map_location=next(self.model.parameters()).device)
        self.model.load_state_dict(state['model'], strict=strict)
        opt_state = state.get('opt', None)
        if with_opt and opt_state is not None:
            self.optimizer.load_state_dict(opt_state)

    def set_lr(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr

    def forward_on_batch(self, batch):
        raise NotImplementedError

    def train_step(self):
        try:
            batch = next(self.current_iter)
        except StopIteration:
            self.current_iter = iter(self.train_dl)
            self.epoch += 1
            batch = next(self.current_iter)
        self.optimizer.zero_grad()
        loss = self.forward_on_batch(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()


class LRFinder:

    def __init__(self, trainer, start_lr=1e-7, end_lr=10, num_it=100, stop_div=True):
        self.trainer = trainer
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.num_it = num_it
        self.stop_div = stop_div

    def run(self):
        self.trainer.save('tmp')
        best_loss = np.inf
        # find
        lrs = [self.start_lr * (self.end_lr / self.start_lr) ** ((i + 1) / self.num_it)
               for i in range(self.num_it)]
        losses = []
        raw_iter, raw_epoch = self.trainer.current_iter, self.trainer.epoch
        self.trainer.current_iter = iter(self.trainer.train_dl)
        for lr in lrs:
            self.trainer.set_lr(lr)
            loss = self.trainer.train_step()
            best_loss = min(loss, best_loss)
            if self.stop_div and (loss > 4 * best_loss or np.isnan(loss)):
                break
            losses.append(loss)
        self.trainer.load('tmp')
        self.trainer.current_iter, self.trainer.epoch = raw_iter, raw_epoch
        return lrs[:len(losses)], losses
