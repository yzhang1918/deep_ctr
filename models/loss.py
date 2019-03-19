import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=.25, gamma=2., reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        bce = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise NotImplementedError
        return focal_loss


class WeightedFocalLoss(nn.Module):

    def __init__(self, finish_weight, alpha=.25, gamma=2.):
        super().__init__()
        self.finish_weight = finish_weight
        self.focal_loss = FocalLoss(alpha, gamma)

    def forward(self, preds, targets):
        l1 = self.focal_loss(preds[:, 0], targets[:, 0])
        l2 = self.focal_loss(preds[:, 1], targets[:, 1])
        l = self.finish_weight * l1 + (1 - self.finish_weight) * l2
        return l


class WeightedBCELoss(nn.Module):

    def __init__(self, finish_weight=.5):
        super().__init__()
        self.finish_weight = finish_weight

    def forward(self, preds, targets):
        l1 = F.binary_cross_entropy_with_logits(preds[:, 0], targets[:, 0])
        l2 = F.binary_cross_entropy_with_logits(preds[:, 1], targets[:, 1])
        l = self.finish_weight * l1 + (1 - self.finish_weight) * l2
        return l
