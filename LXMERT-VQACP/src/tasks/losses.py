import torch
from torch import nn
from torch.nn import functional as F

import src.config as config


def cross_entropy_loss(logits, labels, **kwargs):
    """ Modified cross entropy loss. """
    nll = F.log_softmax(logits, dim=-1)
    loss = -nll * labels / 10
    if 'miu' in kwargs:
        loss = loss * smooth(kwargs['miu'], kwargs['mask'])
    return loss.sum(dim=-1).mean()


def smooth(miu, mask):
    miu_valid = miu * mask
    miu_invalid = miu * (1.0 - mask) # most 1.0
    return miu_invalid + torch.clamp(F.softplus(miu_valid), max=100.0)


class Plain(nn.Module):
    def forward(self, logits, labels, **kwargs):
        if config.loss_type == 'ce':
            loss = cross_entropy_loss(logits, labels, **kwargs)
        else:
            if 'miu' in kwargs:
                loss = F.binary_cross_entropy_with_logits(logits, labels,
                            pos_weight=smooth(kwargs['miu'], kwargs['mask']))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss *= labels.size(1)
        # print(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, epsilon=1e-9, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, logits, labels, **kwargs):
        """
        logits: tensor of shape (N, num_answer)
        label: tensor of shape (N, num_answer)
        """
        logits = F.softmax(logits, dim=-1)
        ce_loss = - (labels * torch.log(logits)).sum(dim=-1)

        pt = torch.exp(-ce_loss)

        loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
