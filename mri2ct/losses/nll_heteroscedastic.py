import torch
import torch.nn as nn
import math
class GaussianNLLLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    def forward(self, pred_mean, pred_log_var, target):
        precision = torch.exp(-pred_log_var)
        loss = 0.5 * (precision * (pred_mean - target) ** 2 + pred_log_var)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
class HeteroscedasticLoss(nn.Module):
    def __init__(self, nll_weight=1.0, reg_weight=0.01):
        super().__init__()
        self.nll_weight = nll_weight
        self.reg_weight = reg_weight
        self.nll_loss = GaussianNLLLoss()
    def forward(self, pred_mean, pred_log_var, target):
        nll = self.nll_loss(pred_mean, pred_log_var, target)
        reg = torch.mean(pred_log_var ** 2)
        return self.nll_weight * nll + self.reg_weight * reg
