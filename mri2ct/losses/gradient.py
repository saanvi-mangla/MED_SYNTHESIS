import torch
import torch.nn as nn
import torch.nn.functional as F
class GradientLoss(nn.Module):
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
    def gradient_3d(self, x):
        grad_x = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        grad_y = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        grad_z = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        return grad_x, grad_y, grad_z
    def forward(self, pred, target):
        pred_grad_x, pred_grad_y, pred_grad_z = self.gradient_3d(pred)
        target_grad_x, target_grad_y, target_grad_z = self.gradient_3d(target)
        if self.loss_type == 'l1':
            loss_x = F.l1_loss(pred_grad_x, target_grad_x)
            loss_y = F.l1_loss(pred_grad_y, target_grad_y)
            loss_z = F.l1_loss(pred_grad_z, target_grad_z)
        else:
            loss_x = F.mse_loss(pred_grad_x, target_grad_x)
            loss_y = F.mse_loss(pred_grad_y, target_grad_y)
            loss_z = F.mse_loss(pred_grad_z, target_grad_z)
        return (loss_x + loss_y + loss_z) / 3.0
class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred_dx = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        pred_dy = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        pred_dz = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        target_dx = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        target_dy = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        target_dz = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        loss = (
            F.l1_loss(pred_dx, target_dx) +
            F.l1_loss(pred_dy, target_dy) +
            F.l1_loss(pred_dz, target_dz)
        ) / 3.0
        return loss
