import torch
import torch.nn as nn
import torch.nn.functional as F
class PerceptualLoss3D(nn.Module):
    def __init__(self, feature_layers=[2, 4, 6]):
        super().__init__()
        self.feature_layers = feature_layers
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv3d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv3d(64, 64, 3, padding=1), nn.ReLU(),
        )
        for param in self.features.parameters():
            param.requires_grad = False
    def forward(self, pred, target):
        pred_features = []
        target_features = []
        x_pred = pred
        x_target = target
        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            if i in self.feature_layers:
                pred_features.append(x_pred)
                target_features.append(x_target)
        loss = 0
        for pf, tf in zip(pred_features, target_features):
            loss += F.l1_loss(pf, tf)
        return loss / len(pred_features)
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = PerceptualLoss3D()
    def forward(self, pred, target):
        return self.loss_fn(pred, target)
