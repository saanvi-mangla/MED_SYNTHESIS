import torch
import torch.nn as nn
class HUWeightedLoss(nn.Module):
    def __init__(self, bone_weight=2.0, soft_tissue_weight=1.5, air_weight=1.0):
        super().__init__()
        self.bone_weight = bone_weight
        self.soft_tissue_weight = soft_tissue_weight
        self.air_weight = air_weight
    def get_tissue_masks(self, ct):
        hu = ct * 1500.0
        bone = hu > 200
        soft_tissue = (hu > -100) & (hu <= 200)
        air = hu <= -100
        return bone, soft_tissue, air
    def forward(self, pred, target):
        bone, soft_tissue, air = self.get_tissue_masks(target)
        loss = torch.abs(pred - target)
        weighted_loss = (
            loss * (bone.float() * self.bone_weight +
                    soft_tissue.float() * self.soft_tissue_weight +
                    air.float() * self.air_weight)
        )
        return weighted_loss.mean()
