import torch
import torch.nn as nn
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        return torch.mean(loss)
class HUWeightedCharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3, bone_weight=2.0, soft_tissue_weight=1.5, 
                 air_weight=1.0):
        super().__init__()
        self.epsilon = epsilon
        self.bone_weight = bone_weight
        self.soft_tissue_weight = soft_tissue_weight
        self.air_weight = air_weight
    def get_tissue_weights(self, ct_target):
        hu_approx = ct_target * 1500
        weights = torch.ones_like(ct_target)
        bone_mask = hu_approx > 200
        weights[bone_mask] = self.bone_weight
        soft_tissue_mask = (hu_approx > -100) & (hu_approx <= 200)
        weights[soft_tissue_mask] = self.soft_tissue_weight
        air_mask = hu_approx <= -100
        weights[air_mask] = self.air_weight
        return weights
    def forward(self, pred, target, tissue_weights=None):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.epsilon * self.epsilon)
        if tissue_weights is None:
            tissue_weights = self.get_tissue_weights(target)
        weighted_loss = loss * tissue_weights
        return torch.mean(weighted_loss)
class AdaptiveCharbonnierLoss(nn.Module):
    def __init__(self, base_epsilon=1e-3, adaptive=True):
        super().__init__()
        self.base_epsilon = base_epsilon
        self.adaptive = adaptive
    def forward(self, pred, target, mask=None):
        diff = pred - target
        if self.adaptive:
            local_var = torch.var(target, dim=[2, 3, 4], keepdim=True)
            epsilon = self.base_epsilon + 0.1 * local_var
        else:
            epsilon = self.base_epsilon
        loss = torch.sqrt(diff * diff + epsilon * epsilon)
        if mask is not None:
            loss = loss * mask
            return torch.sum(loss) / (torch.sum(mask) + 1e-8)
        return torch.mean(loss)
if __name__ == "__main__":
    pred = torch.randn(2, 1, 32, 32, 32)
    target = torch.randn(2, 1, 32, 32, 32)
    print("Testing Charbonnier loss...")
    loss_fn = CharbonnierLoss()
    loss = loss_fn(pred, target)
    print(f"Loss: {loss.item():.4f}")
    print("\nTesting HU-weighted Charbonnier loss...")
    weighted_loss_fn = HUWeightedCharbonnierLoss()
    weighted_loss = weighted_loss_fn(pred, target)
    print(f"Weighted loss: {weighted_loss.item():.4f}")
    print("\nTesting adaptive Charbonnier loss...")
    adaptive_loss_fn = AdaptiveCharbonnierLoss()
    adaptive_loss = adaptive_loss_fn(pred, target)
    print(f"Adaptive loss: {adaptive_loss.item():.4f}")
    l1_loss = torch.nn.functional.l1_loss(pred, target)
    l2_loss = torch.nn.functional.mse_loss(pred, target)
    print(f"\nComparison:")
    print(f"L1 loss: {l1_loss.item():.4f}")
    print(f"L2 loss: {l2_loss.item():.4f}")
    print(f"Charbonnier loss: {loss.item():.4f}")
