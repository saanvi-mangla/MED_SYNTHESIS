import torch
import torch.nn as nn
import torch.nn.functional as F
def hinge_loss_dis(real_pred, fake_pred):
    if isinstance(real_pred, list):
        loss = 0
        for real_p, fake_p in zip(real_pred, fake_pred):
            loss += torch.mean(F.relu(1.0 - real_p))
            loss += torch.mean(F.relu(1.0 + fake_p))
        return loss / len(real_pred)
    else:
        loss_real = torch.mean(F.relu(1.0 - real_pred))
        loss_fake = torch.mean(F.relu(1.0 + fake_pred))
        return loss_real + loss_fake
def hinge_loss_gen(fake_pred):
    if isinstance(fake_pred, list):
        loss = 0
        for fake_p in fake_pred:
            loss += -torch.mean(fake_p)
        return loss / len(fake_pred)
    else:
        return -torch.mean(fake_pred)
def vanilla_gan_loss_dis(real_pred, fake_pred):
    if isinstance(real_pred, list):
        loss = 0
        for real_p, fake_p in zip(real_pred, fake_pred):
            loss += F.binary_cross_entropy_with_logits(
                real_p, torch.ones_like(real_p)
            )
            loss += F.binary_cross_entropy_with_logits(
                fake_p, torch.zeros_like(fake_p)
            )
        return loss / len(real_pred)
    else:
        loss_real = F.binary_cross_entropy_with_logits(
            real_pred, torch.ones_like(real_pred)
        )
        loss_fake = F.binary_cross_entropy_with_logits(
            fake_pred, torch.zeros_like(fake_pred)
        )
        return loss_real + loss_fake
def vanilla_gan_loss_gen(fake_pred):
    if isinstance(fake_pred, list):
        loss = 0
        for fake_p in fake_pred:
            loss += F.binary_cross_entropy_with_logits(
                fake_p, torch.ones_like(fake_p)
            )
        return loss / len(fake_pred)
    else:
        return F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )
def lsgan_loss_dis(real_pred, fake_pred):
    if isinstance(real_pred, list):
        loss = 0
        for real_p, fake_p in zip(real_pred, fake_pred):
            loss += torch.mean((real_p - 1) ** 2)
            loss += torch.mean(fake_p ** 2)
        return loss / len(real_pred)
    else:
        loss_real = torch.mean((real_pred - 1) ** 2)
        loss_fake = torch.mean(fake_pred ** 2)
        return loss_real + loss_fake
def lsgan_loss_gen(fake_pred):
    if isinstance(fake_pred, list):
        loss = 0
        for fake_p in fake_pred:
            loss += torch.mean((fake_p - 1) ** 2)
        return loss / len(fake_pred)
    else:
        return torch.mean((fake_pred - 1) ** 2)
class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
    def forward(self, real_features, fake_features):
        loss = 0
        num_features = 0
        if isinstance(real_features[0], list):
            for real_scale, fake_scale in zip(real_features, fake_features):
                for real_feat, fake_feat in zip(real_scale, fake_scale):
                    loss += self.l1_loss(fake_feat, real_feat.detach())
                    num_features += 1
        else:
            for real_feat, fake_feat in zip(real_features, fake_features):
                loss += self.l1_loss(fake_feat, real_feat.detach())
                num_features += 1
        return loss / num_features if num_features > 0 else loss
class GANLoss:
    def __init__(self, loss_type='hinge'):
        self.loss_type = loss_type
        if loss_type == 'hinge':
            self.dis_loss = hinge_loss_dis
            self.gen_loss = hinge_loss_gen
        elif loss_type == 'vanilla':
            self.dis_loss = vanilla_gan_loss_dis
            self.gen_loss = vanilla_gan_loss_gen
        elif loss_type == 'lsgan':
            self.dis_loss = lsgan_loss_dis
            self.gen_loss = lsgan_loss_gen
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    def discriminator_loss(self, real_pred, fake_pred):
        return self.dis_loss(real_pred, fake_pred)
    def generator_loss(self, fake_pred):
        return self.gen_loss(fake_pred)
if __name__ == "__main__":
    real_pred = torch.randn(4, 1, 8, 8, 8)
    fake_pred = torch.randn(4, 1, 8, 8, 8)
    print("Testing single-scale losses...")
    print(f"Hinge dis loss: {hinge_loss_dis(real_pred, fake_pred).item():.4f}")
    print(f"Hinge gen loss: {hinge_loss_gen(fake_pred).item():.4f}")
    print(f"Vanilla dis loss: {vanilla_gan_loss_dis(real_pred, fake_pred).item():.4f}")
    print(f"Vanilla gen loss: {vanilla_gan_loss_gen(fake_pred).item():.4f}")
    print(f"LSGAN dis loss: {lsgan_loss_dis(real_pred, fake_pred).item():.4f}")
    print(f"LSGAN gen loss: {lsgan_loss_gen(fake_pred).item():.4f}")
    print("\nTesting multi-scale losses...")
    real_pred_ms = [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 4, 4, 4)]
    fake_pred_ms = [torch.randn(4, 1, 8, 8, 8), torch.randn(4, 1, 4, 4, 4)]
    print(f"Multi-scale hinge dis loss: {hinge_loss_dis(real_pred_ms, fake_pred_ms).item():.4f}")
    print(f"Multi-scale hinge gen loss: {hinge_loss_gen(fake_pred_ms).item():.4f}")
    print("\nTesting feature matching loss...")
    fm_loss = FeatureMatchingLoss()
    real_feats = [torch.randn(4, 64, 32, 32, 32), torch.randn(4, 128, 16, 16, 16)]
    fake_feats = [torch.randn(4, 64, 32, 32, 32), torch.randn(4, 128, 16, 16, 16)]
    fm = fm_loss(real_feats, fake_feats)
    print(f"Feature matching loss: {fm.item():.4f}")
