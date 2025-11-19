import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
def enable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout3d):
            module.train()
def disable_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout3d):
            module.eval()
class MCDropout:
    def __init__(self, model, num_samples=20, dropout_rate=0.1):
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
    @torch.no_grad()
    def predict_with_uncertainty(self, x, context=None, progress=True):
        self.model.eval()
        enable_dropout(self.model)
        samples = []
        iterator = range(self.num_samples)
        if progress:
            iterator = tqdm(iterator, desc='MC Dropout Sampling')
        for _ in iterator:
            if context is not None:
                pred = self.model(x, context)
            else:
                pred = self.model(x)
            samples.append(pred.cpu())
        disable_dropout(self.model)
        samples = torch.stack(samples, dim=0)
        mean = torch.mean(samples, dim=0)
        variance = torch.var(samples, dim=0)
        return mean, variance, samples
class MCDropoutDiffusion:
    def __init__(self, diffusion_model, num_samples=10):
        self.diffusion_model = diffusion_model
        self.num_samples = num_samples
    @torch.no_grad()
    def sample_with_uncertainty(self, mri, progress=True):
        samples = []
        iterator = range(self.num_samples)
        if progress:
            iterator = tqdm(iterator, desc='MC Sampling')
        for _ in iterator:
            sample = self.diffusion_model.sample(mri, num_samples=1, progress=False)
            samples.append(sample.cpu())
        samples = torch.stack(samples, dim=0)
        mean_ct = torch.mean(samples, dim=0)
        uncertainty = torch.var(samples, dim=0)
        return mean_ct, uncertainty, samples
if __name__ == "__main__":
    from torch import nn
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 32, 3, padding=1),
                nn.ReLU(),
                nn.Dropout3d(0.2),
                nn.Conv3d(32, 1, 3, padding=1)
            )
        def forward(self, x):
            return self.net(x)
    model = SimpleModel()
    mc_dropout = MCDropout(model, num_samples=10)
    x = torch.randn(1, 1, 32, 32, 32)
    mean, var, samples = mc_dropout.predict_with_uncertainty(x, progress=False)
    print(f"Mean shape: {mean.shape}")
    print(f"Variance shape: {var.shape}")
    print(f"Samples shape: {samples.shape}")
    print(f"Mean uncertainty: {var.mean().item():.6f}")
