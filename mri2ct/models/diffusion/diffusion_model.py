import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model,
                 timesteps=1000,
                 beta_schedule='linear',
                 beta_start=0.0001,
                 beta_end=0.02,
                 prediction_type='noise'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.prediction_type = prediction_type
        if beta_schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, timesteps)
        elif beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - alphas_cumprod))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    def _extract(self, a, t, x_shape):
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    def predict_start_from_noise(self, x_t, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance
    def p_mean_variance(self, x, t, context):
        model_output = self.model(x, t, context)
        if self.prediction_type == 'noise':
            x_start = self.predict_start_from_noise(x, t, model_output)
        else:
            x_start = model_output
        x_start = torch.clamp(x_start, -1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start, x, t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start
    @torch.no_grad()
    def p_sample(self, x, t, context):
        model_mean, _, model_log_variance, _ = self.p_mean_variance(x, t, context)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
    @torch.no_grad()
    def p_sample_loop(self, shape, context, progress=True):
        device = context.device
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        timesteps = range(self.timesteps - 1, -1, -1)
        if progress:
            timesteps = tqdm(timesteps, desc='Sampling')
        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_batch, context)
        return x
    def forward(self, x_start, context):
        batch_size = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.model(x_noisy, t, context)
        if self.prediction_type == 'noise':
            target = noise
        else:
            target = x_start
        loss = F.mse_loss(predicted, target)
        return {
            'loss': loss,
            'predicted': predicted,
            'target': target
        }
    @torch.no_grad()
    def sample(self, mri, num_samples=1, progress=True):
        batch_size = mri.shape[0]
        if num_samples > 1:
            context = mri.repeat_interleave(num_samples, dim=0)
        else:
            context = mri
        shape = (batch_size * num_samples, *mri.shape[1:])
        samples = self.p_sample_loop(shape, context, progress=progress)
        return samples
if __name__ == "__main__":
    from unet_transformer import UNetTransformer3D
    unet = UNetTransformer3D(
        in_channels=1,
        out_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 4),
        num_heads=4
    )
    diffusion = GaussianDiffusion(
        model=unet,
        timesteps=1000,
        beta_schedule='linear'
    )
    x_start = torch.randn(2, 1, 32, 32, 32)
    context = torch.randn(2, 1, 32, 32, 32)
    output = diffusion(x_start, context)
    print(f"Training loss: {output['loss'].item():.4f}")
    samples = diffusion.sample(context[:1], num_samples=1, progress=False)
    print(f"Generated samples shape: {samples.shape}")
