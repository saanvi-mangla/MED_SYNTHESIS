import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
class DDIMSampler:
    def __init__(self, diffusion_model, ddim_steps=50, eta=0.0):
        self.model = diffusion_model.model
        self.diffusion = diffusion_model
        self.ddim_steps = ddim_steps
        self.eta = eta
        self.ddim_timesteps = self._make_ddim_timesteps(
            ddim_steps, diffusion_model.timesteps
        )
        self.ddim_alphas = diffusion_model.alphas_cumprod[self.ddim_timesteps]
        self.ddim_alphas_prev = torch.cat([
            diffusion_model.alphas_cumprod[0:1],
            diffusion_model.alphas_cumprod[self.ddim_timesteps[:-1]]
        ])
        self.ddim_sqrt_one_minus_alphas = torch.sqrt(1.0 - self.ddim_alphas)
        self.ddim_sigmas = (
            eta * torch.sqrt(
                (1 - self.ddim_alphas_prev) / (1 - self.ddim_alphas) *
                (1 - self.ddim_alphas / self.ddim_alphas_prev)
            )
        )
    def _make_ddim_timesteps(self, ddim_steps, ddpm_steps):
        c = ddpm_steps // ddim_steps
        ddim_timesteps = np.asarray(list(range(0, ddpm_steps, c))) + 1
        ddim_timesteps[-1] = ddpm_steps - 1
        return torch.from_numpy(ddim_timesteps).long()
    @torch.no_grad()
    def ddim_sample_step(self, x, t, t_next, context):
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_output = self.model(x, t_batch, context)
        alpha_t = self.diffusion.alphas_cumprod[t]
        if t_next < 0:
            alpha_t_next = torch.tensor(1.0, device=x.device)
        else:
            alpha_t_next = self.diffusion.alphas_cumprod[t_next]
        if self.diffusion.prediction_type == 'noise':
            pred_x0 = (
                x - torch.sqrt(1 - alpha_t) * model_output
            ) / torch.sqrt(alpha_t)
        else:
            pred_x0 = model_output
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        dir_xt = torch.sqrt(1 - alpha_t_next - self.eta ** 2) * model_output
        noise = torch.randn_like(x) if self.eta > 0 and t_next >= 0 else 0
        x_next = (
            torch.sqrt(alpha_t_next) * pred_x0 +
            dir_xt +
            self.eta * noise
        )
        return x_next, pred_x0
    @torch.no_grad()
    def sample(self, mri, num_samples=1, progress=True, return_intermediates=False):
        batch_size = mri.shape[0]
        device = mri.device
        if num_samples > 1:
            context = mri.repeat_interleave(num_samples, dim=0)
        else:
            context = mri
        shape = (batch_size * num_samples, *mri.shape[1:])
        x = torch.randn(shape, device=device)
        intermediates = []
        timesteps = self.ddim_timesteps.flip(0)
        iterator = enumerate(timesteps)
        if progress:
            iterator = tqdm(list(iterator), desc='DDIM Sampling')
        for i, t in iterator:
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x, pred_x0 = self.ddim_sample_step(x, t, t_next, context)
            if return_intermediates:
                intermediates.append(x.cpu())
        if return_intermediates:
            return x, intermediates
        return x
    @torch.no_grad()
    def sample_progressive(self, mri, num_samples=1, save_every=10):
        batch_size = mri.shape[0]
        device = mri.device
        if num_samples > 1:
            context = mri.repeat_interleave(num_samples, dim=0)
        else:
            context = mri
        shape = (batch_size * num_samples, *mri.shape[1:])
        x = torch.randn(shape, device=device)
        results = {'init': x.cpu()}
        timesteps = self.ddim_timesteps.flip(0)
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), desc='Progressive'):
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x, pred_x0 = self.ddim_sample_step(x, t, t_next, context)
            if i % save_every == 0 or i == len(timesteps) - 1:
                results[f'step_{i}'] = x.cpu()
                results[f'pred_x0_step_{i}'] = pred_x0.cpu()
        results['final'] = x.cpu()
        return results
    @torch.no_grad()
    def encode(self, x0, context, num_steps=None):
        if num_steps is None:
            num_steps = self.ddim_steps
        timesteps = self.ddim_timesteps[:num_steps]
        x = x0
        for i, t in enumerate(tqdm(timesteps, desc='DDIM Encoding')):
            t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
            noise_pred = self.model(x, t_batch, context)
            if i + 1 < len(timesteps):
                t_next = timesteps[i + 1]
                alpha_t = self.diffusion.alphas_cumprod[t]
                alpha_t_next = self.diffusion.alphas_cumprod[t_next]
                x = (
                    torch.sqrt(alpha_t_next / alpha_t) * x +
                    (torch.sqrt(1 - alpha_t_next) - torch.sqrt((1 - alpha_t) * alpha_t_next / alpha_t)) * noise_pred
                )
        return x
if __name__ == "__main__":
    from diffusion_model import GaussianDiffusion
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
    ddim = DDIMSampler(diffusion, ddim_steps=50, eta=0.0)
    mri = torch.randn(1, 1, 32, 32, 32)
    print("Sampling with DDIM (50 steps)...")
    samples = ddim.sample(mri, num_samples=1, progress=True)
    print(f"Generated samples shape: {samples.shape}")
    print("\nProgressive sampling...")
    progressive = ddim.sample_progressive(mri, save_every=10)
    print(f"Saved {len(progressive)} intermediate results")
