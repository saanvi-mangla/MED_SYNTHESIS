import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
def mae(pred, target, mask=None):
    diff = torch.abs(pred - target)
    if mask is not None:
        diff = diff * mask
        return torch.sum(diff) / (torch.sum(mask) + 1e-8)
    return torch.mean(diff)
def rmse(pred, target, mask=None):
    diff = (pred - target) ** 2
    if mask is not None:
        diff = diff * mask
        return torch.sqrt(torch.sum(diff) / (torch.sum(mask) + 1e-8))
    return torch.sqrt(torch.mean(diff))
def tissue_specific_mae(pred, target, hu_ranges):
    pred_hu = pred * 1500.0
    target_hu = target * 1500.0
    results = {}
    for name, min_hu, max_hu in hu_ranges:
        mask = (target_hu >= min_hu) & (target_hu < max_hu)
        if mask.sum() > 0:
            tissue_mae = mae(pred, target, mask.float())
            results[name] = tissue_mae.item()
        else:
            results[name] = np.nan
    return results
def compute_psnr(pred, target, data_range=2.0):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    return psnr(target_np, pred_np, data_range=data_range)
def compute_ssim(pred, target, data_range=2.0):
    pred_np = pred.cpu().numpy().squeeze()
    target_np = target.cpu().numpy().squeeze()
    return ssim(target_np, pred_np, data_range=data_range)
def compute_all_metrics(pred, target, mask=None):
    metrics = {
        'mae': mae(pred, target, mask).item(),
        'rmse': rmse(pred, target, mask).item(),
        'psnr': compute_psnr(pred, target),
        'ssim': compute_ssim(pred, target)
    }
    tissue_ranges = [
        ('bone', 200, 2000),
        ('soft_tissue', -100, 200),
        ('air', -1000, -100)
    ]
    tissue_metrics = tissue_specific_mae(pred, target, tissue_ranges)
    metrics.update({f'mae_{k}': v for k, v in tissue_metrics.items()})
    return metrics
