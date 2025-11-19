import torch
import numpy as np
def compute_coverage(pred_mean, pred_std, target, confidence=0.95):
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower_bound = pred_mean - z_score * pred_std
    upper_bound = pred_mean + z_score * pred_std
    within_interval = (target >= lower_bound) & (target <= upper_bound)
    coverage = torch.mean(within_interval.float()).item()
    return coverage
def compute_interval_width(pred_std, confidence=0.95):
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    width = 2 * z_score * pred_std
    return torch.mean(width).item()
def compute_sharpness(pred_std):
    return torch.mean(pred_std).item()
def compute_uncertainty_metrics(pred_mean, pred_std, target, confidence_levels=[0.68, 0.95, 0.99]):
    metrics = {}
    for conf in confidence_levels:
        coverage = compute_coverage(pred_mean, pred_std, target, conf)
        width = compute_interval_width(pred_std, conf)
        metrics[f'coverage_{int(conf*100)}'] = coverage
        metrics[f'width_{int(conf*100)}'] = width
    metrics['sharpness'] = compute_sharpness(pred_std)
    metrics['rmse'] = torch.sqrt(torch.mean((pred_mean - target) ** 2)).item()
    return metrics
