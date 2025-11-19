import torch
import numpy as np
def normalize_uncertainty(uncertainty, method='zscore'):
    if method == 'zscore':
        mean = uncertainty.mean()
        std = uncertainty.std()
        return (uncertainty - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = uncertainty.min()
        max_val = uncertainty.max()
        return (uncertainty - min_val) / (max_val - min_val + 1e-8)
    else:
        return uncertainty
def uncertainty_to_confidence(uncertainty):
    return 1.0 / (1.0 + uncertainty)
def create_coverage_mask(pred_mean, pred_std, target, confidence=0.95):
    from scipy import stats
    z_score = stats.norm.ppf((1 + confidence) / 2)
    lower = pred_mean - z_score * pred_std
    upper = pred_mean + z_score * pred_std
    coverage_mask = ((target >= lower) & (target <= upper)).float()
    return coverage_mask
def save_uncertainty_maps(ct, uncertainty, coverage_mask, output_dir, patient_id):
    import SimpleITK as sitk
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ct_np = ct.cpu().numpy().squeeze()
    unc_np = uncertainty.cpu().numpy().squeeze()
    mask_np = coverage_mask.cpu().numpy().squeeze()
    ct_img = sitk.GetImageFromArray(ct_np)
    unc_img = sitk.GetImageFromArray(unc_np)
    mask_img = sitk.GetImageFromArray(mask_np)
    sitk.WriteImage(ct_img, str(output_dir / f'{patient_id}_ct.nii.gz'))
    sitk.WriteImage(unc_img, str(output_dir / f'{patient_id}_uncertainty.nii.gz'))
    sitk.WriteImage(mask_img, str(output_dir / f'{patient_id}_coverage.nii.gz'))
