import torch
import torch.nn.functional as F
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
def load_segmentation_model(checkpoint_path, device='cuda'):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=10,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
@torch.no_grad()
def evaluate_segmentation_transfer(real_ct, synthetic_ct, segmentation_model, 
                                   ground_truth_seg=None):
    device = real_ct.device
    real_seg = segmentation_model(real_ct)
    synthetic_seg = segmentation_model(synthetic_ct)
    real_seg = torch.argmax(real_seg, dim=1, keepdim=True)
    synthetic_seg = torch.argmax(synthetic_seg, dim=1, keepdim=True)
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean')
    dice = dice_metric(synthetic_seg, real_seg)
    hausdorff = hausdorff_metric(synthetic_seg, real_seg)
    results = {
        'dice': dice.mean().item(),
        'hausdorff': hausdorff.mean().item()
    }
    if ground_truth_seg is not None:
        gt_dice = dice_metric(synthetic_seg, ground_truth_seg)
        results['dice_vs_ground_truth'] = gt_dice.mean().item()
    return results
