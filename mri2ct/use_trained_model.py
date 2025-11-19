#!/usr/bin/env python3
import torch
import torch.nn as nn
import numpy as np
import SimpleITK as sitk
from pathlib import Path
import argparse

class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16):
        super().__init__()
        features = init_features
        self.enc1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = self._block(features * 4, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.dec3 = self._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.dec2 = self._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.dec1 = self._block(features * 2, features)
        self.out = nn.Conv3d(features, out_channels, 1)
    
    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, 3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, 3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        return self.out(dec1)

def load_model(checkpoint_path, device='cpu'):
    print(f"Loading model from {checkpoint_path}...")
    
    model = SimpleUNet3D(in_channels=1, out_channels=1, init_features=16)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    if 'epoch' in checkpoint:
        print(f"   Trained for {checkpoint['epoch']} epochs")
    if 'val_loss' in checkpoint:
        print(f"   Validation loss: {checkpoint['val_loss']:.6f}")
    
    return model

def predict_ct(model, mri_path, output_path, device='cpu'):
    print(f"\nRunning inference...")
    print(f"   Input MRI: {mri_path}")
    print(f"   Output CT: {output_path}")
    
    print("   Loading MRI...")
    mri_image = sitk.ReadImage(str(mri_path))
    mri = sitk.GetArrayFromImage(mri_image)
    original_shape = mri.shape
    original_spacing = mri_image.GetSpacing()
    original_origin = mri_image.GetOrigin()
    original_direction = mri_image.GetDirection()
    
    from scipy.ndimage import zoom
    if mri.shape != (64, 64, 64):
        print(f"   Resizing from {mri.shape} to (64, 64, 64)...")
        scale = [64/s for s in mri.shape]
        mri = zoom(mri, scale, order=1)
    
    print("   Normalizing...")
    mri = (mri - mri.mean()) / (mri.std() + 1e-8)
    
    mri_tensor = torch.FloatTensor(mri).unsqueeze(0).unsqueeze(0).to(device)
    
    print("   Generating synthetic CT...")
    with torch.no_grad():
        ct_pred = model(mri_tensor)
    
    ct_pred = ct_pred.squeeze().cpu().numpy()
    
    ct_pred = ct_pred * 1000.0
    
    if original_shape != (64, 64, 64):
        print(f"   Resizing back to {original_shape}...")
        scale = [o/64 for o in original_shape]
        ct_pred = zoom(ct_pred, scale, order=1)
    
    print("   Saving synthetic CT...")
    ct_image = sitk.GetImageFromArray(ct_pred)
    ct_image.SetSpacing(original_spacing)
    ct_image.SetOrigin(original_origin)
    ct_image.SetDirection(original_direction)
    sitk.WriteImage(ct_image, str(output_path))
    
    print(f"Synthetic CT saved to {output_path}")
    print(f"   Shape: {ct_pred.shape}")
    print(f"   HU range: [{ct_pred.min():.1f}, {ct_pred.max():.1f}]")
    
    return ct_pred

def main():
    parser = argparse.ArgumentParser(description='Use trained model for MRI to CT inference')
    parser.add_argument('--model', type=str, 
                       default='checkpoints/full_training/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                       help='Input MRI file (.nii.gz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output synthetic CT file (.nii.gz)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("="*80)
    print("  MRI to CT Inference - Using Your Trained Model")
    print("="*80)
    print()
    
    model = load_model(args.model, args.device)
    
    ct_pred = predict_ct(model, args.input, args.output, args.device)
    
    print()
    print("="*80)
    print("Inference complete!")
    print("="*80)

if __name__ == '__main__':
    main()
