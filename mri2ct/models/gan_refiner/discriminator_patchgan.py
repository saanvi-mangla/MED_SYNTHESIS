import torch
import torch.nn as nn
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, use_norm=True):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, 4, stride=stride, padding=1),
        ]
        if use_norm:
            layers.append(nn.InstanceNorm3d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, num_layers=3):
        super().__init__()
        layers = []
        layers.append(ConvBlock3D(in_channels, base_channels, stride=2, use_norm=False))
        channels = base_channels
        for i in range(1, num_layers):
            out_channels = min(channels * 2, 512)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(ConvBlock3D(channels, out_channels, stride=stride, use_norm=True))
            channels = out_channels
        layers.append(nn.Conv3d(channels, 1, 4, padding=1))
        self.model = nn.Sequential(*layers)
    def forward(self, ct, mri):
        x = torch.cat([ct, mri], dim=1)
        return self.model(x)
class MultiScalePatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, num_scales=3, num_layers=3):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            PatchGANDiscriminator(in_channels, base_channels, num_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)
    def forward(self, ct, mri):
        outputs = []
        ct_input = ct
        mri_input = mri
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(ct_input, mri_input))
            if i < self.num_scales - 1:
                ct_input = self.downsample(ct_input)
                mri_input = self.downsample(mri_input)
        return outputs
class SpectralNormConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv3d(in_channels, out_channels, 4, stride=stride, padding=1)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        return self.act(self.conv(x))
class SpectralPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, num_layers=4):
        super().__init__()
        layers = []
        channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * min(2 ** i, 8)
            stride = 2 if i < num_layers - 1 else 1
            layers.append(SpectralNormConvBlock3D(channels, out_channels, stride=stride))
            channels = out_channels
        layers.append(
            nn.utils.spectral_norm(nn.Conv3d(channels, 1, 4, padding=1))
        )
        self.model = nn.Sequential(*layers)
    def forward(self, ct, mri):
        x = torch.cat([ct, mri], dim=1)
        return self.model(x)
class MultiScaleSpectralDiscriminator(nn.Module):
    def __init__(self, in_channels=2, base_channels=64, num_scales=2, num_layers=4):
        super().__init__()
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList([
            SpectralPatchGANDiscriminator(in_channels, base_channels, num_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool3d(kernel_size=2, stride=2)
    def forward(self, ct, mri):
        outputs = []
        ct_input = ct
        mri_input = mri
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(ct_input, mri_input))
            if i < self.num_scales - 1:
                ct_input = self.downsample(ct_input)
                mri_input = self.downsample(mri_input)
        return outputs
    def get_features(self, ct, mri):
        features = []
        ct_input = ct
        mri_input = mri
        x = torch.cat([ct_input, mri_input], dim=1)
        for i, disc in enumerate(self.discriminators):
            scale_features = []
            for layer in disc.model:
                x = layer(x)
                scale_features.append(x)
            features.append(scale_features)
            if i < self.num_scales - 1:
                ct_input = self.downsample(ct_input)
                mri_input = self.downsample(mri_input)
                x = torch.cat([ct_input, mri_input], dim=1)
        return features
if __name__ == "__main__":
    disc = PatchGANDiscriminator(in_channels=2, base_channels=32, num_layers=3)
    ct = torch.randn(2, 1, 64, 64, 64)
    mri = torch.randn(2, 1, 64, 64, 64)
    out = disc(ct, mri)
    print(f"Single-scale output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in disc.parameters()) / 1e6:.2f}M")
    print("\nTesting multi-scale discriminator...")
    ms_disc = MultiScalePatchGANDiscriminator(
        in_channels=2, base_channels=32, num_scales=3, num_layers=3
    )
    outputs = ms_disc(ct, mri)
    print(f"Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Scale {i} output shape: {out.shape}")
    print("\nTesting spectral discriminator...")
    spec_disc = MultiScaleSpectralDiscriminator(
        in_channels=2, base_channels=32, num_scales=2, num_layers=4
    )
    outputs = spec_disc(ct, mri)
    print(f"Number of scales: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"Scale {i} output shape: {out.shape}")
