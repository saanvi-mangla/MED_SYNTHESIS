import torch
import torch.nn as nn
import torch.nn.functional as F
class ResBlock3D(nn.Module):
    def __init__(self, channels, use_dropout=False):
        super().__init__()
        layers = [
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(inplace=True)
        ]
        if use_dropout:
            layers.append(nn.Dropout3d(0.5))
        layers.extend([
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels)
        ])
        self.block = nn.Sequential(*layers)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(x + self.block(x))
class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.res = ResBlock3D(out_channels, use_dropout)
    def forward(self, x):
        x = self.conv(x)
        x = self.res(x)
        return x
class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout=False):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.res = ResBlock3D(out_channels * 2, use_dropout)
        self.conv = nn.Conv3d(out_channels * 2, out_channels, 1)
    def forward(self, x, skip):
        x = self.up(x)
        x = self.norm(x)
        x = self.relu(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x)
        x = self.conv(x)
        return x
class ResUNetGenerator(nn.Module):
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 base_channels=64,
                 num_downs=4,
                 use_dropout=False):
        super().__init__()
        self.num_downs = num_downs
        self.conv_in = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 7, padding=3),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for i in range(num_downs):
            out_ch = min(channels * 2, 512)
            self.down_blocks.append(
                DownBlock3D(channels, out_ch, use_dropout)
            )
            channels = out_ch
        self.bottleneck = nn.Sequential(
            ResBlock3D(channels, use_dropout),
            ResBlock3D(channels, use_dropout),
            ResBlock3D(channels, use_dropout)
        )
        self.up_blocks = nn.ModuleList()
        for i in range(num_downs):
            out_ch = max(channels // 2, base_channels)
            self.up_blocks.append(
                UpBlock3D(channels, out_ch, use_dropout)
            )
            channels = out_ch
        self.conv_out = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.InstanceNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, out_channels, 7, padding=3),
            nn.Tanh()
        )
    def forward(self, diffusion_ct, mri):
        x = torch.cat([diffusion_ct, mri], dim=1)
        x = self.conv_in(x)
        skips = []
        for down in self.down_blocks:
            skips.append(x)
            x = down(x)
        x = self.bottleneck(x)
        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip)
        out = self.conv_out(x)
        out = out + diffusion_ct
        return out
class ResUNetGeneratorWithUncertainty(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64, 
                 num_downs=4, use_dropout=False):
        super().__init__()
        self.generator = ResUNetGenerator(
            in_channels, out_channels, base_channels, num_downs, use_dropout
        )
        self.uncertainty_head = nn.Sequential(
            nn.Conv3d(base_channels, base_channels // 2, 3, padding=1),
            nn.InstanceNorm3d(base_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels // 2, 1, 3, padding=1),
            nn.Softplus()
        )
        self.intermediate_features = None
        def hook_fn(module, input, output):
            self.intermediate_features = output
        self.generator.conv_out[0].register_forward_hook(hook_fn)
    def forward(self, diffusion_ct, mri):
        refined_ct = self.generator(diffusion_ct, mri)
        if self.intermediate_features is not None:
            log_var = self.uncertainty_head(self.intermediate_features)
        else:
            log_var = torch.zeros_like(refined_ct)
        return refined_ct, log_var
if __name__ == "__main__":
    generator = ResUNetGenerator(
        in_channels=2,
        out_channels=1,
        base_channels=32,
        num_downs=3
    )
    diffusion_ct = torch.randn(1, 1, 64, 64, 64)
    mri = torch.randn(1, 1, 64, 64, 64)
    refined_ct = generator(diffusion_ct, mri)
    print(f"Input shape: {diffusion_ct.shape}")
    print(f"Output shape: {refined_ct.shape}")
    print(f"Parameters: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M")
    print("\nTesting with uncertainty head...")
    generator_unc = ResUNetGeneratorWithUncertainty(
        in_channels=2,
        out_channels=1,
        base_channels=32,
        num_downs=3
    )
    refined_ct, log_var = generator_unc(diffusion_ct, mri)
    print(f"Refined CT shape: {refined_ct.shape}")
    print(f"Log variance shape: {log_var.shape}")
