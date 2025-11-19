import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.SiLU()
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels)
            )
        else:
            self.time_mlp = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    def forward(self, x, time_emb=None):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb)
            h = h + time_emb[:, :, None, None, None]
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return h + self.shortcut(x)
class AttentionBlock3D(nn.Module):
    def __init__(self, channels, num_heads=8, groups=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    def forward(self, x):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = rearrange(q, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        k = rearrange(k, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        v = rearrange(v, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhqc,bhkc->bhqk', q, k) * scale, dim=-1)
        out = torch.einsum('bhqk,bhkc->bhqc', attn, v)
        out = rearrange(out, 'b h (d x y) c -> b (h c) d x y', d=D, x=H, y=W)
        out = self.proj(out)
        return x + out
class CrossAttentionBlock3D(nn.Module):
    def __init__(self, channels, context_dim, num_heads=8, groups=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(groups, channels)
        self.norm_context = nn.GroupNorm(groups, context_dim)
        self.q = nn.Conv3d(channels, channels, 1)
        self.kv = nn.Conv3d(context_dim, channels * 2, 1)
        self.proj = nn.Conv3d(channels, channels, 1)
    def forward(self, x, context):
        B, C, D, H, W = x.shape
        h = self.norm(x)
        context = self.norm_context(context)
        q = self.q(h)
        kv = self.kv(context)
        k, v = torch.chunk(kv, 2, dim=1)
        q = rearrange(q, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        k = rearrange(k, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        v = rearrange(v, 'b (h c) d x y -> b h (d x y) c', h=self.num_heads)
        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.einsum('bhqc,bhkc->bhqk', q, k) * scale, dim=-1)
        out = torch.einsum('bhqk,bhkc->bhqc', attn, v)
        out = rearrange(out, 'b h (d x y) c -> b (h c) d x y', d=D, x=H, y=W)
        out = self.proj(out)
        return x + out
class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False,
                 has_cross_attention=False, context_dim=None, num_heads=8):
        super().__init__()
        self.res1 = ResidualBlock3D(in_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock3D(out_channels, out_channels, time_emb_dim)
        if has_attention:
            self.attn = AttentionBlock3D(out_channels, num_heads)
        else:
            self.attn = None
        if has_cross_attention and context_dim is not None:
            self.cross_attn = CrossAttentionBlock3D(out_channels, context_dim, num_heads)
        else:
            self.cross_attn = None
        self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)
    def forward(self, x, time_emb, context=None):
        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        if self.attn is not None:
            h = self.attn(h)
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)
        skip = h
        h = self.downsample(h)
        return h, skip
class UpBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, has_attention=False,
                 has_cross_attention=False, context_dim=None, num_heads=8):
        super().__init__()
        self.res1 = ResidualBlock3D(in_channels + out_channels, out_channels, time_emb_dim)
        self.res2 = ResidualBlock3D(out_channels, out_channels, time_emb_dim)
        if has_attention:
            self.attn = AttentionBlock3D(out_channels, num_heads)
        else:
            self.attn = None
        if has_cross_attention and context_dim is not None:
            self.cross_attn = CrossAttentionBlock3D(out_channels, context_dim, num_heads)
        else:
            self.cross_attn = None
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 4, stride=2, padding=1)
    def forward(self, x, skip, time_emb, context=None):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        h = self.res1(x, time_emb)
        h = self.res2(h, time_emb)
        if self.attn is not None:
            h = self.attn(h)
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)
        return h
class UNetTransformer3D(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 model_channels=64,
                 channel_mult=(1, 2, 4, 8),
                 num_heads=8,
                 attention_levels=(2, 3),
                 cross_attention_levels=(1, 2, 3),
                 context_channels=1,
                 time_emb_dim=256,
                 dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.time_emb_dim = time_emb_dim
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.context_encoder = nn.Sequential(
            nn.Conv3d(context_channels, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, model_channels, 3, padding=1),
        )
        self.conv_in = nn.Conv3d(in_channels, model_channels, 3, padding=1)
        self.downs = nn.ModuleList([])
        channels = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            self.downs.append(
                DownBlock3D(
                    ch, out_ch, time_emb_dim,
                    has_attention=level in attention_levels,
                    has_cross_attention=level in cross_attention_levels,
                    context_dim=model_channels,
                    num_heads=num_heads
                )
            )
            ch = out_ch
            channels.append(ch)
        self.mid_block1 = ResidualBlock3D(ch, ch, time_emb_dim)
        self.mid_attn = AttentionBlock3D(ch, num_heads)
        self.mid_cross_attn = CrossAttentionBlock3D(ch, model_channels, num_heads)
        self.mid_block2 = ResidualBlock3D(ch, ch, time_emb_dim)
        self.ups = nn.ModuleList([])
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            self.ups.append(
                UpBlock3D(
                    ch, out_ch, time_emb_dim,
                    has_attention=level in attention_levels,
                    has_cross_attention=level in cross_attention_levels,
                    context_dim=model_channels,
                    num_heads=num_heads
                )
            )
            ch = out_ch
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            nn.Conv3d(model_channels, out_channels, 3, padding=1)
        )
    def forward(self, x, timesteps, context):
        t_emb = self.time_embed(timesteps)
        context_enc = self.context_encoder(context)
        h = self.conv_in(x)
        skips = []
        for down in self.downs:
            h, skip = down(h, t_emb, context_enc)
            skips.append(skip)
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_cross_attn(h, context_enc)
        h = self.mid_block2(h, t_emb)
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, t_emb, context_enc)
        out = self.conv_out(h)
        return out
if __name__ == "__main__":
    model = UNetTransformer3D(
        in_channels=1,
        out_channels=1,
        model_channels=32,
        channel_mult=(1, 2, 4, 8),
        num_heads=4
    )
    x = torch.randn(1, 1, 64, 64, 64)
    t = torch.randint(0, 1000, (1,))
    context = torch.randn(1, 1, 64, 64, 64)
    out = model(x, t, context)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
