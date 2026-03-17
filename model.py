"""UNet architecture with time conditioning for consistency models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional encoding for timesteps."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.cos(), args.sin()], dim=-1)


class TimeMLPEmbedding(nn.Module):
    """Sinusoidal encoding -> 2-layer MLP."""

    def __init__(self, dim):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, t):
        return self.mlp(self.sinusoidal(t))


class ResBlock(nn.Module):
    """ResBlock with time conditioning: GN -> SiLU -> Conv -> (+time) -> GN -> SiLU -> Conv -> residual."""

    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(32, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Single-head self-attention."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        scale = C ** -0.5
        attn = torch.bmm(q.transpose(1, 2), k) * scale
        attn = attn.softmax(dim=-1)
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet for 32x32 images with time conditioning.

    Channel config: [128, 256, 256], 2 ResBlocks per level.
    Attention at 16x16 resolution. Encoder: 32->16->8.
    """

    def __init__(self, in_ch=3, out_ch=3, ch=128, ch_mults=(1, 2, 2), num_res_blocks=2, attn_resolutions=(16,)):
        super().__init__()
        time_dim = ch * 4
        self.time_embed = TimeMLPEmbedding(time_dim)
        channels = [ch * m for m in ch_mults]

        # Input projection
        self.input_conv = nn.Conv2d(in_ch, ch, 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = ch
        self._skip_channels = [ch]  # track for decoder

        for level, cur_ch in enumerate(channels):
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(prev_ch, cur_ch, time_dim))
                resolution = 32 // (2 ** level)
                if resolution in attn_resolutions:
                    attns.append(SelfAttention(cur_ch))
                else:
                    attns.append(nn.Identity())
                prev_ch = cur_ch
                self._skip_channels.append(cur_ch)
            self.down_blocks.append(nn.ModuleDict({"blocks": blocks, "attns": attns}))

            if level < len(channels) - 1:
                self.down_samples.append(Downsample(cur_ch))
                self._skip_channels.append(cur_ch)
            else:
                self.down_samples.append(nn.Identity())

        # Middle
        mid_ch = channels[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim)
        self.mid_attn = SelfAttention(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level in reversed(range(len(channels))):
            cur_ch = channels[level]
            blocks = nn.ModuleList()
            attns = nn.ModuleList()
            for i in range(num_res_blocks + 1):
                skip_ch = self._skip_channels.pop()
                blocks.append(ResBlock(prev_ch + skip_ch, cur_ch, time_dim))
                resolution = 32 // (2 ** level)
                if resolution in attn_resolutions:
                    attns.append(SelfAttention(cur_ch))
                else:
                    attns.append(nn.Identity())
                prev_ch = cur_ch
            self.up_blocks.append(nn.ModuleDict({"blocks": blocks, "attns": attns}))

            if level > 0:
                self.up_samples.append(Upsample(cur_ch))
            else:
                self.up_samples.append(nn.Identity())

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_ch, 3, padding=1)

    def forward(self, x, t):
        """
        Args:
            x: (B, 3, 32, 32) noisy input
            t: (B,) noise level / timestep
        Returns:
            (B, 3, 32, 32) raw network output F_θ(x, t)
        """
        t_emb = self.time_embed(t)
        h = self.input_conv(x)
        skips = [h]

        # Encoder
        for level, (block_dict, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for block, attn in zip(block_dict["blocks"], block_dict["attns"]):
                h = block(h, t_emb)
                h = attn(h)
                skips.append(h)
            if level < len(self.down_blocks) - 1:
                h = downsample(h)
                skips.append(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for block_dict, upsample in zip(self.up_blocks, self.up_samples):
            for block, attn in zip(block_dict["blocks"], block_dict["attns"]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = block(h, t_emb)
                h = attn(h)
            h = upsample(h)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


class ConsistencyModel(nn.Module):
    """
    Wraps UNet with consistency model parameterization:
    f_θ(x, t) = c_skip(t) * x + c_out(t) * F_θ(x, t)
    """

    def __init__(self, sigma_data=0.5, sigma_min=0.002, **unet_kwargs):
        super().__init__()
        self.net = UNet(**unet_kwargs)
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

    def c_skip(self, t):
        return self.sigma_data ** 2 / (t ** 2 + self.sigma_data ** 2)

    def c_out(self, t):
        return self.sigma_data * t / (t ** 2 + self.sigma_data ** 2).sqrt()

    def forward(self, x, t):
        """
        Args:
            x: (B, 3, 32, 32)
            t: (B,) sigma values
        Returns:
            (B, 3, 32, 32) consistency function output
        """
        c_skip = self.c_skip(t)[:, None, None, None]
        c_out = self.c_out(t)[:, None, None, None]
        F_x = self.net(x, t)
        return c_skip * x + c_out * F_x
