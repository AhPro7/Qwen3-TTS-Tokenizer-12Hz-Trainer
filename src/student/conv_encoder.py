# coding=utf-8
"""
Strided Conv Encoder: Raw Audio (24kHz) → Features at 12Hz

Downsampling factor: 24000 / 12 = 2000
Stride chain: 5 × 4 × 4 × 5 × 5 = 2000 ✓

Architecture per ConvBlock:
  Conv1d → GroupNorm → GELU → (optional residual)
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Causal Conv1d block with GroupNorm + GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        causal: bool = False,
    ):
        super().__init__()
        self.stride = stride
        self.causal = causal
        self.pad = (kernel_size - 1) if causal else (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0 if causal else self.pad,
            groups=groups,
        )
        num_groups = min(32, out_channels)
        while out_channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal:
            x = F.pad(x, (self.pad, 0))
        return self.act(self.norm(self.conv(x)))


class ResConvBlock(nn.Module):
    """Residual ConvBlock (no stride) for within-scale processing."""

    def __init__(self, channels: int, kernel_size: int = 7):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=kernel_size),
            ConvBlock(channels, channels, kernel_size=kernel_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ConvEncoder(nn.Module):
    """Strided convolutional encoder: 24kHz waveform → 12Hz feature frames.

    Architecture:
        Input: [B, 1, num_samples]  (raw mono waveform at input_sr)
        Output: [B, d_model, T]     (features at token_rate Hz)

    Downsampling strides: 5 × 4 × 4 × 5 × 5 = 2000
      input_sr=24000, token_rate=12 → ratio=2000 ✓

    Per stage: ConvBlock(stride) → ResConvBlock × n_res
    """

    def __init__(
        self,
        d_model: int = 256,
        input_sr: int = 24_000,
        token_rate: int = 12,
        strides: List[int] = (5, 4, 4, 5, 5),
        n_res_per_stage: int = 2,
        base_channels: int = 64,
    ):
        super().__init__()
        expected_ratio = input_sr // token_rate  # 2000
        actual_ratio = 1
        for s in strides:
            actual_ratio *= s
        assert actual_ratio == expected_ratio, (
            f"Strides product {actual_ratio} ≠ expected {expected_ratio} "
            f"for {input_sr}Hz→{token_rate}Hz"
        )

        self.input_sr   = input_sr
        self.token_rate = token_rate
        self.strides    = list(strides)

        # Channel progression: base → 2x → 4x → 8x → d_model
        channels = [
            1,
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            d_model,
        ]

        stages = []
        for i, stride in enumerate(strides):
            in_ch  = channels[i]
            out_ch = channels[i + 1]
            k_size = stride * 2 + 1  # kernel slightly larger than stride
            stage  = nn.Sequential(
                ConvBlock(in_ch, out_ch, kernel_size=k_size, stride=stride),
                *[ResConvBlock(out_ch) for _ in range(n_res_per_stage)],
            )
            stages.append(stage)

        self.stages  = nn.ModuleList(stages)
        self.out_proj = nn.Linear(d_model, d_model)

        # Compute receptive field for logging
        self._receptive_field = self._calc_receptive_field(strides)

    def _calc_receptive_field(self, strides):
        # Approximate: product of strides * kernel_factor
        return math.prod(strides) * 3  # rough estimate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_samples] or [B, 1, num_samples]
        Returns:
            features: [B, T, d_model]  at token_rate Hz
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, S]
        elif x.dim() == 3 and x.shape[1] != 1:
            x = x.mean(1, keepdim=True)  # multi-channel → mono

        for stage in self.stages:
            x = stage(x)  # [B, ch, T_i]

        # x: [B, d_model, T]
        x = x.transpose(1, 2)       # [B, T, d_model]
        x = self.out_proj(x)        # [B, T, d_model]
        return x

    def get_output_length(self, num_samples: int) -> int:
        """Returns the number of output frames for a given input length."""
        from math import floor
        L = num_samples
        for s in self.strides:
            L = floor(L / s)
        return L

    def info(self) -> dict:
        """Returns architecture info for logging."""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "conv_encoder/input_sr":        self.input_sr,
            "conv_encoder/token_rate_hz":   self.token_rate,
            "conv_encoder/strides":         str(self.strides),
            "conv_encoder/downsample_ratio": math.prod(self.strides),
            "conv_encoder/params_M":        n_params / 1e6,
            "conv_encoder/receptive_field_ms": self._receptive_field * 1000 / self.input_sr,
        }
