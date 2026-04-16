# coding=utf-8
"""
Conformer Encoder for StudentCodec.

ConformerBlock (macaron-style):
  x → FeedForward (½) → MultiHeadSelfAttention → DepthwiseConv → FeedForward (½) → LayerNorm

References:
  - Gulati et al. "Conformer: Convolution-augmented Transformer for Speech" (2020)
  - SoundStream, EnCodec use similar architectures for audio latent modelling
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Rotary Position Embedding ───────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    """RoPE: Rotary Position Embedding (Su et al. 2021).
    Better than learned/sinusoidal for audio (relative position equivariance).
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._max_len = max_seq_len
        self._cos = None
        self._sin = None

    def _build_cache(self, seq_len: int, device, dtype):
        if self._cos is not None and self._cos.shape[0] >= seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos = emb.cos().to(dtype)
        self._sin = emb.sin().to(dtype)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """Apply RoPE to query and key tensors.
        q, k: [B, heads, T, head_dim]
        """
        T = q.shape[2]
        self._build_cache(T, q.device, q.dtype)
        cos = self._cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self._sin[:T].unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


# ─── Attention ────────────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention with RoPE and optional masking."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out  = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D]
        Returns:
            out: [B, T, D]
        """
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                  # each [B, T, H, head]
        q = q.transpose(1, 2)                          # [B, H, T, head]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = self.rope(q, k)

        # SDPA (Flash Attention-compatible)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.drop.p if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out(out)


# ─── Depthwise Conv Module ────────────────────────────────────────────────────

class ConvModule(nn.Module):
    """Conformer convolution module.
    LayerNorm → PointwiseConv (gate) → DepthwiseConv → BatchNorm → Swish → PointwiseConv
    """

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.layer_norm = nn.LayerNorm(d_model)
        self.pw1        = nn.Conv1d(d_model, 2 * d_model, 1)   # gate expansion
        self.dw_conv    = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.bn         = nn.BatchNorm1d(d_model)
        self.pw2        = nn.Conv1d(d_model, d_model, 1)
        self.drop       = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]"""
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)                    # [B, D, T]
        x = self.pw1(x)                           # [B, 2D, T]
        x = F.glu(x, dim=1)                       # [B, D, T]  (gating)
        x = self.dw_conv(x)                       # [B, D, T]
        x = self.bn(x)
        x = F.silu(x)                              # Swish
        x = self.pw2(x)                           # [B, D, T]
        x = self.drop(x.transpose(1, 2))          # [B, T, D]
        return residual + x


# ─── Feed-Forward ─────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    """Macaron-style FFN (half scale at input and output)."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.ff(self.norm(x))


# ─── ConformerBlock ───────────────────────────────────────────────────────────

class ConformerBlock(nn.Module):
    """Single Conformer block (macaron-style):
        FFN(½) → MHSA → ConvModule → FFN(½) → LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1  = FeedForward(d_model, ff_expansion, dropout)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ff2  = FeedForward(d_model, ff_expansion, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.ff1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), attn_mask, key_padding_mask))
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


# ─── ConformerEncoder ─────────────────────────────────────────────────────────

class ConformerEncoder(nn.Module):
    """Stack of ConformerBlocks with optional input projection.

    Args:
        d_model:     hidden dimension
        n_layers:    number of conformer blocks
        n_heads:     attention heads (d_model must be divisible)
        ff_expansion: FFN inner dim = d_model × ff_expansion
        conv_kernel: depthwise conv kernel size (odd number, e.g. 31)
        dropout:     dropout rate
        input_dim:   if provided, adds an initial linear projection
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 4,
        ff_expansion: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model

        if input_dim is not None and input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = nn.Identity()

        self.layers = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_expansion, conv_kernel, dropout)
            for _ in range(n_layers)
        ])

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim]
        Returns:
            [B, T, d_model]
        """
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, attn_mask, key_padding_mask)
        return x

    def info(self) -> dict:
        return {
            "conformer/n_layers":  len(self.layers),
            "conformer/d_model":   self.d_model,
            "conformer/params_M":  self.n_params / 1e6,
        }
