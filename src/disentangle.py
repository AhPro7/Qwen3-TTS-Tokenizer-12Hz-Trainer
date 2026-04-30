# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Shared DisentangledProjection module for speaker/content disentanglement.

Used by: trainer.py, evaluate.py, evaluate_all.py, voice_convert.py

Architecture (AutoVC-inspired bottleneck):
  Speaker path: 2-layer encoder → attention pool → global vector
                → 2-layer decoder → broadcast to all frames.
  Content path: per-frame 2-layer MLP → Vector Quantization (codebook)
                → straight-through estimator → decoder MLP.

The temporal pooling in the speaker path makes it structurally impossible
to encode per-frame content, forcing the content path to carry that
information instead.

The VQ bottleneck on the content path makes it structurally impossible
to encode continuous speaker characteristics (F0, spectral envelope, timbre)
since those must pass through a finite discrete codebook.
This is the same principle used in SpeechTokenizer (codebook 1 = content).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Finite Scalar Quantization (FSQ)
# ---------------------------------------------------------------------------

class FSQ(nn.Module):
    """Finite Scalar Quantization for massive single-codebook capacity without collapse."""
    def __init__(self, levels: list[int] = [8, 8, 8, 8, 8, 8]):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float32))
        basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0)
        self.register_buffer("basis", basis.float())
        self.dim = len(levels)
        self.codebook_size = int(torch.prod(torch.tensor(levels)).item())
        self.last_indices = None

    def forward(self, x: torch.Tensor):
        # x: [B, T, dim]
        bounds = (self.levels - 1) / 2
        
        # Correct FSQ bounding from the paper: bounds * tanh(x / bounds)
        # This prevents the gradient from vanishing too early compared to tanh(x) * bounds
        x_bounded = bounds * torch.tanh(x / bounds)
        
        quantized = torch.round(x_bounded)
        # Straight-through estimator
        quantized_st = x_bounded + (quantized - x_bounded).detach()
        
        # Calculate single discrete index per timestep
        indices = torch.sum((quantized + bounds) * self.basis, dim=-1).long()
        self.last_indices = indices.detach()
        
        # Normalize back to [-1, 1] range for the decoder
        out = quantized_st / bounds
        
        return out, indices, x.new_zeros(())

    @torch.no_grad()
    def codebook_usage(self) -> float:
        """FSQ inherently maps continuously, so usage isn't as easily tracked as EMA VQ, but we return a proxy."""
        if self.last_indices is None:
            return 0.0
        unique = self.last_indices.unique().numel()
        return unique / self.codebook_size


# ---------------------------------------------------------------------------
# DisentangledProjection (Now heavily optimized as a Single Codebook FSQ Bottleneck)
# ---------------------------------------------------------------------------

class DisentangledProjection(nn.Module):
    """
    Refactored to be a Pure Single-Codebook Bottleneck using FSQ.
    Speaker logic is bypassed to force the 1 codebook to memorize everything.
    """
    def __init__(
        self,
        hidden_dim: int = 1024,
        speaker_dim: int = 256,
        content_dim: int = 128,
        codebook_size: int = 1024,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # FSQ Levels: [8, 8, 8, 8, 8, 8] -> 262,144 unique codes.
        levels = [8, 8, 8, 8, 8, 8]
        self.vq = FSQ(levels)
        
        # Pre-VQ: compress hidden_dim -> FSQ dim (6)
        # Replaced ReLU with LayerNorm + SiLU to prevent dead neurons killing the variance
        self.content_pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, self.vq.dim),
        )

        # Post-VQ: expand FSQ dim (6) -> hidden_dim
        self.content_post_proj = nn.Sequential(
            nn.Linear(self.vq.dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Scale initial projections to fully explore the FSQ grid."""
        # FSQ requires large enough inputs to escape the center (index 0).
        # We initialize the pre-projection to have a large variance so it scatters across the levels.
        last_pre = [l for l in self.content_pre_proj if isinstance(l, nn.Linear)][-1]
        nn.init.normal_(last_pre.weight, std=2.0 / self.hidden_dim**0.5) 
        # Add uniform bias spread to immediately hit different bins
        nn.init.uniform_(last_pre.bias, -2.0, 2.0)

        # Content post_proj last layer: small init so it doesn't explode gradients early
        last_post = [l for l in self.content_post_proj if isinstance(l, nn.Linear)][-1]
        nn.init.normal_(last_post.weight, std=0.01)
        nn.init.zeros_(last_post.bias)

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.content_pre_proj[0].weight.dtype

    def _match_dtype(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.weight_dtype:
            return x.to(self.weight_dtype)
        return x

    def encode_content(self, x: torch.Tensor):
        x = self._match_dtype(x)
        pre = self.content_pre_proj(x)
        quantized, indices, commit_loss = self.vq(pre)
        content_emb = self.content_post_proj(quantized)
        return content_emb, indices, commit_loss

    def forward(self, x: torch.Tensor):
        x = self._match_dtype(x)
        content_emb, content_indices, commit_loss = self.encode_content(x)
        return content_emb, content_indices, commit_loss