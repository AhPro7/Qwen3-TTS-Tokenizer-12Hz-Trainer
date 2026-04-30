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
# Vector Quantization layer (rock-solid EMA codebook update)
# ---------------------------------------------------------------------------

class VectorQuantize(nn.Module):
    """Proven EMA-based VQ with Dead-Code Restarts (from stable Run 81)."""
    def __init__(
        self,
        codebook_size: int = 16384,  # Large enough for 1 codebook, small enough to be stable
        dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        restart_threshold: float = 1.0, 
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold

        self.register_buffer("codebook", torch.randn(codebook_size, dim) * (dim ** -0.5))
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_dw", self.codebook.clone())
        self.last_indices = None

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.codebook.dtype

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        distances = torch.cdist(x_flat.unsqueeze(0), self.codebook.unsqueeze(0)).squeeze(0)
        indices_flat = distances.argmin(dim=-1)
        quantized_flat = self.codebook[indices_flat]

        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices_flat, self.codebook_size).float()
                new_cluster_size = one_hot.sum(0)
                new_dw = one_hot.t() @ x_flat

                self.ema_cluster_size.mul_(self.ema_decay).add_(new_cluster_size * (1 - self.ema_decay))
                self.ema_dw.mul_(self.ema_decay).add_(new_dw * (1 - self.ema_decay))
                
                n = self.ema_cluster_size.sum()
                smoothed = ((self.ema_cluster_size + 1e-5) / (n + self.codebook_size * 1e-5) * n)
                self.codebook.copy_(self.ema_dw / smoothed.unsqueeze(1))

                # Dead code restart to prevent collapse
                if self.restart_threshold > 0:
                    dead_mask = self.ema_cluster_size < self.restart_threshold
                    num_dead = dead_mask.sum().item()
                    if num_dead > 0:
                        rand_idx = torch.randint(0, x_flat.shape[0], (int(num_dead),), device=x_flat.device)
                        self.codebook[dead_mask] = x_flat[rand_idx].to(self.codebook.dtype)
                        self.ema_cluster_size[dead_mask] = self.restart_threshold
                        self.ema_dw[dead_mask] = x_flat[rand_idx].to(self.ema_dw.dtype)

        commit_loss = self.commitment_cost * F.mse_loss(x_flat, quantized_flat.detach())
        quantized_st = x_flat + (quantized_flat - x_flat).detach()

        indices = indices_flat.reshape(B, T)
        quantized = quantized_st.reshape(B, T, D)
        self.last_indices = indices.detach()

        return quantized, indices, commit_loss

    @torch.no_grad()
    def codebook_usage(self) -> float:
        if self.last_indices is None: return 0.0
        return self.last_indices.unique().numel() / self.codebook_size


# ---------------------------------------------------------------------------
# Single Codebook Bottleneck (Zero Speaker)
# ---------------------------------------------------------------------------

class DisentangledProjection(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 1024,
        speaker_dim: int = 256,
        content_dim: int = 128,
        codebook_size: int = 16384,  # 16k is highly stable for 1 codebook EMA
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.content_pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, content_dim),
        )

        self.vq = VectorQuantize(
            codebook_size=codebook_size,
            dim=content_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
        )

        self.content_post_proj = nn.Sequential(
            nn.Linear(content_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Identity/Zero init so the unfrozen decoder doesn't instantly explode."""
        last_pre = [l for l in self.content_pre_proj if isinstance(l, nn.Linear)][-1]
        nn.init.normal_(last_pre.weight, std=(self.vq.dim ** -0.5))
        nn.init.zeros_(last_pre.bias)

        last_post = [l for l in self.content_post_proj if isinstance(l, nn.Linear)][-1]
        # Extremely small init to prevent gradient explosion in the decoder
        nn.init.normal_(last_post.weight, std=0.001)
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