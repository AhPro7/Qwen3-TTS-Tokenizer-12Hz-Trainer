# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Shared DisentangledProjection module for speaker/content disentanglement.

Used by: trainer.py, evaluate.py, evaluate_all.py, voice_convert.py

Architecture (AutoVC-inspired bottleneck):
  Speaker path: 2-layer encoder → attention pool → global vector
                → 2-layer decoder → broadcast to all frames.
  Content path: per-frame 2-layer MLP (keeps temporal detail).

The temporal pooling in the speaker path makes it structurally impossible
to encode per-frame content, forcing the content path to carry that
information instead.
"""

import torch
import torch.nn as nn


class DisentangledProjection(nn.Module):
    """AutoVC-style information bottleneck for speaker/content disentanglement.

    Args:
        hidden_dim: Transformer hidden dimension (default: 1024).
        speaker_dim: Speaker bottleneck dimension (default: 256).
    """

    def __init__(self, hidden_dim: int = 1024, speaker_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim

        # Speaker branch: encode → pool → decode
        self.speaker_encoder = nn.Sequential(
            nn.Linear(hidden_dim, speaker_dim),
            nn.LayerNorm(speaker_dim),
            nn.ReLU(),
            nn.Linear(speaker_dim, speaker_dim),
            nn.ReLU(),
        )
        self.speaker_attention = nn.Linear(speaker_dim, 1)  # attention weights
        self.speaker_decoder = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Content branch: per-frame MLP
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Warm-start so combined = speaker_contrib + content_emb ≈ x at step 0.

        Strategy:
          - content_proj layers → near-identity  (content_emb ≈ x)
          - speaker_decoder   → near-zero        (speaker_contrib ≈ 0)
        Result: combined ≈ 0 + x = x  → decoder hears the same signal it was
        trained on → clean audio from step 0, no 'copper mic' artifacts.
        Disentanglement emerges gradually through training.
        """
        # Content path: near-identity init
        for layer in self.content_proj:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)      # identity
                nn.init.zeros_(layer.bias)
                # tiny noise to break symmetry
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)

        # Speaker decoder: near-zero so speaker_contrib ≈ 0 at init
        for layer in self.speaker_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    @property
    def weight_dtype(self) -> torch.dtype:
        """Return the dtype of model weights (handles mixed precision correctly)."""
        return self.speaker_attention.weight.dtype

    def _match_dtype(self, x: torch.Tensor) -> torch.Tensor:
        """Cast input to match weight dtype. Differentiable (gradients flow through)."""
        if x.dtype != self.weight_dtype:
            return x.to(self.weight_dtype)
        return x

    def encode_speaker(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden_dim] → speaker_global: [B, speaker_dim]"""
        x = self._match_dtype(x)
        h = self.speaker_encoder(x)                                 # [B, T, speaker_dim]
        attn = torch.softmax(self.speaker_attention(h), dim=1)      # [B, T, 1]
        # softmax may output float32 under autocast — cast back
        attn = self._match_dtype(attn)
        return (h * attn).sum(dim=1)                                # [B, speaker_dim]

    def decode_speaker(self, speaker_global: torch.Tensor, seq_len: int) -> torch.Tensor:
        """speaker_global: [B, speaker_dim] → [B, T, hidden_dim]"""
        speaker_global = self._match_dtype(speaker_global)
        out = self.speaker_decoder(speaker_global)                  # [B, hidden_dim]
        return out.unsqueeze(1).expand(-1, seq_len, -1)             # [B, T, hidden_dim]

    def encode_content(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden_dim] → content_emb: [B, T, hidden_dim]"""
        return self.content_proj(self._match_dtype(x))

    def forward(self, x: torch.Tensor):
        """Returns (speaker_contribution, content_emb, speaker_global).

        speaker_contribution: [B, T, hidden_dim]  (broadcast from global)
        content_emb:          [B, T, hidden_dim]  (per-frame)
        speaker_global:       [B, speaker_dim]    (for logging / swap)
        """
        x = self._match_dtype(x)
        speaker_global = self.encode_speaker(x)                       # [B, speaker_dim]
        speaker_contribution = self.decode_speaker(speaker_global, x.shape[1])  # [B, T, H]
        content_emb = self.encode_content(x)                          # [B, T, H]
        return speaker_contribution, content_emb, speaker_global
