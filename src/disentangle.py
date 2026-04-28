# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Shared DisentangledProjection module for speaker/content disentanglement.

Used by: trainer.py, evaluate.py, evaluate_all.py, voice_convert.py

Architecture (AutoVC-inspired bottleneck):
  Speaker path: 2-layer encoder → attention pool → global vector
                → 2-layer decoder → broadcast to all frames.
  Content path: per-frame 2-layer MLP (keeps temporal detail).

Disentanglement mechanisms (added on top of Run 81):
  1. GRL (Gradient Reversal Layer): trains content to NOT encode speaker.
  2. Speaker contrastive loss: pushes in-batch speaker embeddings apart.
  3. Content feature dropout: forces decoder to rely on speaker path.

The temporal pooling in the speaker path makes it structurally impossible
to encode per-frame content, forcing the content path to carry that
information instead.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


# ── Gradient Reversal Layer ──────────────────────────────────────────────────
class _GradientReversal(Function):
    """Reverses gradients during backward pass (for adversarial training)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal: forward = identity, backward = -lambda * grad."""
    return _GradientReversal.apply(x, lambda_)


# ── Speaker Adversarial Head ─────────────────────────────────────────────────
class SpeakerAdversarialHead(nn.Module):
    """Classifies speaker from content embeddings (via GRL).

    During forward: tries to predict which batch sample a content frame
    came from. The GRL reverses gradients so the content encoder learns
    to make this IMPOSSIBLE → content becomes speaker-free.
    """

    def __init__(self, hidden_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )
        # Temperature for contrastive similarity
        self.log_temp = nn.Parameter(torch.tensor(2.0).log())

    def forward(self, content_emb: torch.Tensor) -> torch.Tensor:
        """content_emb: [B, T, H] → speaker_repr: [B, proj_dim]"""
        # Time-average → global content representation
        return self.classifier(content_emb.mean(dim=1))


class DisentangledProjection(nn.Module):
    """AutoVC-style information bottleneck for speaker/content disentanglement.

    Args:
        hidden_dim: Transformer hidden dimension (default: 1024).
        speaker_dim: Speaker bottleneck dimension (default: 256).
        content_dropout: Dropout rate for content features during training.
            Forces the decoder to rely on the speaker path for stable
            features (speaker identity). Set to 0.0 to disable.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        speaker_dim: int = 256,
        content_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.content_dropout = content_dropout

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

        # Content dropout (applied during training only)
        self.content_drop = nn.Dropout(p=content_dropout) if content_dropout > 0 else None

        # Speaker adversarial head (GRL)
        self.speaker_adversarial = SpeakerAdversarialHead(hidden_dim, speaker_dim)

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

    def speaker_adversarial_loss(
        self, content_emb: torch.Tensor, grl_lambda: float = 1.0
    ) -> torch.Tensor:
        """Contrastive speaker classification on GRL-reversed content.

        If the adversarial head CAN classify speakers from content, it means
        content still carries speaker info. The GRL reverses the gradient so
        the content encoder learns to strip speaker identity.

        Args:
            content_emb: [B, T, hidden_dim] — content embeddings.
            grl_lambda: GRL strength (ramp from 0→1 during warmup).

        Returns:
            Scalar loss (cross-entropy). Higher = more speaker leakage.
        """
        content_emb = self._match_dtype(content_emb)

        # Reverse gradients through GRL → content encoder learns to
        # make speaker classification impossible
        content_reversed = gradient_reversal(content_emb, grl_lambda)

        # Project to speaker space
        speaker_repr = self.speaker_adversarial(content_reversed)  # [B, proj_dim]

        # Contrastive: each sample is its own class
        # similarity matrix [B, B], labels = [0, 1, ..., B-1]
        temp = self.speaker_adversarial.log_temp.exp().clamp(min=0.01, max=100.0)
        speaker_repr = nn.functional.normalize(speaker_repr, dim=-1)
        sim_matrix = torch.mm(speaker_repr, speaker_repr.t()) * temp
        labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

        loss = nn.functional.cross_entropy(sim_matrix, labels)

        return loss

    def speaker_diversity_loss(
        self, speaker_globals: torch.Tensor, min_norm: float = 2.0
    ) -> torch.Tensor:
        """Push in-batch speaker embeddings apart AND prevent magnitude collapse.

        Two components:
          1. Cosine diversity: penalize high pairwise cosine similarity.
          2. Norm floor: penalize if ||speaker_global|| < min_norm.
             Without this, the encoder collapses magnitude to near-zero
             (cosine is scale-invariant, so small vectors satisfy diversity
             while contributing nothing to the decoder).

        Args:
            speaker_globals: [B, speaker_dim] — speaker embeddings from batch.
            min_norm: Minimum desired L2 norm for speaker embeddings.

        Returns:
            Scalar loss.
        """
        speaker_globals = self._match_dtype(speaker_globals)
        normed = nn.functional.normalize(speaker_globals, dim=-1)
        sim = torch.mm(normed, normed.t())  # [B, B]

        # Exclude diagonal (self-similarity = 1.0 by definition)
        B = sim.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)

        # 1. Cosine diversity: penalize high off-diagonal similarity
        div_loss = (sim[mask] ** 2).mean()

        # 2. Norm floor: penalize if embeddings collapse in magnitude
        # ReLU(min_norm - ||x||) → 0 if norm >= min_norm, positive otherwise
        norms = speaker_globals.norm(dim=-1)  # [B]
        norm_loss = torch.relu(min_norm - norms).mean()

        return div_loss + norm_loss

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

        # Content feature dropout (training only):
        # Forces the decoder to rely on the speaker path for stable features.
        # At inference (eval mode), dropout is disabled automatically.
        if self.content_drop is not None:
            content_emb = self.content_drop(content_emb)

        return speaker_contribution, content_emb, speaker_global
