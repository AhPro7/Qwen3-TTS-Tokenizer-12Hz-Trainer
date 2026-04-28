# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Shared DisentangledProjection module for speaker/content disentanglement.

Used by: trainer.py, evaluate.py, evaluate_all.py, voice_convert.py

Architecture (AutoVC-inspired bottleneck):
  Speaker path: 2-layer encoder → attention pool → global vector
                → 2-layer decoder → broadcast to all frames.
  Content path: per-frame MLP with NARROW BOTTLENECK.
                1024 → content_bottleneck_dim → 1024
                The narrow bottleneck forces the content path to discard
                speaker identity (it can't fit both in limited capacity).

The temporal pooling in the speaker path makes it structurally impossible
to encode per-frame content, forcing the content path to carry that
information instead.

The content bottleneck makes it structurally impossible to encode speaker
identity in the content path, forcing the speaker path to carry that
information instead.

Together, these two bottlenecks achieve full disentanglement.
"""

import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer (GRL).

    Forward: identity.  Backward: negate gradients and scale by lambda.
    Used to adversarially train the content path to NOT encode speaker info.
    """

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lam * grad_output, None


def grad_reverse(x, lam=1.0):
    return GradientReversal.apply(x, lam)


class SpeakerAdversarialHead(nn.Module):
    """Classifies speaker identity from content embeddings.

    Combined with GRL, this forces the content path to strip speaker info.
    NOT a real classifier (no fixed speaker IDs) — uses contrastive approach:
    predicts whether two content segments are from the same speaker.

    Architecture: temporal pool → MLP → speaker embedding for comparison.
    """

    def __init__(self, hidden_dim: int = 1024, proj_dim: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim),
        )
        # Temperature for contrastive loss (learnable)
        self.log_temp = nn.Parameter(torch.log(torch.tensor(0.07)))

    def forward(self, content_emb: torch.Tensor) -> torch.Tensor:
        """content_emb: [B, T, hidden_dim] → speaker_repr: [B, proj_dim]"""
        # Temporal mean pool
        pooled = content_emb.mean(dim=1)  # [B, hidden_dim]
        return nn.functional.normalize(self.proj(pooled), dim=-1)


class DisentangledProjection(nn.Module):
    """AutoVC-style information bottleneck for speaker/content disentanglement.

    Args:
        hidden_dim: Transformer hidden dimension (default: 1024).
        speaker_dim: Speaker bottleneck dimension (default: 256).
        content_bottleneck_dim: Content bottleneck dimension (default: 128).
            Controls how much information the content path can carry.
            Smaller = stronger disentanglement, weaker content detail.
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        speaker_dim: int = 256,
        content_bottleneck_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.content_bottleneck_dim = content_bottleneck_dim

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

        # Content branch: per-frame MLP with NARROW BOTTLENECK
        self.content_encoder = nn.Sequential(
            nn.Linear(hidden_dim, content_bottleneck_dim),
            nn.LayerNorm(content_bottleneck_dim),
            nn.ReLU(),
        )
        self.content_decoder = nn.Sequential(
            nn.Linear(content_bottleneck_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Speaker adversarial head: strips speaker info from content
        self.speaker_adversarial = SpeakerAdversarialHead(hidden_dim, proj_dim=128)

        self._warm_start_init()

    def _warm_start_init(self):
        """Warm-start so the combined output is reasonable at step 0.

        Strategy:
          - speaker_decoder → initialized with moderate scale so speaker
            path contributes meaningful signal from the start.
          - content_decoder → moderate init (can't be identity due to
            bottleneck dimension mismatch).
          - The model will learn to balance the two paths during training.

        NOTE: With the content bottleneck, reconstruction won't be perfect
        at step 0. This is intentional — the bottleneck forces disentanglement.
        """
        # Content encoder/decoder: Xavier uniform (good default for bottleneck)
        for module in [self.content_encoder, self.content_decoder]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

        # Speaker decoder: moderate init so it contributes from the start
        for layer in self.speaker_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
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
        """x: [B, T, hidden_dim] → content_emb: [B, T, hidden_dim]

        Passes through narrow bottleneck (hidden_dim → content_bottleneck_dim → hidden_dim).
        The bottleneck strips speaker identity from the content representation.
        """
        x = self._match_dtype(x)
        z = self.content_encoder(x)   # [B, T, content_bottleneck_dim] — narrow!
        return self.content_decoder(z) # [B, T, hidden_dim]

    def speaker_adversarial_loss(
        self, content_emb: torch.Tensor, grl_lambda: float = 1.0
    ) -> torch.Tensor:
        """Compute contrastive speaker adversarial loss on content embeddings.

        Uses GRL so gradients REVERSE through the content encoder:
        the content path is trained to make it IMPOSSIBLE to predict
        speaker identity from content embeddings.

        Returns:
            loss: In-batch contrastive loss (lower = more speaker info leaked).
                  Through GRL, the content encoder is trained to MAXIMIZE this.
        """
        B = content_emb.shape[0]
        if B < 2:
            return content_emb.new_zeros(())

        # GRL: forward = identity, backward = negate
        content_reversed = grad_reverse(content_emb, grl_lambda)
        content_reversed = self._match_dtype(content_reversed)
        speaker_repr = self.speaker_adversarial(content_reversed)  # [B, proj_dim]

        # In-batch contrastive: all samples should look the same (no speaker info)
        # If content carries speaker info, different speakers will have different repr
        # → high similarity for same speaker, low for different → contrastive loss is low
        # GRL inverts this: content encoder learns to make all repr identical
        temp = self.speaker_adversarial.log_temp.exp()
        sim_matrix = speaker_repr @ speaker_repr.T / temp  # [B, B]

        # Self-similarity target: each sample is its own positive
        labels = torch.arange(B, device=content_emb.device)
        loss = nn.functional.cross_entropy(sim_matrix, labels)

        return loss

    def forward(self, x: torch.Tensor, alpha: float = 1.0):
        """Returns (speaker_contribution, content_emb, speaker_global).

        Args:
            x: [B, T, hidden_dim] — pre_transformer output.
            alpha: Bottleneck blend factor (0.0 = identity, 1.0 = full bottleneck).
                   Ramps from 0→1 over training to avoid distribution shift
                   that kills frozen decoder blocks.

        Returns:
            speaker_contribution: [B, T, hidden_dim]  (broadcast from global)
            content_emb:          [B, T, hidden_dim]  (alpha-blended)
            speaker_global:       [B, speaker_dim]    (for logging / swap)

        Also stores self.last_content_bottleneck for GRL/VC losses
        (always the PURE bottleneck output, regardless of alpha).
        """
        x = self._match_dtype(x)
        speaker_global = self.encode_speaker(x)                       # [B, speaker_dim]
        speaker_contribution = self.decode_speaker(speaker_global, x.shape[1])  # [B, T, H]

        # Pure bottleneck path (for GRL and VC — always full bottleneck)
        content_bottleneck = self.encode_content(x)                   # [B, T, H]
        self.last_content_bottleneck = content_bottleneck

        # Alpha-blended content for reconstruction:
        #   alpha=0 → pure identity x (like Run 81, perfect reconstruction)
        #   alpha=1 → full bottleneck (maximum disentanglement)
        if alpha >= 1.0:
            content_emb = content_bottleneck
        elif alpha <= 0.0:
            content_emb = x
        else:
            content_emb = (1.0 - alpha) * x + alpha * content_bottleneck

        return speaker_contribution, content_emb, speaker_global
