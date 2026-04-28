# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Shared DisentangledProjection module for speaker/content disentanglement.

Used by: trainer.py, evaluate.py, evaluate_all.py, voice_convert.py

Architecture (Residual Disentanglement):
  Speaker path: 2-layer encoder → attention pool → global vector
                → 2-layer decoder → broadcast to all frames.
  Content path: per-frame MLP on the RESIDUAL (x - speaker_contrib.detach()).
                Content never sees speaker info because it's subtracted first.

  combined = speaker_contrib + content_proj(x - speaker_contrib.detach())
           ≈ speaker + (x - speaker)   [when content_proj ≈ identity]
           = x                          → perfect reconstruction from step 0!

  Voice conversion:
    combined_AB = speaker_B + content_proj(x_A - speaker_A.detach())
               ≈ x_A + (speaker_B - speaker_A)
               → shifts the voice from A to B.

Key: The .detach() on speaker_contrib in the content input prevents the content
path from learning to "undo" the subtraction. Content genuinely cannot access
speaker information.
"""

import torch
import torch.nn as nn


class DisentangledProjection(nn.Module):
    """Residual speaker/content disentanglement.

    Args:
        hidden_dim: Transformer hidden dimension (default: 1024).
        speaker_dim: Speaker bottleneck dimension (default: 256).
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        speaker_dim: int = 256,
        content_dropout: float = 0.0,   # kept for CLI compat, not used
    ):
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

        # Content branch: per-frame MLP on RESIDUAL (x - speaker.detach())
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Warm-start for residual formulation.

        Strategy:
          - content_proj → near-identity: content ≈ (x - speaker)
          - speaker_decoder → NORMAL init (Xavier), NOT near-zero!
            With residual: combined = S + identity(x - S) = x regardless of S.
            So speaker can start with any magnitude and reconstruction is perfect.

        The speaker path starts with meaningful magnitude from step 0, avoiding
        the "speaker collapse to near-zero" problem seen in Run 89.
        """
        # Content path: near-identity init
        for layer in self.content_proj:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)      # identity
                nn.init.zeros_(layer.bias)
                # tiny noise to break symmetry
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)

        # Speaker decoder: Xavier init (default), NOT near-zero!
        # With the residual formulation, combined = S + (x - S) = x
        # regardless of S's magnitude → no distribution shift.
        for layer in self.speaker_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
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

    def encode_content(self, x: torch.Tensor, speaker_contribution: torch.Tensor = None) -> torch.Tensor:
        """Encode content from the RESIDUAL (x - speaker_contribution).

        Args:
            x: [B, T, hidden_dim] — raw hidden states.
            speaker_contribution: [B, T, hidden_dim] — speaker contribution to subtract.
                If None, uses x directly (for backward compat / inference without speaker).
        """
        x = self._match_dtype(x)
        if speaker_contribution is not None:
            # Subtract speaker (detached!) so content can't access speaker info
            residual = x - speaker_contribution.detach()
        else:
            residual = x
        return self.content_proj(residual)

    def speaker_diversity_loss(
        self, speaker_globals: torch.Tensor, min_norm: float = 2.0
    ) -> torch.Tensor:
        """Push in-batch speaker embeddings apart AND prevent magnitude collapse.

        Two components:
          1. Cosine diversity: penalize high pairwise cosine similarity.
          2. Norm floor: penalize if ||speaker_global|| < min_norm.

        Args:
            speaker_globals: [B, speaker_dim] — speaker embeddings from batch.
            min_norm: Minimum desired L2 norm for speaker embeddings.

        Returns:
            Scalar loss.
        """
        speaker_globals = self._match_dtype(speaker_globals)
        normed = nn.functional.normalize(speaker_globals, dim=-1)
        sim = torch.mm(normed, normed.t())  # [B, B]

        # Exclude diagonal
        B = sim.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)

        # 1. Cosine diversity
        div_loss = (sim[mask] ** 2).mean()

        # 2. Norm floor
        norms = speaker_globals.norm(dim=-1)  # [B]
        norm_loss = torch.relu(min_norm - norms).mean()

        return div_loss + norm_loss

    def forward(self, x: torch.Tensor):
        """Returns (speaker_contribution, content_emb, speaker_global).

        Residual formulation:
          speaker_contribution = decode(encode_speaker(x))
          content_emb = content_proj(x - speaker_contribution.detach())
          combined = speaker_contribution + content_emb ≈ x
        """
        x = self._match_dtype(x)
        speaker_global = self.encode_speaker(x)                       # [B, speaker_dim]
        speaker_contribution = self.decode_speaker(speaker_global, x.shape[1])  # [B, T, H]

        # RESIDUAL: content sees (x - speaker) → structurally speaker-free
        content_emb = self.encode_content(x, speaker_contribution)    # [B, T, H]

        return speaker_contribution, content_emb, speaker_global
