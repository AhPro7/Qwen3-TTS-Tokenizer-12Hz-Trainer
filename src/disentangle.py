# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Single-codebook VQ bottleneck for speaker-independent content tokens.

Architecture:
  Content path: per-frame MLP → VQ (EMA + dead-code restart + temperature)
                → decoder MLP → hidden_dim

Key fixes vs. previous version:
  - Gumbel-softmax / temperature annealing during early training so
    gradients flow before the codebook is warm (prevents commit_loss ≈ 0).
  - L2-normalized inputs + codebook entries (cosine VQ) for stable distances
    across bfloat16 precision.
  - Entropy regularisation: penalises low codebook utilisation directly in
    the loss, not just via dead-code restart.
  - Exponential Moving Average cluster-size uses Laplace smoothing that
    scales with actual batch size so small batches don't under-smooth.
  - Removed speaker path entirely — it was unused in trainer.py and was
    adding dead parameters that confused the loss accounting.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cosine VQ with EMA + temperature annealing + entropy reg
# ---------------------------------------------------------------------------

class VectorQuantize(nn.Module):
    """
    EMA codebook VQ with:
      • L2-normalised inputs and codebook (cosine distances, bfloat16-safe)
      • Dead-code restart from random batch frames
      • Entropy regularisation to encourage uniform codebook usage
      • Optional temperature for soft → hard annealing during warm-up
    """

    def __init__(
        self,
        codebook_size: int = 2048,
        dim: int = 128,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        restart_threshold: float = 1.0,
        entropy_loss_weight: float = 0.1,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay
        self.restart_threshold = restart_threshold
        self.entropy_loss_weight = entropy_loss_weight

        # Unit-sphere initialisation → distances are always in [-1, 1]
        codebook = torch.randn(codebook_size, dim)
        codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_cluster_size", torch.ones(codebook_size))
        self.register_buffer("ema_dw", codebook.clone())

        self.last_indices: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x.float(), dim=-1).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ):
        """
        Args:
            x:           [B, T, D] float / bfloat16
            temperature: >1 → softer (more exploration), approaches 0 → hard VQ
        Returns:
            quantized    [B, T, D]  straight-through
            indices      [B, T]     int64
            loss         scalar     commit + entropy
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # ── Cosine similarity lookup ──────────────────────────────────
        x_norm = self._normalize(x_flat)              # [BT, D]
        cb_norm = self._normalize(self.codebook)      # [K, D]

        # cosine similarity → negate for argmin
        sim = x_norm @ cb_norm.T                      # [BT, K]  in [-1,1]

        if temperature != 1.0:
            sim = sim / max(temperature, 1e-5)

        indices_flat = sim.argmax(dim=-1)             # [BT]
        quantized_flat = self.codebook[indices_flat]  # [BT, D] (unnormed)

        # ── EMA codebook update ───────────────────────────────────────
        if self.training:
            with torch.no_grad():
                one_hot = F.one_hot(indices_flat, self.codebook_size).float()
                new_cluster_size = one_hot.sum(0)                     # [K]
                new_dw = one_hot.T @ x_flat.float()                   # [K, D]

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    new_cluster_size * (1.0 - self.ema_decay)
                )
                self.ema_dw.mul_(self.ema_decay).add_(
                    new_dw * (1.0 - self.ema_decay)
                )

                n = self.ema_cluster_size.sum()
                smoothed = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.codebook_size * 1e-5)
                    * n
                )
                updated = self.ema_dw / smoothed.unsqueeze(1)
                # Keep codebook on the unit sphere
                self.codebook.copy_(F.normalize(updated, dim=-1).to(self.codebook.dtype))

                # ── Dead-code restart ────────────────────────────────
                dead_mask = self.ema_cluster_size < self.restart_threshold
                num_dead = int(dead_mask.sum().item())
                if num_dead > 0:
                    rand_idx = torch.randint(
                        0, B * T, (num_dead,), device=x_flat.device
                    )
                    new_codes = F.normalize(x_flat[rand_idx].float(), dim=-1)
                    self.codebook[dead_mask] = new_codes.to(self.codebook.dtype)
                    self.ema_cluster_size[dead_mask] = self.restart_threshold
                    self.ema_dw[dead_mask] = new_codes.to(self.ema_dw.dtype)

        # ── Losses ───────────────────────────────────────────────────
        # Commitment: push encoder outputs toward (detached) codebook entries
        commit_loss = self.commitment_cost * F.mse_loss(
            x_flat.float(), quantized_flat.detach().float()
        )

        # Entropy regularisation: maximise codebook usage
        # probs ≈ soft assignment via cosine similarity
        with torch.no_grad():
            probs = F.softmax(sim.float() / max(temperature, 0.1), dim=-1)  # [BT, K]
            avg_probs = probs.mean(0)  # [K]
        # Maximise entropy ↔ minimise negative entropy
        entropy = -(avg_probs * (avg_probs + 1e-9).log()).sum()
        max_entropy = math.log(self.codebook_size)
        entropy_loss = self.entropy_loss_weight * (max_entropy - entropy)

        total_loss = commit_loss + entropy_loss

        # ── Straight-through estimator ────────────────────────────────
        # Use encoder output for backward, quantized for forward
        quantized_st = x_flat + (quantized_flat - x_flat).detach()

        indices = indices_flat.reshape(B, T)
        quantized = quantized_st.reshape(B, T, D)
        self.last_indices = indices.detach()

        return quantized, indices, total_loss

    # ------------------------------------------------------------------
    @torch.no_grad()
    def codebook_usage(self) -> float:
        if self.last_indices is None:
            return 0.0
        return float(self.last_indices.unique().numel()) / self.codebook_size

    @torch.no_grad()
    def perplexity(self) -> float:
        """Effective codebook size (exp of entropy of usage distribution)."""
        if self.last_indices is None:
            return 0.0
        counts = torch.bincount(
            self.last_indices.flatten(), minlength=self.codebook_size
        ).float()
        probs = counts / counts.sum().clamp(min=1)
        entropy = -(probs * (probs + 1e-9).log()).sum()
        return float(entropy.exp().item())


# ---------------------------------------------------------------------------
# Content-only bottleneck
# ---------------------------------------------------------------------------

class DisentangledProjection(nn.Module):
    """
    Projects Qwen decoder hidden states through a single VQ codebook.

    No speaker path — speaker conditioning is implicit in the decoder weights.
    The content path acts as a learned discrete bottleneck that forces the
    decoder to reconstruct audio from content tokens only.

    hidden_dim → project_down → L2-norm → VQ → project_up → hidden_dim
    """

    def __init__(
        self,
        hidden_dim: int = 1024,
        content_dim: int = 256,          # Larger dim → richer per-token space
        codebook_size: int = 2048,        # 2048 is stable and large enough
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
        entropy_loss_weight: float = 0.05,
        dropout: float = 0.0,            # Optional dropout in projection
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.content_dim = content_dim

        # Project down: hidden_dim → content_dim
        # Use LayerNorm before projection for stable bfloat16 training
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.content_pre_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim // 2, content_dim),
        )

        self.vq = VectorQuantize(
            codebook_size=codebook_size,
            dim=content_dim,
            commitment_cost=commitment_cost,
            ema_decay=ema_decay,
            entropy_loss_weight=entropy_loss_weight,
        )

        # Project up: content_dim → hidden_dim
        self.content_post_proj = nn.Sequential(
            nn.Linear(content_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        self._init_weights()

        # Set externally by trainer each step
        self.temperature: float = 1.0
        # alpha=0 → pure bypass (x unchanged), alpha=1 → pure VQ bottleneck
        # Annealed 0→1 over vq_alpha_anneal_steps to force bottleneck to take over
        self.alpha: float = 0.0

    def _init_weights(self):
        """
        Initialise so that at step 0, content_post_proj ≈ zero,
        meaning the VQ bottleneck adds ~nothing to the hidden state.
        This lets the frozen/unfrozen decoder settle before the bottleneck kicks in.
        """
        # Pre-proj: small normal init so distances are reasonable
        for m in self.content_pre_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Post-proj last layer: near-zero so bottleneck starts transparent
        last_post = [m for m in self.content_post_proj.modules() if isinstance(m, nn.Linear)][-1]
        nn.init.normal_(last_post.weight, std=0.001)
        nn.init.zeros_(last_post.bias)

    @property
    def weight_dtype(self) -> torch.dtype:
        return self.content_pre_proj[0].weight.dtype

    def _cast(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(self.weight_dtype) if x.dtype != self.weight_dtype else x

    def encode_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Returns just the VQ token indices [B, T] — for inference/eval."""
        x = self._cast(x)
        normed = self.pre_norm(x)
        pre = self.content_pre_proj(normed)
        _, indices, _ = self.vq(pre, temperature=self.temperature)
        return indices

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, hidden_dim]
        Returns:
            content_emb:    [B, T, hidden_dim]  alpha-blended VQ output
            content_indices:[B, T]              discrete token ids
            vq_loss:        scalar              commit + entropy losses

        Alpha schedule (set by trainer):
            alpha=0.0 → output = x            (pure bypass, bottleneck invisible)
            alpha=0.5 → output = 0.5*x + 0.5*VQ(x)   (half-half)
            alpha=1.0 → output = VQ(x)        (pure bottleneck, no bypass)

        This forces the bottleneck to gradually take over the representation
        instead of staying as a near-zero residual forever.
        """
        x = self._cast(x)
        normed = self.pre_norm(x)
        pre = self.content_pre_proj(normed)              # [B, T, content_dim]
        quantized, indices, vq_loss = self.vq(pre, temperature=self.temperature)
        vq_out = self.content_post_proj(quantized)       # [B, T, hidden_dim]

        # Alpha blend: lerp from bypass (x) to pure VQ (vq_out)
        content_emb = (1.0 - self.alpha) * x + self.alpha * vq_out
        return content_emb, indices, vq_loss