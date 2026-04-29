# coding=utf-8
"""
SingleCodebookVQ: Merge 16 RVQ codebooks into ONE discrete token per frame.

Experiment 95: Pure reconstruction — no speaker disentanglement.

Architecture:
  hidden [B, T, 1024]  (from frozen pre_transformer)
    → LayerNorm + MLP(1024 → vq_dim)         [pre_vq_proj]
    → VectorQuantize(vq_dim, codebook_size)   [single large codebook]
    → MLP(vq_dim → 1024) + LayerNorm         [post_vq_proj]
    → quantized [B, T, 1024]                 [decoder input]

Key design choices:
  - Project to lower dim (256) for VQ: avoids curse of dimensionality.
  - Large codebook (16384): compensates for losing 16 codebooks.
  - EMA codebook updates + dead code reset: prevents codebook collapse.
  - Entropy regularization: maximizes codebook utilization.
  - Alpha warmup: smooth transition from continuous → quantized.
  - Residual warmup: output = hidden + α*(VQ_output - hidden)
    so at α=0 the decoder sees the original hidden (perfect recon).

Token rate: 12.5 Hz (one token per frame).
Bitrate: log2(16384) * 12.5 = 175 bits/sec (vs. 16*10*12.5 = 2000 for RVQ-16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizeEnhanced(nn.Module):
    """Enhanced VQ with EMA, dead code reset, and usage tracking."""

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5,
                 threshold_dead_frames=2):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps
        self.threshold_dead_frames = threshold_dead_frames

        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size,
                         1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum",
                             self.embedding.weight.data.clone())
        self.register_buffer("initted", torch.tensor(False))

        # Track frames since each code was last used (for dead code reset)
        self.register_buffer("frames_since_used",
                             torch.zeros(codebook_size, dtype=torch.long))

    def _init_from_data(self, x_flat):
        """Initialize codebook from first batch (k-means++ style)."""
        if self.initted.item():
            return
        n = min(x_flat.shape[0], self.codebook_size)
        indices = torch.randperm(x_flat.shape[0], device=x_flat.device)[:n]
        data = x_flat[indices].detach().float()
        self.embedding.weight.data[:n] = data.to(self.embedding.weight.dtype)
        self.ema_embed_sum.data[:n] = data.to(self.ema_embed_sum.dtype)
        self.ema_cluster_size.data[:n] = 1.0
        self.initted.fill_(True)

    def _reset_dead_codes(self, x_flat):
        """Reset codes that haven't been used for threshold_dead_frames steps."""
        dead_mask = self.frames_since_used >= self.threshold_dead_frames
        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return 0

        # Pick random vectors from the current batch to replace dead codes
        n_replace = min(num_dead, x_flat.shape[0])
        if n_replace == 0:
            return 0

        replace_indices = torch.randperm(
            x_flat.shape[0], device=x_flat.device
        )[:n_replace]
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:n_replace]

        replacement = x_flat[replace_indices].detach()
        self.embedding.weight.data[dead_indices] = replacement.to(
            self.embedding.weight.dtype
        )
        self.ema_embed_sum.data[dead_indices] = replacement.to(
            self.ema_embed_sum.dtype
        )
        self.ema_cluster_size.data[dead_indices] = 1.0
        self.frames_since_used[dead_indices] = 0

        return n_replace

    def forward(self, x):
        """x: [..., D] → quantized [..., D], indices [...], vq_loss, metrics"""
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        # Init codebook from data on first call
        if self.training:
            self._init_from_data(x_flat)

        # L2 normalized distances for stability
        # dist(a,b) = ||a||^2 + ||b||^2 - 2*a·b
        cb = self.embedding.weight.float()
        dist = (
            x_flat.pow(2).sum(1, keepdim=True)
            + cb.pow(2).sum(1, keepdim=True).t()
            - 2 * x_flat @ cb.t()
        )
        indices = dist.argmin(dim=-1)
        quantized = self.embedding(indices).to(x.dtype)

        # Compute metrics
        with torch.no_grad():
            unique_codes = indices.unique().numel()
            usage_frac = unique_codes / self.codebook_size

            # Perplexity: exp(entropy) of the assignment distribution
            one_hot_counts = F.one_hot(
                indices, self.codebook_size
            ).float().sum(0)
            probs = one_hot_counts / one_hot_counts.sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            perplexity = torch.exp(entropy)

        # EMA codebook update
        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(0)
            embed_sum = (one_hot.t() @ x_flat).float()

            ema_cs = self.ema_cluster_size.float()
            ema_es = self.ema_embed_sum.float()
            ema_cs.lerp_(cluster_size, 1 - self.decay)
            ema_es.lerp_(embed_sum, 1 - self.decay)
            self.ema_cluster_size.copy_(ema_cs)
            self.ema_embed_sum.copy_(ema_es)

            n = ema_cs.sum()
            cluster_size_smoothed = (
                (ema_cs + self.eps) / (n + self.codebook_size * self.eps) * n
            )
            new_weights = ema_es / cluster_size_smoothed.unsqueeze(1)
            self.embedding.weight.data.copy_(
                new_weights.to(self.embedding.weight.dtype)
            )

            # Update dead code tracking
            used_mask = cluster_size > 0
            self.frames_since_used[used_mask] = 0
            self.frames_since_used[~used_mask] += 1

            # Reset dead codes periodically
            num_reset = self._reset_dead_codes(x_flat)

        # Losses
        commitment_loss = F.mse_loss(x_flat, quantized.detach().float())
        codebook_loss = F.mse_loss(quantized.float(), x_flat.detach())

        # Straight-through estimator
        quantized_st = x + (quantized.reshape(shape) - x).detach()
        indices = indices.reshape(shape[:-1])

        metrics = {
            "codebook_usage": usage_frac,
            "unique_codes": unique_codes,
            "perplexity": perplexity.item(),
            "dead_codes": (
                (self.frames_since_used >= self.threshold_dead_frames)
                .sum().item()
            ),
        }

        return (
            quantized_st,
            indices,
            commitment_loss + 0.25 * codebook_loss,
            metrics,
        )

    def decode(self, indices):
        return self.embedding(indices)


class SingleCodebookVQ(nn.Module):
    """Single codebook VQ — Experiment 95.

    Merges all 16 RVQ codebooks into ONE token per frame.
    Pure reconstruction, no speaker disentanglement.

    Args:
        hidden_dim: Input dimension from pre_transformer (1024 for Qwen).
        vq_dim: Quantization space dimension (default 256).
        codebook_size: Number of codes (default 16384).
        decay: EMA decay for codebook updates.
        commitment_weight: Weight for commitment loss component.
        entropy_weight: Weight for entropy regularization (codebook usage).
    """

    def __init__(
        self,
        hidden_dim=1024,
        vq_dim=256,
        codebook_size=16384,
        decay=0.99,
        commitment_weight=1.0,
        entropy_weight=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vq_dim = vq_dim
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.entropy_weight = entropy_weight

        # Pre-VQ: project 1024 → vq_dim
        self.pre_vq = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vq_dim),
        )

        # VQ layer
        self.vq = VectorQuantizeEnhanced(
            vq_dim, codebook_size, decay=decay
        )

        # Post-VQ: project vq_dim → 1024
        self.post_vq = nn.Sequential(
            nn.Linear(vq_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Initialize so that post_vq(pre_vq(x)) ≈ 0.

        Combined with residual warmup: output = hidden + α * VQ_path
        At α=0: output = hidden → perfect reconstruction from step 0.
        """
        # Post-VQ: near-zero output at init
        for layer in self.post_vq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.LayerNorm):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    @property
    def weight_dtype(self):
        return next(self.pre_vq.parameters()).dtype

    def _match_dtype(self, x):
        if x.dtype != self.weight_dtype:
            return x.to(self.weight_dtype)
        return x

    def forward(self, hidden, vq_alpha=1.0):
        """
        Args:
            hidden: [B, T, 1024] from pre_transformer (frozen).
            vq_alpha: 0→1 blend factor (0=bypass VQ, 1=fully quantized).

        Returns:
            output: [B, T, 1024] for decoder.
            tokens: [B, T] discrete token indices.
            vq_loss: scalar VQ loss.
            metrics: dict with codebook usage stats.
        """
        hidden = self._match_dtype(hidden)

        # Project to VQ space
        z = self.pre_vq(hidden)  # [B, T, vq_dim]

        # Quantize
        z_q, tokens, vq_loss, metrics = self.vq(z)

        # Blend in VQ space: smooth transition continuous → quantized
        z_out = z + vq_alpha * (z_q - z)

        # Project back to hidden dim
        vq_decoded = self.post_vq(z_out)  # [B, T, 1024]

        # Residual warmup: output = hidden + α * (vq_decoded - hidden)
        # At α=0: output = hidden (decoder gets original → perfect recon)
        # At α=1: output = vq_decoded (decoder gets VQ output)
        output = hidden + vq_alpha * (vq_decoded - hidden)

        # Entropy regularization: maximize codebook usage
        # Computed from soft distance distribution
        if self.training and self.entropy_weight > 0:
            z_flat = z.reshape(-1, self.vq_dim).float()
            cb = self.vq.embedding.weight.float()
            dist = (
                z_flat.pow(2).sum(1, keepdim=True)
                + cb.pow(2).sum(1, keepdim=True).t()
                - 2 * z_flat @ cb.t()
            )
            # Soft assignment probabilities
            soft_probs = F.softmax(-dist * 10.0, dim=-1)
            avg_probs = soft_probs.mean(0)
            entropy = -(avg_probs * torch.log(avg_probs + 1e-10)).sum()
            max_entropy = torch.log(
                torch.tensor(
                    float(self.codebook_size), device=hidden.device
                )
            )
            # Normalize: 0 = no entropy, 1 = max entropy (uniform)
            entropy_loss = (max_entropy - entropy) / max_entropy
            vq_loss = (
                self.commitment_weight * vq_loss
                + self.entropy_weight * entropy_loss
            )
            metrics["entropy_loss"] = entropy_loss.item()
            metrics["entropy_ratio"] = (entropy / max_entropy).item()
        else:
            vq_loss = self.commitment_weight * vq_loss

        return output, tokens, vq_loss, metrics

    def decode_tokens(self, tokens):
        """Decode from discrete tokens for inference.

        Args:
            tokens: [B, T] discrete token indices.

        Returns:
            output: [B, T, 1024] hidden states for decoder.
        """
        z_q = self.vq.decode(tokens)  # [B, T, vq_dim]
        return self.post_vq(z_q)      # [B, T, 1024]
