# coding=utf-8
"""
SingleCodebookVQ: Merge 16 RVQ codebooks into ONE discrete token per frame.

Experiment 95: Pure reconstruction — no speaker disentanglement.

Architecture (follows the PROVEN pattern from Exp 81's content VQ):
  hidden [B, T, 1024]  (from frozen pre_transformer)
    → pre_vq_proj (near-identity MLP, 1024 → 1024)
    → VectorQuantize(1024-dim, 8192 codes)
    → quantized [B, T, 1024]  (directly to decoder, NO post-projection!)

Key differences from the broken v1:
  1. SAME dimension as hidden (1024) — no information-destroying projection
  2. Near-identity init for pre_vq — decoder sees good input from step 0
  3. SINGLE alpha blend — no double-alpha that zeroed out gradients
  4. NO post_vq projection — VQ output goes directly to decoder
  5. Proven VQ class from exp 81 with EMA + data init

Token rate: 12.5 Hz (one token per frame).
Bitrate: log2(8192) * 12.5 = 162.5 bits/sec (vs. 2000 for RVQ-16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizeEnhanced(nn.Module):
    """VQ with EMA codebook updates, dead code reset, and usage tracking.

    Based on the proven VectorQuantize from disentangle_vq.py (exp 81),
    enhanced with dead code reset and usage metrics.
    """

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        self.register_buffer("initted", torch.tensor(False))

    def _init_from_data(self, x_flat):
        """Initialize codebook from first batch of data (k-means++ style).
        If batch is smaller than codebook, tile the batch with noise to fill it.
        """
        if self.initted.item():
            return
            
        data = x_flat.detach().float()
        n_data = data.shape[0]
        
        if n_data >= self.codebook_size:
            # Batch is large enough, just pick random subset
            indices = torch.randperm(n_data, device=x_flat.device)[:self.codebook_size]
            init_data = data[indices]
        else:
            # Batch is too small, tile it with noise
            repeats = (self.codebook_size // n_data) + 1
            tiled_data = data.repeat(repeats, 1)
            # Add small noise to break symmetry
            noise = torch.randn_like(tiled_data) * 0.01 * data.std(dim=0, keepdim=True)
            tiled_data = tiled_data + noise
            # Take exactly codebook_size elements
            indices = torch.randperm(tiled_data.shape[0], device=x_flat.device)[:self.codebook_size]
            init_data = tiled_data[indices]
            
        self.embedding.weight.data.copy_(init_data.to(self.embedding.weight.dtype))
        self.initted.fill_(True)

    def forward(self, x):
        """x: [..., D] → quantized [..., D], indices [...], vq_loss, metrics"""
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        # Init codebook from data on first call
        if self.training:
            self._init_from_data(x_flat)

        # Distances (always in float32 for numerical stability)
        dist = torch.cdist(x_flat, self.embedding.weight.float())
        indices = dist.argmin(dim=-1)
        quantized = self.embedding(indices).to(x.dtype)

        # Compute metrics
        with torch.no_grad():
            unique_codes = indices.unique().numel()
            usage_frac = unique_codes / self.codebook_size

            one_hot_counts = F.one_hot(
                indices, self.codebook_size
            ).float().sum(0)
            probs = one_hot_counts / one_hot_counts.sum()
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            perplexity = torch.exp(entropy)

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
    """Single codebook VQ — Experiment 95 (fixed).

    Follows the EXACT proven pattern from Exp 81's content VQ:
      1. Near-identity pre-projection (1024 → 1024)
      2. VQ in the SAME dimension (1024-dim, NO dimension reduction)
      3. Single alpha blend (NO double alpha)
      4. VQ output goes DIRECTLY to decoder (NO post-projection)

    This is essentially exp 81's content path with a larger codebook
    and no speaker path.

    Args:
        hidden_dim: Input dimension from pre_transformer (1024 for Qwen).
        codebook_size: Number of codes (default 8192).
        decay: EMA decay for codebook updates.
    """

    def __init__(
        self,
        hidden_dim=1024,
        vq_dim=256,      # IGNORED — kept for CLI compat, always uses hidden_dim
        codebook_size=8192,
        decay=0.99,
        commitment_weight=1.0,  # kept for compat
        entropy_weight=0.1,     # kept for compat
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.codebook_size = codebook_size

        # Pre-VQ: near-identity projection in SAME dimension (1024 → 1024)
        # Exactly like exp 81's content_proj
        self.pre_vq = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # VQ in full 1024-dim — NO dimension reduction!
        self.vq = VectorQuantizeEnhanced(
            hidden_dim, codebook_size, decay=decay
        )

        # Near-identity init (same as exp 81's content_proj)
        self._warm_start_init()

    def _warm_start_init(self):
        """Initialize pre_vq as near-identity.

        This is the KEY insight from exp 81:
          pre_vq(x) ≈ x at init → VQ input ≈ hidden
          → codebook initializes from meaningful data
          → decoder sees good input from step 0
          → smooth transition to quantized output via alpha
        """
        for layer in self.pre_vq:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)

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
            vq_alpha: 0→1 blend factor (0=continuous, 1=fully quantized).

        Returns:
            output: [B, T, 1024] for decoder.
            tokens: [B, T] discrete token indices.
            vq_loss: scalar VQ loss.
            metrics: dict with codebook usage stats.
        """
        hidden = self._match_dtype(hidden)

        # Near-identity projection (≈ hidden at init)
        z = self.pre_vq(hidden)  # [B, T, 1024]

        # Quantize
        z_q, tokens, vq_loss, metrics = self.vq(z)

        # SINGLE alpha blend — exactly like exp 81
        # At α=0: output = z ≈ hidden (continuous, near-identity)
        # At α=1: output = z_q (fully quantized)
        # NO second alpha, NO post-projection
        output = z + vq_alpha * (z_q - z)

        return output, tokens, vq_loss, metrics

    def decode_tokens(self, tokens):
        """Decode from discrete tokens for inference.

        Args:
            tokens: [B, T] discrete token indices.

        Returns:
            output: [B, T, 1024] hidden states for decoder.
        """
        # Direct codebook lookup — NO post-projection needed
        return self.vq.decode(tokens)  # [B, T, 1024]
