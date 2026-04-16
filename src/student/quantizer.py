# coding=utf-8
"""
Vector Quantizers for StudentCodec.

ContentVQ:   Per-frame VQ, 4096 codes, EMA updates (no codebook collapse)
SpeakerVQ:  Utterance-level VQ, 1024 codes, attention-pooled input

Both track:
  - Codebook utilization (% active codes)
  - Perplexity (effective codebook usage)
  - Commitment loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class EMACodebook(nn.Module):
    """VQ codebook with Exponential Moving Average updates.

    EMA-VQ is more stable than straight-through VQ:
    - No learnable codebook (codebook updated via EMA, not gradients)
    - Only commitment loss backpropagates through the encoder
    - Dead code restart: reinitialize rarely-used codes from batch

    Reference: van den Oord et al. "Neural Discrete Representation Learning" (2017)
               Dhariwal et al. "Jukebox" (EMA version)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        dead_code_threshold: int = 2,    # min usages before restarting code
        restart_usage_threshold: float = 1e-4,
    ):
        super().__init__()
        self.vocab_size          = vocab_size
        self.d_model             = d_model
        self.ema_decay           = ema_decay
        self.commitment_weight   = commitment_weight
        self.dead_code_threshold = dead_code_threshold

        # Codebook (not a parameter — updated via EMA)
        self.register_buffer("codebook",      torch.randn(vocab_size, d_model))
        self.register_buffer("ema_count",     torch.ones(vocab_size))
        self.register_buffer("ema_weight",    torch.randn(vocab_size, d_model))
        self.register_buffer("initialized",   torch.tensor(False))

        # Normalize codebook at init
        with torch.no_grad():
            self.codebook = F.normalize(self.codebook, dim=-1)
            self.ema_weight.copy_(self.codebook)

    @torch.no_grad()
    def _init_from_data(self, x_flat: torch.Tensor):
        """Initialize codebook from first batch using k-means++."""
        N = x_flat.shape[0]
        indices = torch.randperm(N, device=x_flat.device)[:self.vocab_size]
        if indices.shape[0] < self.vocab_size:
            # Repeat if batch smaller than vocab
            indices = indices.repeat(self.vocab_size // indices.shape[0] + 1)[:self.vocab_size]
        self.codebook.copy_(F.normalize(x_flat[indices], dim=-1))
        self.ema_weight.copy_(self.codebook)
        self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: [B, T, D] or [B, D] (for speaker global vector)
        Returns:
            z_q:      quantized [same shape as x]
            indices:  codebook indices [B, T] or [B]
            metrics:  dict with commitment_loss, perplexity, utilization
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.d_model)   # [N, D]

        # Initialize codebook from first batch
        if not self.initialized:
            self._init_from_data(x_flat.detach())

        # Normalize for cosine distance (more stable than L2 for VQ)
        x_norm = F.normalize(x_flat, dim=-1)
        cb_norm = F.normalize(self.codebook, dim=-1)

        # Distances: cosine = -(dot product) since normalized
        dists = -torch.mm(x_norm, cb_norm.T)    # [N, vocab]
        indices = torch.argmin(dists, dim=-1)    # [N]

        # Quantize
        z_q_flat = self.codebook[indices]        # [N, D]

        # EMA update (training only)
        if self.training:
            self._ema_update(x_flat.detach(), indices)

        # Straight-through estimator: gradients pass through to encoder
        z_q_flat = x_flat + (z_q_flat - x_flat).detach()

        # Commitment loss
        commit_loss = F.mse_loss(x_flat, z_q_flat.detach())

        # Metrics
        metrics = self._compute_metrics(indices, commit_loss)

        return z_q_flat.reshape(shape), indices.reshape(shape[:-1]), metrics

    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor):
        one_hot = F.one_hot(indices, self.vocab_size).float()     # [N, vocab]
        count_new = one_hot.sum(0)                                  # [vocab]
        weight_new = one_hot.T @ x_flat                            # [vocab, D]

        self.ema_count  = self.ema_decay * self.ema_count  + (1 - self.ema_decay) * count_new
        self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * weight_new

        # Update codebook
        n = self.ema_count.unsqueeze(1).clamp(min=1e-5)
        self.codebook = self.ema_weight / n

        # Dead code restart: replace unused codes with random vectors from batch
        dead_mask = self.ema_count < self.dead_code_threshold
        if dead_mask.any():
            n_dead = dead_mask.sum().item()
            rand_idx = torch.randint(0, x_flat.shape[0], (int(n_dead),), device=x_flat.device)
            self.codebook[dead_mask] = F.normalize(x_flat[rand_idx], dim=-1)
            self.ema_count[dead_mask] = self.dead_code_threshold
            self.ema_weight[dead_mask] = self.codebook[dead_mask]

    @torch.no_grad()
    def _compute_metrics(self, indices: torch.Tensor, commit_loss: torch.Tensor) -> dict:
        one_hot = F.one_hot(indices, self.vocab_size).float()
        avg_prob = one_hot.mean(0)
        perplexity = torch.exp(-torch.sum(avg_prob * torch.log(avg_prob + 1e-10)))
        utilization = (one_hot.sum(0) > 0).float().mean()
        return {
            "commit_loss":  commit_loss,
            "perplexity":   perplexity,
            "utilization":  utilization,
        }


class AttentionPool(nn.Module):
    """Attention pooling: [B, T, D] → [B, D].

    Learns to attend to the most speaker-relevant frames.
    Used to extract a single global speaker representation per utterance.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Linear(d_model, 1, bias=False)
        self.norm  = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x:    [B, T, D]
            mask: [B, T] bool, True = padding (ignored)
        Returns:
            pooled: [B, D]
        """
        x = self.norm(x)
        scores = self.query(x).squeeze(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)                         # [B, D]


class ContentVQ(nn.Module):
    """Per-frame content quantizer.

    Maps continuous per-frame features → discrete content codes.
    1 codebook, 4096 codes → log2(4096) = 12 bits per frame
    At 12Hz: 12 × 12 = 144 bps for content stream
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 4096,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.proj_in    = nn.Linear(d_model, d_model)
        self.codebook   = EMACodebook(vocab_size, d_model, commitment_weight=commitment_weight)
        self.proj_out   = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: [B, T, D]
        Returns:
            z_q:     quantized  [B, T, D]
            codes:   indices    [B, T]
            metrics: dict
        """
        h = self.proj_in(x)
        z_q, codes, metrics = self.codebook(h)
        out = self.proj_out(z_q)
        return out, codes, {f"content_vq/{k}": v for k, v in metrics.items()}

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: [B, T] → [B, T, D]"""
        emb = self.codebook.codebook[codes]     # [B, T, D]
        return self.proj_out(emb)


class SpeakerVQ(nn.Module):
    """Utterance-level speaker quantizer.

    Maps full sequence features → single global speaker token.
    1 codebook, 1024 codes → log2(1024) = 10 bits per utterance

    Voice conversion = swap this single token.
    """

    def __init__(
        self,
        d_model: int = 256,
        vocab_size: int = 1024,
        commitment_weight: float = 0.25,
    ):
        super().__init__()
        self.d_model    = d_model
        self.vocab_size = vocab_size
        self.attn_pool  = AttentionPool(d_model)
        self.proj_in    = nn.Linear(d_model, d_model)
        self.codebook   = EMACodebook(vocab_size, d_model, commitment_weight=commitment_weight)
        self.proj_out   = nn.Linear(d_model, d_model)

    def encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x:    [B, T, D] conformer output
            mask: [B, T] padding mask
        Returns:
            speaker_emb:  [B, D]  (to broadcast across T for conditioning)
            speaker_code: [B]     (the discrete speaker token)
            metrics: dict
        """
        pooled = self.attn_pool(x, mask)     # [B, D]
        h = self.proj_in(pooled)             # [B, D]
        z_q, codes, metrics = self.codebook(h)
        speaker_emb = self.proj_out(z_q)     # [B, D]
        return speaker_emb, codes, {f"speaker_vq/{k}": v for k, v in metrics.items()}

    def decode(self, speaker_code: torch.Tensor) -> torch.Tensor:
        """speaker_code: [B] → speaker_emb: [B, D]"""
        emb = self.codebook.codebook[speaker_code]  # [B, D]
        return self.proj_out(emb)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        return self.encode(x, mask)
