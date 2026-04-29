# coding=utf-8
"""
DisentangledVQ: 2-codebook tokenizer for voice conversion.

Content codebook: per-frame VQ (shared across speakers → strips speaker)
Speaker codebook: global VQ (pooled across time → strips content)

Voice conversion: keep source content tokens, swap speaker token → decode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantize(nn.Module):
    """Simple VQ with straight-through estimator and EMA codebook update."""

    def __init__(self, dim, codebook_size, decay=0.99, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())
        self.register_buffer("initted", torch.tensor(False))

    def _init_from_data(self, x_flat):
        """Initialize codebook from first batch of data (k-means++ style)."""
        if self.initted.item():
            return
        n = min(x_flat.shape[0], self.codebook_size)
        indices = torch.randperm(x_flat.shape[0], device=x_flat.device)[:n]
        data = x_flat[indices].detach().float()
        self.embedding.weight.data[:n] = data.to(self.embedding.weight.dtype)
        self.ema_embed_sum.data[:n] = data.to(self.ema_embed_sum.dtype)
        self.ema_cluster_size.data[:n] = 1.0
        self.initted.fill_(True)

    def forward(self, x):
        """x: [..., D] → quantized [..., D], indices [...], vq_loss scalar"""
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        # Init codebook from data on first call
        if self.training:
            self._init_from_data(x_flat)

        # Distances (always in float32 for numerical stability)
        dist = torch.cdist(x_flat, self.embedding.weight.float())
        indices = dist.argmin(dim=-1)
        quantized = self.embedding(indices).to(x.dtype)

        # EMA codebook update (all in float32, then cast back)
        if self.training:
            one_hot = F.one_hot(indices, self.codebook_size).float()
            cluster_size = one_hot.sum(0)
            embed_sum = one_hot.t() @ x_flat

            # Do EMA in float32 to avoid bf16 lerp_ mismatch
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
            self.embedding.weight.data.copy_(new_weights.to(self.embedding.weight.dtype))

        # Losses
        commitment_loss = F.mse_loss(x_flat, quantized.detach().float())
        codebook_loss = F.mse_loss(quantized.float(), x_flat.detach())

        # Straight-through
        quantized_st = x + (quantized.reshape(shape) - x).detach()
        indices = indices.reshape(shape[:-1])

        return quantized_st, indices, commitment_loss + 0.25 * codebook_loss

    def decode(self, indices):
        return self.embedding(indices)


class DisentangledVQ(nn.Module):
    """2-codebook disentangled tokenizer.

    Args:
        hidden_dim: Input dimension (1024 for Qwen).
        content_codebook_size: Number of content codes (default 1024).
        speaker_codebook_size: Number of speaker codes (default 512).
        speaker_dim: Internal speaker dimension (default 256).
        vq_warmup_steps: Steps to linearly blend VQ with continuous (warm start).
    """

    def __init__(
        self,
        hidden_dim=1024,
        content_codebook_size=1024,
        speaker_codebook_size=512,
        speaker_dim=256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim

        # Content path: project → VQ (per-frame)
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.content_vq = VectorQuantize(hidden_dim, content_codebook_size)

        # Speaker path: encode → pool → VQ (global)
        self.speaker_encoder = nn.Sequential(
            nn.Linear(hidden_dim, speaker_dim),
            nn.LayerNorm(speaker_dim),
            nn.ReLU(),
            nn.Linear(speaker_dim, speaker_dim),
            nn.ReLU(),
        )
        self.speaker_attention = nn.Linear(speaker_dim, 1)
        self.speaker_vq = VectorQuantize(speaker_dim, speaker_codebook_size)
        self.speaker_decoder = nn.Sequential(
            nn.Linear(speaker_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self._warm_start_init()

    def _warm_start_init(self):
        """Content path = near-identity, speaker decoder = near-zero."""
        for layer in self.content_proj:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)

        for layer in self.speaker_decoder:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -1e-3, 1e-3)
                nn.init.zeros_(layer.bias)

    @property
    def weight_dtype(self):
        return self.speaker_attention.weight.dtype

    def _match_dtype(self, x):
        if x.dtype != self.weight_dtype:
            return x.to(self.weight_dtype)
        return x

    def encode_speaker(self, x):
        """x: [B, T, H] → speaker_global: [B, speaker_dim]"""
        x = self._match_dtype(x)
        h = self.speaker_encoder(x)
        attn = torch.softmax(self.speaker_attention(h), dim=1)
        attn = self._match_dtype(attn)
        return (h * attn).sum(dim=1)

    def forward(self, hidden, vq_alpha=1.0):
        """
        Args:
            hidden: [B, T, H] from pre_transformer
            vq_alpha: 0→1 blend factor (0=continuous, 1=fully quantized)

        Returns:
            combined: [B, T, H] for decoder
            content_tokens: [B, T] content code indices
            speaker_tokens: [B] speaker code indices
            vq_loss: scalar VQ loss
            speaker_global: [B, speaker_dim] for logging
        """
        hidden = self._match_dtype(hidden)

        # Content path
        content_continuous = self.content_proj(hidden)  # [B, T, H]
        content_quantized, content_tokens, content_vq_loss = self.content_vq(content_continuous)

        # Blend for warm start
        content = content_continuous + vq_alpha * (content_quantized - content_continuous)

        # Speaker path
        speaker_global = self.encode_speaker(hidden)  # [B, S]
        speaker_q, speaker_tokens, speaker_vq_loss = self.speaker_vq(speaker_global)

        speaker_global_out = speaker_global + vq_alpha * (speaker_q - speaker_global)
        speaker_decoded = self.speaker_decoder(speaker_global_out)  # [B, H]
        speaker_broadcast = speaker_decoded.unsqueeze(1).expand(-1, hidden.shape[1], -1)

        # Combined
        combined = content + speaker_broadcast

        vq_loss = content_vq_loss + speaker_vq_loss

        return combined, content_tokens, speaker_tokens, vq_loss, speaker_global

    def decode_from_tokens(self, content_tokens, speaker_tokens, seq_len=None):
        """Decode from discrete tokens (for inference / voice conversion).

        Args:
            content_tokens: [B, T] content code indices
            speaker_tokens: [B] speaker code indices
        """
        content = self.content_vq.decode(content_tokens)  # [B, T, H]
        speaker = self.speaker_vq.decode(speaker_tokens)  # [B, S]
        speaker_decoded = self.speaker_decoder(speaker)  # [B, H]
        speaker_broadcast = speaker_decoded.unsqueeze(1).expand(-1, content.shape[1], -1)
        return content + speaker_broadcast
