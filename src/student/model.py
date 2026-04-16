# coding=utf-8
"""
StudentCodec: Full self-contained audio codec with disentangled speaker/content.

Architecture:
  Encoder: ConvEncoder (24kHz→12Hz) + ConformerEncoder
  Quantizer: ContentVQ (4096 codes, per-frame) + SpeakerVQ (1024 codes, per-utterance)
  Decoder: HiFiGAN-Lite (12Hz→24kHz)

Inference: No Qwen needed. Only this model.
Voice conversion: swap speaker_code integer.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .conv_encoder import ConvEncoder
from .conformer import ConformerEncoder
from .quantizer import ContentVQ, SpeakerVQ


# ─── HiFiGAN-Lite Decoder ─────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """HiFiGAN residual block with multiple dilations."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple = (1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size - 1) * d // 2
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=pad)),
                nn.LeakyReLU(0.1),
                nn.utils.weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size-1)//2)),
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = x + conv(x)
        return x


class HiFiGANLiteDecoder(nn.Module):
    """Lightweight HiFiGAN upsampler: 12Hz features → 24kHz waveform.

    Upsampling chain: [8, 5, 4, 3] × ... → total = ?
    Actually for 12Hz→24kHz: need 24000/12 = 2000× total.
    We use: [8, 5, 5, 4, 5] to get 8×5×5×4×5=4000... hmm, no.
    Let me use [5,4,4,5,5] = 2000 ✓ (matching conv encoder)

    Channels: 256 → 128 → 64 → 32 → 16 → 1
    """

    def __init__(
        self,
        d_model: int = 256,
        upsample_rates: List[int] = (5, 4, 4, 5, 5),
        upsample_channels: List[int] = (256, 128, 64, 32, 16),
        resblock_kernel_sizes: List[int] = (3, 7, 11),
        resblock_dilation_sizes: List[Tuple] = ((1,3,5), (1,3,5), (1,3,5)),
    ):
        super().__init__()
        assert len(upsample_rates) == len(upsample_channels), (
            "upsample_rates and upsample_channels must have same length"
        )

        self.pre_conv = nn.utils.weight_norm(
            nn.Conv1d(d_model, upsample_channels[0], kernel_size=7, padding=3)
        )

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        in_ch = upsample_channels[0]
        for i, (rate, out_ch) in enumerate(zip(upsample_rates, upsample_channels)):
            self.ups.append(nn.utils.weight_norm(
                nn.ConvTranspose1d(
                    in_ch, out_ch,
                    kernel_size=rate * 2,
                    stride=rate,
                    padding=rate // 2,
                )
            ))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(out_ch, k, d))
            in_ch = out_ch

        self.post_conv = nn.utils.weight_norm(
            nn.Conv1d(in_ch, 1, kernel_size=7, padding=3)
        )
        self.n_resblocks_per_up = len(resblock_kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] at 12Hz
        Returns:
            wav: [B, 1, num_samples] at 24kHz
        """
        x = x.transpose(1, 2)  # [B, d_model, T]
        x = self.pre_conv(x)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = None
            for j in range(self.n_resblocks_per_up):
                rb = self.resblocks[i * self.n_resblocks_per_up + j]
                xs = rb(x) if xs is None else xs + rb(x)
            x = xs / self.n_resblocks_per_up

        x = F.leaky_relu(x, 0.1)
        x = self.post_conv(x)
        return torch.tanh(x)  # [B, 1, S]


# ─── StudentCodec ─────────────────────────────────────────────────────────────

class StudentCodec(nn.Module):
    """
    Self-contained audio codec with disentangled speaker/content.

    Inference usage (no Qwen dependency):
        codec = StudentCodec.from_pretrained("checkpoint-dir")
        content_codes, speaker_code = codec.encode("audio.wav")
        wav = codec.decode(content_codes, speaker_code=target_speaker_code)
    """

    CONFIG_NAME = "student_config.json"
    WEIGHTS_NAME = "student_model.safetensors"

    def __init__(
        self,
        # ConvEncoder
        input_sr: int = 24_000,
        token_rate: int = 12,
        enc_strides: Tuple[int, ...] = (5, 4, 4, 5, 5),
        enc_base_channels: int = 64,
        enc_n_res: int = 2,
        # Conformer
        d_model: int = 256,
        conformer_layers: int = 6,
        conformer_heads: int = 4,
        conformer_ff_expansion: int = 4,
        conformer_conv_kernel: int = 31,
        dropout: float = 0.1,
        # Quantizers
        content_vocab_size: int = 4096,
        speaker_vocab_size: int = 1024,
        vq_commitment_weight: float = 0.25,
        # Decoder
        dec_upsample_rates: Tuple[int, ...] = (5, 4, 4, 5, 5),
        dec_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        dec_resblock_kernels: Tuple[int, ...] = (3, 7, 11),
        # Teacher projection (for distillation during training)
        teacher_hidden_dim: int = 1024,   # Qwen's actual pre_transformer output dim
    ):
        super().__init__()
        self.input_sr            = input_sr
        self.token_rate          = token_rate
        self.d_model             = d_model
        self.content_vocab_size  = content_vocab_size
        self.speaker_vocab_size  = speaker_vocab_size

        # ── Encoder ──────────────────────────────────────────────
        self.conv_encoder = ConvEncoder(
            d_model=d_model,
            input_sr=input_sr,
            token_rate=token_rate,
            strides=enc_strides,
            n_res_per_stage=enc_n_res,
            base_channels=enc_base_channels,
        )
        self.conformer = ConformerEncoder(
            d_model=d_model,
            n_layers=conformer_layers,
            n_heads=conformer_heads,
            ff_expansion=conformer_ff_expansion,
            conv_kernel=conformer_conv_kernel,
            dropout=dropout,
        )

        # ── Quantizers ───────────────────────────────────────────
        self.content_vq = ContentVQ(
            d_model=d_model,
            vocab_size=content_vocab_size,
            commitment_weight=vq_commitment_weight,
        )
        self.speaker_vq = SpeakerVQ(
            d_model=d_model,
            vocab_size=speaker_vocab_size,
            commitment_weight=vq_commitment_weight,
        )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = HiFiGANLiteDecoder(
            d_model=d_model,
            upsample_rates=list(dec_upsample_rates),
            upsample_channels=list(dec_channels),
            resblock_kernel_sizes=list(dec_resblock_kernels),
        )

        # ── Distillation projection (training only) ───────────────
        # Projects student d_model → teacher hidden_dim for feature KD loss
        self.teacher_proj = nn.Linear(d_model, teacher_hidden_dim)

    def encode_features(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Raw audio → continuous features [B, T, d_model]."""
        x = self.conv_encoder(audio)          # [B, T, d_model]
        x = self.conformer(x)                 # [B, T, d_model]
        return x

    def quantize(
        self,
        features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Features → discrete codes + quantized embeddings.

        Returns:
            content_emb:   [B, T, d_model]
            speaker_emb:   [B, d_model]     (global, 1 per utterance)
            content_codes: [B, T]           (discrete content tokens)
            speaker_code:  [B]              (discrete speaker token)
            metrics:       dict for logging
        """
        content_emb, content_codes, c_metrics = self.content_vq(features)
        speaker_emb, speaker_code, s_metrics  = self.speaker_vq(features, padding_mask)
        return content_emb, speaker_emb, content_codes, speaker_code, {**c_metrics, **s_metrics}

    def decode(
        self,
        content_emb: torch.Tensor,
        speaker_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Decode content + speaker embeddings → waveform.

        Args:
            content_emb: [B, T, d_model]
            speaker_emb: [B, d_model]      (will be broadcast to T)
        Returns:
            wav: [B, num_samples]
        """
        speaker_expanded = speaker_emb.unsqueeze(1).expand_as(content_emb)
        combined = content_emb + speaker_expanded          # [B, T, d_model]
        wav = self.decoder(combined)                       # [B, 1, S]
        return wav.squeeze(1).clamp(-1, 1)                 # [B, S]

    def forward(
        self,
        audio: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Full forward pass (training).

        Returns dict with:
          wav:              reconstructed waveform [B, S]
          features:         conformer output [B, T, d_model]  (for distill loss)
          teacher_proj:     projected features [B, T, teacher_hidden_dim]
          content_codes:    [B, T]
          speaker_code:     [B]
          vq_metrics:       dict for logging
        """
        features = self.encode_features(audio, padding_mask)

        content_emb, speaker_emb, content_codes, speaker_code, vq_metrics = \
            self.quantize(features, padding_mask)

        wav = self.decode(content_emb, speaker_emb)

        return {
            "wav":           wav,
            "features":      features,
            "teacher_proj":  self.teacher_proj(features),
            "content_codes": content_codes,
            "speaker_code":  speaker_code,
            "vq_metrics":    vq_metrics,
        }

    # ── Convenience inference methods ────────────────────────────

    @torch.inference_mode()
    def encode_audio(
        self,
        audio: Union[torch.Tensor, "np.ndarray"],  # noqa: F821
        sr: int = 24_000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """audio → (content_codes [T], speaker_code int)"""
        import numpy as np
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if sr != self.input_sr:
            import torchaudio
            audio = torchaudio.functional.resample(audio, sr, self.input_sr)

        features = self.encode_features(audio)
        _, _, content_codes, speaker_code, _ = self.quantize(features)
        return content_codes.squeeze(0), speaker_code.squeeze(0)

    @torch.inference_mode()
    def decode_codes(
        self,
        content_codes: torch.Tensor,
        speaker_code: Union[torch.Tensor, int],
    ) -> torch.Tensor:
        """content_codes [T] + speaker_code → wav [S]"""
        if isinstance(speaker_code, int):
            speaker_code = torch.tensor([speaker_code], device=content_codes.device)

        content_emb = self.content_vq.decode(content_codes.unsqueeze(0))
        speaker_emb = self.speaker_vq.decode(speaker_code.unsqueeze(0) if speaker_code.dim() == 0 else speaker_code)
        return self.decode(content_emb, speaker_emb).squeeze(0)

    def info(self) -> dict:
        """Full model info for logging."""
        n_total    = sum(p.numel() for p in self.parameters())
        n_enc      = sum(p.numel() for p in self.conv_encoder.parameters())
        n_conf     = sum(p.numel() for p in self.conformer.parameters())
        n_content  = sum(p.numel() for p in self.content_vq.parameters())
        n_speaker  = sum(p.numel() for p in self.speaker_vq.parameters())
        n_dec      = sum(p.numel() for p in self.decoder.parameters())
        return {
            "model/total_params_M":    n_total   / 1e6,
            "model/enc_conv_params_M": n_enc     / 1e6,
            "model/enc_conf_params_M": n_conf    / 1e6,
            "model/content_vq_params_M": n_content / 1e6,
            "model/speaker_vq_params_M": n_speaker / 1e6,
            "model/decoder_params_M":  n_dec     / 1e6,
            "model/content_vocab_size": self.content_vocab_size,
            "model/speaker_vocab_size": self.speaker_vocab_size,
            "model/token_rate_hz":     self.token_rate,
            "model/input_sr":          self.input_sr,
        }

    # ── Checkpoint I/O ───────────────────────────────────────────

    def save_pretrained(self, save_dir: str):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Config
        config = {
            "input_sr": self.input_sr, "token_rate": self.token_rate,
            "d_model": self.d_model,
            "content_vocab_size": self.content_vocab_size,
            "speaker_vocab_size": self.speaker_vocab_size,
        }
        with open(save_dir / self.CONFIG_NAME, "w") as f:
            json.dump(config, f, indent=2)

        # Weights
        save_file(self.state_dict(), str(save_dir / self.WEIGHTS_NAME))
        print(f"Saved StudentCodec to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir: str, device: str = "cpu") -> "StudentCodec":
        load_dir = Path(load_dir)
        with open(load_dir / cls.CONFIG_NAME) as f:
            config = json.load(f)
        model = cls(**config)
        weights = load_file(str(load_dir / cls.WEIGHTS_NAME), device=device)
        model.load_state_dict(weights, strict=True)
        return model.eval()
