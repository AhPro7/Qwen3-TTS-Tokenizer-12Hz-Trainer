# coding=utf-8
"""
Distillation + audio losses for StudentCodec training.

Losses:
  L_feat      — Feature distillation: MSE(student_proj, teacher_hidden)
  L_acoustic  — Acoustic distillation: MelLoss(student_wav, teacher_wav)
  L_mel       — Ground truth mel: MelLoss(student_wav, target_wav)
  L_commit    — VQ commitment (from quantizers)
  L_adv       — GAN adversarial (phase 2)
  L_fm        — Feature matching (phase 2)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# ─── Multi-Resolution Mel Loss ────────────────────────────────────────────────

class MelSpectrogramLoss(nn.Module):
    """Single-scale mel spectrogram L1 loss."""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        power: float = 1.0,
    ):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max or sample_rate // 2,
            power=power,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target: [B, S] waveforms"""
        self.mel = self.mel.to(pred.device)
        pred_mel   = self.mel(pred).clamp(min=1e-5).log()
        target_mel = self.mel(target).clamp(min=1e-5).log()
        return F.l1_loss(pred_mel, target_mel)


class MultiResolutionMelLoss(nn.Module):
    """Multi-scale mel loss covering 7 frequency resolutions.
    
    Covers coarse (prosody) to fine (timbre) acoustic features.
    """

    SCALES = [
        # (n_fft, hop_length, n_mels)
        (2048, 512,  128),   # 46ms — prosody, intonation
        (1024, 256,  80),    # 23ms — phoneme-level
        (512,  128,  64),    # 12ms — fine phonetics
        (256,  64,   32),    # 6ms  — pitch fine structure
        (128,  32,   16),    # 3ms  — glottal pulse
        (64,   16,   8),     # 1.5ms — very fine
        (32,   8,    4),     # 0.7ms — noise floor
    ]

    def __init__(self, sample_rate: int = 24_000):
        super().__init__()
        self.losses = nn.ModuleList([
            MelSpectrogramLoss(sample_rate, n_fft, hop, n_mels)
            for n_fft, hop, n_mels in self.SCALES
        ])

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns mean across all scales."""
        return torch.stack([loss(pred, target) for loss in self.losses]).mean()


# ─── Feature Distillation Loss ────────────────────────────────────────────────

def feature_distillation_loss(
    student_proj: torch.Tensor,    # [B, T_s, teacher_dim]
    teacher_hidden: torch.Tensor,  # [B, T_t, teacher_dim]
) -> torch.Tensor:
    """MSE between student projected features and teacher hidden states.
    
    Handles temporal length mismatch by adaptive average pooling.
    This teaches the student to build the same internal representations
    as the teacher, without needing the teacher at inference.
    """
    T_s = student_proj.shape[1]
    T_t = teacher_hidden.shape[1]

    if T_s != T_t:
        # Align lengths via interpolation
        student_proj = student_proj.permute(0, 2, 1)  # [B, D, T_s]
        student_proj = F.interpolate(student_proj, size=T_t, mode="linear", align_corners=False)
        student_proj = student_proj.permute(0, 2, 1)  # [B, T_t, D]

    return F.mse_loss(student_proj, teacher_hidden.detach())


# ─── VQ Losses ────────────────────────────────────────────────────────────────

def vq_loss(vq_metrics: dict) -> Tuple[torch.Tensor, dict]:
    """Aggregate commitment losses from content + speaker VQs.
    
    Returns combined scalar loss + per-component dict for logging.
    """
    content_commit = vq_metrics.get("content_vq/commit_loss", torch.tensor(0.0))
    speaker_commit = vq_metrics.get("speaker_vq/commit_loss", torch.tensor(0.0))
    total = content_commit + speaker_commit

    log = {
        "vq/content_commit":     content_commit.item() if torch.is_tensor(content_commit) else content_commit,
        "vq/speaker_commit":     speaker_commit.item() if torch.is_tensor(speaker_commit) else speaker_commit,
        "vq/content_perplexity": vq_metrics.get("content_vq/perplexity", torch.tensor(0.0)),
        "vq/speaker_perplexity": vq_metrics.get("speaker_vq/perplexity", torch.tensor(0.0)),
        "vq/content_utilization": vq_metrics.get("content_vq/utilization", torch.tensor(0.0)),
        "vq/speaker_utilization": vq_metrics.get("speaker_vq/utilization", torch.tensor(0.0)),
    }
    return total, log


# ─── GAN Losses ───────────────────────────────────────────────────────────────

def discriminator_loss(
    real_outputs: List[List[torch.Tensor]],
    fake_outputs: List[List[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LS-GAN discriminator loss (Mao et al. 2017)."""
    loss_real = loss_fake = 0.0
    for dr, dg in zip(real_outputs, fake_outputs):
        r, g = dr[-1], dg[-1]
        loss_real = loss_real + F.mse_loss(r, torch.ones_like(r))
        loss_fake = loss_fake + F.mse_loss(g, torch.zeros_like(g))
    total = loss_real + loss_fake
    return total, loss_real, loss_fake


def generator_adversarial_loss(
    fake_outputs: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Generator loss: fool discriminator."""
    loss = 0.0
    for dg in fake_outputs:
        g = dg[-1]
        loss = loss + F.mse_loss(g, torch.ones_like(g))
    return loss


def feature_matching_loss(
    real_outputs: List[List[torch.Tensor]],
    fake_outputs: List[List[torch.Tensor]],
) -> torch.Tensor:
    """Feature matching L1 loss across discriminator internal features."""
    loss = 0.0
    n = 0
    for dr, dg in zip(real_outputs, fake_outputs):
        for r_feat, g_feat in zip(dr[:-1], dg[:-1]):
            loss = loss + F.l1_loss(g_feat, r_feat.detach())
            n += 1
    return loss / max(n, 1)


# ─── Global RMS Loss ──────────────────────────────────────────────────────────

def global_rms_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize global loudness mismatch in dB."""
    pred_rms   = torch.sqrt(pred.pow(2).mean(-1) + 1e-8)
    target_rms = torch.sqrt(target.pow(2).mean(-1) + 1e-8)
    pred_db    = 20 * torch.log10(pred_rms)
    target_db  = 20 * torch.log10(target_rms)
    return F.l1_loss(pred_db, target_db)


# ─── Loss Scheduler (curriculum) ─────────────────────────────────────────────

class LambdaScheduler:
    """Linearly interpolate lambda weights between phases.
    
    Phase 0 (0–warmup_steps):    KD heavy, no GAN
    Phase 1 (warmup–main_steps): balanced KD + mel
    Phase 2 (main_steps+):       GAN enabled, KD regularizes
    """

    def __init__(
        self,
        warmup_steps: int = 2000,
        main_steps: int   = 5000,
        # Phase 0 → Phase 1 → Phase 2
        lambda_feat:     Tuple = (10.0, 2.0, 1.0),
        lambda_acoustic: Tuple = (5.0,  10.0, 10.0),
        lambda_mel:      Tuple = (15.0, 15.0, 15.0),
        lambda_commit:   Tuple = (0.25, 0.25, 0.25),
        lambda_adv:      Tuple = (0.0,  0.0,  0.3),
        lambda_fm:       Tuple = (0.0,  0.0,  3.0),
        lambda_rms:      Tuple = (1.0,  1.0,  1.0),
    ):
        self.warmup_steps = warmup_steps
        self.main_steps   = main_steps
        self.schedules = {
            "feat":     lambda_feat,
            "acoustic": lambda_acoustic,
            "mel":      lambda_mel,
            "commit":   lambda_commit,
            "adv":      lambda_adv,
            "fm":       lambda_fm,
            "rms":      lambda_rms,
        }

    def get(self, step: int) -> dict:
        """Return current lambda values for a given global step."""
        if step < self.warmup_steps:
            t = step / max(self.warmup_steps, 1)
            phase_idx = 0
        elif step < self.main_steps:
            t = (step - self.warmup_steps) / max(self.main_steps - self.warmup_steps, 1)
            phase_idx = 1
        else:
            t = 1.0
            phase_idx = 2

        result = {}
        for name, vals in self.schedules.items():
            if phase_idx < 2:
                v0, v1 = vals[phase_idx], vals[phase_idx + 1]
                result[name] = v0 + t * (v1 - v0)
            else:
                result[name] = vals[2]
        return result

    def use_gan(self, step: int) -> bool:
        return step >= self.main_steps
