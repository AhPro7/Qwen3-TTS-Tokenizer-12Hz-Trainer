# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3TTSTokenizerV2Decoder Fine-tuning Script

Supports GAN training (MPD + MSD) and/or reconstruction-only training.
Optionally adds a 48kHz decoder block on top of the base 24kHz decoder.
Can train only the new decoder blocks or the entire decoder.

Usage:
    # Single GPU (GAN + 48kHz decoder block)
    python trainer.py \
        --train_shards "data/train-{000000..000010}.tar" \
        --output_dir output/run1

    # Reconstruction only (no GAN)
    python trainer.py \
        --train_shards "data/train-*.tar" \
        --no-use_gan \
        --output_dir output/run1

    # Train full decoder (24kHz, no GAN)
    python trainer.py \
        --train_shards "data/train-*.tar" \
        --no-use_gan \
        --no-add_48k_decoder_block \
        --train_full_decoder \
        --output_dir output/run1
"""

import argparse
import gc
import glob
import json
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from safetensors.torch import load_file, save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Suppress MPS-backend STFT resize deprecation warning (PyTorch internal bug, harmless)
warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized",
    module="torch.functional",
)

# Add project root to path so `xcodec2` is importable as a package
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Append xcodec2 dir at the END for any bare imports inside xcodec2 modules
_xcodec2_dir = str(Path(__file__).parent.parent / "xcodec2")
if _xcodec2_dir not in sys.path:
    sys.path.append(_xcodec2_dir)

from xcodec2.criterions import (
    MultiResolutionMelSpectrogramLoss,
)
from xcodec2.module import (
    HiFiGANMultiPeriodDiscriminator,
    SpecDiscriminator,
)
from dataset import create_webdataset_loader
from losses import (
    global_rms_loss,
    generator_adversarial_loss,
    discriminator_loss,
    feature_matching_loss,
    d_r1_loss,
)
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

from disentangle import DisentangledProjection

BASE_SAMPLE_RATE = 24_000  # Hz, base Qwen3-TTS-Tokenizer output rate


def expand_shards(path: str, print_fn=print) -> "str | list[str]":
    """Expand glob wildcards to a sorted file list, or return the path unchanged."""
    if "*" in path and "{" not in path:
        expanded = sorted(glob.glob(path))
        if not expanded:
            print_fn(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print_fn(f"Found {len(expanded)} tar files")
        return expanded
    return path


def align_audio(
    pred_audio: torch.Tensor, target_audio: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Squeeze channel dim if present and truncate both tensors to the same length."""
    pred = pred_audio.squeeze(1) if pred_audio.dim() == 3 else pred_audio
    target = target_audio.squeeze(1) if target_audio.dim() == 3 else target_audio
    min_len = min(pred.shape[-1], target.shape[-1])
    return pred[..., :min_len], target[..., :min_len], min_len


def apply_length_mask(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    min_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-out padding regions beyond each sample's valid length."""
    mask = torch.arange(min_len, device=pred.device)[None, :] < lengths[:, None]
    return pred * mask, target * mask


def compute_grad_norm(module: nn.Module) -> float:
    """Compute L2 gradient norm across all parameters with gradients."""
    return (
        sum(
            p.grad.norm().item() ** 2 for p in module.parameters() if p.grad is not None
        )
        ** 0.5
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3TTSTokenizerV2Decoder fine-tuning script"
    )

    # Data
    parser.add_argument(
        "--train_shards",
        type=str,
        required=True,
        help="WebDataset shard pattern for training data",
    )
    parser.add_argument(
        "--val_shards",
        type=str,
        default=None,
        help="WebDataset shard pattern for validation data",
    )

    # Model
    parser.add_argument(
        "--decoder_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Base 24kHz decoder model path",
    )
    parser.add_argument(
        "--extra_upsample_rate",
        type=int,
        default=2,
        help="Additional upsample rate to append when --add_48k_decoder_block is set (default: 2 for 48kHz)",
    )
    parser.add_argument(
        "--num_decoder_block_frozen",
        type=int,
        default=None,
        help="Number of decoder blocks to freeze (default: base_num_decoder_modules - 2). Ignored when --train_full_decoder is set.",
    )
    parser.add_argument(
        "--use_gan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable GAN training (MPD + MSD discriminators). Use --no-use_gan to disable.",
    )
    parser.add_argument(
        "--add_48k_decoder_block",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append extra_upsample_rate to upsample_rates to target 48kHz. Use --no-add_48k_decoder_block to fine-tune the base 24kHz decoder.",
    )
    parser.add_argument(
        "--train_full_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train the entire Qwen3TTSTokenizerV2Decoder (including pre_conv, pre_transformer, upsample). When False, only trains the new/unfrozen decoder blocks.",
    )

    # Checkpoint resume
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help=(
            "Resume training from a checkpoint directory. "
            "Always loads decoder_block.safetensors (generator weights). "
            "Also restores discriminator and optimizer/scheduler states if present."
        ),
    )

    parser.add_argument(
        "--no_resume_optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not resume optimizer state when resuming from a checkpoint.",
    )

    # Training settings
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--lr_g", type=float, default=1e-4, help="Generator learning rate"
    )
    parser.add_argument(
        "--lr_d", type=float, default=2e-4, help="Discriminator learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of warmup steps (lr goes from 0 to target lr)"
    )
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--beta1_g", type=float, default=0.8, help="Generator Adam beta1"
    )
    parser.add_argument(
        "--beta2_g", type=float, default=0.99, help="Generator Adam beta2"
    )
    parser.add_argument(
        "--beta1_d", type=float, default=0.8, help="Discriminator Adam beta1"
    )
    parser.add_argument(
        "--beta2_d", type=float, default=0.99, help="Discriminator Adam beta2"
    )
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # GAN loss weights
    parser.add_argument(
        "--lambda_adv", type=float, default=1.0, help="Adversarial loss weight"
    )
    parser.add_argument(
        "--lambda_fm", type=float, default=1.0, help="Feature matching loss weight"
    )
    parser.add_argument(
        "--lambda_d_mpd", type=float, default=1.0, help="MPD discriminator loss weight"
    )
    parser.add_argument(
        "--lambda_d_msd", type=float, default=1.0, help="MSD discriminator loss weight"
    )

    # R1 gradient penalty (lazy discriminator regularization, StyleGAN2-style)
    parser.add_argument(
        "--r1",
        type=float,
        default=10.0,
        help="Weight of R1 gradient penalty on discriminators. 0 disables R1.",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=0,
        help=(
            "Apply R1 discriminator regularization every N optimizer steps "
            "(lazy regularization, StyleGAN2 default: 16)."
        ),
    )

    parser.add_argument(
        "--lambda_multi_res_mel",
        type=float,
        default=15.0,
        help="Multi-resolution mel loss weight (inworld-ai style, 7 scales). 0=disabled",
    )
    parser.add_argument(
        "--lambda_global_rms",
        type=float,
        default=1.0,
        help="Global dB RMS loss weight (inworld-ai style). 0=disabled",
    )
    parser.add_argument(
        "--lambda_orth",
        type=float,
        default=0.5,
        help="Orthogonality cosine loss weight (speaker ⊥ content)",
    )
    parser.add_argument(
        "--lambda_consistency",
        type=float,
        default=0.0,
        help=(
            "Content consistency loss weight: MSE(content_emb, original_hidden). "
            "Penalizes the content path from deviating too far. "
            "0 disables (recommended — warm-start init provides the same benefit). "
        ),
    )
    parser.add_argument(
        "--lambda_speaker_id",
        type=float,
        default=1.0,
        help=(
            "Speaker identity preservation loss weight. "
            "Re-extracts speaker embedding from the combined hidden and ensures "
            "it matches the original speaker global vector. 0 disables."
        ),
    )
    parser.add_argument(
        "--lambda_cycle",
        type=float,
        default=0.5,
        help=(
            "In-batch speaker swap cycle consistency loss weight. "
            "Swaps speakers within a batch, re-extracts, and ensures consistency. "
            "Trains the model to handle voice conversion during training. 0 disables."
        ),
    )

    # Architecture
    parser.add_argument(
        "--speaker_dim",
        type=int,
        default=256,
        help="Speaker bottleneck dimension in DisentangledProjection (default: 256)",
    )
    parser.add_argument(
        "--content_dim",
        type=int,
        default=128,
        help="VQ codebook vector dimension for content path (default: 128)",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=1024,
        help="Number of discrete codes in content VQ codebook (default: 1024)",
    )
    parser.add_argument(
        "--vq_commitment_cost",
        type=float,
        default=0.25,
        help="VQ commitment loss weight (default: 0.25)",
    )

    # Data settings
    parser.add_argument(
        "--max_audio_length",
        type=float,
        default=5.0,
        help="Maximum audio length (seconds)",
    )
    parser.add_argument(
        "--min_audio_length",
        type=float,
        default=1.0,
        help="Minimum audio length (seconds)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of DataLoader workers"
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/decoder_block_48k_gan",
        help="Output directory",
    )
    parser.add_argument(
        "--save_every", type=int, default=5000, help="Checkpoint save interval (steps)"
    )
    parser.add_argument(
        "--eval_every", type=int, default=1000, help="Evaluation interval (steps)"
    )
    parser.add_argument(
        "--log_every", type=int, default=10, help="Log output interval (steps)"
    )
    parser.add_argument(
        "--log_grad_norms", action="store_true", default=False,
        help="Log per-loss gradient norms (expensive, off by default)"
    )

    # Logging
    parser.add_argument("--log_with", type=str, default="wandb", help="Logging method")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen3-tts-decoder-block-48k",
        help="WandB project",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="WandB run name"
    )
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")

    # Reference discriminator checkpoint
    parser.add_argument(
        "--ref_discriminator_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory containing discriminator.pt. "
            "Loads frozen reference MPD/MSD and logs their dg scores in eval_step."
        ),
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None, help="Maximum training steps"
    )

    parser.add_argument(
        "--disentangle_warmup_steps",
        type=int,
        default=500,
        help=(
            "Number of steps before disentanglement losses (orth, speaker_id, cycle) "
            "reach full weight. Linearly ramps from 0 to target lambda. "
            "Prevents GAN collapse by letting the model stabilize first."
        ),
    )
    parser.add_argument(
        "--spike_skip_threshold",
        type=float,
        default=3.0,
        help=(
            "Skip generator update when mel loss exceeds this multiple of the "
            "running EMA. Protects against multi-speaker chunk poisoning. "
            "0 disables spike detection."
        ),
    )
    parser.add_argument(
        "--spike_ema_decay",
        type=float,
        default=0.99,
        help="EMA decay factor for mel loss running average (spike detection).",
    )

    return parser.parse_args()


# =============================================================================
# DecoderTrainingWrapper
# =============================================================================

class DecoderTrainingWrapper(nn.Module):
    """Wraps Qwen3TTSTokenizerV2Decoder for efficient training.

    When train_full_decoder=False: runs frozen layers under torch.no_grad() to save VRAM,
    and only computes gradients for the unfrozen decoder blocks.
    When train_full_decoder=True: runs the entire decoder with gradients.

    disentangle.forward() returns 5 values:
        speaker_contrib, content_emb, speaker_global, content_indices, commit_loss
    All are stored as instance attributes so the training loop can read them
    without re-running the model.
    """

    def __init__(
        self,
        decoder: Qwen3TTSTokenizerV2Decoder,
        num_frozen_decoder_modules: int,
        train_full_decoder: bool = False,
        speaker_dim: int = 256,
        content_dim: int = 128,
        codebook_size: int = 1024,
        vq_commitment_cost: float = 0.25,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_frozen = num_frozen_decoder_modules
        self.train_full_decoder = train_full_decoder

        hidden_dim = 1024  # actual output dim of pre_transformer
        self.disentangle = DisentangledProjection(
            hidden_dim=hidden_dim,
            speaker_dim=speaker_dim,
            content_dim=content_dim,
            codebook_size=codebook_size,
            commitment_cost=vq_commitment_cost,
        )

        # Stored after each forward pass — read by the training loop
        self.last_content_emb = None
        self.last_original_hidden = None  # [B, T, hidden_dim] pre-disentangle
        self.last_content_indices = None  # [B, T] discrete VQ token indices
        self.last_commit_loss = None      # scalar VQ commitment loss

    def _run_disentangle(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run single codebook bottleneck, store results, return new hidden."""
        content_emb, content_indices, commit_loss = self.disentangle(hidden)

        self.last_content_emb = content_emb
        self.last_original_hidden = hidden.detach()
        self.last_content_indices = content_indices
        self.last_commit_loss = commit_loss

        return content_emb

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} layers of codes, "
                f"got {codes.shape[1]}"
            )

        if self.train_full_decoder:
            # Full decoder with gradients
            hidden = self.decoder.quantizer.decode(codes)
            hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
            hidden = self.decoder.pre_transformer(
                inputs_embeds=hidden
            ).last_hidden_state

            hidden = self._run_disentangle(hidden)

            hidden = hidden.permute(0, 2, 1)
            for blocks in self.decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in self.decoder.decoder:
                wav = block(wav)

        else:
            # Frozen encoder: no_grad for quantizer/pre_conv/pre_transformer
            with torch.no_grad():
                hidden = self.decoder.quantizer.decode(codes)
                hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
                hidden = self.decoder.pre_transformer(
                    inputs_embeds=hidden
                ).last_hidden_state

            # DisentangledProjection: WITH gradients
            hidden = self._run_disentangle(hidden)

            # Frozen decoder blocks: requires_grad=False so no update,
            # but graph flows through so gradients reach DisentangledProjection.
            hidden = hidden.permute(0, 2, 1)
            for blocks in self.decoder.upsample:
                for block in blocks:
                    hidden = block(hidden)
            wav = hidden
            for block in self.decoder.decoder[: self.num_frozen]:
                wav = block(wav)

            # Trainable decoder tail
            for block in self.decoder.decoder[self.num_frozen :]:
                wav = block(wav)

        return wav.clamp(min=-1, max=1)


# =============================================================================
# Model / discriminator creation
# =============================================================================

def create_model(args, accelerator):
    """Create decoder model, optionally adding 48kHz decoder block."""
    accelerator.print(f"Loading base decoder from {args.decoder_model_path}...")

    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.decoder_model_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="cpu",
    )
    base_decoder = tokenizer.model.decoder
    base_state_dict = base_decoder.state_dict()
    base_num_decoder_modules = len(base_decoder.decoder)

    accelerator.print(
        f"Base decoder: upsample_rates={list(base_decoder.config.upsample_rates)}, "
        f"decoder modules={base_num_decoder_modules}"
    )

    # Build config (optionally adding 48kHz decoder block)
    config_dict = base_decoder.config.to_dict()
    base_upsample_rates = list(config_dict["upsample_rates"])
    if args.add_48k_decoder_block:
        new_upsample_rates = base_upsample_rates + [args.extra_upsample_rate]
        config_dict["upsample_rates"] = new_upsample_rates
        accelerator.print(f"New upsample_rates (48kHz): {new_upsample_rates}")
    else:
        new_upsample_rates = base_upsample_rates
        accelerator.print(
            f"Using base upsample_rates (no 48k block): {base_upsample_rates}"
        )
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
    if accelerator.device.type == "cuda":
        decoder_config._attn_implementation = "flash_attention_2"

    decoder = Qwen3TTSTokenizerV2Decoder(decoder_config).to(torch.bfloat16)
    missing_keys, unexpected_keys = decoder.load_state_dict(
        base_state_dict, strict=False
    )
    accelerator.print(
        f"Weight loading: {len(missing_keys)} missing keys (new blocks), "
        f"{len(unexpected_keys)} unexpected keys (old final layers)"
    )

    del tokenizer, base_decoder, base_state_dict
    gc.collect()

    # Freeze/unfreeze parameters
    if args.train_full_decoder:
        if args.num_decoder_block_frozen is not None:
            accelerator.print(
                "WARNING: --num_decoder_block_frozen is ignored when --train_full_decoder is set."
            )
        for param in decoder.parameters():
            param.requires_grad = True
        num_frozen = 0
    else:
        if args.num_decoder_block_frozen is not None:
            num_frozen = args.num_decoder_block_frozen
            if num_frozen < 0 or num_frozen > len(decoder.decoder):
                raise ValueError(
                    f"--num_decoder_block_frozen must be in [0, {len(decoder.decoder)}], got {num_frozen}"
                )
        else:
            num_frozen = base_num_decoder_modules - 2
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(num_frozen, len(decoder.decoder)):
            for param in decoder.decoder[i].parameters():
                param.requires_grad = True

    trainable_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_decoder = sum(p.numel() for p in decoder.parameters())
    accelerator.print(
        f"Decoder trainable: {trainable_decoder:,} / {total_decoder:,} "
        f"({trainable_decoder / total_decoder * 100:.4f}%)"
    )

    wrapper = DecoderTrainingWrapper(
        decoder,
        num_frozen,
        train_full_decoder=args.train_full_decoder,
        speaker_dim=args.speaker_dim,
        content_dim=args.content_dim,
        codebook_size=args.codebook_size,
        vq_commitment_cost=args.vq_commitment_cost,
    )

    # DisentangledProjection is always trainable
    dis_params = sum(p.numel() for p in wrapper.disentangle.parameters())
    total_trainable = trainable_decoder + dis_params
    accelerator.print(
        f"DisentangledProjection: {dis_params:,} params (all trainable)"
    )
    accelerator.print(
        f"Total trainable: {total_trainable:,} "
        f"(decoder: {trainable_decoder:,} + disentangle: {dis_params:,})"
    )
    accelerator.print(
        f"VQ content codebook: size={args.codebook_size}, dim={args.content_dim}"
    )

    # Load weights from checkpoint
    if args.resume_from:
        checkpoint_path = Path(args.resume_from) / "decoder_block.safetensors"
        if checkpoint_path.exists():
            accelerator.print(f"Loading generator weights from {checkpoint_path}...")
            trained_weights = load_file(str(checkpoint_path))
            missing, unexpected = decoder.load_state_dict(trained_weights, strict=False)
            accelerator.print(
                f"Generator weights loaded: {len(missing)} missing, "
                f"{len(unexpected)} unexpected"
            )
        else:
            accelerator.print(
                f"WARNING: decoder_block.safetensors not found at {checkpoint_path}"
            )

        dis_path = Path(args.resume_from) / "disentangle.safetensors"
        if dis_path.exists():
            accelerator.print(f"Loading DisentangledProjection from {dis_path}...")
            try:
                dis_weights = load_file(str(dis_path))
                wrapper.disentangle.load_state_dict(dis_weights, strict=False)
                accelerator.print("DisentangledProjection weights loaded ✓")
            except Exception as e:
                accelerator.print(
                    f"WARNING: DisentangledProjection load failed (architecture change?): {e}\n"
                    f"Starting DisentangledProjection from warm-start init."
                )
        else:
            accelerator.print(
                "WARNING: disentangle.safetensors not found — "
                "starting DisentangledProjection from warm-start init."
            )

    # Cast DisentangledProjection to bf16 to match model dtype under mixed precision
    wrapper.disentangle = wrapper.disentangle.to(torch.bfloat16)

    return wrapper, num_frozen, base_upsample_rates, new_upsample_rates


def create_discriminators(accelerator):
    """Create MPD and SpecDiscriminator discriminators."""
    mpd = HiFiGANMultiPeriodDiscriminator(
        periods=[2, 3, 5, 7, 11],
        max_downsample_channels=512,
        channels=16,
        channel_increasing_factor=4,
    )

    msd = SpecDiscriminator(
        stft_params={
            "fft_sizes": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "hop_sizes": [39, 63, 103, 167, 271, 438, 709, 1148],
            "win_lengths": [78, 126, 206, 334, 542, 876, 1418, 2296],
            "window": "hann_window",
        },
        in_channels=1,
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=32,
        max_downsample_channels=512,
        downsample_scales=[2, 2, 2],
        use_weight_norm=True,
    )

    mpd_params = sum(p.numel() for p in mpd.parameters())
    msd_params = sum(p.numel() for p in msd.parameters())
    accelerator.print(
        f"Discriminator params: MPD={mpd_params:,}, SpecDisc={msd_params:,}, "
        f"Total={mpd_params + msd_params:,}"
    )

    return mpd, msd


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def eval_step(
    model: nn.Module,
    mel_loss_fn: MultiResolutionMelSpectrogramLoss,
    dataloader: DataLoader,
    accelerator: Accelerator,
    mpd: "nn.Module | None" = None,
    msd: "nn.Module | None" = None,
    ref_mpd: "nn.Module | None" = None,
    ref_msd: "nn.Module | None" = None,
    max_batches: int = 50,
) -> dict:
    """Evaluation (mel loss + optional discriminator stats)."""
    model.eval()
    if mpd is not None:
        mpd.eval()
    if msd is not None:
        msd.eval()

    total_mel_loss = 0.0
    total_dr_mpd = 0.0
    total_dg_mpd = 0.0
    total_dr_msd = 0.0
    total_dg_msd = 0.0
    total_ref_dg_mpd = 0.0
    total_ref_dg_msd = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_batches >= max_batches:
                break

            audio_codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
            target_audio = batch["audio"].to(accelerator.device)
            audio_lengths = batch["audio_lengths"].to(accelerator.device)

            pred_48k = model(audio_codes)

            pred, target, min_len = align_audio(pred_48k, target_audio)
            pred, target = apply_length_mask(pred, target, audio_lengths, min_len)

            mel_loss = mel_loss_fn(pred, target)
            total_mel_loss += mel_loss.item()

            pred_wav = pred.unsqueeze(1)
            target_wav = target.unsqueeze(1)

            if mpd is not None and msd is not None:
                _, dr_mpd, dg_mpd = discriminator_loss(
                    mpd(target_wav), mpd(pred_wav)
                )
                _, dr_msd, dg_msd = discriminator_loss(
                    msd(target_wav), msd(pred_wav)
                )
                total_dr_mpd += dr_mpd.item()
                total_dg_mpd += dg_mpd.item()
                total_dr_msd += dr_msd.item()
                total_dg_msd += dg_msd.item()

            if ref_mpd is not None and ref_msd is not None:
                _, _, ref_dg_mpd = discriminator_loss(
                    ref_mpd(target_wav), ref_mpd(pred_wav)
                )
                _, _, ref_dg_msd = discriminator_loss(
                    ref_msd(target_wav), ref_msd(pred_wav)
                )
                total_ref_dg_mpd += ref_dg_mpd.item()
                total_ref_dg_msd += ref_dg_msd.item()

            num_batches += 1

    model.train()
    if mpd is not None:
        mpd.train()
    if msd is not None:
        msd.train()

    n = max(num_batches, 1)
    result = {"val/loss_multi_res_mel": total_mel_loss / n}
    if mpd is not None and msd is not None:
        result.update(
            {
                "val/d_mpd/dr": total_dr_mpd / n,
                "val/d_mpd/dg": total_dg_mpd / n,
                "val/d_msd/dr": total_dr_msd / n,
                "val/d_msd/dg": total_dg_msd / n,
            }
        )
    if ref_mpd is not None and ref_msd is not None:
        result.update(
            {
                "val/ref_d_mpd/dg": total_ref_dg_mpd / n,
                "val/ref_d_msd/dg": total_ref_dg_msd / n,
            }
        )
    return result


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    mpd: "nn.Module | None",
    msd: "nn.Module | None",
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: "torch.optim.Optimizer | None",
    scheduler_g,
    scheduler_d,
    step: int,
    epoch: int,
    args,
    accelerator: Accelerator,
    num_frozen: int,
    base_upsample_rates: list,
    new_upsample_rates: list,
    is_best: bool = False,
):
    """Save checkpoint (generator weights + optional discriminator + training state)."""
    if not accelerator.is_main_process:
        return

    output_dir = Path(args.output_dir)
    checkpoint_name = "checkpoint-best" if is_best else f"checkpoint-step-{step}"
    checkpoint_dir = output_dir / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Generator trainable weights
    unwrapped_model = accelerator.unwrap_model(model)
    decoder = unwrapped_model.decoder
    if args.train_full_decoder:
        trainable_state_dict = {k: v.cpu() for k, v in decoder.state_dict().items()}
    else:
        trainable_state_dict = {}
        for k, v in decoder.state_dict().items():
            parts = k.split(".", 2)
            if (
                parts[0] == "decoder"
                and len(parts) > 1
                and parts[1].isdigit()
                and int(parts[1]) >= num_frozen
            ):
                trainable_state_dict[k] = v.cpu()
    save_file(trainable_state_dict, str(checkpoint_dir / "decoder_block.safetensors"))

    # DisentangledProjection weights
    if hasattr(unwrapped_model, "disentangle"):
        disentangle_state = {
            k: v.cpu() for k, v in unwrapped_model.disentangle.state_dict().items()
        }
        save_file(disentangle_state, str(checkpoint_dir / "disentangle.safetensors"))

    # Discriminator weights
    if args.use_gan and mpd is not None and msd is not None:
        unwrapped_mpd = accelerator.unwrap_model(mpd)
        unwrapped_msd = accelerator.unwrap_model(msd)
        torch.save(
            {
                "mpd": unwrapped_mpd.state_dict(),
                "msd": unwrapped_msd.state_dict(),
            },
            checkpoint_dir / "discriminator.pt",
        )

    # Training state
    torch.save(
        {
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict() if optimizer_d else None,
            "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
            "step": step,
            "epoch": epoch,
        },
        checkpoint_dir / "training_state.pt",
    )

    # Config
    config_dict = {
        "base_upsample_rates": base_upsample_rates,
        "new_upsample_rates": new_upsample_rates,
        "extra_upsample_rate": args.extra_upsample_rate,
        "num_frozen_decoder_modules": num_frozen,
        "use_gan": args.use_gan,
        "add_48k_decoder_block": args.add_48k_decoder_block,
        "train_full_decoder": args.train_full_decoder,
        "speaker_dim": args.speaker_dim,
        "content_dim": args.content_dim,
        "codebook_size": args.codebook_size,
        "vq_commitment_cost": args.vq_commitment_cost,
        "step": step,
        "epoch": epoch,
        "training_type": "gan" if args.use_gan else "reconstruction",
        "lambda_adv": args.lambda_adv,
        "lambda_fm": args.lambda_fm,
        "lambda_d_mpd": args.lambda_d_mpd,
        "lambda_d_msd": args.lambda_d_msd,
        "lambda_multi_res_mel": args.lambda_multi_res_mel,
        "lambda_global_rms": args.lambda_global_rms,
        "lambda_orth": args.lambda_orth,
        "lambda_speaker_id": args.lambda_speaker_id,
        "lambda_cycle": args.lambda_cycle,
        "beta1_g": args.beta1_g,
        "beta2_g": args.beta2_g,
        "beta1_d": args.beta1_d,
        "beta2_d": args.beta2_d,
        "r1": args.r1,
        "d_reg_every": args.d_reg_every,
    }
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    accelerator.print(f"Saved checkpoint to {checkpoint_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
        project_dir=args.output_dir,
    )

    set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Create generator
    model, num_frozen, base_upsample_rates, new_upsample_rates = create_model(
        args, accelerator
    )

    # Create discriminators
    if args.use_gan:
        mpd, msd = create_discriminators(accelerator)
    else:
        mpd, msd = None, None
        accelerator.print("GAN disabled: skipping discriminator creation.")

    # Reference discriminators (frozen, for eval logging)
    ref_mpd, ref_msd = None, None
    if args.ref_discriminator_checkpoint:
        ref_disc_path = Path(args.ref_discriminator_checkpoint) / "discriminator.pt"
        if ref_disc_path.exists():
            ref_mpd, ref_msd = create_discriminators(accelerator)
            disc_state = torch.load(ref_disc_path, map_location="cpu")
            ref_mpd.load_state_dict(disc_state["mpd"])
            ref_msd.load_state_dict(disc_state["msd"])
            ref_mpd.to(accelerator.device).eval()
            ref_msd.to(accelerator.device).eval()
            for p in ref_mpd.parameters():
                p.requires_grad_(False)
            for p in ref_msd.parameters():
                p.requires_grad_(False)
            accelerator.print(
                f"Loaded reference discriminators from {ref_disc_path}"
            )
        else:
            accelerator.print(
                f"WARNING: ref_discriminator_checkpoint specified but "
                f"{ref_disc_path} not found. Skipping reference discriminators."
            )

    # Mel loss
    target_sample_rate = BASE_SAMPLE_RATE * (
        args.extra_upsample_rate if args.add_48k_decoder_block else 1
    )
    multi_res_mel_loss_fn = MultiResolutionMelSpectrogramLoss(
        sample_rate=target_sample_rate
    ).to(accelerator.device)

    # Training data
    accelerator.print(f"Loading training data: {args.train_shards}...")
    shard_pattern = expand_shards(args.train_shards, accelerator.print)
    train_dataloader = create_webdataset_loader(
        shard_pattern=shard_pattern,
        target_sample_rate=target_sample_rate,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=1000,
    )

    # Validation data (optional)
    val_dataloader = None
    if args.val_shards:
        shard_pattern = expand_shards(args.val_shards, accelerator.print)
        val_dataloader = create_webdataset_loader(
            shard_pattern=shard_pattern,
            target_sample_rate=target_sample_rate,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_buffer=0,
        )

    # Optimizers
    optimizer_g = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_g,
        betas=(args.beta1_g, args.beta2_g),
        weight_decay=args.weight_decay,
    )
    if args.use_gan:
        if args.r1 > 0 and args.d_reg_every > 1:
            d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        else:
            d_reg_ratio = 1.0
        optimizer_d = AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=args.lr_d * d_reg_ratio,
            betas=(args.beta1_d ** d_reg_ratio, args.beta2_d ** d_reg_ratio),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer_d = None

    # Schedulers
    if args.max_train_steps:
        total_steps = args.max_train_steps
    else:
        try:
            total_steps = (
                len(train_dataloader)
                * args.num_epochs
                // args.gradient_accumulation_steps
            )
        except TypeError:
            accelerator.print(
                "WARNING: Cannot determine dataset length. "
                "Please specify --max_train_steps."
            )
            total_steps = 500000

    warmup_steps = args.warmup_steps
    cosine_steps_g = max(1, total_steps - warmup_steps)
    if warmup_steps > 0:
        scheduler_g = SequentialLR(
            optimizer_g,
            schedulers=[
                LinearLR(optimizer_g, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer_g, T_max=cosine_steps_g, eta_min=args.lr_g * 0.1),
            ],
            milestones=[warmup_steps],
        )
    else:
        scheduler_g = CosineAnnealingLR(
            optimizer_g, T_max=total_steps, eta_min=args.lr_g * 0.1
        )

    if args.use_gan:
        cosine_steps_d = max(1, total_steps - warmup_steps)
        if warmup_steps > 0:
            scheduler_d = SequentialLR(
                optimizer_d,
                schedulers=[
                    LinearLR(optimizer_d, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
                    CosineAnnealingLR(optimizer_d, T_max=cosine_steps_d, eta_min=args.lr_d * 0.1),
                ],
                milestones=[warmup_steps],
            )
        else:
            scheduler_d = CosineAnnealingLR(
                optimizer_d, T_max=total_steps, eta_min=args.lr_d * 0.1
            )
    else:
        scheduler_d = None

    accelerator.print(f"Total training steps: {total_steps}")

    # Prepare with Accelerate
    model, optimizer_g, train_dataloader, scheduler_g = accelerator.prepare(
        model, optimizer_g, train_dataloader, scheduler_g
    )
    if args.use_gan:
        mpd, optimizer_d, scheduler_d = accelerator.prepare(
            mpd, optimizer_d, scheduler_d
        )
        msd = accelerator.prepare(msd)
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    # Cache discriminator params and dtype
    if args.use_gan:
        disc_params = list(mpd.parameters()) + list(msd.parameters())
        disc_dtype = next(mpd.parameters()).dtype
    else:
        disc_params = []
        disc_dtype = next(model.parameters()).dtype

    # Initialize tracker
    if args.log_with and accelerator.is_main_process:
        tracker_config = {
            "batch_size": args.batch_size,
            "lr_g": args.lr_g,
            "lr_d": args.lr_d if args.use_gan else None,
            "use_gan": args.use_gan,
            "add_48k_decoder_block": args.add_48k_decoder_block,
            "train_full_decoder": args.train_full_decoder,
            "lambda_adv": args.lambda_adv,
            "lambda_fm": args.lambda_fm,
            "lambda_d_mpd": args.lambda_d_mpd,
            "lambda_d_msd": args.lambda_d_msd,
            "lambda_multi_res_mel": args.lambda_multi_res_mel,
            "lambda_global_rms": args.lambda_global_rms,
            "lambda_orth": args.lambda_orth,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "extra_upsample_rate": args.extra_upsample_rate,
            "max_audio_length": args.max_audio_length,
            "training_type": "gan" if args.use_gan else "reconstruction",
            "r1": args.r1,
            "d_reg_every": args.d_reg_every,
            "content_dim": args.content_dim,
            "codebook_size": args.codebook_size,
            "vq_commitment_cost": args.vq_commitment_cost,
        }
        if args.log_with == "wandb":
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=tracker_config,
                init_kwargs={
                    "wandb": {
                        "name": args.wandb_run_name,
                        "entity": args.wandb_entity,
                        "dir": args.output_dir,
                    }
                },
            )
        else:
            accelerator.init_trackers(
                project_name=args.wandb_project, config=tracker_config
            )
    elif args.log_with:
        accelerator.init_trackers(project_name=args.wandb_project)

    # Resume training state
    start_step = 0
    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming training state from {args.resume_from}...")
        checkpoint_dir = Path(args.resume_from)

        prev_num_frozen = None
        config_path = checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                ckpt_config = json.load(f)
            prev_num_frozen = ckpt_config.get("num_frozen_decoder_modules")

        num_frozen_changed = (
            prev_num_frozen is not None and prev_num_frozen != num_frozen
        )
        if num_frozen_changed:
            accelerator.print(
                f"NOTE: num_frozen changed ({prev_num_frozen} -> {num_frozen}). "
                f"Generator optimizer/scheduler state will NOT be restored."
            )

        disc_path = checkpoint_dir / "discriminator.pt"
        if args.use_gan and disc_path.exists():
            disc_state = torch.load(disc_path, map_location="cpu")
            accelerator.unwrap_model(mpd).load_state_dict(disc_state["mpd"])
            accelerator.unwrap_model(msd).load_state_dict(disc_state["msd"])

        training_state_path = checkpoint_dir / "training_state.pt"
        if not training_state_path.exists():
            accelerator.print(
                f"WARNING: training_state.pt not found in {checkpoint_dir}. "
                f"Cannot resume optimizer/scheduler state or step/epoch count."
            )
        else:
            training_state = torch.load(training_state_path, map_location="cpu")
            start_step = training_state["step"]
            start_epoch = training_state["epoch"]

            if not args.no_resume_optimizer:
                if not num_frozen_changed:
                    optimizer_g.load_state_dict(training_state["optimizer_g"])
                    if training_state["scheduler_g"] and scheduler_g:
                        scheduler_g.load_state_dict(training_state["scheduler_g"])

                if (
                    args.use_gan
                    and optimizer_d is not None
                    and training_state.get("optimizer_d")
                ):
                    optimizer_d.load_state_dict(training_state["optimizer_d"])
                if args.use_gan and training_state.get("scheduler_d") and scheduler_d:
                    scheduler_d.load_state_dict(training_state["scheduler_d"])

        accelerator.print(f"Resumed from step {start_step}, epoch {start_epoch}")

    # =========================================================================
    # Training loop
    # =========================================================================
    global_step = start_step
    best_val_loss = float("inf")

    model.train()
    if args.use_gan:
        mpd.train()
        msd.train()

    mpd_grad_norm = 0.0
    msd_grad_norm = 0.0
    gen_grad_norm = 0.0
    r1_loss_val = 0.0
    total_audio_sec = 0
    spike_skipped = 0
    mel_ema = None

    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\n{'=' * 50}")
        accelerator.print(f"Epoch {epoch + 1}/{args.num_epochs}")
        accelerator.print(f"{'=' * 50}")

        progress_bar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for step, batch in enumerate(progress_bar):
            audio_codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
            target_audio = batch["audio"].to(accelerator.device)
            audio_lengths = batch["audio_lengths"].to(accelerator.device)

            # Generator forward
            pred_48k = model(audio_codes)

            # Align shapes
            pred, target, min_len = align_audio(pred_48k, target_audio)
            pred, target = apply_length_mask(pred, target, audio_lengths, min_len)

            # GAN loss placeholders
            loss_d = pred.new_zeros(())
            loss_d_mpd = pred.new_zeros(())
            loss_d_msd = pred.new_zeros(())
            loss_g_adv = pred.new_zeros(())
            loss_fm = pred.new_zeros(())
            loss_fm_mpd = pred.new_zeros(())
            loss_fm_msd = pred.new_zeros(())
            dr_mpd = pred.new_zeros(())
            dg_mpd = pred.new_zeros(())
            dr_msd = pred.new_zeros(())
            dg_msd = pred.new_zeros(())

            if args.use_gan:
                pred_wav = pred.unsqueeze(1).to(dtype=disc_dtype)
                target_wav = target.unsqueeze(1).to(dtype=disc_dtype)

            accumulate_models = [model] + ([mpd, msd] if args.use_gan else [])
            with accelerator.accumulate(*accumulate_models):

                # ─── Discriminator update ──────────────────────────────────
                if args.use_gan:
                    mpd_real_outputs = mpd(target_wav)
                    mpd_fake_outputs = mpd(pred_wav.detach())
                    loss_d_mpd, dr_mpd, dg_mpd = discriminator_loss(
                        mpd_real_outputs, mpd_fake_outputs
                    )

                    msd_real_outputs = msd(target_wav)
                    msd_fake_outputs = msd(pred_wav.detach())
                    loss_d_msd, dr_msd, dg_msd = discriminator_loss(
                        msd_real_outputs, msd_fake_outputs
                    )

                    loss_d = (
                        args.lambda_d_mpd * loss_d_mpd
                        + args.lambda_d_msd * loss_d_msd
                    )

                    optimizer_d.zero_grad()
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        mpd_grad_norm = compute_grad_norm(mpd)
                        msd_grad_norm = compute_grad_norm(msd)
                    accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                    optimizer_d.step()
                    scheduler_d.step()

                # ─── R1 regularization ────────────────────────────────────
                if (
                    args.use_gan
                    and args.r1 > 0
                    and args.d_reg_every > 0
                    and accelerator.sync_gradients
                    and (global_step + 1) % args.d_reg_every == 0
                ):
                    target_wav_r1 = target_wav.detach().float().requires_grad_(True)

                    mpd_real_r1 = mpd(target_wav_r1)
                    r1_mpd = d_r1_loss(mpd_real_r1, target_wav_r1)

                    msd_real_r1 = msd(target_wav_r1)
                    r1_msd = d_r1_loss(msd_real_r1, target_wav_r1)

                    r1_total = r1_mpd + r1_msd
                    r1_loss_scaled = (
                        (args.r1 / 2) * r1_total * args.d_reg_every
                        + 0.0 * mpd_real_r1[0][-1].sum()
                        + 0.0 * msd_real_r1[0][-1].sum()
                    )

                    optimizer_d.zero_grad()
                    accelerator.backward(r1_loss_scaled)
                    accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                    optimizer_d.step()
                    r1_loss_val = r1_total.item()

                # ─── Generator adversarial + FM ───────────────────────────
                if args.use_gan:
                    mpd_fake_outputs_g = mpd(pred_wav)
                    with torch.no_grad():
                        mpd_real_outputs_g = mpd(target_wav)
                    loss_g_adv_mpd = generator_adversarial_loss(mpd_fake_outputs_g)
                    loss_fm_mpd = feature_matching_loss(
                        mpd_real_outputs_g, mpd_fake_outputs_g
                    )

                    msd_fake_outputs_g = msd(pred_wav)
                    with torch.no_grad():
                        msd_real_outputs_g = msd(target_wav)
                    loss_g_adv_msd = generator_adversarial_loss(msd_fake_outputs_g)
                    loss_fm_msd = feature_matching_loss(
                        msd_real_outputs_g, msd_fake_outputs_g
                    )

                    loss_g_adv = loss_g_adv_mpd + loss_g_adv_msd
                    loss_fm = loss_fm_mpd + loss_fm_msd

                # ─── Reconstruction losses ────────────────────────────────
                if args.lambda_multi_res_mel > 0:
                    loss_multi_res_mel = multi_res_mel_loss_fn(pred, target)
                else:
                    loss_multi_res_mel = pred.new_zeros(())

                if args.lambda_global_rms > 0:
                    loss_global_rms = global_rms_loss(pred, target)
                else:
                    loss_global_rms = pred.new_zeros(())

                # ─── Bottleneck losses ────────────────────────────────────────
                unwrapped_model = accelerator.unwrap_model(model)
                content_emb = unwrapped_model.last_content_emb
                original_hidden = unwrapped_model.last_original_hidden
                # VQ losses (from discrete content codebook)
                commit_loss = unwrapped_model.last_commit_loss

                # Content consistency: MSE(content_emb, original_hidden)
                if args.lambda_consistency > 0:
                    loss_consistency = torch.nn.functional.mse_loss(
                        content_emb, original_hidden
                    )
                else:
                    loss_consistency = content_emb.new_zeros(())

                # Total generator loss (commit_loss is VQ bottleneck loss)
                loss_g = (
                    args.lambda_adv * loss_g_adv
                    + args.lambda_fm * loss_fm
                    + args.lambda_multi_res_mel * loss_multi_res_mel
                    + args.lambda_global_rms * loss_global_rms
                    + args.lambda_consistency * loss_consistency
                    + commit_loss  # VQ commitment loss (no extra lambda, built-in weight)
                )

                # ─── Spike detection ──────────────────────────────────────
                mel_val = loss_multi_res_mel.item()
                skip_this_batch = False
                if args.spike_skip_threshold > 0:
                    if mel_ema is None:
                        mel_ema = mel_val
                    else:
                        if mel_val > args.spike_skip_threshold * mel_ema and global_step > 50:
                            skip_this_batch = True
                            spike_skipped += 1
                            if accelerator.is_main_process:
                                accelerator.print(
                                    f"  ⚠ SPIKE SKIP step {global_step}: "
                                    f"mel={mel_val:.2f} > {args.spike_skip_threshold}"
                                    f"×EMA({mel_ema:.2f})="
                                    f"{args.spike_skip_threshold * mel_ema:.2f}. "
                                    f"Skipping G update. (total skipped: {spike_skipped})"
                                )
                        if not skip_this_batch:
                            mel_ema = (
                                args.spike_ema_decay * mel_ema
                                + (1 - args.spike_ema_decay) * mel_val
                            )

                # ─── Per-component gradient norms (optional) ───────────────
                g_grad_norms = {}
                should_log_grad = (
                    args.log_grad_norms
                    and accelerator.sync_gradients
                    and (global_step + 1) % args.log_every == 0
                    and not skip_this_batch
                )
                if should_log_grad:
                    gen_params = [p for p in model.parameters() if p.requires_grad]
                    for gn_name, gn_loss, gn_lam in [
                        ("g/grad_norm_adv", loss_g_adv, args.lambda_adv),
                        ("g/grad_norm_fm", loss_fm, args.lambda_fm),
                        ("g/grad_norm_multi_res_mel", loss_multi_res_mel, args.lambda_multi_res_mel),
                        ("g/grad_norm_global_rms", loss_global_rms, args.lambda_global_rms),
                    ]:
                        if gn_lam > 0:
                            grads = torch.autograd.grad(
                                gn_lam * gn_loss,
                                gen_params,
                                retain_graph=True,
                                allow_unused=True,
                            )
                            total_norm = (
                                sum(
                                    g.detach().norm() ** 2
                                    for g in grads
                                    if g is not None
                                )
                                ** 0.5
                            )
                            g_grad_norms[gn_name] = total_norm.item()

                # ─── Generator step ───────────────────────────────────────
                if skip_this_batch:
                    optimizer_g.zero_grad()
                    scheduler_g.step()
                else:
                    optimizer_g.zero_grad()
                    accelerator.backward(loss_g)
                    if accelerator.sync_gradients:
                        gen_grad_norm = compute_grad_norm(model)
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer_g.step()
                    scheduler_g.step()

            # ── Log / eval / save (only on real sync steps) ───────────────
            if accelerator.sync_gradients:
                global_step += 1
                audio_sec = audio_lengths.sum().item() / target_sample_rate
                total_audio_sec += audio_sec

                if global_step % args.log_every == 0:
                    log_dict = {
                        "g/loss_total": loss_g.item(),
                        "g/loss_multi_res_mel": loss_multi_res_mel.item(),
                        "g/loss_global_rms": loss_global_rms.item(),
                        "g/loss_consistency": loss_consistency.item(),
                        "g/vq_commit_loss": commit_loss.item(),
                        "g/vq_codebook_usage": unwrapped_model.disentangle.vq.codebook_usage(),
                        "g/grad_norm": gen_grad_norm,
                        "train/lr/generator": scheduler_g.get_last_lr()[0],
                        "train/audio_sec": audio_sec,
                        "train/total_audio_hour": total_audio_sec / 3600.0,
                        "train/mel_ema": mel_ema if mel_ema is not None else 0.0,
                        "train/spike_skipped_total": spike_skipped,
                        "train/batch_skipped": 1.0 if skip_this_batch else 0.0,
                    }
                    log_dict.update(g_grad_norms)
                    if args.use_gan:
                        log_dict.update(
                            {
                                "d/loss_total": loss_d.item(),
                                "d/mpd_loss": loss_d_mpd.item(),
                                "d/msd_loss": loss_d_msd.item(),
                                "d_mpd/grad_norm": mpd_grad_norm,
                                "d_msd/grad_norm": msd_grad_norm,
                                "d_mpd/dr": dr_mpd.item(),
                                "d_mpd/dg": dg_mpd.item(),
                                "d_msd/dr": dr_msd.item(),
                                "d_msd/dg": dg_msd.item(),
                                "g/loss_adv": loss_g_adv.item(),
                                "g/loss_fm": loss_fm.item(),
                                "g/loss_fm_mpd": loss_fm_mpd.item(),
                                "g/loss_fm_msd": loss_fm_msd.item(),
                                "train/lr/discriminator": scheduler_d.get_last_lr()[0],
                                "d/r1_loss": r1_loss_val,
                            }
                        )
                    accelerator.log(log_dict, step=global_step)

                    progress_bar.set_postfix(
                        g=loss_g.item(),
                        mel=loss_multi_res_mel.item(),
                        vq=commit_loss.item(),
                        **(
                            {"d": loss_d.item(), "adv": loss_g_adv.item()}
                            if args.use_gan
                            else {}
                        ),
                    )

                # Evaluation
                if (
                    val_dataloader
                    and global_step % args.eval_every == 0
                    and global_step > 0
                ):
                    val_losses = eval_step(
                        model,
                        multi_res_mel_loss_fn,
                        val_dataloader,
                        accelerator,
                        mpd=mpd if args.use_gan else None,
                        msd=msd if args.use_gan else None,
                        ref_mpd=ref_mpd,
                        ref_msd=ref_msd,
                    )
                    accelerator.print(
                        f"\nStep {global_step} - Validation: {val_losses}"
                    )
                    accelerator.log(val_losses, step=global_step)

                    # W&B Voice Conversion Table
                    if accelerator.is_main_process:
                        try:
                            import wandb
                            sample_batch = next(iter(val_dataloader))
                            sample_codes = sample_batch["audio_codes"].to(accelerator.device).transpose(1, 2)
                            sample_targets = sample_batch["audio"].float().cpu()
                            sample_sr = target_sample_rate

                            unwrapped = accelerator.unwrap_model(model)
                            dec = unwrapped.decoder
                            dis = unwrapped.disentangle

                            def _to_hidden(c):
                                h = dec.quantizer.decode(c)
                                h = dec.pre_conv(h).transpose(1, 2)
                                return dec.pre_transformer(inputs_embeds=h).last_hidden_state

                            def _decode_hidden(hidden):
                                x = hidden.permute(0, 2, 1)
                                for blocks in dec.upsample:
                                    for block in blocks:
                                        x = block(x)
                                wav = x
                                for block in dec.decoder:
                                    wav = block(wav)
                                return wav.clamp(-1, 1).squeeze().float().cpu().numpy()

                            B = sample_codes.shape[0]
                            vc_pairs = []
                            if B >= 2:
                                vc_pairs.append((0, 1))
                            if B >= 3:
                                vc_pairs.append((1, 2))
                                vc_pairs.append((0, 2))
                            elif B >= 2:
                                vc_pairs.append((1, 0))
                            while len(vc_pairs) < 3 and len(vc_pairs) > 0:
                                vc_pairs.append(vc_pairs[0])

                            vc_table = wandb.Table(columns=[
                                "Original Audio",
                                "Reconstructed Audio",
                            ])

                            with torch.inference_mode(), accelerator.autocast():
                                unique_indices = sorted(set(
                                    idx for pair in vc_pairs[:3] for idx in pair
                                ))
                                recon_cache = {}
                                hidden_cache = {}
                                for idx in unique_indices:
                                    h = _to_hidden(sample_codes[idx:idx+1])
                                    hidden_cache[idx] = h
                                    # FSQ bottleneck
                                    cnt_c, _, _ = dis(h)
                                    recon_hidden = cnt_c
                                    recon_cache[idx] = _decode_hidden(recon_hidden)

                                for src_i in unique_indices[:3]: # Just log up to 3 reconstructions
                                    try:
                                        src_gt_np = sample_targets[src_i].numpy()
                                        src_recon_np = recon_cache[src_i]

                                        vc_table.add_data(
                                            wandb.Audio(src_gt_np, sample_rate=sample_sr,
                                                        caption=f"Original #{src_i}"),
                                            wandb.Audio(src_recon_np, sample_rate=sample_sr,
                                                        caption=f"Recon #{src_i}"),
                                        )
                                    except Exception as row_e:
                                        accelerator.print(f"Table row failed: {row_e}")

                            log_media = {"voice_conversion_table": vc_table}
                            for i in range(min(2, B)):
                                if i in recon_cache:
                                    log_media[f"val/audio_original_{i}"] = wandb.Audio(
                                        sample_targets[i].numpy(), sample_rate=sample_sr
                                    )
                                    log_media[f"val/audio_recon_{i}"] = wandb.Audio(
                                        recon_cache[i], sample_rate=sample_sr
                                    )
                            accelerator.log(
                                log_media,
                                step=global_step,
                            )
                        except Exception as e:
                            accelerator.print(f"Audio logging failed: {e}")

                    if val_losses["val/loss_multi_res_mel"] < best_val_loss:
                        best_val_loss = val_losses["val/loss_multi_res_mel"]
                        save_checkpoint(
                            model, mpd, msd, optimizer_g, optimizer_d,
                            scheduler_g, scheduler_d, global_step, epoch,
                            args, accelerator, num_frozen,
                            base_upsample_rates, new_upsample_rates,
                            is_best=True,
                        )

                # Periodic checkpoint
                if global_step % args.save_every == 0 and global_step > 0:
                    save_checkpoint(
                        model, mpd, msd, optimizer_g, optimizer_d,
                        scheduler_g, scheduler_d, global_step, epoch,
                        args, accelerator, num_frozen,
                        base_upsample_rates, new_upsample_rates,
                    )

                if args.max_train_steps and global_step >= args.max_train_steps:
                    break

        # End-of-epoch checkpoint
        save_checkpoint(
            model, mpd, msd, optimizer_g, optimizer_d,
            scheduler_g, scheduler_d, global_step, epoch,
            args, accelerator, num_frozen,
            base_upsample_rates, new_upsample_rates,
        )

        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    # Final checkpoint
    save_checkpoint(
        model, mpd, msd, optimizer_g, optimizer_d,
        scheduler_g, scheduler_d, global_step, args.num_epochs,
        args, accelerator, num_frozen,
        base_upsample_rates, new_upsample_rates,
    )

    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()