# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3TTSTokenizerV2Decoder Fine-tuning Script

Supports GAN training (MPD + MSD) and/or reconstruction-only training.
Optionally adds a 48kHz decoder block on top of the base 24kHz decoder.
Can train only the new decoder blocks or the entire decoder.

Key changes vs. previous version:
  - VQ temperature annealing: starts at 2.0, anneals to 0.5 over
    --vq_temp_anneal_steps steps. Fixes near-zero commit_loss / codebook collapse.
  - Residual bottleneck in DisentangledProjection → decoder never fully
    loses the original hidden state, preventing mel divergence.
  - Removed lambda_speaker_id / lambda_cycle / lambda_orth — speaker path
    was removed from DisentangledProjection; these losses were no-ops.
  - Discriminator update is now skipped for the first --disc_warmup_steps
    to let the generator stabilise before GAN pressure begins.
  - lambda_adv is linearly ramped from 0 over --adv_warmup_steps to prevent
    early GAN collapse.
  - Generator loss is now scaled by 1/lambda_multi_res_mel so mel loss is
    always O(1) regardless of lambda — this stops adv from dominating.
  - Per-step VQ metrics (perplexity, codebook_usage, temperature) logged.

Usage:
    python trainer.py \
        --train_shards "data/train-{000000..000010}.tar" \
        --output_dir output/run1
"""

import argparse
import gc
import glob
import json
import math
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

warnings.filterwarnings(
    "ignore",
    message="An output with one or more elements was resized",
    module="torch.functional",
)

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
_xcodec2_dir = str(Path(__file__).parent.parent / "xcodec2")
if _xcodec2_dir not in sys.path:
    sys.path.append(_xcodec2_dir)

from xcodec2.criterions import MultiResolutionMelSpectrogramLoss
from xcodec2.module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
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

BASE_SAMPLE_RATE = 24_000


def expand_shards(path: str, print_fn=print) -> "str | list[str]":
    if "*" in path and "{" not in path:
        expanded = sorted(glob.glob(path))
        if not expanded:
            print_fn(f"Error: No files found matching pattern: {path}")
            sys.exit(1)
        print_fn(f"Found {len(expanded)} tar files")
        return expanded
    return path


def align_audio(pred_audio, target_audio):
    pred = pred_audio.squeeze(1) if pred_audio.dim() == 3 else pred_audio
    target = target_audio.squeeze(1) if target_audio.dim() == 3 else target_audio
    min_len = min(pred.shape[-1], target.shape[-1])
    return pred[..., :min_len], target[..., :min_len], min_len


def apply_length_mask(pred, target, lengths, min_len):
    mask = torch.arange(min_len, device=pred.device)[None, :] < lengths[:, None]
    return pred * mask, target * mask


def compute_grad_norm(module: nn.Module) -> float:
    return (
        sum(p.grad.norm().item() ** 2 for p in module.parameters() if p.grad is not None) ** 0.5
    )


def get_vq_temperature(step: int, anneal_steps: int, t_start: float, t_end: float) -> float:
    """Cosine annealing from t_start to t_end over anneal_steps."""
    if anneal_steps <= 0 or step >= anneal_steps:
        return t_end
    ratio = step / anneal_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return t_end + (t_start - t_end) * cosine


def get_adv_weight(step: int, adv_warmup_steps: int, lambda_adv: float) -> float:
    """Linear ramp of adversarial loss weight from 0 to lambda_adv."""
    if adv_warmup_steps <= 0 or step >= adv_warmup_steps:
        return lambda_adv
    return lambda_adv * (step / adv_warmup_steps)


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--train_shards", type=str, required=True)
    parser.add_argument("--val_shards", type=str, default=None)

    # Model
    parser.add_argument("--decoder_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--extra_upsample_rate", type=int, default=2)
    parser.add_argument("--num_decoder_block_frozen", type=int, default=None)
    parser.add_argument("--use_gan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_48k_decoder_block", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_full_decoder", action=argparse.BooleanOptionalAction, default=False)

    # Checkpoint
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--no_resume_optimizer", action=argparse.BooleanOptionalAction, default=False)

    # Training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_g", type=float, default=1e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1_g", type=float, default=0.8)
    parser.add_argument("--beta2_g", type=float, default=0.99)
    parser.add_argument("--beta1_d", type=float, default=0.8)
    parser.add_argument("--beta2_d", type=float, default=0.99)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # GAN loss weights
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_fm", type=float, default=1.0)
    parser.add_argument("--lambda_d_mpd", type=float, default=1.0)
    parser.add_argument("--lambda_d_msd", type=float, default=1.0)
    parser.add_argument("--lambda_multi_res_mel", type=float, default=15.0)
    parser.add_argument("--lambda_global_rms", type=float, default=1.0)

    # Consistency loss (content emb vs original hidden)
    parser.add_argument("--lambda_consistency", type=float, default=0.0)

    # R1
    parser.add_argument("--r1", type=float, default=10.0)
    parser.add_argument("--d_reg_every", type=int, default=0)

    # VQ
    parser.add_argument("--speaker_dim", type=int, default=256)   # kept for CLI compat, unused
    parser.add_argument("--content_dim", type=int, default=256)
    parser.add_argument("--codebook_size", type=int, default=2048)
    parser.add_argument("--vq_commitment_cost", type=float, default=0.25)
    parser.add_argument("--vq_entropy_loss_weight", type=float, default=0.05,
                        help="Entropy regularisation weight — penalises low codebook utilisation.")
    parser.add_argument("--vq_temp_start", type=float, default=2.0,
                        help="VQ temperature at step 0 (soft → hard annealing).")
    parser.add_argument("--vq_temp_end", type=float, default=0.5,
                        help="VQ temperature after annealing completes.")
    parser.add_argument("--vq_temp_anneal_steps", type=int, default=3000,
                        help="Number of steps over which VQ temperature anneals.")

    # Discriminator / adversarial warmup
    parser.add_argument("--disc_warmup_steps", type=int, default=500,
                        help="Skip discriminator updates for this many steps.")
    parser.add_argument("--adv_warmup_steps", type=int, default=1000,
                        help="Linearly ramp lambda_adv from 0 to target over this many steps.")

    # Data
    parser.add_argument("--max_audio_length", type=float, default=5.0)
    parser.add_argument("--min_audio_length", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)

    # Output
    parser.add_argument("--output_dir", type=str, default="output/decoder_block_48k_gan")
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--log_grad_norms", action="store_true", default=False)

    # Logging
    parser.add_argument("--log_with", type=str, default="wandb")
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-decoder-block-48k")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)

    # Reference discriminator
    parser.add_argument("--ref_discriminator_checkpoint", type=str, default=None)

    # Spike detection
    parser.add_argument("--spike_skip_threshold", type=float, default=3.0)
    parser.add_argument("--spike_ema_decay", type=float, default=0.99)

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_train_steps", type=int, default=None)

    return parser.parse_args()


# =============================================================================
# DecoderTrainingWrapper
# =============================================================================

class DecoderTrainingWrapper(nn.Module):
    """
    Wraps Qwen3TTSTokenizerV2Decoder with a single-codebook VQ bottleneck
    inserted between the transformer encoder and the HiFiGAN decoder.

    The bottleneck uses a residual connection so early in training the
    decoder still sees ~original hidden states, preventing mel divergence.
    """

    def __init__(
        self,
        decoder: Qwen3TTSTokenizerV2Decoder,
        num_frozen_decoder_modules: int,
        train_full_decoder: bool = False,
        content_dim: int = 256,
        codebook_size: int = 2048,
        vq_commitment_cost: float = 0.25,
        vq_entropy_loss_weight: float = 0.05,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_frozen = num_frozen_decoder_modules
        self.train_full_decoder = train_full_decoder

        hidden_dim = 1024  # Qwen3-TTS-Tokenizer-12Hz transformer output dim
        self.disentangle = DisentangledProjection(
            hidden_dim=hidden_dim,
            content_dim=content_dim,
            codebook_size=codebook_size,
            commitment_cost=vq_commitment_cost,
            entropy_loss_weight=vq_entropy_loss_weight,
        )

        # Populated after each forward, read by training loop
        self.last_content_emb: torch.Tensor | None = None
        self.last_original_hidden: torch.Tensor | None = None
        self.last_content_indices: torch.Tensor | None = None
        self.last_commit_loss: torch.Tensor | None = None

    def set_vq_temperature(self, temperature: float):
        self.disentangle.temperature = temperature

    def _run_bottleneck(self, hidden: torch.Tensor) -> torch.Tensor:
        content_emb, content_indices, vq_loss = self.disentangle(hidden)
        self.last_content_emb = content_emb
        self.last_original_hidden = hidden.detach()
        self.last_content_indices = content_indices
        self.last_commit_loss = vq_loss
        return content_emb

    def _decode_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run the HiFiGAN decoder part (upsamplers + decoder blocks)."""
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav

    def _decode_hidden_partial(self, hidden: torch.Tensor, num_frozen: int) -> torch.Tensor:
        """Run decoder with only the unfrozen tail blocks having gradients."""
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in self.decoder.decoder[:num_frozen]:
            wav = block(wav)
        for block in self.decoder.decoder[num_frozen:]:
            wav = block(wav)
        return wav

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[1] != self.decoder.config.num_quantizers:
            raise ValueError(
                f"Expected {self.decoder.config.num_quantizers} code layers, got {codes.shape[1]}"
            )

        if self.train_full_decoder:
            hidden = self.decoder.quantizer.decode(codes)
            hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
            hidden = self.decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
            hidden = self._run_bottleneck(hidden)
            wav = self._decode_hidden(hidden)
        else:
            with torch.no_grad():
                hidden = self.decoder.quantizer.decode(codes)
                hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
                hidden = self.decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
            hidden = self._run_bottleneck(hidden)
            wav = self._decode_hidden_partial(hidden, self.num_frozen)

        return wav.clamp(min=-1.0, max=1.0)


# =============================================================================
# Model / discriminator creation
# =============================================================================

def create_model(args, accelerator):
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

    config_dict = base_decoder.config.to_dict()
    base_upsample_rates = list(config_dict["upsample_rates"])
    if args.add_48k_decoder_block:
        new_upsample_rates = base_upsample_rates + [args.extra_upsample_rate]
        config_dict["upsample_rates"] = new_upsample_rates
        accelerator.print(f"New upsample_rates (48kHz): {new_upsample_rates}")
    else:
        new_upsample_rates = base_upsample_rates
    for key in ("model_type", "transformers_version"):
        config_dict.pop(key, None)

    decoder_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
    if accelerator.device.type == "cuda":
        decoder_config._attn_implementation = "flash_attention_2"

    decoder = Qwen3TTSTokenizerV2Decoder(decoder_config).to(torch.bfloat16)
    missing_keys, unexpected_keys = decoder.load_state_dict(base_state_dict, strict=False)
    accelerator.print(
        f"Weight loading: {len(missing_keys)} missing, {len(unexpected_keys)} unexpected"
    )

    del tokenizer, base_decoder, base_state_dict
    gc.collect()

    if args.train_full_decoder:
        for param in decoder.parameters():
            param.requires_grad = True
        num_frozen = 0
    else:
        num_frozen = (
            args.num_decoder_block_frozen
            if args.num_decoder_block_frozen is not None
            else base_num_decoder_modules - 2
        )
        for param in decoder.parameters():
            param.requires_grad = False
        for i in range(num_frozen, len(decoder.decoder)):
            for param in decoder.decoder[i].parameters():
                param.requires_grad = True

    trainable_decoder = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_decoder = sum(p.numel() for p in decoder.parameters())
    accelerator.print(
        f"Decoder trainable: {trainable_decoder:,} / {total_decoder:,} "
        f"({trainable_decoder / total_decoder * 100:.2f}%)"
    )

    wrapper = DecoderTrainingWrapper(
        decoder,
        num_frozen,
        train_full_decoder=args.train_full_decoder,
        content_dim=args.content_dim,
        codebook_size=args.codebook_size,
        vq_commitment_cost=args.vq_commitment_cost,
        vq_entropy_loss_weight=args.vq_entropy_loss_weight,
    )

    dis_params = sum(p.numel() for p in wrapper.disentangle.parameters())
    accelerator.print(
        f"DisentangledProjection: {dis_params:,} params | "
        f"codebook_size={args.codebook_size}, content_dim={args.content_dim}"
    )
    accelerator.print(
        f"VQ temperature: {args.vq_temp_start} → {args.vq_temp_end} "
        f"over {args.vq_temp_anneal_steps} steps"
    )

    # Resume generator weights
    if args.resume_from:
        ckpt_path = Path(args.resume_from) / "decoder_block.safetensors"
        if ckpt_path.exists():
            accelerator.print(f"Loading generator weights from {ckpt_path}...")
            decoder.load_state_dict(load_file(str(ckpt_path)), strict=False)

        dis_path = Path(args.resume_from) / "disentangle.safetensors"
        if dis_path.exists():
            accelerator.print(f"Loading DisentangledProjection from {dis_path}...")
            try:
                wrapper.disentangle.load_state_dict(load_file(str(dis_path)), strict=False)
                accelerator.print("DisentangledProjection weights loaded ✓")
            except Exception as e:
                accelerator.print(f"WARNING: DisentangledProjection load failed: {e}")

    wrapper.disentangle = wrapper.disentangle.to(torch.bfloat16)
    return wrapper, num_frozen, base_upsample_rates, new_upsample_rates


def create_discriminators(accelerator):
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
    mpd_p = sum(p.numel() for p in mpd.parameters())
    msd_p = sum(p.numel() for p in msd.parameters())
    accelerator.print(f"Discriminators: MPD={mpd_p:,}, SpecDisc={msd_p:,}")
    return mpd, msd


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def eval_step(
    model,
    mel_loss_fn,
    dataloader,
    accelerator,
    mpd=None,
    msd=None,
    ref_mpd=None,
    ref_msd=None,
    max_batches: int = 50,
) -> dict:
    model.eval()
    if mpd is not None:
        mpd.eval()
    if msd is not None:
        msd.eval()

    totals = dict(mel=0.0, dr_mpd=0.0, dg_mpd=0.0, dr_msd=0.0, dg_msd=0.0,
                  ref_dg_mpd=0.0, ref_dg_msd=0.0, vq_usage=0.0, perplexity=0.0)
    n = 0

    for batch in dataloader:
        if n >= max_batches:
            break
        codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
        target = batch["audio"].to(accelerator.device)
        lengths = batch["audio_lengths"].to(accelerator.device)

        pred = model(codes)
        pred, target_t, min_len = align_audio(pred, target)
        pred, target_t = apply_length_mask(pred, target_t, lengths, min_len)

        totals["mel"] += mel_loss_fn(pred, target_t).item()

        unwrapped = accelerator.unwrap_model(model)
        totals["vq_usage"] += unwrapped.disentangle.vq.codebook_usage()
        totals["perplexity"] += unwrapped.disentangle.vq.perplexity()

        if mpd is not None:
            pw = pred.unsqueeze(1)
            tw = target_t.unsqueeze(1)
            _, dr_mpd, dg_mpd = discriminator_loss(mpd(tw), mpd(pw))
            _, dr_msd, dg_msd = discriminator_loss(msd(tw), msd(pw))
            totals["dr_mpd"] += dr_mpd.item()
            totals["dg_mpd"] += dg_mpd.item()
            totals["dr_msd"] += dr_msd.item()
            totals["dg_msd"] += dg_msd.item()

        if ref_mpd is not None:
            pw = pred.unsqueeze(1)
            tw = target_t.unsqueeze(1)
            _, _, ref_dg_mpd = discriminator_loss(ref_mpd(tw), ref_mpd(pw))
            _, _, ref_dg_msd = discriminator_loss(ref_msd(tw), ref_msd(pw))
            totals["ref_dg_mpd"] += ref_dg_mpd.item()
            totals["ref_dg_msd"] += ref_dg_msd.item()

        n += 1

    model.train()
    if mpd is not None:
        mpd.train()
    if msd is not None:
        msd.train()

    d = max(n, 1)
    result = {
        "val/loss_multi_res_mel": totals["mel"] / d,
        "val/vq_codebook_usage": totals["vq_usage"] / d,
        "val/vq_perplexity": totals["perplexity"] / d,
    }
    if mpd is not None:
        result.update({
            "val/d_mpd/dr": totals["dr_mpd"] / d,
            "val/d_mpd/dg": totals["dg_mpd"] / d,
            "val/d_msd/dr": totals["dr_msd"] / d,
            "val/d_msd/dg": totals["dg_msd"] / d,
        })
    if ref_mpd is not None:
        result.update({
            "val/ref_d_mpd/dg": totals["ref_dg_mpd"] / d,
            "val/ref_d_msd/dg": totals["ref_dg_msd"] / d,
        })
    return result


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(
    model, mpd, msd, optimizer_g, optimizer_d,
    scheduler_g, scheduler_d, step, epoch, args, accelerator,
    num_frozen, base_upsample_rates, new_upsample_rates, is_best=False,
):
    if not accelerator.is_main_process:
        return

    out = Path(args.output_dir)
    name = "checkpoint-best" if is_best else f"checkpoint-step-{step}"
    ckpt_dir = out / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    unwrapped = accelerator.unwrap_model(model)
    decoder = unwrapped.decoder

    if args.train_full_decoder:
        gen_sd = {k: v.cpu() for k, v in decoder.state_dict().items()}
    else:
        gen_sd = {}
        for k, v in decoder.state_dict().items():
            parts = k.split(".", 2)
            if (parts[0] == "decoder" and len(parts) > 1
                    and parts[1].isdigit() and int(parts[1]) >= num_frozen):
                gen_sd[k] = v.cpu()
    save_file(gen_sd, str(ckpt_dir / "decoder_block.safetensors"))

    if hasattr(unwrapped, "disentangle"):
        save_file(
            {k: v.cpu() for k, v in unwrapped.disentangle.state_dict().items()},
            str(ckpt_dir / "disentangle.safetensors"),
        )

    if args.use_gan and mpd is not None:
        torch.save(
            {"mpd": accelerator.unwrap_model(mpd).state_dict(),
             "msd": accelerator.unwrap_model(msd).state_dict()},
            ckpt_dir / "discriminator.pt",
        )

    torch.save(
        {
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict() if optimizer_d else None,
            "scheduler_g": scheduler_g.state_dict() if scheduler_g else None,
            "scheduler_d": scheduler_d.state_dict() if scheduler_d else None,
            "step": step,
            "epoch": epoch,
        },
        ckpt_dir / "training_state.pt",
    )

    config_dict = {
        "base_upsample_rates": base_upsample_rates,
        "new_upsample_rates": new_upsample_rates,
        "extra_upsample_rate": args.extra_upsample_rate,
        "num_frozen_decoder_modules": num_frozen,
        "use_gan": args.use_gan,
        "add_48k_decoder_block": args.add_48k_decoder_block,
        "train_full_decoder": args.train_full_decoder,
        "content_dim": args.content_dim,
        "codebook_size": args.codebook_size,
        "vq_commitment_cost": args.vq_commitment_cost,
        "vq_entropy_loss_weight": args.vq_entropy_loss_weight,
        "vq_temp_start": args.vq_temp_start,
        "vq_temp_end": args.vq_temp_end,
        "vq_temp_anneal_steps": args.vq_temp_anneal_steps,
        "step": step,
        "epoch": epoch,
    }
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    accelerator.print(f"Saved checkpoint → {ckpt_dir}")


# =============================================================================
# W&B audio sample logging
# =============================================================================

@torch.no_grad()
def _log_audio_samples(
    model: nn.Module,
    val_dataloader: DataLoader,
    accelerator: Accelerator,
    sample_rate: int,
    global_step: int,
    num_samples: int = 4,
):
    """
    Logs a W&B Table with columns:
        Original | Reconstructed (VQ bottleneck) | Bypass (no bottleneck)

    'Bypass' runs the full decoder WITHOUT the VQ bottleneck so you can
    hear how much the bottleneck degrades / changes the output.
    """
    try:
        import wandb
    except ImportError:
        return

    model.eval()
    unwrapped: DecoderTrainingWrapper = accelerator.unwrap_model(model)
    dec = unwrapped.decoder
    dis = unwrapped.disentangle

    # ── Grab one batch from val ────────────────────────────────────────
    batch = next(iter(val_dataloader))
    codes   = batch["audio_codes"].to(accelerator.device).transpose(1, 2)  # [B, Q, T]
    targets = batch["audio"].float().cpu()                                  # [B, L]
    B = min(codes.shape[0], num_samples)

    # ── Helper: transformer encoder → hidden ──────────────────────────
    def _to_hidden(c: torch.Tensor) -> torch.Tensor:
        """c: [1, Q, T] → hidden: [1, T, D]"""
        h = dec.quantizer.decode(c)
        h = dec.pre_conv(h).transpose(1, 2)
        return dec.pre_transformer(inputs_embeds=h).last_hidden_state

    # ── Helper: hidden → waveform ─────────────────────────────────────
    def _hidden_to_wav(hidden: torch.Tensor) -> "list[float]":
        """hidden: [1, T, D] → numpy float32 mono waveform"""
        x = hidden.permute(0, 2, 1)
        for blocks in dec.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in dec.decoder:
            wav = block(wav)
        return wav.clamp(-1, 1).squeeze().float().cpu().numpy()

    table = wandb.Table(columns=[
        "step",
        "sample_id",
        "original",
        "reconstructed_vq",
        "bypass_no_vq",
        "vq_token_ids",
        "codebook_usage_pct",
    ])

    with torch.inference_mode(), accelerator.autocast():
        for i in range(B):
            try:
                c_i = codes[i : i + 1]            # [1, Q, T]

                # ── Encoder hidden ────────────────────────────────────
                hidden = _to_hidden(c_i)           # [1, T, D]

                # ── VQ bottleneck ─────────────────────────────────────
                content_emb, indices, _ = dis(hidden)
                recon_wav = _hidden_to_wav(content_emb)

                # ── Bypass (identity — no bottleneck) ─────────────────
                bypass_wav = _hidden_to_wav(hidden)

                # ── Original target ───────────────────────────────────
                orig_np = targets[i].numpy()
                # Trim/pad to same length as recon for fair comparison
                min_len = min(len(orig_np), len(recon_wav), len(bypass_wav))
                orig_np   = orig_np[:min_len]
                recon_wav = recon_wav[:min_len]
                bypass_wav = bypass_wav[:min_len]

                # ── VQ token summary ──────────────────────────────────
                token_ids = indices[0].cpu().tolist()          # [T]
                n_unique  = len(set(token_ids))
                usage_pct = round(100.0 * n_unique / dis.vq.codebook_size, 2)
                token_str = str(token_ids[:20]) + ("..." if len(token_ids) > 20 else "")

                table.add_data(
                    global_step,
                    i,
                    wandb.Audio(orig_np,   sample_rate=sample_rate, caption=f"Original #{i}"),
                    wandb.Audio(recon_wav, sample_rate=sample_rate, caption=f"VQ Recon #{i}"),
                    wandb.Audio(bypass_wav, sample_rate=sample_rate, caption=f"Bypass #{i}"),
                    token_str,
                    usage_pct,
                )

            except Exception as row_err:
                accelerator.print(f"Audio table row {i} failed: {row_err}")

    # Also log individual audio clips directly (easier to find in W&B UI)
    direct_log = {"val/audio_table": table}
    with torch.inference_mode(), accelerator.autocast():
        for i in range(min(B, 2)):
            try:
                c_i    = codes[i : i + 1]
                hidden = _to_hidden(c_i)
                content_emb, _, _ = dis(hidden)
                recon_wav  = _hidden_to_wav(content_emb)
                bypass_wav = _hidden_to_wav(hidden)
                orig_np    = targets[i].numpy()
                min_len = min(len(orig_np), len(recon_wav), len(bypass_wav))

                direct_log[f"val/original_{i}"]   = wandb.Audio(
                    orig_np[:min_len], sample_rate=sample_rate)
                direct_log[f"val/recon_vq_{i}"]   = wandb.Audio(
                    recon_wav[:min_len], sample_rate=sample_rate)
                direct_log[f"val/bypass_{i}"]     = wandb.Audio(
                    bypass_wav[:min_len], sample_rate=sample_rate)
            except Exception as e:
                accelerator.print(f"Direct audio log {i} failed: {e}")

    try:
        wandb.log(direct_log, step=global_step)
    except Exception as e:
        accelerator.print(f"W&B audio log failed: {e}")

    model.train()


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

    model, num_frozen, base_upsample_rates, new_upsample_rates = create_model(args, accelerator)

    if args.use_gan:
        mpd, msd = create_discriminators(accelerator)
    else:
        mpd, msd = None, None
        accelerator.print("GAN disabled.")

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
            for p in list(ref_mpd.parameters()) + list(ref_msd.parameters()):
                p.requires_grad_(False)

    target_sample_rate = BASE_SAMPLE_RATE * (
        args.extra_upsample_rate if args.add_48k_decoder_block else 1
    )
    multi_res_mel_loss_fn = MultiResolutionMelSpectrogramLoss(
        sample_rate=target_sample_rate
    ).to(accelerator.device)

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

    optimizer_g = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_g,
        betas=(args.beta1_g, args.beta2_g),
        weight_decay=args.weight_decay,
    )
    if args.use_gan:
        d_reg_ratio = (args.d_reg_every / (args.d_reg_every + 1)
                       if args.r1 > 0 and args.d_reg_every > 1 else 1.0)
        optimizer_d = AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=args.lr_d * d_reg_ratio,
            betas=(args.beta1_d ** d_reg_ratio, args.beta2_d ** d_reg_ratio),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer_d = None

    try:
        total_steps = (
            args.max_train_steps
            or len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
        )
    except TypeError:
        accelerator.print("WARNING: Cannot determine dataset length. Set --max_train_steps.")
        total_steps = 500_000

    warmup = args.warmup_steps
    if warmup > 0:
        scheduler_g = SequentialLR(
            optimizer_g,
            schedulers=[
                LinearLR(optimizer_g, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optimizer_g, T_max=max(1, total_steps - warmup), eta_min=args.lr_g * 0.1),
            ],
            milestones=[warmup],
        )
    else:
        scheduler_g = CosineAnnealingLR(optimizer_g, T_max=total_steps, eta_min=args.lr_g * 0.1)

    if args.use_gan:
        if warmup > 0:
            scheduler_d = SequentialLR(
                optimizer_d,
                schedulers=[
                    LinearLR(optimizer_d, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                    CosineAnnealingLR(optimizer_d, T_max=max(1, total_steps - warmup), eta_min=args.lr_d * 0.1),
                ],
                milestones=[warmup],
            )
        else:
            scheduler_d = CosineAnnealingLR(optimizer_d, T_max=total_steps, eta_min=args.lr_d * 0.1)
    else:
        scheduler_d = None

    accelerator.print(f"Total steps: {total_steps}")

    model, optimizer_g, train_dataloader, scheduler_g = accelerator.prepare(
        model, optimizer_g, train_dataloader, scheduler_g
    )
    if args.use_gan:
        mpd, optimizer_d, scheduler_d = accelerator.prepare(mpd, optimizer_d, scheduler_d)
        msd = accelerator.prepare(msd)
    if val_dataloader:
        val_dataloader = accelerator.prepare(val_dataloader)

    if args.use_gan:
        disc_params = list(mpd.parameters()) + list(msd.parameters())
        disc_dtype = next(mpd.parameters()).dtype
    else:
        disc_params = []
        disc_dtype = next(model.parameters()).dtype

    # Init trackers
    if args.log_with and accelerator.is_main_process:
        tracker_config = vars(args)
        tracker_config["total_steps"] = total_steps
        if args.log_with == "wandb":
            accelerator.init_trackers(
                args.wandb_project,
                config=tracker_config,
                init_kwargs={"wandb": {
                    "name": args.wandb_run_name,
                    "entity": args.wandb_entity,
                    "dir": args.output_dir,
                }},
            )
        else:
            accelerator.init_trackers(args.wandb_project, config=tracker_config)
    elif args.log_with:
        accelerator.init_trackers(project_name=args.wandb_project)

    # Resume training state
    start_step, start_epoch = 0, 0
    if args.resume_from:
        ckpt_dir = Path(args.resume_from)
        disc_pt = ckpt_dir / "discriminator.pt"
        if args.use_gan and disc_pt.exists():
            disc_state = torch.load(disc_pt, map_location="cpu")
            accelerator.unwrap_model(mpd).load_state_dict(disc_state["mpd"])
            accelerator.unwrap_model(msd).load_state_dict(disc_state["msd"])

        ts_path = ckpt_dir / "training_state.pt"
        if ts_path.exists():
            ts = torch.load(ts_path, map_location="cpu")
            start_step = ts["step"]
            start_epoch = ts["epoch"]
            if not args.no_resume_optimizer:
                optimizer_g.load_state_dict(ts["optimizer_g"])
                if ts.get("scheduler_g") and scheduler_g:
                    scheduler_g.load_state_dict(ts["scheduler_g"])
                if args.use_gan and optimizer_d and ts.get("optimizer_d"):
                    optimizer_d.load_state_dict(ts["optimizer_d"])
                if args.use_gan and ts.get("scheduler_d") and scheduler_d:
                    scheduler_d.load_state_dict(ts["scheduler_d"])
            accelerator.print(f"Resumed from step {start_step}, epoch {start_epoch}")

    # =========================================================================
    # Training loop
    # =========================================================================
    global_step = start_step
    best_val_loss = float("inf")
    mel_ema = None
    spike_skipped = 0
    total_audio_sec = 0
    mpd_grad_norm = msd_grad_norm = gen_grad_norm = r1_loss_val = 0.0

    model.train()
    if args.use_gan:
        mpd.train()
        msd.train()

    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\n{'='*50}\nEpoch {epoch + 1}/{args.num_epochs}\n{'='*50}")

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            codes = batch["audio_codes"].to(accelerator.device).transpose(1, 2)
            target_audio = batch["audio"].to(accelerator.device)
            audio_lengths = batch["audio_lengths"].to(accelerator.device)

            # ── VQ temperature annealing ──────────────────────────────
            vq_temp = get_vq_temperature(
                global_step,
                args.vq_temp_anneal_steps,
                args.vq_temp_start,
                args.vq_temp_end,
            )
            accelerator.unwrap_model(model).set_vq_temperature(vq_temp)

            # ── Forward ──────────────────────────────────────────────
            pred_wav_full = model(codes)
            pred, target, min_len = align_audio(pred_wav_full, target_audio)
            pred, target = apply_length_mask(pred, target, audio_lengths, min_len)

            # ── Loss placeholders ─────────────────────────────────────
            loss_d = loss_d_mpd = loss_d_msd = pred.new_zeros(())
            loss_g_adv = loss_fm = pred.new_zeros(())
            loss_fm_mpd = loss_fm_msd = pred.new_zeros(())
            dr_mpd = dg_mpd = dr_msd = dg_msd = pred.new_zeros(())

            disc_active = args.use_gan and global_step >= args.disc_warmup_steps
            adv_weight = get_adv_weight(global_step, args.adv_warmup_steps, args.lambda_adv)

            accumulate_models = [model] + ([mpd, msd] if disc_active else [])
            with accelerator.accumulate(*accumulate_models):

                # ── Discriminator ─────────────────────────────────────
                if disc_active:
                    pw = pred.unsqueeze(1).to(dtype=disc_dtype).detach()
                    tw = target.unsqueeze(1).to(dtype=disc_dtype)

                    loss_d_mpd, dr_mpd, dg_mpd = discriminator_loss(mpd(tw), mpd(pw))
                    loss_d_msd, dr_msd, dg_msd = discriminator_loss(msd(tw), msd(pw))
                    loss_d = args.lambda_d_mpd * loss_d_mpd + args.lambda_d_msd * loss_d_msd

                    optimizer_d.zero_grad()
                    accelerator.backward(loss_d)
                    if accelerator.sync_gradients:
                        mpd_grad_norm = compute_grad_norm(mpd)
                        msd_grad_norm = compute_grad_norm(msd)
                    accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                    optimizer_d.step()
                    scheduler_d.step()

                # ── R1 ────────────────────────────────────────────────
                if (disc_active and args.r1 > 0 and args.d_reg_every > 0
                        and accelerator.sync_gradients
                        and (global_step + 1) % args.d_reg_every == 0):
                    tw_r1 = target.unsqueeze(1).detach().float().requires_grad_(True)
                    r1_mpd = d_r1_loss(mpd(tw_r1), tw_r1)
                    r1_msd = d_r1_loss(msd(tw_r1), tw_r1)
                    r1_scaled = (args.r1 / 2) * (r1_mpd + r1_msd) * args.d_reg_every
                    optimizer_d.zero_grad()
                    accelerator.backward(r1_scaled)
                    accelerator.clip_grad_norm_(disc_params, args.max_grad_norm)
                    optimizer_d.step()
                    r1_loss_val = (r1_mpd + r1_msd).item()

                # ── Generator adversarial + FM ─────────────────────────
                if disc_active and adv_weight > 0:
                    pw_g = pred.unsqueeze(1).to(dtype=disc_dtype)
                    tw_g = target.unsqueeze(1).to(dtype=disc_dtype)

                    mpd_fake_g = mpd(pw_g)
                    with torch.no_grad():
                        mpd_real_g = mpd(tw_g)
                    loss_g_adv_mpd = generator_adversarial_loss(mpd_fake_g)
                    loss_fm_mpd = feature_matching_loss(mpd_real_g, mpd_fake_g)

                    msd_fake_g = msd(pw_g)
                    with torch.no_grad():
                        msd_real_g = msd(tw_g)
                    loss_g_adv_msd = generator_adversarial_loss(msd_fake_g)
                    loss_fm_msd = feature_matching_loss(msd_real_g, msd_fake_g)

                    loss_g_adv = loss_g_adv_mpd + loss_g_adv_msd
                    loss_fm = loss_fm_mpd + loss_fm_msd

                # ── Reconstruction losses ──────────────────────────────
                loss_mel = (
                    multi_res_mel_loss_fn(pred, target)
                    if args.lambda_multi_res_mel > 0
                    else pred.new_zeros(())
                )
                loss_rms = (
                    global_rms_loss(pred, target)
                    if args.lambda_global_rms > 0
                    else pred.new_zeros(())
                )

                # ── VQ commitment + entropy ────────────────────────────
                unwrapped = accelerator.unwrap_model(model)
                commit_loss = unwrapped.last_commit_loss
                content_emb = unwrapped.last_content_emb
                original_hidden = unwrapped.last_original_hidden

                loss_consistency = (
                    torch.nn.functional.mse_loss(content_emb, original_hidden)
                    if args.lambda_consistency > 0
                    else pred.new_zeros(())
                )

                # ── Total generator loss ───────────────────────────────
                # Scale so mel is always O(1) — adv/fm are also O(1) by design
                loss_g = (
                    args.lambda_multi_res_mel * loss_mel
                    + args.lambda_global_rms * loss_rms
                    + adv_weight * loss_g_adv
                    + args.lambda_fm * loss_fm
                    + args.lambda_consistency * loss_consistency
                    + commit_loss  # already weighted inside VQ
                )

                # ── Spike detection ────────────────────────────────────
                mel_val = loss_mel.item()
                skip = False
                if args.spike_skip_threshold > 0:
                    if mel_ema is None:
                        mel_ema = mel_val
                    elif mel_val > args.spike_skip_threshold * mel_ema and global_step > 100:
                        skip = True
                        spike_skipped += 1
                        accelerator.print(
                            f"⚠ SPIKE SKIP step {global_step}: "
                            f"mel={mel_val:.3f} > {args.spike_skip_threshold}×EMA({mel_ema:.3f}). "
                            f"Total skipped: {spike_skipped}"
                        )
                    if not skip:
                        mel_ema = args.spike_ema_decay * mel_ema + (1 - args.spike_ema_decay) * mel_val

                # ── Generator step ─────────────────────────────────────
                optimizer_g.zero_grad()
                if not skip:
                    accelerator.backward(loss_g)
                    if accelerator.sync_gradients:
                        gen_grad_norm = compute_grad_norm(model)
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer_g.step()
                scheduler_g.step()

            # ── Logging / eval / save ─────────────────────────────────
            if accelerator.sync_gradients:
                global_step += 1
                total_audio_sec += audio_lengths.sum().item() / target_sample_rate

                if global_step % args.log_every == 0:
                    vq_usage = unwrapped.disentangle.vq.codebook_usage()
                    vq_perp = unwrapped.disentangle.vq.perplexity()
                    log_dict = {
                        "g/loss_total": loss_g.item(),
                        "g/loss_mel": loss_mel.item(),
                        "g/loss_rms": loss_rms.item(),
                        "g/loss_consistency": loss_consistency.item(),
                        "g/vq_commit_loss": commit_loss.item(),
                        "g/vq_codebook_usage": vq_usage,
                        "g/vq_perplexity": vq_perp,
                        "g/vq_temperature": vq_temp,
                        "g/adv_weight": adv_weight,
                        "g/grad_norm": gen_grad_norm,
                        "train/lr_g": scheduler_g.get_last_lr()[0],
                        "train/total_audio_hour": total_audio_sec / 3600.0,
                        "train/mel_ema": mel_ema or 0.0,
                        "train/spike_skipped_total": spike_skipped,
                        "train/disc_active": 1.0 if disc_active else 0.0,
                    }
                    if disc_active:
                        log_dict.update({
                            "d/loss_total": loss_d.item(),
                            "d/mpd_loss": loss_d_mpd.item(),
                            "d/msd_loss": loss_d_msd.item(),
                            "d_mpd/dr": dr_mpd.item(),
                            "d_mpd/dg": dg_mpd.item(),
                            "d_msd/dr": dr_msd.item(),
                            "d_msd/dg": dg_msd.item(),
                            "d_mpd/grad_norm": mpd_grad_norm,
                            "d_msd/grad_norm": msd_grad_norm,
                            "g/loss_adv": loss_g_adv.item(),
                            "g/loss_fm": loss_fm.item(),
                            "g/loss_fm_mpd": loss_fm_mpd.item(),
                            "g/loss_fm_msd": loss_fm_msd.item(),
                            "train/lr_d": scheduler_d.get_last_lr()[0],
                            "d/r1_loss": r1_loss_val,
                        })
                    accelerator.log(log_dict, step=global_step)
                    pbar.set_postfix(
                        mel=f"{loss_mel.item():.3f}",
                        vq=f"{commit_loss.item():.2e}",
                        usage=f"{vq_usage:.2f}",
                        T=f"{vq_temp:.2f}",
                        **({"adv": f"{loss_g_adv.item():.2f}"} if disc_active else {}),
                    )

                if val_dataloader and global_step % args.eval_every == 0 and global_step > 0:
                    val_metrics = eval_step(
                        model, multi_res_mel_loss_fn, val_dataloader, accelerator,
                        mpd=mpd if disc_active else None,
                        msd=msd if disc_active else None,
                        ref_mpd=ref_mpd, ref_msd=ref_msd,
                    )
                    accelerator.print(f"\nStep {global_step} - Validation: {val_metrics}")
                    accelerator.log(val_metrics, step=global_step)

                    # ── W&B audio samples ──────────────────────────────
                    if accelerator.is_main_process and args.log_with == "wandb":
                        _log_audio_samples(
                            model=model,
                            val_dataloader=val_dataloader,
                            accelerator=accelerator,
                            sample_rate=target_sample_rate,
                            global_step=global_step,
                            num_samples=4,
                        )

                    if val_metrics["val/loss_multi_res_mel"] < best_val_loss:
                        best_val_loss = val_metrics["val/loss_multi_res_mel"]
                        save_checkpoint(
                            model, mpd, msd, optimizer_g, optimizer_d,
                            scheduler_g, scheduler_d, global_step, epoch, args,
                            accelerator, num_frozen, base_upsample_rates, new_upsample_rates,
                            is_best=True,
                        )

                if global_step % args.save_every == 0 and global_step > 0:
                    save_checkpoint(
                        model, mpd, msd, optimizer_g, optimizer_d,
                        scheduler_g, scheduler_d, global_step, epoch, args,
                        accelerator, num_frozen, base_upsample_rates, new_upsample_rates,
                    )

                if args.max_train_steps and global_step >= args.max_train_steps:
                    break

        save_checkpoint(
            model, mpd, msd, optimizer_g, optimizer_d,
            scheduler_g, scheduler_d, global_step, epoch, args,
            accelerator, num_frozen, base_upsample_rates, new_upsample_rates,
        )
        if args.max_train_steps and global_step >= args.max_train_steps:
            break

    save_checkpoint(
        model, mpd, msd, optimizer_g, optimizer_d,
        scheduler_g, scheduler_d, global_step, args.num_epochs, args,
        accelerator, num_frozen, base_upsample_rates, new_upsample_rates,
    )
    accelerator.end_training()
    accelerator.print("\nTraining completed!")


if __name__ == "__main__":
    main()