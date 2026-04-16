#!/usr/bin/env python3
# coding=utf-8
"""
StudentCodec Distillation Trainer

Training pipeline:
  Phase 0 (0–2k steps):   KD-heavy, no GAN — student learns from teacher
  Phase 1 (2k–5k steps):  Balanced KD + mel  — transition
  Phase 2 (5k+ steps):    GAN enabled + KD regularization — perceptual polish

WandB logs (every log_every steps):
  Losses:
    g/loss_total, g/loss_feat, g/loss_acoustic, g/loss_mel,
    g/loss_rms, g/loss_vq, g/loss_adv, g/loss_fm,
    d/loss_total, d/loss_mpd, d/loss_msd
  Lambdas:
    lambda/feat, lambda/acoustic, lambda/mel, lambda/commit, lambda/adv
  VQ Health:
    vq/content_perplexity, vq/speaker_perplexity,
    vq/content_utilization, vq/speaker_utilization
  Training:
    train/lr, train/step, train/phase, train/audio_hours
    train/grad_norm_generator, train/grad_norm_discriminator
  Audio (every eval_every steps):
    audio/sample{1-4}_pred, audio/sample{1-4}_target
    audio/voice_convert_0to1 (swap speaker between samples 0 and 1)
    audio/teacher_recon      (what Qwen produces, quality reference)
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchaudio
from accelerate import Accelerator
from accelerate.utils import set_seed

sys.path.insert(0, str(Path(__file__).parent.parent.parent))          # project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))  # src/ for dataset, etc.
sys.path.append(str(Path(__file__).parent.parent.parent / "xcodec2"))

from dataset import create_webdataset_loader
from student.model import StudentCodec
from student.losses import (
    MultiResolutionMelLoss,
    feature_distillation_loss,
    vq_loss,
    discriminator_loss,
    generator_adversarial_loss,
    feature_matching_loss,
    global_rms_loss,
    LambdaScheduler,
)


# ─── Teacher Loading ──────────────────────────────────────────────────────────

def load_teacher(model_path: str, device, dtype):
    """Load frozen Qwen teacher. Returns the tokenizer object."""
    print(f"Loading teacher: {model_path}")
    from qwen_tts import Qwen3TTSTokenizer
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        model_path,
        attn_implementation="eager",
        dtype=dtype,
        device_map=str(device) if str(device) != "cpu" else None,
    )
    tokenizer.model.eval()
    for p in tokenizer.model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in tokenizer.model.parameters())
    print(f"  Teacher loaded and frozen ({n_params/1e6:.1f}M params)")
    return tokenizer


@torch.no_grad()
def teacher_encode_decode(teacher, audio: torch.Tensor, sr: int = 24_000):
    """Get teacher hidden states and reconstructed audio.

    Returns:
        teacher_hidden: [B, T, hidden_dim]
        teacher_wav:    [B, S]
    """
    decoder = teacher.model.decoder
    encoded = teacher.encode(audios=audio.cpu().numpy().tolist(), sr=sr)
    codes   = encoded.audio_codes   # list of [seq_len, num_quantizers]

    # Stack into batch tensor [B, num_quantizers, seq_len]
    max_len = max(c.shape[0] for c in codes)
    codes_batch = torch.zeros(len(codes), codes[0].shape[-1], max_len, dtype=torch.long)
    for i, c in enumerate(codes):
        codes_batch[i, :, :c.shape[0]] = c.T
    codes_batch = codes_batch.to(audio.device)

    hidden = decoder.quantizer.decode(codes_batch)
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state  # [B, T, H]

    # Decode
    x = hidden.permute(0, 2, 1)
    for blocks in decoder.upsample:
        for block in blocks:
            x = block(x)
    wav = x
    for block in decoder.decoder:
        wav = block(wav)
    teacher_wav = wav.clamp(-1, 1).squeeze(1)  # [B, S]

    return hidden, teacher_wav


# ─── Discriminators (reuse from xcodec2) ─────────────────────────────────────

def build_discriminators(use_gan: bool):
    if not use_gan:
        return None, None
    from xcodec2.module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
    mpd = HiFiGANMultiPeriodDiscriminator()
    msd = SpecDiscriminator()
    return mpd, msd


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("StudentCodec Distillation Trainer")

    # Data
    p.add_argument("--train_shards", type=str, required=True)
    p.add_argument("--val_shards",   type=str, default=None)
    p.add_argument("--max_audio_length", type=float, default=7.0)
    p.add_argument("--min_audio_length", type=float, default=1.0)
    p.add_argument("--target_sample_rate", type=int, default=24_000)

    # Teacher
    p.add_argument("--teacher_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")

    # Student model
    p.add_argument("--d_model",         type=int, default=256)
    p.add_argument("--conformer_layers", type=int, default=6)
    p.add_argument("--conformer_heads",  type=int, default=4)
    p.add_argument("--content_vocab",    type=int, default=4096)
    p.add_argument("--speaker_vocab",    type=int, default=1024)

    # Training
    p.add_argument("--output_dir",    type=str, default="output/student")
    p.add_argument("--resume_from",   type=str, default=None)
    p.add_argument("--batch_size",    type=int, default=8)
    p.add_argument("--grad_accum",    type=int, default=2)
    p.add_argument("--max_steps",     type=int, default=10_000)
    p.add_argument("--warmup_steps",  type=int, default=2_000)
    p.add_argument("--main_steps",    type=int, default=5_000)
    p.add_argument("--lr_g",    type=float, default=2e-4)
    p.add_argument("--lr_d",    type=float, default=4e-4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16")
    p.add_argument("--num_workers", type=int, default=4)

    # Logging / saving
    p.add_argument("--log_every",  type=int, default=10)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--wandb_project",  type=str, default="StudentCodec")
    p.add_argument("--wandb_run_name", type=str, default="student-v1")

    return p.parse_args()


# ─── Main Training Loop ───────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.grad_accum,
        log_with="wandb",
        project_dir=args.output_dir,
    )

    accelerator.init_trackers(
        args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": args.wandb_run_name}},
    )

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    # ── Build student model ───────────────────────────────────────
    student = StudentCodec(
        input_sr=args.target_sample_rate,
        d_model=args.d_model,
        conformer_layers=args.conformer_layers,
        conformer_heads=args.conformer_heads,
        content_vocab_size=args.content_vocab,
        speaker_vocab_size=args.speaker_vocab,
    )
    accelerator.print("\n" + "═"*60)
    accelerator.print("  StudentCodec Architecture")
    accelerator.print("═"*60)
    for k, v in student.info().items():
        accelerator.print(f"  {k:<40} {v}")
    accelerator.print("═"*60 + "\n")

    # ── Losses ───────────────────────────────────────────────────
    mel_loss_fn = MultiResolutionMelLoss(sample_rate=args.target_sample_rate)
    lambda_sched = LambdaScheduler(
        warmup_steps=args.warmup_steps,
        main_steps=args.main_steps,
    )

    # ── Discriminators ───────────────────────────────────────────
    mpd, msd = build_discriminators(use_gan=False)  # enabled later in phase 2

    # ── Optimizers ───────────────────────────────────────────────
    opt_g = torch.optim.AdamW(
        student.parameters(), lr=args.lr_g,
        betas=(0.8, 0.99), weight_decay=0.01,
    )
    sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_g, T_max=args.max_steps, eta_min=1e-6,
    )

    # ── Data ─────────────────────────────────────────────────────
    import glob
    train_pattern = sorted(glob.glob(args.train_shards)) or args.train_shards
    train_loader  = create_webdataset_loader(
        shard_pattern=train_pattern,
        target_sample_rate=args.target_sample_rate,
        max_audio_length=args.max_audio_length,
        min_audio_length=args.min_audio_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_buffer=500,
    )

    val_loader = None
    if args.val_shards:
        val_pattern = sorted(glob.glob(args.val_shards)) or args.val_shards
        val_loader = create_webdataset_loader(
            shard_pattern=val_pattern,
            target_sample_rate=args.target_sample_rate,
            max_audio_length=args.max_audio_length,
            min_audio_length=args.min_audio_length,
            batch_size=args.batch_size,
            num_workers=2,
            shuffle_buffer=0,
        )

    # ── Accelerate prepare ───────────────────────────────────────
    student, opt_g, sched_g, train_loader = accelerator.prepare(
        student, opt_g, sched_g, train_loader
    )

    # ── Load teacher (after accelerator.prepare to avoid wrapping) ──
    teacher = load_teacher(
        args.teacher_path,
        device=accelerator.device,
        dtype=dtype,
    )

    # ── Discriminator setup (phase 2) ────────────────────────────
    opt_d = None
    gan_enabled = False

    def _enable_gan():
        nonlocal mpd, msd, opt_d, gan_enabled
        if gan_enabled:
            return
        accelerator.print("\n🔥 Phase 2: Enabling GAN discriminators!")
        mpd, msd = build_discriminators(use_gan=True)
        mpd = mpd.to(accelerator.device)
        msd = msd.to(accelerator.device)
        opt_d = torch.optim.AdamW(
            list(mpd.parameters()) + list(msd.parameters()),
            lr=args.lr_d, betas=(0.8, 0.99),
        )
        mpd, msd, opt_d = accelerator.prepare(mpd, msd, opt_d)
        gan_enabled = True

    # ── Training State ───────────────────────────────────────────
    global_step     = 0
    best_val_loss   = float("inf")
    total_audio_sec = 0.0
    output_dir      = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accelerator.print(f"Training for {args.max_steps:,} steps on {accelerator.device}")
    accelerator.print(f"  Phase 0: KD only (0 → {args.warmup_steps:,})")
    accelerator.print(f"  Phase 1: KD + mel ({args.warmup_steps:,} → {args.main_steps:,})")
    accelerator.print(f"  Phase 2: GAN + KD ({args.main_steps:,} → {args.max_steps:,})")

    # ── Training loop ─────────────────────────────────────────────
    t0 = time.time()

    for epoch in range(1000):
        for batch in train_loader:
            if global_step >= args.max_steps:
                break

            audio        = batch["audio"].to(accelerator.device)            # [B, S]
            audio_lengths = batch["audio_lengths"].to(accelerator.device)   # [B]

            lambdas = lambda_sched.get(global_step)

            # Enable GAN at phase 2
            if global_step >= args.main_steps:
                _enable_gan()

            with accelerator.accumulate(student):

                # ── Teacher forward (no grad) ─────────────────────
                with torch.no_grad():
                    try:
                        teacher_hidden, teacher_wav = teacher_encode_decode(
                            teacher, audio, sr=args.target_sample_rate
                        )
                        has_teacher = True
                    except Exception as e:
                        # Teacher may fail on some samples — skip KD for that batch
                        accelerator.print(f"Teacher encode failed (skipping KD): {e}")
                        teacher_hidden = None
                        teacher_wav    = None
                        has_teacher    = False

                # ── Student forward ───────────────────────────────
                out = student(audio)
                student_wav     = out["wav"]                  # [B, S]
                student_feat    = out["features"]             # [B, T, d_model]
                teacher_proj    = out["teacher_proj"]         # [B, T, teacher_dim]
                vq_metrics      = out["vq_metrics"]

                # ── Compute losses ────────────────────────────────
                losses = {}

                # Feature distillation
                if has_teacher and lambdas["feat"] > 0:
                    l_feat = feature_distillation_loss(teacher_proj, teacher_hidden)
                    losses["feat"] = lambdas["feat"] * l_feat
                else:
                    l_feat = torch.tensor(0.0, device=audio.device)

                # Acoustic distillation (student vs teacher audio)
                min_len = min(student_wav.shape[-1], teacher_wav.shape[-1]) if has_teacher else 0
                if has_teacher and lambdas["acoustic"] > 0 and min_len > 0:
                    l_acoustic = mel_loss_fn(
                        student_wav[..., :min_len],
                        teacher_wav[..., :min_len],
                    )
                    losses["acoustic"] = lambdas["acoustic"] * l_acoustic
                else:
                    l_acoustic = torch.tensor(0.0, device=audio.device)

                # Ground truth mel loss
                min_len_gt = min(student_wav.shape[-1], audio.shape[-1])
                l_mel = mel_loss_fn(student_wav[..., :min_len_gt], audio[..., :min_len_gt])
                losses["mel"] = lambdas["mel"] * l_mel

                # Global RMS
                l_rms = global_rms_loss(student_wav[..., :min_len_gt], audio[..., :min_len_gt])
                losses["rms"] = lambdas["rms"] * l_rms

                # VQ commitment
                l_vq, vq_log = vq_loss(vq_metrics)
                losses["vq"] = lambdas["commit"] * l_vq

                # ── GAN (phase 2 only) ────────────────────────────
                l_adv = torch.tensor(0.0, device=audio.device)
                l_fm  = torch.tensor(0.0, device=audio.device)
                l_d   = torch.tensor(0.0, device=audio.device)

                if gan_enabled and lambdas["adv"] > 0:
                    target_wav = audio[..., :min_len_gt].unsqueeze(1)
                    pred_wav   = student_wav[..., :min_len_gt].unsqueeze(1)

                    # Discriminator step
                    with torch.no_grad():
                        pred_wav_d = pred_wav.detach()
                    mpd_real = mpd(target_wav)
                    mpd_fake = mpd(pred_wav_d)
                    msd_real = msd(target_wav)
                    msd_fake = msd(pred_wav_d)
                    l_d_mpd, _, _ = discriminator_loss(mpd_real, mpd_fake)
                    l_d_msd, _, _ = discriminator_loss(msd_real, msd_fake)
                    l_d = l_d_mpd + l_d_msd

                    opt_d.zero_grad()
                    accelerator.backward(l_d)
                    accelerator.clip_grad_norm_(
                        list(mpd.parameters()) + list(msd.parameters()),
                        args.max_grad_norm,
                    )
                    opt_d.step()

                    # Generator adversarial
                    mpd_fake_g = mpd(pred_wav)
                    msd_fake_g = msd(pred_wav)
                    l_adv = generator_adversarial_loss(mpd_fake_g) + \
                            generator_adversarial_loss(msd_fake_g)

                    mpd_real_g = mpd(target_wav)
                    msd_real_g = msd(target_wav)
                    l_fm = feature_matching_loss(mpd_real_g, mpd_fake_g) + \
                           feature_matching_loss(msd_real_g, msd_fake_g)

                    losses["adv"] = lambdas["adv"] * l_adv
                    losses["fm"]  = lambdas["fm"]  * l_fm

                # ── Generator update ──────────────────────────────
                loss_g = sum(losses.values())

                opt_g.zero_grad()
                accelerator.backward(loss_g)

                grad_norm_g = 0.0
                if accelerator.sync_gradients:
                    grad_norm_g = compute_grad_norm(accelerator.unwrap_model(student))
                    accelerator.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                opt_g.step()
                sched_g.step()

            # ── Count step ───────────────────────────────────────
            if accelerator.sync_gradients:
                global_step += 1
                total_audio_sec += audio_lengths.sum().item() / args.target_sample_rate

                current_phase = (
                    "0-KD_only" if global_step < args.warmup_steps else
                    "1-balanced" if global_step < args.main_steps else
                    "2-GAN"
                )

                # ── WandB logging ─────────────────────────────────
                if global_step % args.log_every == 0 and accelerator.is_main_process:
                    elapsed = time.time() - t0
                    steps_per_sec = global_step / max(elapsed, 1)
                    eta_sec = (args.max_steps - global_step) / max(steps_per_sec, 1e-6)

                    log = {
                        # Core losses
                        "g/loss_total":     loss_g.item(),
                        "g/loss_feat":      l_feat.item(),
                        "g/loss_acoustic":  l_acoustic.item(),
                        "g/loss_mel":       l_mel.item(),
                        "g/loss_rms":       l_rms.item(),
                        "g/loss_vq":        l_vq.item(),
                        "g/loss_adv":       l_adv.item(),
                        "g/loss_fm":        l_fm.item(),
                        "d/loss_total":     l_d.item(),
                        # Lambda schedule
                        "lambda/feat":      lambdas["feat"],
                        "lambda/acoustic":  lambdas["acoustic"],
                        "lambda/mel":       lambdas["mel"],
                        "lambda/commit":    lambdas["commit"],
                        "lambda/adv":       lambdas["adv"],
                        # VQ health
                        **{k: (v.item() if torch.is_tensor(v) else v) for k, v in vq_log.items()},
                        # Training stats
                        "train/lr":              sched_g.get_last_lr()[0],
                        "train/grad_norm_g":     grad_norm_g,
                        "train/phase":           current_phase,
                        "train/audio_hours":     total_audio_sec / 3600.0,
                        "train/steps_per_sec":   steps_per_sec,
                        "train/eta_hours":       eta_sec / 3600.0,
                    }
                    accelerator.log(log, step=global_step)

                    # Print to console
                    accelerator.print(
                        f"Step {global_step:5d} | Phase {current_phase} | "
                        f"mel={l_mel.item():.4f} | feat={l_feat.item():.4f} | "
                        f"vq_util={vq_log.get('vq/content_utilization', 0):.2f} | "
                        f"lr={sched_g.get_last_lr()[0]:.2e} | "
                        f"ETA {eta_sec/3600:.1f}h"
                    )

                # ── Audio + Eval logging ──────────────────────────
                if global_step % args.eval_every == 0 and accelerator.is_main_process:
                    _log_audio(
                        student=accelerator.unwrap_model(student),
                        teacher=teacher,
                        val_loader=val_loader or train_loader,
                        mel_loss_fn=mel_loss_fn,
                        accelerator=accelerator,
                        global_step=global_step,
                        sr=args.target_sample_rate,
                    )

                # ── Checkpoint ───────────────────────────────────
                if global_step % args.save_every == 0:
                    _save_checkpoint(
                        student=accelerator.unwrap_model(student),
                        opt_g=opt_g,
                        step=global_step,
                        output_dir=output_dir,
                        accelerator=accelerator,
                        is_best=False,
                    )

        if global_step >= args.max_steps:
            break

    # Save final checkpoint
    _save_checkpoint(
        student=accelerator.unwrap_model(student),
        opt_g=opt_g,
        step=global_step,
        output_dir=output_dir,
        accelerator=accelerator,
        is_best=True,
    )
    accelerator.end_training()
    accelerator.print("✅ Training complete!")


# ─── Audio Logging ────────────────────────────────────────────────────────────

@torch.inference_mode()
def _log_audio(student, teacher, val_loader, mel_loss_fn, accelerator, global_step, sr):
    """Log rich audio samples to WandB:
      - 4× (pred, target) reconstruction pairs
      - 1× teacher reconstruction (quality reference)
      - 1× voice conversion demo (swap speaker between samples)
    """
    try:
        import wandb
        student.eval()

        batch = next(iter(val_loader))
        audio = batch["audio"].to(accelerator.device)   # [B, S]
        n_log = min(4, audio.shape[0])

        # Student reconstructions
        with torch.inference_mode():
            out = student(audio[:n_log])
        preds = out["wav"]

        # Teacher reconstructions
        _, teacher_wav = teacher_encode_decode(teacher, audio[:2], sr=sr)

        audio_log = {}

        for i in range(n_log):
            pred_np = preds[i].float().cpu().numpy()
            tgt_np  = audio[i].float().cpu().numpy()
            audio_log[f"audio/sample{i+1}_pred"]   = wandb.Audio(pred_np, sample_rate=sr, caption=f"step{global_step} student recon #{i+1}")
            audio_log[f"audio/sample{i+1}_target"] = wandb.Audio(tgt_np,  sample_rate=sr, caption=f"step{global_step} target #{i+1}")

        # Teacher reference (first sample)
        if teacher_wav is not None and teacher_wav.shape[0] > 0:
            audio_log["audio/teacher_recon"] = wandb.Audio(
                teacher_wav[0].float().cpu().numpy(),
                sample_rate=sr,
                caption=f"step{global_step} teacher (Qwen) reconstruction",
            )

        # Voice conversion demo: content[0] + speaker[1]
        if audio.shape[0] >= 2:
            try:
                feat_0 = student.encode_features(audio[0:1])
                feat_1 = student.encode_features(audio[1:2])

                content_emb_0, _, _, _,  _ = student.quantize(feat_0)
                _, speaker_emb_1, _, spk_1, _ = student.quantize(feat_1)

                vc_wav = student.decode(content_emb_0, speaker_emb_1)
                audio_log["audio/voice_convert_0to1"] = wandb.Audio(
                    vc_wav.squeeze().float().cpu().numpy(),
                    sample_rate=sr,
                    caption=f"step{global_step} VC: content[0] + speaker[1] (code={spk_1[0].item()})",
                )

                # Original samples for comparison
                audio_log["audio/vc_source"] = wandb.Audio(
                    audio[0].float().cpu().numpy(), sample_rate=sr,
                    caption=f"step{global_step} VC source (content donor)",
                )
                audio_log["audio/vc_speaker"] = wandb.Audio(
                    audio[1].float().cpu().numpy(), sample_rate=sr,
                    caption=f"step{global_step} VC speaker (voice donor)",
                )
            except Exception as vc_e:
                accelerator.print(f"VC logging failed: {vc_e}")

        # Speaker code histogram (for monitoring codebook usage)
        try:
            feat_b = student.encode_features(audio)
            _, _, _, spk_codes, _ = student.quantize(feat_b)
            code_table = wandb.Table(
                columns=["sample", "speaker_code"],
                data=[[i, c.item()] for i, c in enumerate(spk_codes)],
            )
            audio_log["vq/speaker_codes_batch"] = code_table
        except Exception:
            pass

        accelerator.log(audio_log, step=global_step)
        student.train()

    except Exception as e:
        accelerator.print(f"Audio logging failed: {e}")
        student.train()


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def _save_checkpoint(student, opt_g, step, output_dir, accelerator, is_best):
    tag = "checkpoint-best" if is_best else f"checkpoint-step-{step}"
    ckpt_dir = output_dir / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        student.save_pretrained(str(ckpt_dir))
        accelerator.print(f"  ✅ Saved {'best ' if is_best else ''}checkpoint → {ckpt_dir}")


if __name__ == "__main__":
    main()
