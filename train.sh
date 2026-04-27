#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=81

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Reconstruction + Disentanglement (NO GAN)
# ═══════════════════════════════════════════════════════════════════════════════
#
# WHY: Runs 77-79 all collapsed because the GAN discriminator detects any
# hidden-state perturbation from DisentangledProjection and kills the generator.
# This is a fundamental incompatibility — you CANNOT train disentanglement and
# a GAN from scratch simultaneously on a pretrained decoder.
#
# STRATEGY: Remove the discriminator entirely. Train ONLY with:
#   - Multi-resolution mel loss (reconstruction fidelity)
#   - Global RMS loss (volume matching)
#   - Orthogonality loss (speaker ⊥ content)
#   - Speaker identity loss (re-extraction consistency)
#   - Cycle consistency loss (speaker swap training)
#
# Without a GAN, there is NO adversarial collapse risk. The training is
# purely supervised and will converge reliably.
#
# EXPECTED: mel loss should steadily decrease to ~1.5-2.0 over 5000-10000 steps.
# Audio quality won't be as crisp as GAN-trained (no high-freq detail),
# but the disentanglement will actually WORK.
#
# NEXT: Once Phase 1 converges (mel < 2.0), run train_phase2.sh to add GAN
# for audio quality polish. The discriminator will now see a stable model
# and won't collapse.
# ═══════════════════════════════════════════════════════════════════════════════

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --resume_from  "${OUTPUT_DIR}/run18/checkpoint-step-218750" \
    \
    --no-use_gan \
    --train_full_decoder \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_audio_length 7.0 \
    --min_audio_length 1.0 \
    \
    --lr_g 2e-4 \
    --beta1_g 0.8 \
    --beta2_g 0.99 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --max_train_steps 10000 \
    --save_every 1000 \
    --eval_every 200 \
    --log_every 5 \
    \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    1.0  \
    --lambda_orth          0.5  \
    --lambda_consistency   0.0  \
    --lambda_speaker_id    0.5  \
    --lambda_cycle         0.2  \
    --disentangle_warmup_steps 1000 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --no_resume_optimizer \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-Phase1-NoGAN"
