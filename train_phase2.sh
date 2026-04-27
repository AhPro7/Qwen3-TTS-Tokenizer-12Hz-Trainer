#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=81

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: GAN Polish (run AFTER Phase 1 converges)
# ═══════════════════════════════════════════════════════════════════════════════
#
# PREREQUISITE: Phase 1 (train.sh / Run 80) must have converged with
#   val/loss_multi_res_mel < 2.0 before running this.
#
# STRATEGY: Resume from Phase 1 checkpoint, add discriminator with very
# conservative settings. The model already reconstructs well AND has
# working disentanglement — the GAN only needs to add high-frequency detail.
#
# Key differences from Phase 1:
#   - GAN enabled (--use_gan, default)
#   - Very low adversarial weight (0.05) — GAN is a polish, not the driver
#   - Low FM loss (1.0) — prevents the FM cascade that killed Runs 77-79
#   - Low LR (2e-5) — gentle fine-tuning on an already-good model
#   - Disentanglement at full strength from step 0 (already trained)
# ═══════════════════════════════════════════════════════════════════════════════

# ⚠️  UPDATE THIS PATH to point to your best Phase 1 checkpoint!
PHASE1_CHECKPOINT="${OUTPUT_DIR}/run80/checkpoint-best"

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --resume_from  "${PHASE1_CHECKPOINT}" \
    \
    --use_gan \
    --train_full_decoder \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --max_audio_length 7.0 \
    --min_audio_length 1.0 \
    \
    --lr_g 2e-5 \
    --lr_d 5e-5 \
    --beta1_g 0.8 \
    --beta2_g 0.99 \
    --beta1_d 0.8 \
    --beta2_d 0.99 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --max_train_steps 5000 \
    --save_every 500 \
    --eval_every 100 \
    --log_every 5 \
    \
    --lambda_adv           0.05 \
    --lambda_fm            1.0  \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    1.0  \
    --lambda_d_mpd         0.01 \
    --lambda_d_msd         0.1  \
    --lambda_orth          0.5  \
    --lambda_consistency   0.0  \
    --lambda_speaker_id    0.5  \
    --lambda_cycle         0.2  \
    --disentangle_warmup_steps 0 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --no_resume_optimizer \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-Phase2-GAN"
