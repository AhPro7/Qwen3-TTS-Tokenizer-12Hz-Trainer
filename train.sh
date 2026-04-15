#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=55

# ── PoC Rationale ─────────────────────────────────────────────────────────────
# Dataset: 50h, clips 0-7s → use max_audio_length 7.0 (no cropping waste)
# 50h ÷ avg5s × 95% ≈ 34k clips  →  34k ÷ batch8 ≈ 4250 steps/epoch
# 10 000 steps ≈ 2.3 epochs ≈ ~2-3h on A100  (enough for clear PoC signal)
#
# max_audio_length 7.0 : match real data — avoids random-crop information loss
# batch 8 vs 16       : 7s seqs are ~2× longer → half batch to keep VRAM safe
# lr_g 1e-4        : DisentangledProjection is fresh → needs higher LR
# lr_d 2e-4        : Standard 2× generator LR (HiFiGAN convention)
# warmup_steps 300 : Protect pretrained decoder from large early updates
# lambda_orth 0.5  : 5× stronger push for speaker/content separation
# train_full_decoder: Let decoder adapt to the new disentangled hidden repr
# ──────────────────────────────────────────────────────────────────────────────

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --resume_from  "${OUTPUT_DIR}/run18/checkpoint-step-218750" \
    \
    --train_full_decoder \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --max_audio_length 7.0 \
    --min_audio_length 1.0 \
    \
    --lr_g 1e-4 \
    --lr_d 2e-4 \
    --beta1_g 0.8 \
    --beta2_g 0.99 \
    --beta1_d 0.8 \
    --beta2_d 0.99 \
    --warmup_steps 300 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --max_train_steps 10000 \
    --save_every 500 \
    --eval_every 100 \
    --log_every 5 \
    \
    --lambda_adv           0.3  \
    --lambda_fm            3.0  \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    1.0  \
    --lambda_d_mpd         0.01 \
    --lambda_d_msd         0.1  \
    --lambda_orth          0.5  \
    \
    --no_resume_optimizer \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-PoC"
