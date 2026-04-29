#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=98

# ── Rationale ─────────────────────────────────────────────────────────────────
# EXPERIMENT 98: SINGLE CODEBOOK VQ
#
# Follows the PROVEN pattern from Exp 81's content VQ:
#   1. Near-identity pre-projection (1024 → 1024) — NOT 256-dim!
#   2. VQ in same 1024-dim — codebook inits from meaningful hidden states
#   3. Single alpha blend (NO double-alpha that killed v1)
#   4. Direct VQ output to decoder (NO near-zero post_vq)
#
# Fixes from failed v1:
#   - v1 bug: double alpha (α²) → VQ got no gradients → codebook collapsed
#   - v1 bug: near-zero post_vq init → decoder got zeros → empty audio
#   - v1 bug: 256-dim projection → random init destroyed hidden structure
#
# 8192 codes × 1024-dim, same architecture as exp 81 content path.
# ──────────────────────────────────────────────────────────────────────────────

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    \
    --single_codebook \
    --codebook_size 8192 \
    --vq_dim 256 \
    --entropy_weight 0.1 \
    \
    --batch_size 4 \
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
    --max_train_steps 20000 \
    --save_every 1000 \
    --eval_every 200 \
    --log_every 5 \
    \
    --lambda_adv           0.3  \
    --lambda_fm            3.0  \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    5.0  \
    --lambda_d_mpd         0.01 \
    --lambda_d_msd         0.1  \
    --lambda_vq            1.0  \
    --lambda_orth          0.0  \
    --lambda_consistency   0.0  \
    --lambda_speaker_id    0.0  \
    --lambda_cycle         0.0  \
    --lambda_speaker_adv   0.0  \
    --lambda_speaker_div   0.0  \
    --content_dropout      0.0  \
    --disentangle_warmup_steps 500 \
    --prematch_prob        0.0  \
    --prematch_k           4    \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-SingleCodebook-8192-FIXED"
