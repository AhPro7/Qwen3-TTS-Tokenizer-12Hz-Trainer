#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=95

# ── Rationale ─────────────────────────────────────────────────────────────────
# EXPERIMENT 95: SINGLE CODEBOOK VQ
#
# Merge 16 RVQ codebooks into ONE large codebook (16384 codes).
# Produces ONE discrete token per frame at 12.5 Hz.
# Pure reconstruction — no speaker disentanglement.
#
# Architecture:
#   frozen encoder → hidden [B, T, 1024]
#     → LayerNorm + MLP(1024 → 256)  [pre_vq_proj]
#     → VQ(dim=256, size=16384)      [single codebook]
#     → MLP(256 → 1024) + LayerNorm  [post_vq_proj]
#     → upsample + decoder blocks → waveform
#
# VQ alpha ramps 0→1 over 1000 steps (warm start: continuous first, then VQ).
# Entropy regularization prevents codebook collapse.
# Dead code reset reinitializes unused codes from batch data.
#
# Bitrate: log2(16384) * 12.5 = 175 bits/sec
# (vs. 16 * 10 * 12.5 = 2000 bits/sec for 16-codebook RVQ)
# ──────────────────────────────────────────────────────────────────────────────

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    \
    --single_codebook \
    --codebook_size 16384 \
    --vq_dim 256 \
    --entropy_weight 0.1 \
    \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_audio_length 7.0 \
    --min_audio_length 1.0 \
    \
    --lr_g 2e-4 \
    --lr_d 2e-4 \
    --beta1_g 0.8 \
    --beta2_g 0.99 \
    --beta1_d 0.8 \
    --beta2_d 0.99 \
    --warmup_steps 500 \
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
    --disentangle_warmup_steps 1000 \
    --prematch_prob        0.0  \
    --prematch_k           4    \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-SingleCodebook-16384"
