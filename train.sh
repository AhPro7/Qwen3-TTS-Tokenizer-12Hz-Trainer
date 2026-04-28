#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=90

# ── Rationale ─────────────────────────────────────────────────────────────────
# RESIDUAL DISENTANGLEMENT (from Run 81 debug):
#
# Problem: content carries everything, speaker encoder produces same
# embedding for all speakers (cosine 0.95), voice conversion does nothing.
#
# Fix: content_proj sees (x - speaker_contrib.detach()), NOT raw x.
# Content structurally CAN'T carry speaker info (it's subtracted + detached).
# combined = speaker + content(x - speaker) = speaker + (x - speaker) = x
# → perfect reconstruction regardless of speaker magnitude.
#
# Speaker decoder uses Xavier init (not near-zero) → starts with meaningful
# contribution. Diversity + norm floor prevent collapse.
# ──────────────────────────────────────────────────────────────────────────────

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
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
    --max_train_steps 10000 \
    --save_every 500 \
    --eval_every 100 \
    --log_every 5 \
    \
    --lambda_adv           0.3  \
    --lambda_fm            3.0  \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    5.0  \
    --lambda_d_mpd         0.01 \
    --lambda_d_msd         0.1  \
    --lambda_orth          1.0  \
    --lambda_consistency   0.0  \
    --lambda_speaker_id    1.0  \
    --lambda_cycle         0.5  \
    --lambda_speaker_adv   0.0  \
    --lambda_speaker_div   1.0  \
    --content_dropout      0.0  \
    --disentangle_warmup_steps 500 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-ResidualDisentangle"
