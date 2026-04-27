#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=82

# ── Rationale ─────────────────────────────────────────────────────────────────
# Run 81 had amazing reconstruction but ZERO speaker transfer — the content
# path (1024→1024→1024 identity init) leaked all speaker information.
#
# This run adds three fixes:
#   1. Content bottleneck (1024→128→1024): forces content to drop speaker info
#   2. Speaker adversarial (GRL): actively strips speaker from content path
#   3. VC mel verification: decodes swapped-speaker audio and verifies speaker
#      transfer at the waveform level (the "cloning discriminator")
#
# Expect: temporary reconstruction quality drop that recovers with training.
# Speaker transfer should emerge after warmup period.
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
    --lambda_speaker_adv   1.0  \
    --lambda_vc_mel        2.0  \
    --vc_mel_every         5    \
    --disentangle_warmup_steps 500 \
    \
    --content_bottleneck_dim 128 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-ContentBottleneck-VCMel"

