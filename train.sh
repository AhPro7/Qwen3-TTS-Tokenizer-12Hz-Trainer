#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Settings ─────────────────────────────────────────────────────────────────
TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=86

# ── Rationale ─────────────────────────────────────────────────────────────────
# Run 84 mel stuck at 7-8 despite 2000 steps of reconstruction-only training.
#
# ROOT CAUSE: When resuming from Run 81 with strict=False, the old
# speaker_decoder weights (near-zero, because Run 81 never needed the speaker
# path) were loaded and overwrote the fresh Xavier init. This killed the
# speaker path — it couldn't contribute to reconstruction, so the content
# bottleneck (256-dim) had to carry EVERYTHING alone. Not enough capacity.
#
# Fixes:
#   --no_resume_disentangle: fresh DisentangledProjection with non-zero
#     speaker_decoder (Xavier gain=0.5). Both paths contribute from step 0.
#   --content_bottleneck_dim 512: wider bottleneck (50% retention).
#     256-dim was too tight — mel plateaued at ~7.
#   --gan_start_step 3000: longer reconstruction pre-training.
# ──────────────────────────────────────────────────────────────────────────────

uv run accelerate launch "${SCRIPT_DIR}/src/trainer.py" \
    --train_shards "${TRAIN_SHARDS}" \
    --val_shards   "${VAL_SHARDS}"   \
    --output_dir   "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    \
    --resume_from  "${OUTPUT_DIR}/run81/checkpoint-step-1000" \
    --no_resume_optimizer \
    --no_resume_discriminator \
    --no_resume_disentangle \
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
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --max_train_steps 10000 \
    --save_every 500 \
    --eval_every 100 \
    --log_every 5 \
    --gan_start_step 3000 \
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
    --lambda_speaker_adv   0.5  \
    --lambda_vc_mel        1.0  \
    --vc_mel_every         5    \
    --disentangle_warmup_steps 1500 \
    \
    --content_bottleneck_dim 512 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-FreshDisentangle-BN512"

