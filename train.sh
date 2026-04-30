#!/bin/bash
# Run 228 — Fixed VQ bottleneck training
# Key changes from Run 227:
#   - VQ temperature annealing (2.0 → 0.5 over 3000 steps): fixes commit_loss ≈ 0
#   - Residual bottleneck in DisentangledProjection: prevents mel divergence
#   - disc_warmup_steps=1000: let generator stabilize before GAN pressure
#   - adv_warmup_steps=2000: ramp adv loss slowly to prevent GAN collapse
#   - Removed lambda_orth / lambda_speaker_id / lambda_cycle (no speaker path)
#   - codebook_size=2048, content_dim=256: stable and LLM-compatible
#   - lambda_adv=1.0 (was 0.3): adv is ramped slowly anyway, no need to pre-scale
#   - lambda_fm=2.0: feature matching is the most stable GAN signal

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SHARDS="${SCRIPT_DIR}/datasets3/train/*.tar"
VAL_SHARDS="${SCRIPT_DIR}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/qwen-tokenzier-v2"
RUN_NUMBER=230

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
    --lr_g 2e-5 \
    --lr_d 1e-4 \
    --beta1_g 0.8 \
    --beta2_g 0.99 \
    --beta1_d 0.8 \
    --beta2_d 0.99 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --max_train_steps 10000 \
    --save_every 1000 \
    --eval_every 100 \
    --log_every 5 \
    \
    --lambda_adv           1.0  \
    --lambda_fm            2.0  \
    --lambda_multi_res_mel 15.0 \
    --lambda_global_rms    1.0  \
    --lambda_d_mpd         1.0  \
    --lambda_d_msd         1.0  \
    --lambda_consistency   0.0  \
    \
    --disc_warmup_steps    1000 \
    --adv_warmup_steps     2000 \
    \
    --codebook_size        2048 \
    --content_dim          256  \
    --vq_commitment_cost   0.25 \
    --vq_entropy_loss_weight 0.05 \
    --vq_temp_start        2.0  \
    --vq_temp_end          0.5  \
    --vq_temp_anneal_steps 3000 \
    --vq_alpha_anneal_steps 5000 \
    \
    --spike_skip_threshold 3.0 \
    --spike_ema_decay      0.99 \
    \
    --train_full_decoder \
    --mixed_precision bf16 \
    --wandb_project  Qwen3-TTS-Tokenizer-12Hz-Trainer \
    --wandb_run_name "Run${RUN_NUMBER}-VQTempFix"