#!/bin/bash
# Student Codec Distillation Training
# Qwen3-TTS-Tokenizer-12Hz → StudentCodec (Conformer + ContentVQ + SpeakerVQ)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TRAIN_SHARDS="${PROJECT_ROOT}/datasets3/train/*.tar"
VAL_SHARDS="${PROJECT_ROOT}/datasets3/val/*.tar"
OUTPUT_DIR="/content/drive/MyDrive/student-codec"
RUN_NUMBER=1

# ── Phase schedule ────────────────────────────────────────────
# Steps:  0 → 2000  Phase 0: KD only (no GAN)        ~30 min
# Steps: 2000→5000  Phase 1: KD + mel balanced        ~45 min
# Steps: 5000→10000 Phase 2: GAN + KD regularization  ~75 min
# Total: 10000 steps ≈ 2.5h on A100

uv run accelerate launch \
    --mixed_precision fp16 \
    "${PROJECT_ROOT}/src/student/trainer.py" \
    \
    --train_shards  "${TRAIN_SHARDS}" \
    --val_shards    "${VAL_SHARDS}"   \
    --output_dir    "${OUTPUT_DIR}/run${RUN_NUMBER}" \
    --max_audio_length 7.0 \
    --min_audio_length 1.0 \
    --target_sample_rate 24000 \
    \
    --teacher_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
    \
    --d_model          256 \
    --conformer_layers 6   \
    --conformer_heads  4   \
    --content_vocab    4096 \
    --speaker_vocab    1024 \
    \
    --batch_size 8 \
    --grad_accum 2 \
    --max_steps  10000 \
    --warmup_steps 2000 \
    --main_steps   5000 \
    --lr_g 2e-4 \
    --lr_d 4e-4 \
    --max_grad_norm 1.0 \
    \
    --log_every  10  \
    --eval_every 250 \
    --save_every 500 \
    \
    --mixed_precision fp16 \
    --wandb_project  StudentCodec \
    --wandb_run_name "Run${RUN_NUMBER}-distill"
