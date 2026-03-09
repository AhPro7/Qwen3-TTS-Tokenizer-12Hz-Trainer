#!/usr/bin/env python3
"""
Convert HuggingFace dataset to WebDataset format for Qwen3TTS training.

Input:  HuggingFace dataset with audio field (voice['audio']['array'] / ['sampling_rate'])
Output: WebDataset tar shards under <output_dir>/train/ and <output_dir>/val/
        Each sample contains:
          - .flac  : 48kHz mono 16-bit FLAC audio
          - .npy   : flattened int32 codec codes (shape: seq_len * 16,)

Usage:
    python scripts/hf_to_webdataset.py <dataset> <output_dir> [options]

Example:
    python scripts/hf_to_webdataset.py \
        "my-org/my-voice-dataset" ./output \
        --shard-size 1000 \
        --val-percent 5.0
"""

import argparse
import io
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import webdataset as wds
from datasets import load_dataset
from loguru import logger
from qwen_tts import Qwen3TTSTokenizer
from resampy import resample
from tqdm import tqdm


TOKENIZER_SR = 24000  # Qwen3TTS tokenizer input sample rate
OUTPUT_SR = 48000     # Output FLAC sample rate


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer(device: str) -> Qwen3TTSTokenizer:
    logger.info("Loading Qwen3TTS tokenizer (Qwen/Qwen3-TTS-Tokenizer-12Hz)...")
    return Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        dtype=torch.bfloat16,
        device_map=device,
    )


# ---------------------------------------------------------------------------
# Audio utilities
# ---------------------------------------------------------------------------

def prepare_audio(
    audio_array: np.ndarray, sr: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to mono float32 and resample to TOKENIZER_SR and OUTPUT_SR.

    Returns:
        (audio_24k, audio_48k) both as float32 ndarray
    """
    # Mono
    if audio_array.ndim > 1:
        audio_array = np.mean(audio_array, axis=-1)

    audio = audio_array.astype(np.float32)

    # Normalize if clipped
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val

    audio_24k = resample(audio, sr, TOKENIZER_SR) if sr != TOKENIZER_SR else audio.copy()
    audio_48k = resample(audio, sr, OUTPUT_SR) if sr != OUTPUT_SR else audio.copy()

    return audio_24k, audio_48k


def encode_flac(audio_48k: np.ndarray) -> bytes:
    """Encode float32 audio array to 16-bit FLAC bytes at OUTPUT_SR."""
    buf = io.BytesIO()
    audio_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(buf, audio_int16, OUTPUT_SR, format="flac", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def encode_npy(codes_np: np.ndarray) -> bytes:
    """Serialize numpy array to bytes."""
    buf = io.BytesIO()
    np.save(buf, codes_np)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Shard writer helper
# ---------------------------------------------------------------------------

class ShardWriter:
    """Writes WebDataset .tar shards to a directory, rolling over at shard_size."""

    def __init__(self, output_dir: Path, shard_size: int):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.shard_idx = 0
        self.count = 0
        self._writer: wds.TarWriter | None = None
        self._open_shard()

    def _open_shard(self):
        if self._writer is not None:
            self._writer.close()
        path = str(self.output_dir / f"shard-{self.shard_idx:06d}.tar")
        self._writer = wds.TarWriter(path)
        logger.debug(f"Opened shard: {path}")

    def write(self, key: str, flac_bytes: bytes, npy_bytes: bytes):
        if self.count > 0 and self.count % self.shard_size == 0:
            self.shard_idx += 1
            self._open_shard()
        self._writer.write({"__key__": key, "flac": flac_bytes, "npy": npy_bytes})
        self.count += 1

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def num_shards(self) -> int:
        return self.shard_idx + 1 if self.count > 0 else 0


# ---------------------------------------------------------------------------
# Tokenizer batching
# ---------------------------------------------------------------------------

@torch.inference_mode()
def tokenize_batch(
    audio_list: list[np.ndarray],
    tokenizer: Qwen3TTSTokenizer,
) -> list[np.ndarray]:
    """Encode a batch of 24kHz float32 audio arrays.

    Returns a list of int32 numpy arrays, each flattened (seq_len * 16,).
    """
    encoded = tokenizer.encode(audios=audio_list, sr=TOKENIZER_SR)
    return [
        codes.cpu().numpy().flatten().astype(np.int32)
        for codes in encoded.audio_codes
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace dataset to WebDataset format for Qwen3TTS training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        help="HuggingFace dataset name (e.g. 'my-org/my-dataset') or local path",
    )
    parser.add_argument(
        "output_dir",
        help="Output root directory; train/ and val/ sub-dirs are created here",
    )
    parser.add_argument(
        "--dataset-config",
        default=None,
        metavar="CONFIG",
        help="Dataset configuration / subset name",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="Number of samples per WebDataset shard (tar file)",
    )
    parser.add_argument(
        "--val-percent",
        type=float,
        default=0.1,
        help="Percentage of samples assigned to the validation set",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Skip audio clips longer than this many seconds",
    )
    parser.add_argument(
        "--batch-duration",
        type=float,
        default=160.0,
        help="Accumulate this many seconds of audio before calling the tokenizer",
    )
    parser.add_argument(
        "--shuffle-buffer",
        type=int,
        default=100,
        help="Streaming shuffle buffer size passed to dataset.shuffle()",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling and train/val splitting",
    )
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Device: {device}")

    tokenizer = load_tokenizer(device)

    # Prepare output directories
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset} (split={args.split})")
    load_kwargs: dict = {"split": args.split, "streaming": True}
    if args.dataset_config:
        load_kwargs["name"] = args.dataset_config
    dataset = load_dataset(args.dataset, **load_kwargs)
    shuffled = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train_writer = ShardWriter(train_dir, args.shard_size)
    val_writer = ShardWriter(val_dir, args.shard_size)

    rng = np.random.default_rng(args.seed)
    val_fraction = args.val_percent / 100.0

    # Pending batch accumulator
    pending_ids: list[int] = []
    pending_24k: list[np.ndarray] = []
    pending_48k: list[np.ndarray] = []
    pending_duration = 0.0
    global_idx = 0

    def flush():
        if not pending_24k:
            return
        codes_list = tokenize_batch(pending_24k, tokenizer)
        for idx, audio_48k, codes_np in zip(pending_ids, pending_48k, codes_list):
            flac_bytes = encode_flac(audio_48k)
            npy_bytes = encode_npy(codes_np)
            key = f"{idx:08d}"
            if rng.random() < val_fraction:
                val_writer.write(key, flac_bytes, npy_bytes)
            else:
                train_writer.write(key, flac_bytes, npy_bytes)
        pending_ids.clear()
        pending_24k.clear()
        pending_48k.clear()

    skipped = 0
    try:
        with tqdm(desc="Processing", unit="samples") as pbar:
            for voice in shuffled:
                audio_array = voice["audio"]["array"]
                sr = voice["audio"]["sampling_rate"]

                try:
                    audio_24k, audio_48k = prepare_audio(audio_array, sr)
                except Exception as e:
                    logger.warning(f"Audio processing failed (idx={global_idx}): {e}")
                    skipped += 1
                    continue

                duration = len(audio_24k) / TOKENIZER_SR
                if duration > args.max_duration:
                    skipped += 1
                    continue

                pending_ids.append(global_idx)
                pending_24k.append(audio_24k)
                pending_48k.append(audio_48k)
                pending_duration += duration
                global_idx += 1
                pbar.update(1)

                if pending_duration >= args.batch_duration:
                    flush()
                    pending_duration = 0.0

        flush()  # Remaining samples
    finally:
        train_writer.close()
        val_writer.close()

    logger.info(
        f"Done!\n"
        f"  Train : {train_writer.count:,} samples in {train_writer.num_shards} shard(s)\n"
        f"  Val   : {val_writer.count:,} samples in {val_writer.num_shards} shard(s)\n"
        f"  Skipped: {skipped:,}"
    )


if __name__ == "__main__":
    main()
