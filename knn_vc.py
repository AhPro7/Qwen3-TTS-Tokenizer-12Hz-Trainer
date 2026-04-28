#!/usr/bin/env python3
"""
kNN-VC: Voice Conversion via k-Nearest Neighbor Feature Matching.

Uses the frozen Qwen3-TTS-Tokenizer-12Hz model. No training needed.

How it works:
  1. Encode source audio → hidden features [T_s, 1024]
  2. Encode target audio(s) → feature pool [T_t, 1024]
  3. For each source frame, find the nearest neighbor in target pool
  4. Decode matched features → converted waveform

The target speaker's features naturally carry their vocal characteristics,
so the output has A's content spoken in B's voice.

Usage:
    # Single target utterance
    python knn_vc.py --source source.wav --target target.wav --output converted.wav

    # Multiple target utterances (better quality with larger feature pool)
    python knn_vc.py --source source.wav --target target1.wav target2.wav target3.wav

    # With target folder
    python knn_vc.py --source source.wav --target_dir /path/to/speaker_B_audios/
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent / "src"))

from qwen_tts import Qwen3TTSTokenizer

TOKENIZER_SR = 24_000


def load_audio(path):
    """Load and resample audio to 24kHz mono."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != TOKENIZER_SR:
        resampler = torchaudio.transforms.Resample(sr, TOKENIZER_SR)
        audio = resampler(torch.from_numpy(audio).float()).numpy()
    return audio


class KnnVoiceConverter:
    """kNN-VC using Qwen3-TTS hidden features."""

    def __init__(self, device="auto", dtype=torch.bfloat16):
        self.device = self._resolve_device(device)
        self.dtype = dtype

        print("Loading Qwen3-TTS-Tokenizer-12Hz...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz",
            attn_implementation="eager",
            dtype=dtype,
            device_map="cpu" if self.device.type == "cpu" else str(self.device),
        )
        self.decoder = self.tokenizer.model.decoder
        self.output_sr = self.tokenizer.get_output_sample_rate()
        print(f"Ready. Output: {self.output_sr} Hz")

    def _resolve_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(device)

    @torch.no_grad()
    def encode(self, audio: np.ndarray) -> torch.Tensor:
        """Audio → hidden features [1, T, 1024]"""
        encoded = self.tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
        codes = encoded.audio_codes[0].T.unsqueeze(0).to(self.device)

        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        return hidden

    @torch.no_grad()
    def decode(self, hidden: torch.Tensor) -> np.ndarray:
        """Hidden [1, T, 1024] → waveform."""
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(-1, 1).squeeze().cpu().float().numpy()

    def build_target_pool(self, target_audios: list) -> torch.Tensor:
        """Build feature pool from one or more target speaker utterances.

        More utterances = larger pool = better matching quality.

        Args:
            target_audios: List of audio file paths.

        Returns:
            pool: [N, 1024] — all target features concatenated.
        """
        all_features = []
        for path in target_audios:
            print(f"  Encoding target: {os.path.basename(path)}")
            audio = load_audio(path)
            hidden = self.encode(audio)  # [1, T, 1024]
            all_features.append(hidden.squeeze(0))  # [T, 1024]

        pool = torch.cat(all_features, dim=0)  # [N, 1024]
        print(f"  Target pool: {pool.shape[0]} frames")
        return pool

    @torch.no_grad()
    def convert(
        self,
        source_audio: np.ndarray,
        target_pool: torch.Tensor,
        k: int = 1,
        temperature: float = 10.0,
    ) -> np.ndarray:
        """Voice conversion via kNN feature matching.

        Args:
            source_audio: Source audio (numpy).
            target_pool: Target speaker feature pool [N, 1024].
            k: Number of nearest neighbors (1=hard match, 4=soft blend).
            temperature: Softmax temperature for k>1 weighting.

        Returns:
            Converted waveform (numpy).
        """
        # Encode source
        source_hidden = self.encode(source_audio)  # [1, T_s, 1024]
        src = source_hidden.squeeze(0)  # [T_s, 1024]

        # L2-normalize for cosine matching
        src_norm = F.normalize(src, dim=-1)
        tgt_norm = F.normalize(target_pool, dim=-1)

        # Cosine similarity [T_s, N]
        sim = torch.mm(src_norm, tgt_norm.t())

        # Top-k
        k = min(k, target_pool.shape[0])
        topk_sim, topk_idx = sim.topk(k, dim=-1)  # [T_s, k]

        if k == 1:
            # Hard match: just take the nearest neighbor
            matched = target_pool[topk_idx.squeeze(-1)]  # [T_s, 1024]
        else:
            # Soft match: weighted average of k neighbors
            weights = F.softmax(topk_sim * temperature, dim=-1)  # [T_s, k]
            matched = torch.zeros_like(src)
            for i in range(k):
                matched += weights[:, i:i+1] * target_pool[topk_idx[:, i]]

        matched = matched.unsqueeze(0)  # [1, T_s, 1024]

        # Decode
        return self.decode(matched)


def parse_args():
    parser = argparse.ArgumentParser(description="kNN-VC: Voice Conversion")
    parser.add_argument("--source", required=True, help="Source audio (content donor)")
    parser.add_argument("--target", nargs="+", default=None, help="Target audio(s) (speaker donor)")
    parser.add_argument("--target_dir", default=None, help="Directory with target speaker audio files")
    parser.add_argument("--output", default="knn_vc_output.wav", help="Output file path")
    parser.add_argument("--k", type=int, default=1, help="Number of nearest neighbors (1=best for voice change)")
    parser.add_argument("--temperature", type=float, default=10.0, help="Softmax temperature for k>1")
    parser.add_argument("--save_recon", action="store_true", help="Also save reconstruction of source and target")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main():
    args = parse_args()

    # Collect target files
    target_files = []
    if args.target:
        target_files = args.target
    if args.target_dir:
        for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
            target_files.extend(glob.glob(os.path.join(args.target_dir, ext)))
    if not target_files:
        print("ERROR: No target audio files. Use --target or --target_dir.")
        sys.exit(1)

    print("=" * 60)
    print("kNN Voice Conversion")
    print("=" * 60)

    converter = KnnVoiceConverter(device=args.device)

    # Build target pool
    print(f"\nBuilding target speaker pool ({len(target_files)} file(s)):")
    target_pool = converter.build_target_pool(target_files)

    # Load source
    print(f"\nSource: {args.source}")
    source_audio = load_audio(args.source)
    print(f"  Duration: {len(source_audio)/TOKENIZER_SR:.2f}s")

    # Convert
    print(f"\nConverting (k={args.k})...")
    t0 = time.perf_counter()
    converted = converter.convert(source_audio, target_pool, k=args.k, temperature=args.temperature)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.2f}s")
    print(f"  Output: {len(converted)} samples, {len(converted)/converter.output_sr:.2f}s")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sf.write(args.output, converted, converter.output_sr)
    print(f"\n✅ Saved → {args.output}")

    # Optional: save reconstructions
    if args.save_recon:
        parent = Path(args.output).parent
        stem = Path(args.output).stem

        src_hidden = converter.encode(source_audio)
        recon_src = converter.decode(src_hidden)
        p = parent / f"{stem}_recon_source.wav"
        sf.write(str(p), recon_src, converter.output_sr)
        print(f"  Source recon → {p}")

        tgt_audio = load_audio(target_files[0])
        tgt_hidden = converter.encode(tgt_audio)
        recon_tgt = converter.decode(tgt_hidden)
        p = parent / f"{stem}_recon_target.wav"
        sf.write(str(p), recon_tgt, converter.output_sr)
        print(f"  Target recon → {p}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
