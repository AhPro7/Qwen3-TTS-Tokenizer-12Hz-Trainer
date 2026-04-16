#!/usr/bin/env python3
# coding=utf-8
"""
Student Codec Voice Conversion Inference

Voice conversion = change 1 integer (speaker token).
No Qwen needed. Only the 34M student model.

Usage:
    python src/student_convert.py \
        --checkpoint output/student/run1/checkpoint-best \
        --source_audio source.wav \
        --target_audio target_speaker.wav \
        --output_audio converted.wav

    # Use a specific speaker code directly:
    python src/student_convert.py \
        --checkpoint output/student/run1/checkpoint-best \
        --source_audio source.wav \
        --speaker_code 472 \
        --output_audio converted.wav

    # Build a speaker library:
    python src/student_convert.py \
        --checkpoint output/student/run1/checkpoint-best \
        --build_speaker_library speakers/*.wav \
        --library_out speaker_library.json

    # Reconstruct (no conversion):
    python src/student_convert.py \
        --checkpoint output/student/run1/checkpoint-best \
        --source_audio source.wav \
        --reconstruct
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent))
from student.model import StudentCodec


def load_audio(path: str, sr: int = 24_000) -> np.ndarray:
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if orig_sr != sr:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, orig_sr, sr)
        audio = audio_t.squeeze(0).numpy()
    return audio


def save_audio(path: str, audio: np.ndarray, sr: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr)
    print(f"  → {path}  ({len(audio)/sr:.2f}s @ {sr}Hz)")


class StudentVoiceConverter:
    """Voice conversion engine using the trained StudentCodec."""

    def __init__(self, checkpoint: str, device: str = "auto"):
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        print(f"\nLoading StudentCodec from {checkpoint}...")
        self.model = StudentCodec.from_pretrained(checkpoint, device=str(device))
        self.model = self.model.to(self.device).eval()
        self.sr    = self.model.input_sr

        n = sum(p.numel() for p in self.model.parameters())
        print(f"  {n/1e6:.1f}M params | {self.sr}Hz | "
              f"content codebook={self.model.content_vocab_size} | "
              f"speaker codebook={self.model.speaker_vocab_size}")

    @torch.inference_mode()
    def get_speaker_code(self, audio_path: str) -> int:
        """Extract discrete speaker token from an audio file."""
        audio = load_audio(audio_path, self.sr)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        _, speaker_code = self.model.encode_audio(audio_t, sr=self.sr)
        code = speaker_code.item()
        print(f"  Speaker code from {Path(audio_path).name}: {code}")
        return code

    @torch.inference_mode()
    def convert(
        self,
        source_path: str,
        target_speaker: Union[str, int],
        output_path: str,
    ):
        """Voice conversion: source content + target speaker → output."""
        print(f"\n{'='*60}")
        print("Voice Conversion")
        print(f"  Source (content):  {source_path}")
        if isinstance(target_speaker, str):
            print(f"  Target (speaker):  {target_speaker}")
        else:
            print(f"  Target speaker code: {target_speaker}")
        print(f"{'='*60}")

        # Load source
        src_audio = load_audio(source_path, self.sr)
        src_t = torch.from_numpy(src_audio).unsqueeze(0).to(self.device)

        # Encode source → content tokens
        content_codes, src_speaker_code = self.model.encode_audio(src_t)
        print(f"  Content tokens:      {content_codes.shape[0]} @ {self.model.token_rate}Hz")
        print(f"  Source speaker code: {src_speaker_code.item()}")

        # Get target speaker code
        if isinstance(target_speaker, str):
            tgt_speaker_code = self.get_speaker_code(target_speaker)
        else:
            tgt_speaker_code = target_speaker
        print(f"  Target speaker code: {tgt_speaker_code}")

        # Decode: source content + target speaker
        wav = self.model.decode_codes(
            content_codes,
            speaker_code=tgt_speaker_code,
        )
        output = wav.float().cpu().numpy()

        save_audio(output_path, output, self.sr)
        print(f"\n✅ Converted: {source_path} → {output_path}")
        print(f"   Speaker: {src_speaker_code.item()} → {tgt_speaker_code}")

        return output

    @torch.inference_mode()
    def reconstruct(self, audio_path: str, output_path: str):
        """Reconstruct audio without conversion (quality check)."""
        audio = load_audio(audio_path, self.sr)
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)
        content_codes, speaker_code = self.model.encode_audio(audio_t)
        wav = self.model.decode_codes(content_codes, speaker_code=speaker_code.item())
        output = wav.float().cpu().numpy()
        save_audio(output_path, output, self.sr)
        print(f"✅ Reconstructed → {output_path}  (speaker code={speaker_code.item()})")
        return output

    @torch.inference_mode()
    def build_speaker_library(
        self,
        audio_files: List[str],
        output_path: str,
    ) -> Dict[str, int]:
        """Build a speaker library: filename → speaker_code mapping."""
        print(f"\nBuilding speaker library from {len(audio_files)} files...")
        library = {}
        for f in audio_files:
            try:
                code = self.get_speaker_code(f)
                library[Path(f).stem] = code
            except Exception as e:
                print(f"  Failed {f}: {e}")

        with open(output_path, "w") as fp:
            json.dump(library, fp, indent=2)

        print(f"\n✅ Speaker library saved → {output_path}")
        print(f"   {len(library)} speakers, "
              f"{len(set(library.values()))} unique codes "
              f"({len(set(library.values()))/self.model.speaker_vocab_size*100:.1f}% of vocab)")

        # Show distribution
        from collections import Counter
        counts = Counter(library.values())
        print(f"   Top 5 codes: {counts.most_common(5)}")
        return library


def parse_args():
    p = argparse.ArgumentParser("Student Codec Voice Conversion")
    p.add_argument("--checkpoint",    type=str, required=True)
    p.add_argument("--source_audio",  type=str, default=None)
    p.add_argument("--target_audio",  type=str, default=None)
    p.add_argument("--speaker_code",  type=int, default=None)
    p.add_argument("--output_audio",  type=str, default="converted.wav")
    p.add_argument("--reconstruct",   action="store_true")
    p.add_argument("--build_speaker_library", nargs="+", default=None)
    p.add_argument("--library_out",   type=str, default="speaker_library.json")
    p.add_argument("--device",        type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    vc = StudentVoiceConverter(args.checkpoint, device=args.device)

    if args.build_speaker_library:
        vc.build_speaker_library(args.build_speaker_library, args.library_out)
        return

    if args.reconstruct:
        assert args.source_audio, "--source_audio required for --reconstruct"
        vc.reconstruct(args.source_audio, args.output_audio)
        return

    assert args.source_audio, "--source_audio required"
    assert args.target_audio or args.speaker_code is not None, (
        "Provide --target_audio or --speaker_code"
    )

    target = args.target_audio if args.target_audio else args.speaker_code
    vc.convert(args.source_audio, target, args.output_audio)


if __name__ == "__main__":
    main()
