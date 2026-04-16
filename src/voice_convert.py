#!/usr/bin/env python3
# coding=utf-8
# SPDX-License-Identifier: Apache-2.0

"""
Voice Conversion Inference Script (v2 — bottleneck architecture)

Takes two audio files:
  - source (content donor): the words/content we want to keep
  - target (speaker donor): the voice/timbre we want to transfer

Uses the bottleneck DisentangledProjection:
  Speaker path: 1024 → 256 → attention pool → global vector → broadcast
  Content path: per-frame MLP 1024 → 1024

Usage:
    python src/voice_convert.py \
        --checkpoint output/run51/checkpoint-best \
        --source_audio source.wav \
        --target_audio target_speaker.wav \
        --output_audio converted.wav \
        --save_reconstructions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)


TOKENIZER_SR = 24_000  # Hz, encoder input sample rate


class DisentangledProjection(nn.Module):
    """Must match the architecture in trainer.py exactly."""

    def __init__(self, hidden_dim: int = 1024, speaker_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim

        # Speaker branch
        self.speaker_encoder = nn.Sequential(
            nn.Linear(hidden_dim, speaker_dim),
            nn.ReLU(),
        )
        self.speaker_attention = nn.Linear(speaker_dim, 1)
        self.speaker_decoder = nn.Linear(speaker_dim, hidden_dim)

        # Content branch
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._warm_start_init()

    def _warm_start_init(self):
        """Must match trainer.py exactly for checkpoint compatibility."""
        for layer in self.content_proj:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)
        nn.init.uniform_(self.speaker_decoder.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.speaker_decoder.bias)

    def encode_speaker(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden_dim] → speaker_global: [B, speaker_dim]"""
        h = self.speaker_encoder(x)
        attn = torch.softmax(self.speaker_attention(h), dim=1)
        return (h * attn).sum(dim=1)

    def decode_speaker(self, speaker_global: torch.Tensor, seq_len: int) -> torch.Tensor:
        """speaker_global: [B, speaker_dim] → [B, T, hidden_dim]"""
        out = self.speaker_decoder(speaker_global)
        return out.unsqueeze(1).expand(-1, seq_len, -1)

    def encode_content(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hidden_dim] → content_emb: [B, T, hidden_dim]"""
        return self.content_proj(x)

    def forward(self, x: torch.Tensor):
        speaker_global = self.encode_speaker(x)
        speaker_contribution = self.decode_speaker(speaker_global, x.shape[1])
        content_emb = self.encode_content(x)
        return speaker_contribution, content_emb, speaker_global


class VoiceConverter:
    """Voice conversion using bottleneck DisentangledProjection.

    The speaker path has a 256-dim bottleneck + temporal pooling, making
    it structurally impossible to encode per-frame content.
    """

    def __init__(
        self,
        checkpoint: str,
        model_path: Optional[str] = None,
        base_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        # Load tokenizer
        load_from = model_path or base_model_path
        print(f"Loading tokenizer from {load_from}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            load_from,
            attn_implementation="eager",
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )

        checkpoint_path = Path(checkpoint)

        # Load config & rebuild decoder if needed
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                ckpt_config = json.load(f)
            print(f"Checkpoint step: {ckpt_config.get('step', '?')}")

            add_48k = ckpt_config.get("add_48k_decoder_block", False)
            new_upsample_rates = ckpt_config.get("new_upsample_rates")

            if add_48k and new_upsample_rates:
                self._rebuild_decoder(new_upsample_rates)

            # Load decoder weights
            decoder_weights_path = checkpoint_path / "decoder_block.safetensors"
            if decoder_weights_path.exists():
                print(f"Loading decoder weights...")
                trained_weights = load_file(str(decoder_weights_path))
                missing, unexpected = self.tokenizer.model.decoder.load_state_dict(
                    trained_weights, strict=False
                )
                print(f"  {len(trained_weights)} keys loaded, {len(missing)} missing")

            if add_48k:
                extra = ckpt_config.get("extra_upsample_rate", 2)
                new_rate = self.tokenizer.config.output_sample_rate * extra
                self.tokenizer.config.output_sample_rate = new_rate
                self.tokenizer.config.decode_upsample_rate = (
                    self.tokenizer.config.decode_upsample_rate * extra
                )
                self.tokenizer.model.output_sample_rate = new_rate
                self.tokenizer.model.decode_upsample_rate = (
                    self.tokenizer.config.decode_upsample_rate
                )

        # Load DisentangledProjection
        hidden_dim = 1024
        speaker_dim = 256
        self.disentangle = DisentangledProjection(hidden_dim, speaker_dim).to(self.device).to(self.dtype)

        disentangle_path = checkpoint_path / "disentangle.safetensors"
        if disentangle_path.exists():
            print(f"Loading DisentangledProjection...")
            dis_weights = load_file(str(disentangle_path))
            self.disentangle.load_state_dict(dis_weights)
            print(f"  {len(dis_weights)} keys loaded ✓")
        else:
            print(
                "⚠️  WARNING: disentangle.safetensors not found!\n"
                "   Retrain with updated trainer.py to generate this file."
            )

        self.disentangle.eval()
        self.output_sr = self.tokenizer.get_output_sample_rate()
        print(f"\nReady. Output: {self.output_sr} Hz")

    def _rebuild_decoder(self, new_upsample_rates):
        base_config = self.tokenizer.model.decoder.config
        config_dict = base_config.to_dict()
        config_dict["upsample_rates"] = new_upsample_rates
        for key in ("model_type", "transformers_version"):
            config_dict.pop(key, None)

        new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
        new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)
        new_decoder.load_state_dict(
            self.tokenizer.model.decoder.state_dict(), strict=False
        )
        new_decoder = new_decoder.to(self.device).to(self.dtype).eval()
        self.tokenizer.model.decoder = new_decoder
        print(f"  Rebuilt decoder: upsample_rates={new_upsample_rates}")

    def _resolve_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype):
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]

    def _load_audio(self, path: str) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != TOKENIZER_SR:
            resampler = torchaudio.transforms.Resample(sr, TOKENIZER_SR).to(self.device)
            audio = resampler(torch.from_numpy(audio).float().to(self.device)).cpu().numpy()
        return audio

    @torch.inference_mode()
    def _encode_to_codes(self, audio: np.ndarray) -> torch.Tensor:
        """audio → codes [1, num_quantizers, seq_len]"""
        encoded = self.tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
        codes = encoded.audio_codes[0]  # [seq_len, num_quantizers]
        codes = codes.T                  # [num_quantizers, seq_len]
        return codes.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def _codes_to_hidden(self, codes: torch.Tensor) -> torch.Tensor:
        """codes → hidden [1, seq_len, 1024]"""
        decoder = self.tokenizer.model.decoder
        hidden = decoder.quantizer.decode(codes)
        hidden = decoder.pre_conv(hidden).transpose(1, 2)
        hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        return hidden

    @torch.inference_mode()
    def _hidden_to_waveform(self, hidden: torch.Tensor) -> np.ndarray:
        """hidden [1, T, 1024] → waveform"""
        decoder = self.tokenizer.model.decoder
        x = hidden.permute(0, 2, 1)
        for blocks in decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in decoder.decoder:
            wav = block(wav)
        return wav.clamp(-1, 1).squeeze().cpu().float().numpy()

    @torch.inference_mode()
    def convert(self, source_path: str, target_path: str) -> Tuple[np.ndarray, int]:
        """Voice conversion: source content + target speaker → output.

        1. Encode both audios to hidden representations
        2. Extract content from source (per-frame)
        3. Extract speaker from target (global 256-dim vector)
        4. Combine and decode
        """
        print(f"\n{'='*60}")
        print("Voice Conversion (bottleneck architecture)")
        print(f"{'='*60}")

        # Load audios
        print(f"Source (content): {source_path}")
        src_audio = self._load_audio(source_path)
        print(f"  {len(src_audio)/TOKENIZER_SR:.2f}s")

        print(f"Target (speaker): {target_path}")
        tgt_audio = self._load_audio(target_path)
        print(f"  {len(tgt_audio)/TOKENIZER_SR:.2f}s")

        # Encode → codes → hidden
        print("\nEncoding...")
        src_codes = self._encode_to_codes(src_audio)
        tgt_codes = self._encode_to_codes(tgt_audio)
        print(f"  Source codes: {list(src_codes.shape)}")
        print(f"  Target codes: {list(tgt_codes.shape)}")

        src_hidden = self._codes_to_hidden(src_codes)
        tgt_hidden = self._codes_to_hidden(tgt_codes)
        print(f"  Source hidden: {list(src_hidden.shape)}")
        print(f"  Target hidden: {list(tgt_hidden.shape)}")

        # Disentangle
        print("\nDisentangling...")
        content_A = self.disentangle.encode_content(src_hidden)        # [1, T_src, 1024]
        speaker_B = self.disentangle.encode_speaker(tgt_hidden)        # [1, 256] global!
        speaker_B_contrib = self.disentangle.decode_speaker(
            speaker_B, content_A.shape[1]
        )  # [1, T_src, 1024]

        print(f"  Content (source): {list(content_A.shape)}  (per-frame)")
        print(f"  Speaker (target): {list(speaker_B.shape)}  (global bottleneck)")
        print(f"  Speaker decoded:  {list(speaker_B_contrib.shape)}  (broadcast)")

        # Combine and decode
        combined = speaker_B_contrib + content_A
        print(f"\nDecoding combined: {list(combined.shape)}...")
        waveform = self._hidden_to_waveform(combined)
        duration = len(waveform) / self.output_sr
        print(f"  Output: {len(waveform)} samples, {duration:.2f}s @ {self.output_sr}Hz")

        return waveform, self.output_sr

    @torch.inference_mode()
    def reconstruct(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Normal reconstruction (for A/B comparison)."""
        audio = self._load_audio(audio_path)
        codes = self._encode_to_codes(audio)
        hidden = self._codes_to_hidden(codes)

        speaker_contrib, content_emb, _ = self.disentangle(hidden)
        combined = speaker_contrib + content_emb

        return self._hidden_to_waveform(combined), self.output_sr


def parse_args():
    parser = argparse.ArgumentParser(description="Voice Conversion (bottleneck v2)")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")

    parser.add_argument("--source_audio", type=str, required=True,
                        help="Content donor audio")
    parser.add_argument("--target_audio", type=str, required=True,
                        help="Speaker donor audio")
    parser.add_argument("--output_audio", type=str, default="converted.wav")
    parser.add_argument("--save_reconstructions", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])

    return parser.parse_args()


def main():
    args = parse_args()

    converter = VoiceConverter(
        checkpoint=args.checkpoint,
        model_path=args.model_path,
        base_model_path=args.base_model_path,
        device=args.device,
        dtype=args.dtype,
    )

    # Voice conversion
    waveform, sr = converter.convert(args.source_audio, args.target_audio)

    output_path = Path(args.output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, sr)
    print(f"\n✅ Converted → {output_path}")

    # Reconstructions for comparison
    if args.save_reconstructions:
        stem = output_path.stem
        parent = output_path.parent

        print("\nReconstructions for A/B comparison:")

        recon, sr = converter.reconstruct(args.source_audio)
        p = parent / f"{stem}_recon_source.wav"
        sf.write(str(p), recon, sr)
        print(f"  Source recon → {p}")

        recon, sr = converter.reconstruct(args.target_audio)
        p = parent / f"{stem}_recon_target.wav"
        sf.write(str(p), recon, sr)
        print(f"  Target recon → {p}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
