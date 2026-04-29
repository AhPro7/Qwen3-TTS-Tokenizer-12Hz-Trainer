#!/usr/bin/env python3
# coding=utf-8
"""
Single Codebook VQ Inference Script (Experiment 95)

Loads a checkpoint trained with --single_codebook and reconstructs audio
through the single-codebook VQ bottleneck.

Modes:
  1. Reconstruct: audio → codes → hidden → VQ(single token) → decode → audio
  2. Encode: audio → single codebook tokens (.npy)
  3. Decode: tokens (.npy) → audio

Usage:
    # Reconstruct (round-trip)
    python src/inference_single_codebook.py \\
        --checkpoint /path/to/checkpoint-best \\
        --input_audio input.wav \\
        --output_audio output.wav

    # Encode to tokens
    python src/inference_single_codebook.py \\
        --checkpoint /path/to/checkpoint-best \\
        --input_audio input.wav \\
        --save_tokens tokens.npy

    # Decode from tokens
    python src/inference_single_codebook.py \\
        --checkpoint /path/to/checkpoint-best \\
        --input_tokens tokens.npy \\
        --output_audio output.wav
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)
from single_codebook_vq import SingleCodebookVQ

TOKENIZER_SR = 24_000


class SingleCodebookTokenizer:
    """Tokenizer using single codebook VQ (Experiment 95).

    Produces ONE discrete token per frame at 12.5 Hz.
    """

    def __init__(
        self,
        checkpoint: str,
        base_model_path: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        device: str = "auto",
        dtype: str = "bfloat16",
    ):
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)

        # Load base tokenizer
        print(f"Loading base model from {base_model_path}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            base_model_path,
            attn_implementation="eager",
            dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )
        self.decoder = self.tokenizer.model.decoder

        # Load checkpoint config
        checkpoint_path = Path(checkpoint)
        config_path = checkpoint_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            self.config = {}

        # Verify this is a single codebook checkpoint
        if not self.config.get("single_codebook", False):
            print("⚠️  WARNING: Checkpoint may not be a single codebook model!")

        # Load decoder weights
        decoder_path = checkpoint_path / "decoder_block.safetensors"
        if decoder_path.exists():
            weights = load_file(str(decoder_path))
            self.decoder.load_state_dict(weights, strict=False)
            print(f"  Loaded {len(weights)} decoder keys")

        # Handle 48k decoder rebuild
        add_48k = self.config.get("add_48k_decoder_block", False)
        if add_48k:
            new_upsample_rates = self.config.get("new_upsample_rates")
            if new_upsample_rates:
                self._rebuild_decoder(new_upsample_rates)
            extra = self.config.get("extra_upsample_rate", 2)
            new_rate = self.tokenizer.config.output_sample_rate * extra
            self.tokenizer.config.output_sample_rate = new_rate
            self.tokenizer.config.decode_upsample_rate = (
                self.tokenizer.config.decode_upsample_rate * extra
            )
            self.tokenizer.model.output_sample_rate = new_rate
            self.tokenizer.model.decode_upsample_rate = (
                self.tokenizer.config.decode_upsample_rate
            )

        # Create and load SingleCodebookVQ
        codebook_size = self.config.get("codebook_size", 16384)
        vq_dim = self.config.get("vq_dim", 256)
        entropy_weight = self.config.get("entropy_weight", 0.1)

        self.single_vq = SingleCodebookVQ(
            hidden_dim=1024,
            vq_dim=vq_dim,
            codebook_size=codebook_size,
            entropy_weight=entropy_weight,
        ).to(self.device).to(self.dtype)

        svq_path = checkpoint_path / "single_codebook_vq.safetensors"
        if svq_path.exists():
            svq_weights = load_file(str(svq_path))
            self.single_vq.load_state_dict(svq_weights)
            print(f"  Loaded SingleCodebookVQ ✓ (codebook={codebook_size}, dim={vq_dim})")
        else:
            print("  ⚠️  single_codebook_vq.safetensors not found!")

        self.single_vq.eval()
        self.output_sr = self.tokenizer.get_output_sample_rate()

        print(f"\n{'='*55}")
        print(f"  Single Codebook Tokenizer Ready")
        print(f"  Codebook size: {codebook_size:,}")
        print(f"  VQ dimension:  {vq_dim}")
        print(f"  Token rate:    12.5 Hz")
        print(f"  Bitrate:       {int(np.log2(codebook_size) * 12.5)} bits/sec")
        print(f"  Output SR:     {self.output_sr} Hz")
        print(f"{'='*55}\n")

    def _resolve_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _resolve_dtype(self, dtype):
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[dtype]

    def _rebuild_decoder(self, new_upsample_rates):
        base_config = self.decoder.config
        config_dict = base_config.to_dict()
        config_dict["upsample_rates"] = new_upsample_rates
        for key in ("model_type", "transformers_version"):
            config_dict.pop(key, None)
        new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
        new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)
        new_decoder.load_state_dict(self.decoder.state_dict(), strict=False)
        new_decoder = new_decoder.to(self.device).to(self.dtype).eval()
        self.tokenizer.model.decoder = new_decoder
        self.decoder = new_decoder
        print(f"  Rebuilt decoder: upsample_rates={new_upsample_rates}")

    def _load_audio(self, path: str) -> np.ndarray:
        import torchaudio
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        if sr != TOKENIZER_SR:
            resampler = torchaudio.transforms.Resample(sr, TOKENIZER_SR).to(self.device)
            audio = resampler(
                torch.from_numpy(audio).float().to(self.device)
            ).cpu().numpy()
        return audio

    @torch.inference_mode()
    def _audio_to_hidden(self, audio: np.ndarray) -> torch.Tensor:
        """audio → codes → hidden [1, T, 1024]"""
        encoded = self.tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
        codes = encoded.audio_codes[0].T.unsqueeze(0).to(self.device)

        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        hidden = self.decoder.pre_transformer(
            inputs_embeds=hidden
        ).last_hidden_state
        return hidden

    @torch.inference_mode()
    def _hidden_to_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [B, T, 1024] → tokens [B, T]"""
        output, tokens, _, _ = self.single_vq(hidden, vq_alpha=1.0)
        return tokens

    @torch.inference_mode()
    def _tokens_to_hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B, T] → hidden [B, T, 1024]"""
        return self.single_vq.decode_tokens(tokens)

    @torch.inference_mode()
    def _hidden_to_waveform(self, hidden: torch.Tensor) -> np.ndarray:
        """hidden [1, T, 1024] → waveform"""
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(-1, 1).squeeze().cpu().float().numpy()

    @torch.inference_mode()
    def encode(self, audio_path: str) -> np.ndarray:
        """Encode audio to single codebook tokens.

        Returns:
            tokens: numpy array of shape [seq_len] with integer token IDs.
        """
        audio = self._load_audio(audio_path)
        hidden = self._audio_to_hidden(audio)
        tokens = self._hidden_to_tokens(hidden)
        return tokens.squeeze(0).cpu().numpy()

    @torch.inference_mode()
    def decode(self, tokens: np.ndarray) -> Tuple[np.ndarray, int]:
        """Decode single codebook tokens to audio.

        Args:
            tokens: numpy array of shape [seq_len].

        Returns:
            (waveform, sample_rate)
        """
        tokens_t = torch.from_numpy(tokens).long().unsqueeze(0).to(self.device)
        hidden = self._tokens_to_hidden(tokens_t)
        wav = self._hidden_to_waveform(hidden)
        return wav, self.output_sr

    @torch.inference_mode()
    def reconstruct(self, audio_path: str) -> Tuple[np.ndarray, int, dict]:
        """Full round-trip: audio → VQ tokens → audio.

        Returns:
            (waveform, sample_rate, timing_info)
        """
        audio = self._load_audio(audio_path)
        input_dur = len(audio) / TOKENIZER_SR

        t0 = time.perf_counter()
        hidden = self._audio_to_hidden(audio)
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        tokens = self._hidden_to_tokens(hidden)
        t_quantize = time.perf_counter() - t0

        t0 = time.perf_counter()
        decoded_hidden = self._tokens_to_hidden(tokens)
        wav = self._hidden_to_waveform(decoded_hidden)
        t_decode = time.perf_counter() - t0

        total_time = t_encode + t_quantize + t_decode
        rtf = total_time / input_dur if input_dur > 0 else 0

        unique_tokens = int(np.unique(tokens.cpu().numpy()).shape[0])
        total_tokens = int(tokens.numel())

        timing = {
            "input_duration_s": round(input_dur, 3),
            "output_duration_s": round(len(wav) / self.output_sr, 3),
            "encode_time_s": round(t_encode, 4),
            "quantize_time_s": round(t_quantize, 4),
            "decode_time_s": round(t_decode, 4),
            "total_time_s": round(total_time, 4),
            "rtf": round(rtf, 4),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "token_rate_hz": 12.5,
        }
        return wav, self.output_sr, timing


def main():
    parser = argparse.ArgumentParser(
        description="Single Codebook VQ Inference (Experiment 95)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--base_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz"
    )
    parser.add_argument("--input_audio", type=str, default=None)
    parser.add_argument("--input_tokens", type=str, default=None)
    parser.add_argument("--output_audio", type=str, default="output_single_vq.wav")
    parser.add_argument("--save_tokens", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
    )

    args = parser.parse_args()

    if args.input_audio is None and args.input_tokens is None:
        print("Error: Either --input_audio or --input_tokens must be specified")
        sys.exit(1)

    # Load model
    tokenizer = SingleCodebookTokenizer(
        checkpoint=args.checkpoint,
        base_model_path=args.base_model_path,
        device=args.device,
        dtype=args.dtype,
    )

    if args.input_audio:
        print(f"Input audio: {args.input_audio}")

        # Encode to tokens
        if args.save_tokens:
            tokens = tokenizer.encode(args.input_audio)
            np.save(args.save_tokens, tokens)
            print(f"  Tokens saved: {args.save_tokens} (shape: {tokens.shape})")
            print(f"  Unique tokens: {np.unique(tokens).shape[0]}")

        # Reconstruct
        wav, sr, timing = tokenizer.reconstruct(args.input_audio)
        print(f"\n  Reconstruction timing:")
        for k, v in timing.items():
            print(f"    {k}: {v}")

    elif args.input_tokens:
        print(f"Input tokens: {args.input_tokens}")
        tokens = np.load(args.input_tokens)
        print(f"  Shape: {tokens.shape}")
        wav, sr = tokenizer.decode(tokens)

    # Save output
    output_path = Path(args.output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wav, sr)
    print(f"\n✅ Output → {output_path} ({len(wav)/sr:.2f}s @ {sr}Hz)")


if __name__ == "__main__":
    main()
