#!/usr/bin/env python3
# coding=utf-8
"""
Model Evaluation & Comparison Script

Loads multiple checkpoints (e.g., Original Qwen, Kanade 12.5, 25, 25clean), 
prints model info, and runs inference on a folder of audio files.
Outputs: model card, reconstruction samples, voice conversion samples, RTF benchmarks,
and a comparison table across all models.

Usage:
    python src/evaluate.py \
        --models "Original" "12.5:output/run_12.5/checkpoint-best" "25:output/run_25/checkpoint-best" \
        --audio_dir /path/to/test_audios/ \
        --output_dir ./eval_output \
        --num_samples 4
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

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

TOKENIZER_SR = 24_000


# ── DisentangledProjection (must match trainer.py) ──────────────────────────
class DisentangledProjection(nn.Module):
    def __init__(self, hidden_dim: int = 1024, speaker_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.speaker_dim = speaker_dim
        self.speaker_encoder = nn.Sequential(
            nn.Linear(hidden_dim, speaker_dim), nn.ReLU(),
        )
        self.speaker_attention = nn.Linear(speaker_dim, 1)
        self.speaker_decoder = nn.Linear(speaker_dim, hidden_dim)
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._warm_start_init()

    def _warm_start_init(self):
        for layer in self.content_proj:
            if isinstance(layer, nn.Linear):
                nn.init.eye_(layer.weight)
                nn.init.zeros_(layer.bias)
                with torch.no_grad():
                    layer.weight.add_(torch.randn_like(layer.weight) * 1e-3)
        nn.init.uniform_(self.speaker_decoder.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.speaker_decoder.bias)

    def encode_speaker(self, x):
        h = self.speaker_encoder(x)
        attn = torch.softmax(self.speaker_attention(h), dim=1)
        return (h * attn).sum(dim=1)

    def decode_speaker(self, speaker_global, seq_len):
        out = self.speaker_decoder(speaker_global)
        return out.unsqueeze(1).expand(-1, seq_len, -1)

    def encode_content(self, x):
        return self.content_proj(x)

    def forward(self, x):
        speaker_global = self.encode_speaker(x)
        speaker_contribution = self.decode_speaker(speaker_global, x.shape[1])
        content_emb = self.encode_content(x)
        return speaker_contribution, content_emb, speaker_global


# ── Model Loader ────────────────────────────────────────────────────────────
class ModelEvaluator:
    def __init__(self, name: str, checkpoint: str, base_model: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
                 device: str = "auto", dtype: str = "bfloat16"):
        self.name = name
        self.device = self._resolve_device(device)
        self.dtype = {"float32": torch.float32, "float16": torch.float16,
                      "bfloat16": torch.bfloat16}[dtype]
        self.is_original = (checkpoint.lower() == "original")
        self.checkpoint_path = None if self.is_original else Path(checkpoint)

        # Load config
        self.config = {}
        if not self.is_original:
            config_path = self.checkpoint_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    self.config = json.load(f)

        # Load tokenizer
        print(f"Loading base model: {base_model} for '{self.name}'")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            base_model, attn_implementation="eager", dtype=self.dtype,
            device_map=str(self.device) if self.device.type != "cpu" else None,
        )

        # Rebuild decoder if 48k
        add_48k = self.config.get("add_48k_decoder_block", False)
        new_upsample_rates = self.config.get("new_upsample_rates")
        if add_48k and new_upsample_rates:
            self._rebuild_decoder(new_upsample_rates)

        # Load decoder weights
        if not self.is_original:
            decoder_path = self.checkpoint_path / "decoder_block.safetensors"
            if decoder_path.exists():
                weights = load_file(str(decoder_path))
                self.tokenizer.model.decoder.load_state_dict(weights, strict=False)
                print(f"  Loaded {len(weights)} decoder keys")

        if add_48k:
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

        # Load DisentangledProjection
        self.disentangle = DisentangledProjection(1024, 256).to(self.device).to(self.dtype)
        self.has_disentangle = False
        if not self.is_original:
            dis_path = self.checkpoint_path / "disentangle.safetensors"
            if dis_path.exists():
                self.disentangle.load_state_dict(load_file(str(dis_path)))
                self.has_disentangle = True
                print(f"  Loaded DisentangledProjection ✓")
            else:
                print(f"  ⚠️  disentangle.safetensors not found — VC disabled")
        self.disentangle.eval()

        self.output_sr = self.tokenizer.get_output_sample_rate()
        self.decoder = self.tokenizer.model.decoder

        # Collect model info
        self._collect_model_info()

    def _resolve_device(self, device):
        if device == "auto":
            if torch.cuda.is_available(): return torch.device("cuda")
            elif torch.backends.mps.is_available(): return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

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

    def _collect_model_info(self):
        dec = self.decoder
        self.info = {
            "name": self.name,
            "checkpoint": str(self.checkpoint_path) if self.checkpoint_path else "Original",
            "step": self.config.get("step", "N/A"),
            "epoch": self.config.get("epoch", "N/A"),
            "training_type": self.config.get("training_type", "N/A"),
            "output_sample_rate": self.output_sr,
            "token_rate_hz": 12,
            "num_codebooks": dec.config.num_quantizers,
            "codebook_size": getattr(dec.config, "codebook_size", "N/A"),
            "hidden_dim": 1024,
            "upsample_rates": list(dec.config.upsample_rates),
            "add_48k_block": self.config.get("add_48k_decoder_block", False),
            "has_disentangle": self.has_disentangle,
            "speaker_dim": 256 if self.has_disentangle else "N/A",
            "device": str(self.device),
            "dtype": str(self.dtype),
        }
        # Parameter counts
        total = sum(p.numel() for p in dec.parameters())
        dis_params = sum(p.numel() for p in self.disentangle.parameters()) if self.has_disentangle else 0
        self.info["decoder_params"] = total
        self.info["decoder_params_m"] = f"{total / 1e6:.1f}M"
        self.info["disentangle_params"] = dis_params
        self.info["disentangle_params_m"] = f"{dis_params / 1e6:.1f}M"
        self.info["total_params_m"] = f"{(total + dis_params) / 1e6:.1f}M"
        # Training config
        for key in ["lambda_adv", "lambda_fm", "lambda_multi_res_mel",
                     "lambda_global_rms", "lambda_orth", "train_full_decoder"]:
            if key in self.config:
                self.info[key] = self.config[key]

    def print_model_card(self):
        print("\n" + "=" * 65)
        print(f"  MODEL CARD: {self.name}")
        print("=" * 65)
        sections = [
            ("Checkpoint", [
                ("Path", self.info["checkpoint"]),
                ("Step", self.info["step"]),
                ("Epoch", self.info["epoch"]),
                ("Training type", self.info["training_type"]),
            ]),
            ("Architecture", [
                ("Codebooks", f"{self.info['num_codebooks']} (RVQ)"),
                ("Token rate", f"{self.info['token_rate_hz']} Hz"),
                ("Output sample rate", f"{self.info['output_sample_rate']} Hz"),
                ("Upsample rates", str(self.info["upsample_rates"])),
                ("Hidden dim", self.info["hidden_dim"]),
                ("48kHz block", "✅ Yes" if self.info["add_48k_block"] else "❌ No"),
            ]),
            ("Parameters", [
                ("Decoder", self.info["decoder_params_m"]),
                ("DisentangledProjection", self.info["disentangle_params_m"]),
                ("Total", self.info["total_params_m"]),
            ]),
            ("Disentanglement", [
                ("Has DisentangledProjection", "✅ Yes" if self.info["has_disentangle"] else "❌ No"),
                ("Speaker bottleneck dim", self.info["speaker_dim"]),
                ("Content dim", "1024 (per-frame)"),
            ]),
        ]
        if "lambda_orth" in self.info:
            sections.append(("Training Hyperparameters", [
                ("λ_adv", self.info.get("lambda_adv", "N/A")),
                ("λ_fm", self.info.get("lambda_fm", "N/A")),
                ("λ_mel", self.info.get("lambda_multi_res_mel", "N/A")),
                ("λ_rms", self.info.get("lambda_global_rms", "N/A")),
                ("λ_orth", self.info.get("lambda_orth", "N/A")),
                ("Full decoder trained", self.info.get("train_full_decoder", "N/A")),
            ]))
        for section_name, items in sections:
            print(f"\n  [{section_name}]")
            for label, value in items:
                print(f"    {label:.<35} {value}")
        print("\n" + "=" * 65)

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
        encoded = self.tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
        codes = encoded.audio_codes[0].T.unsqueeze(0).to(self.device)
        return codes

    @torch.inference_mode()
    def _codes_to_hidden(self, codes: torch.Tensor) -> torch.Tensor:
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        return hidden

    @torch.inference_mode()
    def _hidden_to_waveform(self, hidden: torch.Tensor) -> np.ndarray:
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks:
                x = block(x)
        wav = x
        for block in self.decoder.decoder:
            wav = block(wav)
        return wav.clamp(-1, 1).squeeze().cpu().float().numpy()

    @torch.inference_mode()
    def reconstruct(self, audio_path: str):
        """Reconstruct audio through the full pipeline. Returns (waveform, sr, timing_info)."""
        audio = self._load_audio(audio_path)
        input_dur = len(audio) / TOKENIZER_SR

        t0 = time.perf_counter()
        codes = self._encode_to_codes(audio)
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        hidden = self._codes_to_hidden(codes)
        if self.has_disentangle:
            speaker_contrib, content_emb, speaker_global = self.disentangle(hidden)
            hidden = speaker_contrib + content_emb
        wav = self._hidden_to_waveform(hidden)
        t_decode = time.perf_counter() - t0

        output_dur = len(wav) / self.output_sr
        total_time = t_encode + t_decode
        rtf = total_time / input_dur if input_dur > 0 else 0

        timing = {
            "input_duration_s": round(input_dur, 3),
            "output_duration_s": round(output_dur, 3),
            "encode_time_s": round(t_encode, 4),
            "decode_time_s": round(t_decode, 4),
            "total_time_s": round(total_time, 4),
            "rtf": round(rtf, 4),
            "codes_shape": list(codes.shape),
        }
        return wav, self.output_sr, timing

    @torch.inference_mode()
    def voice_convert(self, source_path: str, target_path: str):
        """Voice conversion: source content + target speaker."""
        if not self.has_disentangle:
            raise RuntimeError("DisentangledProjection not loaded — cannot do VC")

        src_audio = self._load_audio(source_path)
        tgt_audio = self._load_audio(target_path)

        t0 = time.perf_counter()
        src_codes = self._encode_to_codes(src_audio)
        tgt_codes = self._encode_to_codes(tgt_audio)
        src_hidden = self._codes_to_hidden(src_codes)
        tgt_hidden = self._codes_to_hidden(tgt_codes)

        content_A = self.disentangle.encode_content(src_hidden)
        speaker_B = self.disentangle.encode_speaker(tgt_hidden)
        speaker_B_contrib = self.disentangle.decode_speaker(speaker_B, content_A.shape[1])

        combined = speaker_B_contrib + content_A
        wav = self._hidden_to_waveform(combined)
        vc_time = time.perf_counter() - t0

        # Cosine similarity between speakers
        src_spk = self.disentangle.encode_speaker(src_hidden).squeeze().cpu().float().numpy()
        tgt_spk = speaker_B.squeeze().cpu().float().numpy()
        cos_sim = float(np.dot(src_spk, tgt_spk) / (np.linalg.norm(src_spk) * np.linalg.norm(tgt_spk) + 1e-8))

        return wav, self.output_sr, {
            "vc_time_s": round(vc_time, 4),
            "speaker_cosine_sim": round(cos_sim, 4),
            "source_dur_s": round(len(src_audio) / TOKENIZER_SR, 3),
            "target_dur_s": round(len(tgt_audio) / TOKENIZER_SR, 3),
            "output_dur_s": round(len(wav) / self.output_sr, 3),
        }

    def save_model_card(self, output_dir: Path):
        card_path = output_dir / "model_card.json"
        with open(card_path, "w") as f:
            json.dump(self.info, f, indent=2, default=str)
        print(f"  Model card → {card_path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Model Evaluation & Comparison")
    parser.add_argument(
        "--models", type=str, nargs="+", required=True,
        help="List of models to evaluate in format 'Name:Path' or 'Original'. "
             "Example: Original 12.5:outputs/12.5 25:outputs/25 25clean:outputs/25clean"
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder with test .wav/.flac files")
    parser.add_argument("--output_dir", type=str, default="./eval_output")
    parser.add_argument("--num_samples", type=int, default=4, help="Max samples to process")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find audio files
    audio_dir = Path(args.audio_dir)
    audio_files = sorted([
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".wav", ".flac", ".mp3", ".ogg")
    ])[:args.num_samples]

    if not audio_files:
        print(f"\n❌ No audio files found in {audio_dir}")
        return

    print(f"\nFound {len(audio_files)} audio files to evaluate.")
    
    # Process multiple models
    models_to_run = []
    for m in args.models:
        if ":" in m:
            name, path = m.split(":", 1)
            models_to_run.append((name, path))
        else:
            models_to_run.append((m, m))  # e.g. ("Original", "Original")

    all_reports = {}

    for (model_name, model_path) in models_to_run:
        print("\n" + "#" * 70)
        print(f"# EVALUATING MODEL: {model_name}")
        print("#" * 70)
        
        model_out_dir = output_dir / model_name.replace(" ", "_")
        model_out_dir.mkdir(exist_ok=True)
        
        # Load model
        evaluator = ModelEvaluator(
            name=model_name, checkpoint=model_path, base_model=args.base_model,
            device=args.device, dtype=args.dtype,
        )
        evaluator.print_model_card()
        evaluator.save_model_card(model_out_dir)

        # ── Reconstruction ──────────────────────────────────────────────────
        recon_dir = model_out_dir / "reconstructions"
        recon_dir.mkdir(exist_ok=True)
        all_timings = []

        print("\n  [RECONSTRUCTION]")
        for i, audio_file in enumerate(audio_files):
            wav, sr, timing = evaluator.reconstruct(str(audio_file))

            out_path = recon_dir / f"recon_{audio_file.stem}.wav"
            sf.write(str(out_path), wav, sr)
            all_timings.append(timing)
            print(f"    {audio_file.name:.<30} RTF: {timing['rtf']:.4f}x")

        # RTF summary
        avg_rtf = np.mean([t["rtf"] for t in all_timings])
        total_audio = sum(t["input_duration_s"] for t in all_timings)
        total_compute = sum(t["total_time_s"] for t in all_timings)

        # ── Voice Conversion ────────────────────────────────────────────────
        if evaluator.has_disentangle and len(audio_files) >= 2:
            vc_dir = model_out_dir / "voice_conversions"
            vc_dir.mkdir(exist_ok=True)

            print("\n  [VOICE CONVERSION]")
            pairs = []
            for i in range(min(len(audio_files), 3)):
                for j in range(min(len(audio_files), 3)):
                    if i != j:
                        pairs.append((i, j))

            for src_idx, tgt_idx in pairs[:6]:
                src_file = audio_files[src_idx]
                tgt_file = audio_files[tgt_idx]
                wav, sr, vc_info = evaluator.voice_convert(str(src_file), str(tgt_file))
                out_path = vc_dir / f"vc_{src_file.stem}_to_{tgt_file.stem}.wav"
                sf.write(str(out_path), wav, sr)
                print(f"    {src_file.name} → {tgt_file.name:.<20} Sim: {vc_info['speaker_cosine_sim']:.4f}")
        else:
            if not evaluator.has_disentangle:
                print("\n  [VOICE CONVERSION] ⚠️ Skipped (No DisentangledProjection)")

        # ── Save report ─────────────────────────────────────────────────────
        report = {
            "model_info": evaluator.info,
            "reconstruction_timings": all_timings,
            "avg_rtf": round(avg_rtf, 4),
            "total_audio_s": round(total_audio, 2),
            "throughput_x_realtime": round(total_audio / total_compute, 1) if total_compute > 0 else 0,
        }
        report_path = model_out_dir / "eval_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        all_reports[model_name] = report
        
        # Free memory before next model
        del evaluator
        torch.cuda.empty_cache()

    # ── Final Comparison Table ──────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 75)
    
    header = f"| {'Model Name':<15} | {'Params (M)':<12} | {'VC Enabled?':<12} | {'Avg RTF':<10} | {'Output Hz':<10} |"
    print(header)
    print("|" + "-" * 17 + "|" + "-" * 14 + "|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    
    for name, rep in all_reports.items():
        info = rep["model_info"]
        vc_enabled = "✅ Yes" if info.get("has_disentangle", False) else "❌ No"
        print(f"| {name:<15} | {info.get('total_params_m', 'N/A'):<12} | {vc_enabled:<12} | {rep['avg_rtf']:<10.4f} | {info.get('output_sample_rate', 'N/A'):<10} |")
    
    print("=" * 75)
    print(f"\nAll outputs saved in: {output_dir}")
    print("\nDone! ✅\n")


if __name__ == "__main__":
    main()
