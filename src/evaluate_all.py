#!/usr/bin/env python3
# coding=utf-8
"""
Unified Evaluation Script (Run in Colab)

Compares 5 models:
1. Original Qwen (Qwen3-TTS-Tokenizer-12Hz)
2. Custom Trained Qwen (with DisentangledProjection)
3. Kanade 12.5Hz
4. Kanade 25Hz
5. Kanade 25Hz Clean

Make sure to install kanade-tokenizer before running:
pip install git+https://github.com/frothywater/kanade-tokenizer
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from safetensors.torch import load_file

# Add repo root to path for Qwen imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)

QWEN_SR = 24000

# ==============================================================================
# Qwen Custom Components
# ==============================================================================
class DisentangledProjection(nn.Module):
    def __init__(self, hidden_dim: int = 1024, speaker_dim: int = 256):
        super().__init__()
        self.speaker_encoder = nn.Sequential(nn.Linear(hidden_dim, speaker_dim), nn.ReLU())
        self.speaker_attention = nn.Linear(speaker_dim, 1)
        self.speaker_decoder = nn.Linear(speaker_dim, hidden_dim)
        self.content_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
        )

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


# ==============================================================================
# Evaluators
# ==============================================================================
class EvaluatorBase:
    def __init__(self, name, device="cuda"):
        self.name = name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.total_audio_s = 0.0
        self.total_compute_s = 0.0
        self.has_vc = False
        self.params_m = "0.0M"
        self.token_rate = "Unknown"
        self.output_hz = "Unknown"
    
    def load_audio(self, path: str, target_sr: int) -> torch.Tensor:
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1: audio = audio.mean(axis=-1)
        audio = torch.from_numpy(audio).float().to(self.device)
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
        return audio

class QwenEvaluator(EvaluatorBase):
    def __init__(self, name, checkpoint_dir=None, device="cuda"):
        super().__init__(name, device)
        self.is_original = (checkpoint_dir is None)
        self.dtype = torch.bfloat16
        
        print(f"\nLoading {self.name}...")
        self.tokenizer = Qwen3TTSTokenizer.from_pretrained(
            "Qwen/Qwen3-TTS-Tokenizer-12Hz", attn_implementation="eager", 
            dtype=self.dtype, device_map=str(self.device) if self.device.type != "cpu" else None
        )
        self.decoder = self.tokenizer.model.decoder
        self.has_vc = not self.is_original
        self.token_rate = "12 Hz"
        self.output_hz = "24000"

        # Load weights if custom
        if not self.is_original:
            ckpt_path = Path(checkpoint_dir)
            dec_path = ckpt_path / "decoder_block.safetensors"
            dis_path = ckpt_path / "disentangle.safetensors"
            
            # Check 48k config
            config_path = ckpt_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                if cfg.get("add_48k_decoder_block", False):
                    self._rebuild_decoder(cfg["new_upsample_rates"])
                    self.output_hz = "48000"
            
            if dec_path.exists():
                self.decoder.load_state_dict(load_file(str(dec_path)), strict=False)
            
            self.disentangle = DisentangledProjection(1024, 256).to(self.device).to(self.dtype)
            if dis_path.exists():
                self.disentangle.load_state_dict(load_file(str(dis_path)))
            self.disentangle.eval()
        
        # Calculate params
        total = sum(p.numel() for p in self.decoder.parameters())
        if self.has_vc:
            total += sum(p.numel() for p in self.disentangle.parameters())
        self.params_m = f"{total / 1e6:.1f}M"

    def _rebuild_decoder(self, new_upsample_rates):
        base_config = self.decoder.config
        config_dict = base_config.to_dict()
        config_dict["upsample_rates"] = new_upsample_rates
        for key in ("model_type", "transformers_version"): config_dict.pop(key, None)
        new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
        new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)
        new_decoder.load_state_dict(self.decoder.state_dict(), strict=False)
        self.decoder = new_decoder.to(self.device).to(self.dtype).eval()
        self.tokenizer.model.decoder = self.decoder

    @torch.inference_mode()
    def reconstruct(self, audio_path):
        audio_np, sr = sf.read(audio_path, dtype="float32")
        if audio_np.ndim > 1: audio_np = audio_np.mean(axis=-1)
        if sr != QWEN_SR:
            audio_np = torchaudio.functional.resample(torch.from_numpy(audio_np), sr, QWEN_SR).numpy()
            
        input_dur = len(audio_np) / QWEN_SR
        
        t0 = time.perf_counter()
        codes = self.tokenizer.encode(audios=[audio_np], sr=QWEN_SR).audio_codes[0].T.unsqueeze(0).to(self.device)
        
        hidden = self.decoder.quantizer.decode(codes)
        hidden = self.decoder.pre_conv(hidden).transpose(1, 2)
        hidden = self.decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
        
        if self.has_vc:
            speaker_contrib, content_emb, _ = self.disentangle(hidden)
            hidden = speaker_contrib + content_emb
            
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks: x = block(x)
        for block in self.decoder.decoder: x = block(x)
        wav = x.clamp(-1, 1).squeeze().cpu().float().numpy()
        
        compute_dur = time.perf_counter() - t0
        self.total_audio_s += input_dur
        self.total_compute_s += compute_dur
        
        return wav, int(self.output_hz), compute_dur / input_dur

    @torch.inference_mode()
    def voice_convert(self, src_path, tgt_path):
        if not self.has_vc: return None, 0, 0
        src_np, _ = sf.read(src_path, dtype="float32")
        tgt_np, _ = sf.read(tgt_path, dtype="float32")
        if src_np.ndim > 1: src_np = src_np.mean(axis=-1)
        if tgt_np.ndim > 1: tgt_np = tgt_np.mean(axis=-1)
        
        t0 = time.perf_counter()
        src_codes = self.tokenizer.encode(audios=[src_np], sr=QWEN_SR).audio_codes[0].T.unsqueeze(0).to(self.device)
        tgt_codes = self.tokenizer.encode(audios=[tgt_np], sr=QWEN_SR).audio_codes[0].T.unsqueeze(0).to(self.device)
        
        def to_hidden(c):
            h = self.decoder.quantizer.decode(c)
            h = self.decoder.pre_conv(h).transpose(1, 2)
            return self.decoder.pre_transformer(inputs_embeds=h).last_hidden_state

        src_h = to_hidden(src_codes)
        tgt_h = to_hidden(tgt_codes)
        
        content_A = self.disentangle.encode_content(src_h)
        speaker_B = self.disentangle.encode_speaker(tgt_h)
        speaker_B_contrib = self.disentangle.decode_speaker(speaker_B, content_A.shape[1])
        
        hidden = speaker_B_contrib + content_A
        x = hidden.permute(0, 2, 1)
        for blocks in self.decoder.upsample:
            for block in blocks: x = block(x)
        for block in self.decoder.decoder: x = block(x)
        wav = x.clamp(-1, 1).squeeze().cpu().float().numpy()
        
        return wav, int(self.output_hz), time.perf_counter() - t0

class KanadeEvaluator(EvaluatorBase):
    def __init__(self, repo_id, name, device="cuda"):
        super().__init__(name, device)
        try:
            from kanade_tokenizer import KanadeModel, load_vocoder
        except ImportError:
            raise ImportError("Please run `pip install git+https://github.com/frothywater/kanade-tokenizer` first.")
            
        print(f"\nLoading {self.name} ({repo_id})...")
        self.model = KanadeModel.from_pretrained(repo_id).eval().to(self.device)
        self.vocoder = load_vocoder(self.model.config.vocoder_name).to(self.device)
        self.has_vc = True
        self.token_rate = "25 Hz" if "25" in repo_id else "12.5 Hz"
        self.output_hz = str(self.model.config.sample_rate)
        
        # Calculate params
        total = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.vocoder.parameters())
        self.params_m = f"{total / 1e6:.1f}M"

    @torch.inference_mode()
    def reconstruct(self, audio_path):
        from kanade_tokenizer import load_audio, vocode
        audio = load_audio(audio_path, sample_rate=self.model.config.sample_rate).to(self.device)
        input_dur = audio.shape[-1] / self.model.config.sample_rate
        
        t0 = time.perf_counter()
        features = self.model.encode(audio)
        mel = self.model.decode(content_token_indices=features.content_token_indices, global_embedding=features.global_embedding)
        wav = vocode(self.vocoder, mel.unsqueeze(0)).squeeze().cpu().numpy()
        compute_dur = time.perf_counter() - t0
        
        self.total_audio_s += input_dur
        self.total_compute_s += compute_dur
        return wav, int(self.output_hz), compute_dur / input_dur

    @torch.inference_mode()
    def voice_convert(self, src_path, tgt_path):
        from kanade_tokenizer import load_audio, vocode
        src_aud = load_audio(src_path, sample_rate=self.model.config.sample_rate).to(self.device)
        tgt_aud = load_audio(tgt_path, sample_rate=self.model.config.sample_rate).to(self.device)
        
        t0 = time.perf_counter()
        feat_src = self.model.encode(src_aud)
        feat_tgt = self.model.encode(tgt_aud)
        
        mel = self.model.decode(content_token_indices=feat_src.content_token_indices, global_embedding=feat_tgt.global_embedding)
        wav = vocode(self.vocoder, mel.unsqueeze(0)).squeeze().cpu().numpy()
        return wav, int(self.output_hz), time.perf_counter() - t0

# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Model Comparison")
    parser.add_argument("--qwen_checkpoint", type=str, required=True, help="Path to your custom Qwen checkpoint")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder with test audios")
    parser.add_argument("--output_dir", type=str, default="./unified_eval_output")
    args = parser.parse_args()

    audio_files = sorted([str(p) for p in Path(args.audio_dir).glob("*") if p.suffix.lower() in [".wav", ".mp3", ".flac"]])
    if not audio_files:
        print(f"❌ No audio files found in {args.audio_dir}")
        return
    print(f"Found {len(audio_files)} test audio files.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define all models
    model_configs = [
        {"name": "Original Qwen", "type": "qwen", "ckpt": None},
        {"name": "Custom Qwen (Trained)", "type": "qwen", "ckpt": args.qwen_checkpoint},
        {"name": "Kanade 12.5Hz", "type": "kanade", "repo": "frothywater/kanade-12.5hz"},
        {"name": "Kanade 25Hz", "type": "kanade", "repo": "frothywater/kanade-25hz"},
        {"name": "Kanade 25Hz Clean", "type": "kanade", "repo": "frothywater/kanade-25hz-clean"},
    ]

    results = []

    for cfg in model_configs:
        print("\n" + "="*60)
        
        # Instantiate evaluator dynamically
        try:
            if cfg["type"] == "qwen":
                evaluator = QwenEvaluator(cfg["name"], checkpoint_dir=cfg["ckpt"])
            else:
                evaluator = KanadeEvaluator(cfg["repo"], cfg["name"])
        except ImportError as e:
            print(f"⚠️ Skipping {cfg['name']} due to missing dependencies: {e}")
            continue

        model_out = out_dir / cfg["name"].replace(" ", "_")
        (model_out / "reconstruction").mkdir(parents=True, exist_ok=True)
        (model_out / "voice_conversion").mkdir(parents=True, exist_ok=True)

        print(f"\n[Reconstruction]")
        for aud in audio_files[:4]:  # limit to 4 to save time
            name = Path(aud).name
            wav, sr, rtf = evaluator.reconstruct(aud)
            sf.write(str(model_out / "reconstruction" / f"recon_{name}.wav"), wav, sr)
            print(f"  {name:.<30} RTF: {rtf:.3f}x")

        print(f"\n[Voice Conversion]")
        if evaluator.has_vc and len(audio_files) >= 2:
            pairs = [(0, 1), (1, 0), (0, 2)] # Do 3 swaps
            for i, j in pairs:
                if i < len(audio_files) and j < len(audio_files):
                    src, tgt = audio_files[i], audio_files[j]
                    src_name, tgt_name = Path(src).stem, Path(tgt).stem
                    wav, sr, time_s = evaluator.voice_convert(src, tgt)
                    sf.write(str(model_out / "voice_conversion" / f"vc_{src_name}_TO_{tgt_name}.wav"), wav, sr)
                    print(f"  {src_name} → {tgt_name:.<20} Time: {time_s:.2f}s")
        else:
            print("  ⚠️ Skipped (No Disentanglement module in base model)")

        avg_rtf = evaluator.total_compute_s / evaluator.total_audio_s if evaluator.total_audio_s > 0 else 0
        
        results.append({
            "Model Name": evaluator.name,
            "Params": evaluator.params_m,
            "Token Rate": evaluator.token_rate,
            "Output SR": evaluator.output_hz,
            "VC Enabled": "✅" if evaluator.has_vc else "❌",
            "Avg RTF": f"{avg_rtf:.3f}x"
        })

        # Free VRAM between models
        del evaluator
        torch.cuda.empty_cache()

    # Print Comparison Table
    print("\n\n" + "="*80)
    print("  🔥 UNIFIED MODEL COMPARISON 🔥")
    print("="*80)
    header = f"| {'Model Name':<25} | {'Params':<8} | {'Token Rate':<10} | {'Output Hz':<9} | {'VC?':<3} | {'Avg RTF':<8} |"
    print(header)
    print("|" + "-"*27 + "|" + "-"*10 + "|" + "-"*12 + "|" + "-"*11 + "|" + "-"*5 + "|" + "-"*10 + "|")
    
    for r in results:
        print(f"| {r['Model Name']:<25} | {r['Params']:<8} | {r['Token Rate']:<10} | {r['Output SR']:<9} | {r['VC Enabled']:<3} | {r['Avg RTF']:<8} |")
    print("="*80)
    print(f"\nAll audio outputs saved to: {out_dir}")

if __name__ == "__main__":
    main()
