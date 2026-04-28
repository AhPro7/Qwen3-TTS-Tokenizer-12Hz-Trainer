#!/usr/bin/env python3
"""
Debug script: Analyze speaker/content disentanglement in Run 81 checkpoint.

Loads the checkpoint, encodes two audio samples, and tests:
1. Reconstruction (same speaker + same content → should sound original)
2. Voice conversion (swap speaker tokens → should change voice)
3. Speaker embedding analysis (cosine similarity, norms)
4. Content vs speaker contribution analysis (how much does each path contribute?)
5. Zero-speaker test (what if we zero out speaker contribution?)
6. Random-speaker test (what if we use random speaker tokens?)

Saves all outputs as WAV files for listening comparison.

Usage (in Colab):
    python debug_disentangle.py \
        --checkpoint /content/drive/MyDrive/qwen-tokenzier-v2/run81/checkpoint-step-1000 \
        --audio_a /path/to/speaker_A.wav \
        --audio_b /path/to/speaker_B.wav \
        --output_dir ./debug_output
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).parent / "src"))

from safetensors.torch import load_file
from qwen_tts import Qwen3TTSTokenizer
from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2DecoderConfig,
)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
)
from disentangle import DisentangledProjection

TOKENIZER_SR = 24_000


def load_audio(path, device="cpu"):
    """Load and resample audio to 24kHz mono."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if sr != TOKENIZER_SR:
        resampler = torchaudio.transforms.Resample(sr, TOKENIZER_SR)
        audio = resampler(torch.from_numpy(audio).float()).numpy()
    return audio


def setup_model(checkpoint_path, device, dtype=torch.bfloat16):
    """Load tokenizer + decoder + DisentangledProjection from checkpoint."""
    ckpt = Path(checkpoint_path)
    config_path = ckpt / "config.json"
    
    with open(config_path) as f:
        config = json.load(f)
    
    base_model = config.get("decoder_model_path", "Qwen/Qwen3-TTS-Tokenizer-12Hz")
    print(f"Loading base model: {base_model}")
    
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        base_model, attn_implementation="eager", dtype=dtype, device_map="cpu",
    )
    decoder = tokenizer.model.decoder
    
    # Rebuild decoder if needed (48k block)
    new_upsample_rates = config.get("new_upsample_rates")
    if new_upsample_rates:
        config_dict = decoder.config.to_dict()
        config_dict["upsample_rates"] = new_upsample_rates
        for k in ("model_type", "transformers_version"):
            config_dict.pop(k, None)
        new_config = Qwen3TTSTokenizerV2DecoderConfig(**config_dict)
        new_decoder = Qwen3TTSTokenizerV2Decoder(new_config)
        new_decoder.load_state_dict(decoder.state_dict(), strict=False)
        decoder = new_decoder
        print(f"  Rebuilt decoder: upsample_rates={new_upsample_rates}")
    
    # Load trained decoder weights
    dec_path = ckpt / "decoder_block.safetensors"
    if dec_path.exists():
        weights = load_file(str(dec_path))
        decoder.load_state_dict(weights, strict=False)
        print(f"  Loaded {len(weights)} decoder weight keys")
    
    decoder = decoder.to(device).to(dtype).eval()
    
    # Load DisentangledProjection
    speaker_dim = config.get("speaker_dim", 256)
    dis = DisentangledProjection(1024, speaker_dim).to(device).to(dtype)
    
    dis_path = ckpt / "disentangle.safetensors"
    if dis_path.exists():
        dis_weights = load_file(str(dis_path))
        dis.load_state_dict(dis_weights)
        print(f"  Loaded {len(dis_weights)} disentangle weight keys")
    else:
        print("  ⚠️ No disentangle.safetensors found!")
    
    dis.eval()
    
    output_sr = tokenizer.get_output_sample_rate()
    print(f"  Output SR: {output_sr} Hz")
    
    return tokenizer, decoder, dis, output_sr


@torch.no_grad()
def encode_to_hidden(tokenizer, decoder, audio, device, dtype):
    """audio (numpy) → hidden [1, T, 1024]"""
    encoded = tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
    codes = encoded.audio_codes[0].T.unsqueeze(0).to(device)  # [1, Q, T]
    
    hidden = decoder.quantizer.decode(codes)
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    return hidden


@torch.no_grad()
def decode_hidden(decoder, hidden):
    """hidden [1, T, 1024] → waveform numpy"""
    x = hidden.permute(0, 2, 1)
    for blocks in decoder.upsample:
        for block in blocks:
            x = block(x)
    wav = x
    for block in decoder.decoder:
        wav = block(wav)
    return wav.clamp(-1, 1).squeeze().cpu().float().numpy()


def cosine_sim(a, b):
    """Cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(
        a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)
    ).item()


def main():
    parser = argparse.ArgumentParser(description="Debug disentanglement")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--audio_a", required=True, help="Audio A (speaker A)")
    parser.add_argument("--audio_b", required=True, help="Audio B (speaker B)")
    parser.add_argument("--output_dir", default="./debug_output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = torch.bfloat16
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ── Load model ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading model...")
    print("=" * 70)
    tokenizer, decoder, dis, output_sr = setup_model(args.checkpoint, device, dtype)
    
    # ── Load audio ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Loading audio...")
    print("=" * 70)
    audio_a = load_audio(args.audio_a)
    audio_b = load_audio(args.audio_b)
    print(f"Audio A: {len(audio_a)/TOKENIZER_SR:.2f}s")
    print(f"Audio B: {len(audio_b)/TOKENIZER_SR:.2f}s")
    
    # ── Encode to hidden ────────────────────────────────────────────────────
    print("\nEncoding to hidden states...")
    hidden_a = encode_to_hidden(tokenizer, decoder, audio_a, device, dtype)
    hidden_b = encode_to_hidden(tokenizer, decoder, audio_b, device, dtype)
    print(f"Hidden A: {list(hidden_a.shape)}")
    print(f"Hidden B: {list(hidden_b.shape)}")
    
    # ── Disentangle ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Disentangling...")
    print("=" * 70)
    
    with torch.no_grad():
        speaker_contrib_a, content_a, speaker_global_a = dis(hidden_a)
        speaker_contrib_b, content_b, speaker_global_b = dis(hidden_b)
    
    # ── Analysis 1: Norms ───────────────────────────────────────────────────
    print("\n── Contribution Norms (per-frame average) ──")
    print(f"  Hidden A norm:          {hidden_a.norm(dim=-1).mean().item():.4f}")
    print(f"  Hidden B norm:          {hidden_b.norm(dim=-1).mean().item():.4f}")
    print(f"  Content A norm:         {content_a.norm(dim=-1).mean().item():.4f}")
    print(f"  Content B norm:         {content_b.norm(dim=-1).mean().item():.4f}")
    print(f"  Speaker contrib A norm: {speaker_contrib_a.norm(dim=-1).mean().item():.4f}")
    print(f"  Speaker contrib B norm: {speaker_contrib_b.norm(dim=-1).mean().item():.4f}")
    print(f"  Speaker global A norm:  {speaker_global_a.norm().item():.4f}")
    print(f"  Speaker global B norm:  {speaker_global_b.norm().item():.4f}")
    
    ratio_a = speaker_contrib_a.norm(dim=-1).mean() / content_a.norm(dim=-1).mean()
    ratio_b = speaker_contrib_b.norm(dim=-1).mean() / content_b.norm(dim=-1).mean()
    print(f"\n  Speaker/Content ratio A: {ratio_a.item():.6f}")
    print(f"  Speaker/Content ratio B: {ratio_b.item():.6f}")
    print(f"  → If ratio << 0.01, speaker path contributes NOTHING")
    
    # ── Analysis 2: Cosine similarities ─────────────────────────────────────
    print("\n── Cosine Similarities ──")
    print(f"  content_A vs hidden_A:      {cosine_sim(content_a, hidden_a):.6f}")
    print(f"  content_B vs hidden_B:      {cosine_sim(content_b, hidden_b):.6f}")
    print(f"  → If ≈1.0, content is just copying the full hidden (identity leak)")
    
    print(f"\n  speaker_global A vs B:      {cosine_sim(speaker_global_a, speaker_global_b):.6f}")
    print(f"  → If ≈1.0, speaker encoder can't tell speakers apart")
    
    # Check if content carries speaker info (cross-speaker similarity)
    content_a_mean = content_a.mean(dim=1)  # [1, 1024]
    content_b_mean = content_b.mean(dim=1)  # [1, 1024]
    print(f"\n  content_A_mean vs content_B_mean: {cosine_sim(content_a_mean, content_b_mean):.6f}")
    print(f"  → High = content carries speaker-agnostic info (good)")
    print(f"  → But if content ≈ hidden (identity), it also carries speaker info (bad)")
    
    # ── Analysis 3: Combined = speaker + content ────────────────────────────
    combined_a = speaker_contrib_a + content_a
    combined_b = speaker_contrib_b + content_b
    print(f"\n  combined_A vs hidden_A:     {cosine_sim(combined_a, hidden_a):.6f}")
    print(f"  combined_B vs hidden_B:     {cosine_sim(combined_b, hidden_b):.6f}")
    print(f"  → Should be ≈1.0 for good reconstruction")
    
    # ── Test 1: Normal reconstruction ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Test 1: Normal Reconstruction")
    print("=" * 70)
    
    wav_recon_a = decode_hidden(decoder, combined_a)
    wav_recon_b = decode_hidden(decoder, combined_b)
    sf.write(os.path.join(args.output_dir, "01_recon_A.wav"), wav_recon_a, output_sr)
    sf.write(os.path.join(args.output_dir, "01_recon_B.wav"), wav_recon_b, output_sr)
    print("  Saved: 01_recon_A.wav, 01_recon_B.wav")
    
    # ── Test 2: Voice Conversion (swap speakers) ────────────────────────────
    print("\n" + "=" * 70)
    print("Test 2: Voice Conversion (swap speakers)")
    print("=" * 70)
    
    # A's content + B's speaker
    vc_ab_contrib = dis.decode_speaker(speaker_global_b, content_a.shape[1])
    vc_ab = vc_ab_contrib + content_a
    wav_vc_ab = decode_hidden(decoder, vc_ab)
    sf.write(os.path.join(args.output_dir, "02_vc_contentA_speakerB.wav"), wav_vc_ab, output_sr)
    
    # B's content + A's speaker
    vc_ba_contrib = dis.decode_speaker(speaker_global_a, content_b.shape[1])
    vc_ba = vc_ba_contrib + content_b
    wav_vc_ba = decode_hidden(decoder, vc_ba)
    sf.write(os.path.join(args.output_dir, "02_vc_contentB_speakerA.wav"), wav_vc_ba, output_sr)
    print("  Saved: 02_vc_contentA_speakerB.wav, 02_vc_contentB_speakerA.wav")
    print("  → Listen: does the voice actually change?")
    
    # ── Test 3: Zero speaker contribution ───────────────────────────────────
    print("\n" + "=" * 70)
    print("Test 3: Zero out speaker contribution")
    print("=" * 70)
    
    wav_no_spk_a = decode_hidden(decoder, content_a)  # Only content, no speaker
    wav_no_spk_b = decode_hidden(decoder, content_b)
    sf.write(os.path.join(args.output_dir, "03_no_speaker_A.wav"), wav_no_spk_a, output_sr)
    sf.write(os.path.join(args.output_dir, "03_no_speaker_B.wav"), wav_no_spk_b, output_sr)
    print("  Saved: 03_no_speaker_A.wav, 03_no_speaker_B.wav")
    print("  → If sounds identical to recon, speaker path is useless")
    
    # ── Test 4: Random speaker tokens ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Test 4: Random speaker embedding")
    print("=" * 70)
    
    random_speaker = torch.randn_like(speaker_global_a)
    random_contrib = dis.decode_speaker(random_speaker, content_a.shape[1])
    wav_rand_a = decode_hidden(decoder, random_contrib + content_a)
    sf.write(os.path.join(args.output_dir, "04_random_speaker_A.wav"), wav_rand_a, output_sr)
    print("  Saved: 04_random_speaker_A.wav")
    print("  → If sounds identical to recon, speaker path contributes nothing")
    
    # ── Test 5: Amplified speaker contribution ──────────────────────────────
    print("\n" + "=" * 70)
    print("Test 5: Amplified speaker contribution (10x)")
    print("=" * 70)
    
    wav_amp_a = decode_hidden(decoder, speaker_contrib_a * 10 + content_a)
    wav_amp_b = decode_hidden(decoder, speaker_contrib_b * 10 + content_b)
    sf.write(os.path.join(args.output_dir, "05_amp10x_speaker_A.wav"), wav_amp_a, output_sr)
    sf.write(os.path.join(args.output_dir, "05_amp10x_speaker_B.wav"), wav_amp_b, output_sr)
    print("  Saved: 05_amp10x_speaker_A.wav, 05_amp10x_speaker_B.wav")
    print("  → Shows what the speaker path actually encodes")
    
    # ── Test 6: Content only from original hidden (bypass DisentangledProjection) ──
    print("\n" + "=" * 70)
    print("Test 6: Bypass disentangle (decode raw hidden)")
    print("=" * 70)
    
    wav_raw_a = decode_hidden(decoder, hidden_a)
    wav_raw_b = decode_hidden(decoder, hidden_b)
    sf.write(os.path.join(args.output_dir, "06_raw_hidden_A.wav"), wav_raw_a, output_sr)
    sf.write(os.path.join(args.output_dir, "06_raw_hidden_B.wav"), wav_raw_b, output_sr)
    print("  Saved: 06_raw_hidden_A.wav, 06_raw_hidden_B.wav")
    print("  → This is what the decoder produces WITHOUT disentanglement")
    
    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    
    content_identity = cosine_sim(content_a, hidden_a)
    speaker_ratio = ratio_a.item()
    speaker_diff = 1.0 - cosine_sim(speaker_global_a, speaker_global_b)
    
    print(f"\n  Content ≈ Hidden (identity leak):  {content_identity:.4f}  ", end="")
    if content_identity > 0.99:
        print("⚠️ SEVERE — content is copying everything including speaker!")
    elif content_identity > 0.95:
        print("⚠️ HIGH — content carries most speaker info")
    else:
        print("✅ OK — content has diverged from raw hidden")
    
    print(f"  Speaker/Content ratio:             {speaker_ratio:.6f}  ", end="")
    if speaker_ratio < 0.01:
        print("⚠️ DEAD — speaker path contributes < 1%")
    elif speaker_ratio < 0.1:
        print("⚠️ WEAK — speaker path barely contributes")
    else:
        print("✅ OK — speaker path has meaningful contribution")
    
    print(f"  Speaker A vs B difference:         {speaker_diff:.4f}  ", end="")
    if speaker_diff < 0.05:
        print("⚠️ — speakers look identical to the encoder")
    else:
        print("✅ OK — encoder can distinguish speakers")
    
    print(f"\n  Output directory: {os.path.abspath(args.output_dir)}")
    print(f"\n  Files to compare:")
    print(f"    01_recon_A.wav vs 03_no_speaker_A.wav → is speaker path doing anything?")
    print(f"    01_recon_A.wav vs 02_vc_contentA_speakerB.wav → did voice change?")
    print(f"    01_recon_A.wav vs 06_raw_hidden_A.wav → is disentangle changing the signal?")
    print("=" * 70)


if __name__ == "__main__":
    main()
