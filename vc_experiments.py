#!/usr/bin/env python3
"""
Voice Conversion Experiments: Training-free approaches using the BASE Qwen tokenizer.

No checkpoint, no DisentangledProjection. Just the original frozen model.

Tests 3 inference-time VC methods:
  1. AdaIN: Normalize source hidden stats → target hidden stats
  2. kNN-VC: Replace each source frame with nearest neighbor from target
  3. Hybrid: AdaIN + kNN blend

Usage:
    python vc_experiments.py \
        --audio_a audios/speaker_A.wav \
        --audio_b audios/speaker_B.wav \
        --output_dir ./vc_experiments_output
"""

import argparse
import os
import sys
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


@torch.no_grad()
def encode_full(tokenizer, audio, device):
    """Encode audio through full pipeline → hidden [1, T, 1024]"""
    decoder = tokenizer.model.decoder
    encoded = tokenizer.encode(audios=[audio], sr=TOKENIZER_SR)
    codes = encoded.audio_codes[0].T.unsqueeze(0).to(device)
    
    hidden = decoder.quantizer.decode(codes)
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    return hidden, codes


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


@torch.no_grad()
def decode_from_pre_conv(decoder, pre_conv_out):
    """pre_conv output [1, T, D] → through pre_transformer → upsample → decoder → waveform"""
    hidden = decoder.pre_transformer(inputs_embeds=pre_conv_out).last_hidden_state
    return decode_hidden(decoder, hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# VC Method 1: AdaIN (Adaptive Instance Normalization)
# ═══════════════════════════════════════════════════════════════════════════════
def adain_vc(hidden_src, hidden_tgt, eps=1e-5):
    """Transfer global statistics from target to source hidden.
    
    This transfers the vocal tract characteristics (mean, variance of 
    hidden features) from the target speaker to the source content.
    
    output = (src - mean_src) / std_src * std_tgt + mean_tgt
    """
    # Compute per-feature statistics across time
    mean_src = hidden_src.mean(dim=1, keepdim=True)   # [1, 1, 1024]
    std_src = hidden_src.std(dim=1, keepdim=True) + eps
    mean_tgt = hidden_tgt.mean(dim=1, keepdim=True)
    std_tgt = hidden_tgt.std(dim=1, keepdim=True) + eps
    
    # Normalize and re-scale
    normalized = (hidden_src - mean_src) / std_src
    return normalized * std_tgt + mean_tgt


# ═══════════════════════════════════════════════════════════════════════════════
# VC Method 2: kNN Feature Matching
# ═══════════════════════════════════════════════════════════════════════════════
def knn_vc(hidden_src, hidden_tgt, k=4):
    """Replace each source frame with average of k nearest neighbors from target.
    
    This is the approach used by kNN-VC (Baas et al., 2023).
    Each source frame is matched to the closest target frames,
    which naturally carries the target speaker's characteristics.
    """
    # hidden_src: [1, T_s, D], hidden_tgt: [1, T_t, D]
    src = F.normalize(hidden_src.squeeze(0), dim=-1)  # [T_s, D]
    tgt = F.normalize(hidden_tgt.squeeze(0), dim=-1)  # [T_t, D]
    
    # Cosine similarity matrix [T_s, T_t]
    sim = torch.mm(src, tgt.t())
    
    # Top-k nearest neighbors
    k = min(k, tgt.shape[0])
    topk_sim, topk_idx = sim.topk(k, dim=-1)  # [T_s, k]
    
    # Weighted average of top-k target frames
    weights = F.softmax(topk_sim * 10, dim=-1)  # temperature-scaled weights
    
    # Gather target frames and compute weighted average
    tgt_raw = hidden_tgt.squeeze(0)  # [T_t, D] (unnormalized)
    matched = torch.zeros_like(hidden_src.squeeze(0))
    for i in range(k):
        matched += weights[:, i:i+1] * tgt_raw[topk_idx[:, i]]
    
    return matched.unsqueeze(0)


# ═══════════════════════════════════════════════════════════════════════════════
# VC Method 3: Hybrid (AdaIN + kNN blend)
# ═══════════════════════════════════════════════════════════════════════════════
def hybrid_vc(hidden_src, hidden_tgt, knn_weight=0.3, k=4):
    """Blend AdaIN (global style) with kNN (local matching).
    
    AdaIN transfers global vocal characteristics.
    kNN transfers fine-grained phoneme-level features.
    """
    adain_result = adain_vc(hidden_src, hidden_tgt)
    knn_result = knn_vc(hidden_src, hidden_tgt, k=k)
    return (1 - knn_weight) * adain_result + knn_weight * knn_result


# ═══════════════════════════════════════════════════════════════════════════════
# VC Method 4: Code-level manipulation
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def code_swap_vc(decoder, codes_src, codes_tgt, swap_codebooks=(0,)):
    """Swap specific codebook codes between source and target.
    
    Higher codebooks typically carry more fine-grained / speaker info.
    Lower codebooks carry content.
    
    Args:
        codes_src: [1, Q, T_s] source codes
        codes_tgt: [1, Q, T_t] target codes  
        swap_codebooks: which codebook indices to take from target
    """
    T_s = codes_src.shape[2]
    T_t = codes_tgt.shape[2]
    
    # Build mixed codes: start with source
    mixed = codes_src.clone()
    
    for cb in swap_codebooks:
        if T_t >= T_s:
            mixed[0, cb, :] = codes_tgt[0, cb, :T_s]
        else:
            # Repeat target codes to match source length
            repeat = (T_s // T_t) + 1
            expanded = codes_tgt[0, cb].repeat(repeat)[:T_s]
            mixed[0, cb, :] = expanded
    
    # Decode mixed codes
    hidden = decoder.quantizer.decode(mixed)
    hidden = decoder.pre_conv(hidden).transpose(1, 2)
    hidden = decoder.pre_transformer(inputs_embeds=hidden).last_hidden_state
    return decode_hidden(decoder, hidden), mixed


def main():
    parser = argparse.ArgumentParser(description="VC Experiments (training-free)")
    parser.add_argument("--audio_a", required=True, help="Audio A (source content)")
    parser.add_argument("--audio_b", required=True, help="Audio B (target speaker)")
    parser.add_argument("--output_dir", default="./vc_experiments_output")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    dtype = torch.bfloat16
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load BASE model (no checkpoint needed!)
    print("=" * 70)
    print("Loading BASE Qwen3-TTS-Tokenizer-12Hz (no checkpoint)")
    print("=" * 70)
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        attn_implementation="eager",
        dtype=dtype,
        device_map="cpu" if device.type == "cpu" else str(device),
    )
    decoder = tokenizer.model.decoder
    output_sr = tokenizer.get_output_sample_rate()
    print(f"Output SR: {output_sr} Hz")
    
    # Load audio
    print("\n" + "=" * 70)
    print("Loading audio...")
    print("=" * 70)
    audio_a = load_audio(args.audio_a)
    audio_b = load_audio(args.audio_b)
    print(f"Audio A (content): {len(audio_a)/TOKENIZER_SR:.2f}s")
    print(f"Audio B (speaker): {len(audio_b)/TOKENIZER_SR:.2f}s")
    
    # Encode
    print("\nEncoding...")
    hidden_a, codes_a = encode_full(tokenizer, audio_a, device)
    hidden_b, codes_b = encode_full(tokenizer, audio_b, device)
    print(f"Hidden A: {list(hidden_a.shape)}")
    print(f"Hidden B: {list(hidden_b.shape)}")
    print(f"Codes A: {list(codes_a.shape)}")
    print(f"Codes B: {list(codes_b.shape)}")
    
    # Reference: normal reconstruction
    print("\n" + "=" * 70)
    print("Reference: Normal Reconstruction")
    print("=" * 70)
    wav_a = decode_hidden(decoder, hidden_a)
    wav_b = decode_hidden(decoder, hidden_b)
    sf.write(os.path.join(args.output_dir, "00_recon_A.wav"), wav_a, output_sr)
    sf.write(os.path.join(args.output_dir, "00_recon_B.wav"), wav_b, output_sr)
    print("  Saved: 00_recon_A.wav, 00_recon_B.wav")
    
    # VC Method 1: AdaIN
    print("\n" + "=" * 70)
    print("Method 1: AdaIN (global statistics transfer)")
    print("=" * 70)
    hidden_adain = adain_vc(hidden_a, hidden_b)
    wav_adain = decode_hidden(decoder, hidden_adain)
    sf.write(os.path.join(args.output_dir, "01_adain_A_to_B.wav"), wav_adain, output_sr)
    
    hidden_adain_rev = adain_vc(hidden_b, hidden_a)
    wav_adain_rev = decode_hidden(decoder, hidden_adain_rev)
    sf.write(os.path.join(args.output_dir, "01_adain_B_to_A.wav"), wav_adain_rev, output_sr)
    print("  Saved: 01_adain_A_to_B.wav, 01_adain_B_to_A.wav")
    
    # VC Method 2: kNN
    print("\n" + "=" * 70)
    print("Method 2: kNN Feature Matching")
    print("=" * 70)
    for k in [1, 4, 8]:
        hidden_knn = knn_vc(hidden_a, hidden_b, k=k)
        wav_knn = decode_hidden(decoder, hidden_knn)
        sf.write(os.path.join(args.output_dir, f"02_knn_k{k}_A_to_B.wav"), wav_knn, output_sr)
        print(f"  Saved: 02_knn_k{k}_A_to_B.wav")
    
    # VC Method 3: Hybrid
    print("\n" + "=" * 70)
    print("Method 3: Hybrid (AdaIN + kNN)")
    print("=" * 70)
    for w in [0.2, 0.5, 0.8]:
        hidden_hybrid = hybrid_vc(hidden_a, hidden_b, knn_weight=w)
        wav_hybrid = decode_hidden(decoder, hidden_hybrid)
        sf.write(os.path.join(args.output_dir, f"03_hybrid_knn{w:.1f}_A_to_B.wav"), wav_hybrid, output_sr)
        print(f"  Saved: 03_hybrid_knn{w:.1f}_A_to_B.wav")
    
    # VC Method 4: Code-level swap
    print("\n" + "=" * 70)
    print("Method 4: Codebook-level swap")
    print("=" * 70)
    num_cb = codes_a.shape[1]
    print(f"  Total codebooks: {num_cb}")
    
    # Try swapping different codebook combinations
    swap_configs = [
        ((0,), "cb0_from_tgt"),
        ((num_cb-1,), f"cb{num_cb-1}_from_tgt"),
        (tuple(range(num_cb//2, num_cb)), f"cb{num_cb//2}to{num_cb-1}_from_tgt"),
        (tuple(range(0, num_cb//2)), f"cb0to{num_cb//2-1}_from_tgt"),
    ]
    
    for swap_cbs, label in swap_configs:
        wav_code, mixed = code_swap_vc(decoder, codes_a, codes_b, swap_codebooks=swap_cbs)
        sf.write(os.path.join(args.output_dir, f"04_code_{label}.wav"), wav_code, output_sr)
        print(f"  Saved: 04_code_{label}.wav (swapped codebooks {swap_cbs})")
    
    print("\n" + "=" * 70)
    print("DONE! Compare outputs:")
    print("=" * 70)
    print(f"  Output: {os.path.abspath(args.output_dir)}")
    print()
    print("  Method 1 (AdaIN): Global voice transfer. Should change pitch/tone.")
    print("  Method 2 (kNN): Frame matching. Best for timbre transfer.")
    print("  Method 3 (Hybrid): Blend of both.")
    print("  Method 4 (Codes): Swap codebooks. Tests which codebooks carry speaker.")
    print()
    print("  Listen to all and tell me which sounds most like B's voice")
    print("  with A's content!")
    print("=" * 70)


if __name__ == "__main__":
    main()
