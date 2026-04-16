#!/usr/bin/env python3
# coding=utf-8
"""
Model Inspector — Qwen3-TTS-Tokenizer-12Hz
Prints a full breakdown of the model architecture, codebook details,
parameter counts per component, and audio I/O specs.

Usage:
    python scripts/model_info.py
    python scripts/model_info.py --model_path Qwen/Qwen3-TTS-Tokenizer-12Hz
    python scripts/model_info.py --checkpoint output/run55/checkpoint-best
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "xcodec2"))


def fmt(n: int) -> str:
    """Format large numbers nicely."""
    if n >= 1_000_000_000:
        return f"{n/1e9:.3f}B"
    if n >= 1_000_000:
        return f"{n/1e6:.2f}M"
    if n >= 1_000:
        return f"{n/1e3:.1f}K"
    return str(n)


def count_params(module: nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def section(title: str, width: int = 62):
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print(f"{'─'*width}")


def row(label: str, value, width: int = 38):
    print(f"  {label:<{width}} {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint dir to load decoder weights from")
    args = parser.parse_args()

    print(f"\n{'═'*62}")
    print(f"  Qwen3-TTS-Tokenizer Model Inspector")
    print(f"{'═'*62}")

    # ── Load tokenizer ────────────────────────────────────────────
    from qwen_tts import Qwen3TTSTokenizer
    print(f"\nLoading from: {args.model_path}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.model_path,
        attn_implementation="eager",
        dtype=torch.float32,
        device_map=None,
    )
    model = tokenizer.model
    decoder = model.decoder
    cfg = decoder.config

    # ── I/O Specs ─────────────────────────────────────────────────
    section("Audio I/O Specifications")
    input_sr = getattr(tokenizer.config, "input_sample_rate", 24_000)
    output_sr = getattr(tokenizer.config, "output_sample_rate", 24_000)
    try:
        output_sr = tokenizer.get_output_sample_rate()
    except Exception:
        pass
    row("Input sample rate",  f"{input_sr:,} Hz")
    row("Output sample rate", f"{output_sr:,} Hz")
    row("Token rate",         f"{cfg.frame_shift:.1f} Hz  (1 token = {1000/cfg.frame_shift:.1f} ms)"
        if hasattr(cfg, "frame_shift") else "12 Hz  (1 token = 83.3 ms)")
    row("Upsample chain",     f"{list(cfg.upsample_rates)}")
    total_upsample = 1
    for r in cfg.upsample_rates:
        total_upsample *= r
    token_rate = 12  # 12Hz codecs
    computed_sr = token_rate * total_upsample
    row("Total upsample factor", f"×{total_upsample}  (12 Hz tokens × {total_upsample} = {computed_sr:,} Hz output)")

    # ── Codebook / Quantizer ──────────────────────────────────────
    section("Codebook / Quantizer")
    num_q = cfg.num_quantizers
    codebook_size = cfg.codebook_size
    codebook_dim  = cfg.codebook_dim if hasattr(cfg, "codebook_dim") else "?"
    n_q_semantic  = getattr(cfg, "n_q_semantic", "?")

    row("Num quantizers (codebooks)", num_q)
    row("Codebook size (vocab/book)", f"{codebook_size:,}")
    row("Codebook dim",               codebook_dim)
    row("Semantic quantizers",        n_q_semantic)
    row("Acoustic (RVQ) quantizers",  f"{num_q} - {n_q_semantic} = {num_q - n_q_semantic}"
        if isinstance(n_q_semantic, int) else "?")
    row("Total vocab entries",        f"{num_q} × {codebook_size:,} = {num_q * codebook_size:,}")
    row("Bits per token",             f"{num_q} × log2({codebook_size}) = {num_q * (codebook_size.bit_length()-1)} bits")
    row("Bitrate @ 12Hz",             f"12 × {num_q} × log2({codebook_size}) = {12 * num_q * (codebook_size.bit_length()-1)} bps")

    # ── Transformer (pre_transformer) ─────────────────────────────
    section("Pre-Transformer (hidden representation)")
    hidden_size    = getattr(cfg, "hidden_size", "?")
    num_layers     = getattr(cfg, "num_hidden_layers", "?")
    num_heads      = getattr(cfg, "num_attention_heads", "?")
    intermediate   = getattr(cfg, "intermediate_size", "?")
    max_pos        = getattr(cfg, "max_position_embeddings", "?")

    row("Hidden size",             hidden_size)
    row("Transformer layers",      num_layers)
    row("Attention heads",         num_heads)
    row("Intermediate (FFN) size", intermediate)
    row("Max position embeddings", max_pos)

    # ── Decoder (HiFiGAN-style) ───────────────────────────────────
    section("HiFiGAN Decoder Blocks")
    row("Upsample rates",       list(cfg.upsample_rates))
    row("Upsample kernels",     list(getattr(cfg, "upsample_kernel_sizes", "?")))
    row("Resblock kernel sizes",list(getattr(cfg, "resblock_kernel_sizes", "?")))
    row("Resblock dilation sizes", list(getattr(cfg, "resblock_dilation_sizes", "?")))
    row("Num decoder blocks",   len(decoder.decoder))
    row("Pre-conv channels",    getattr(cfg, "pre_conv_channels", "?"))

    # ── Parameter counts ─────────────────────────────────────────
    section("Parameter Counts")

    total_all, train_all = count_params(decoder)
    row("Total decoder params",      f"{fmt(total_all)}  ({total_all:,})")

    if hasattr(decoder, "quantizer"):
        n, _ = count_params(decoder.quantizer)
        row("  └─ quantizer",        fmt(n))
    if hasattr(decoder, "pre_conv"):
        n, _ = count_params(decoder.pre_conv)
        row("  └─ pre_conv",         fmt(n))
    if hasattr(decoder, "pre_transformer"):
        n, _ = count_params(decoder.pre_transformer)
        row("  └─ pre_transformer",  fmt(n))
    if hasattr(decoder, "upsample"):
        n = sum(p.numel() for blocks in decoder.upsample for b in blocks for p in b.parameters())
        row("  └─ upsample blocks",  fmt(n))
    if hasattr(decoder, "decoder"):
        for i, block in enumerate(decoder.decoder):
            n, _ = count_params(block)
            row(f"  └─ decoder[{i}]",  fmt(n))

    # Encoder (if available)
    if hasattr(model, "encoder"):
        n, _ = count_params(model.encoder)
        row("Encoder params",        fmt(n))

    total_model, _ = count_params(model)
    row("Total model params",        f"{fmt(total_model)}  ({total_model:,})")

    # ── Load checkpoint info ──────────────────────────────────────
    if args.checkpoint:
        section(f"Checkpoint: {Path(args.checkpoint).name}")
        ckpt_path = Path(args.checkpoint)

        config_file = ckpt_path / "config.json"
        if config_file.exists():
            with open(config_file) as f:
                ckpt_cfg = json.load(f)
            for k, v in ckpt_cfg.items():
                row(k, v)

        for fname in ["decoder_block.safetensors", "disentangle.safetensors", "discriminator.pt"]:
            fpath = ckpt_path / fname
            if fpath.exists():
                size_mb = fpath.stat().st_size / 1e6
                row(f"  {fname}", f"✓  {size_mb:.1f} MB")
            else:
                row(f"  {fname}", "✗  not found")

    # ── Summary ───────────────────────────────────────────────────
    section("Quick Summary")
    print(f"""
  Model   : Qwen3-TTS-Tokenizer-12Hz
  Codec   : {num_q} codebooks × {codebook_size:,} codes  (RVQ)
  Token   : 12 Hz  (83.3 ms/token)
  Bitrate : {12 * num_q * (codebook_size.bit_length()-1)} bps
  Decoder : {fmt(total_all)} params  (HiFiGAN ×{total_upsample})
  Total   : {fmt(total_model)} params
""")


if __name__ == "__main__":
    main()
