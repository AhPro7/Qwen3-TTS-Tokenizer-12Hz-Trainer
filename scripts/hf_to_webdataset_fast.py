#!/usr/bin/env python3
import argparse, io, os, warnings
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from qwen_tts import Qwen3TTSTokenizer

TOKENIZER_SR = 24000
OUTPUT_SR = 48000
VAD_SR = 16000
VAD_WINDOW = 512

# ---- Resample cache ----
_resamplers = {}
def resample_torch(audio_np, orig_sr, target_sr, device):
    if orig_sr == target_sr:
        return audio_np
    key = (orig_sr, target_sr)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
    t = torch.from_numpy(audio_np).float().to(device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return _resamplers[key](t).squeeze(0).cpu().numpy()

# ---- Normalization ----
def normalize_audio(wav, sr):
    BS = 0.4
    if len(wav) < sr * BS:
        wav = np.pad(wav, (0, int(sr * BS) - len(wav)))
    meter = pyln.Meter(sr, block_size=BS)
    loudness = meter.integrated_loudness(wav)
    if loudness == float("-inf"):
        return wav.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = pyln.normalize.loudness(wav, loudness, -23.0).astype(np.float32)
        if np.max(np.abs(out)) >= 1.0:
            out = pyln.normalize.peak(out, -1.0).astype(np.float32)
    return out

# ---- VAD (GPU torch mode) ----
def load_vad(device):
    logger.info("Loading SileroVAD (torch/GPU)...")
    model, _ = torch.hub.load("litagin02/silero-vad", "silero_vad", onnx=False, trust_repo=True)
    return model.to(device).eval()

def vad_process(audio, sr, vad_model, device, silence_gap_s=0.1, min_voiced_s=1.0):
    pad = int(sr * 0.5)
    padded = np.pad(audio, (pad, pad))
    audio_16k = resample_torch(padded, sr, VAD_SR, device)
    t = torch.from_numpy(audio_16k).float().to(device)
    vad_model.reset_states()
    probs = []
    with torch.inference_mode():
        for i in range(0, len(t), VAD_WINDOW):
            chunk = t[i:i+VAD_WINDOW]
            probs.append(vad_model(chunk, VAD_SR).item() if len(chunk)==VAD_WINDOW else 0.0)
    if probs: probs[-1] = 0.0
    probs = np.array(probs)
    hi = probs >= 0.5
    lo_or_hi = (probs <= 0.35) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: return audio
    cnt = np.cumsum(lo_or_hi)
    binary = np.where(cnt, hi[ind[cnt-1]], False).astype(int)
    vi = np.where(binary==1)[0]
    if not len(vi): return audio
    breaks = np.where(np.diff(vi)>1)[0]
    starts = np.concatenate([[vi[0]], vi[breaks+1]])
    ends = np.concatenate([vi[breaks], [vi[-1]]])
    min_s = int(min_voiced_s * sr)
    gap = np.zeros(int(silence_gap_s * sr), dtype=np.float32)
    segs = []
    for fs, fe in zip(starts, ends):
        s = max(0, int(fs * VAD_WINDOW / VAD_SR * sr))
        e = min(len(padded), int((fe+1) * VAD_WINDOW / VAD_SR * sr))
        if e - s >= min_s:
            segs.append(padded[s:e])
    if not segs: return audio
    out = segs[0]
    for seg in segs[1:]: out = np.concatenate([out, gap, seg])
    return out

# ---- Assembler ----
class Assembler:
    def __init__(self, dur_s, gap_s):
        self.dur = int(dur_s * OUTPUT_SR)
        self.gap = np.zeros(int(gap_s * OUTPUT_SR), dtype=np.float32)
        self.tail = np.array([], dtype=np.float32)
    def add(self, a):
        buf = np.concatenate([self.tail, self.gap, a]) if len(self.tail) else a
        chunks, buf = [], buf
        while len(buf) >= self.dur:
            chunks.append(buf[:self.dur].copy())
            buf = buf[self.dur:]
        self.tail = buf
        return chunks
    def flush(self):
        if not len(self.tail): return []
        p = np.zeros(self.dur, dtype=np.float32)
        p[:len(self.tail)] = self.tail
        self.tail = np.array([], dtype=np.float32)
        return [p]

# ---- Encode ----
def enc_flac(a):
    buf = io.BytesIO()
    sf.write(buf, (np.clip(a,-1,1)*32767).astype(np.int16), OUTPUT_SR, format="flac", subtype="PCM_16")
    return buf.getvalue()

def enc_npy(c):
    buf = io.BytesIO()
    np.save(buf, c)
    return buf.getvalue()

# ---- Shard writer ----
class ShardWriter:
    def __init__(self, d, size):
        self.d, self.size, self.idx, self.count, self._w = d, size, 0, 0, None
        self._open()
    def _open(self):
        if self._w: self._w.close()
        p = str(self.d / f"shard-{self.idx:06d}.tar")
        self._w = wds.TarWriter(p)
        logger.debug(f"Shard: {p}")
    def write(self, key, flac, npy):
        if self.count and self.count % self.size == 0:
            self.idx += 1; self._open()
        self._w.write({"__key__": key, "flac": flac, "npy": npy})
        self.count += 1
    def close(self):
        if self._w: self._w.close(); self._w = None
    @property
    def num_shards(self): return self.idx + 1 if self.count else 0

# ---- Tokenizer ----
def load_tokenizer(device):
    logger.info("Loading Qwen3TTS tokenizer...")
    return Qwen3TTSTokenizer.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz", dtype=torch.bfloat16, device_map=device)

@torch.inference_mode()
def tokenize_batch(audios, tokenizer):
    enc = tokenizer.encode(audios=audios, sr=TOKENIZER_SR)
    return [c.cpu().numpy().flatten().astype(np.int32) for c in enc.audio_codes]

# ---- Worker thread ----
def worker(dataset, assembler, vad_model, device, gap, min_v, stage_size, rng, queue, pbar, skipped):
    staging = []
    def flush(force=False):
        if not staging: return
        rng.shuffle(staging)
        cut = len(staging) if force else len(staging)//2
        b, staging[:] = staging[:cut], staging[cut:]
        if b: queue.put(b)

    for voice in dataset:
        arr = voice["audio"]["array"]
        sr  = voice["audio"]["sampling_rate"]
        try:
            if arr.ndim > 1: arr = arr.mean(-1)
            arr = arr.astype(np.float32)
            arr = normalize_audio(arr, sr)
            arr = vad_process(arr, sr, vad_model, device, gap, min_v)
            arr48 = resample_torch(arr, sr, OUTPUT_SR, device)
        except Exception as e:
            logger.warning(f"Skip: {e}"); skipped[0] += 1; continue

        for c48 in assembler.add(arr48):
            c24 = resample_torch(c48, OUTPUT_SR, TOKENIZER_SR, device)
            staging.append((c24, c48))
            pbar.update(1)
        if len(staging) >= stage_size:
            flush()

    for c48 in assembler.flush():
        c24 = resample_torch(c48, OUTPUT_SR, TOKENIZER_SR, device)
        staging.append((c24, c48))
        pbar.update(1)
    flush(force=True)
    queue.put(None)

# ---- Main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("output_dir")
    ap.add_argument("--split", default="train")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--shard-size", type=int, default=1000)
    ap.add_argument("--val-percent", type=float, default=5.0)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--stage-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--silence-gap", type=float, default=0.1)
    ap.add_argument("--min-voiced", type=float, default=1.0)
    ap.add_argument("--tokenizer-batch", type=int, default=32)
    ap.add_argument("--queue-size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

    tokenizer = load_tokenizer(device)
    vad_model  = load_vad(device)

    out = Path(args.output_dir)
    (out/"train").mkdir(parents=True, exist_ok=True)
    (out/"val").mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading dataset (non-streaming)...")
    kw = {"split": args.split}
    if args.dataset_config: kw["name"] = args.dataset_config
    ds = load_dataset(args.dataset, **kw).shuffle(seed=args.seed)
    logger.info(f"Loaded {len(ds):,} samples")

    tw = ShardWriter(out/"train", args.shard_size)
    vw = ShardWriter(out/"val",   args.shard_size)
    rng_stg  = np.random.default_rng(args.seed)
    rng_spl  = np.random.default_rng(args.seed+1)
    val_frac = args.val_percent / 100.0
    asm      = Assembler(args.duration, args.silence_gap)
    gkey     = [0]
    skipped  = [0]
    q        = Queue(maxsize=args.queue_size)

    try:
        with tqdm(desc="Processing", unit="chunks") as pbar:
            t = Thread(target=worker,
                       args=(ds, asm, vad_model, device, args.silence_gap,
                             args.min_voiced, args.stage_size, rng_stg, q, pbar, skipped),
                       daemon=True)
            t.start()
            while True:
                batch = q.get()
                if batch is None: break
                for i in range(0, len(batch), args.tokenizer_batch):
                    sub = batch[i:i+args.tokenizer_batch]
                    codes = tokenize_batch([c24 for c24,_ in sub], tokenizer)
                    for (_, c48), code in zip(sub, codes):
                        key = f"{gkey[0]:08d}"; gkey[0] += 1
                        flac, npy = enc_flac(c48), enc_npy(code)
                        (vw if rng_spl.random()<val_frac else tw).write(key, flac, npy)
            t.join()
    finally:
        tw.close(); vw.close()

    logger.info(f"Done! Train: {tw.count:,} in {tw.num_shards} shards | Val: {vw.count:,} in {vw.num_shards} shards | Skipped: {skipped[0]:,}")

if __name__ == "__main__":
    main()
