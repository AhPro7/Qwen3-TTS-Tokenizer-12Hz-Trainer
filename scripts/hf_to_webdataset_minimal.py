#!/usr/bin/env python3
"""
Minimal fast HuggingFace -> WebDataset converter.
NO VAD, NO loudness normalization — just resample, chunk, tokenize.
"""
import argparse, io, os, sys, warnings

# ── Auto-repair datasets/features/audio.py ──────────────────────────
# The HF datasets >= 4.5 requires torchcodec (FFmpeg) for audio
# decoding. This function detects a corrupted or torchcodec-dependent
# audio.py and rewrites it with a soundfile-based version.
# --------------------------------------------------------------------
def _repair_datasets_audio():
    """Fix datasets audio.py to use soundfile instead of torchcodec."""
    import importlib.util, site

    # Locate audio.py
    audio_path = None
    search = []
    try:
        search += site.getsitepackages()
    except Exception:
        pass
    try:
        search.append(site.getusersitepackages())
    except Exception:
        pass
    for sp in search:
        c = os.path.join(sp, "datasets", "features", "audio.py")
        if os.path.isfile(c):
            audio_path = c
            break
    if audio_path is None:
        spec = importlib.util.find_spec("datasets")
        if spec and spec.origin:
            c = os.path.join(os.path.dirname(spec.origin), "features", "audio.py")
            if os.path.isfile(c):
                audio_path = c
    if audio_path is None:
        return  # let the import fail naturally

    with open(audio_path, "r") as f:
        source = f.read()

    # Check if repair is needed: syntax error OR torchcodec dependency
    needs_repair = False
    try:
        compile(source, audio_path, "exec")
    except SyntaxError:
        needs_repair = True
    if not needs_repair and "torchcodec" not in source:
        return  # file is fine and already patched
    needs_repair = True  # torchcodec present → replace with soundfile

    corrected = '''\
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional, Union

import numpy as np
import pyarrow as pa

from .. import config
from ..download.download_config import DownloadConfig
from ..table import array_cast
from ..utils.file_utils import is_local_path, xopen
from ..utils.py_utils import no_op_if_value_is_null, string_to_dict

if TYPE_CHECKING:
    from .features import FeatureType


@dataclass
class Audio:
    """Audio Feature — patched to use soundfile (no torchcodec dependency)."""

    sampling_rate: Optional[int] = None
    decode: bool = True
    num_channels: Optional[int] = None
    stream_index: Optional[int] = None
    id: Optional[str] = field(default=None, repr=False)
    # Automatically constructed
    dtype: ClassVar[str] = "dict"
    pa_type: ClassVar[Any] = pa.struct({"bytes": pa.binary(), "path": pa.string()})
    _type: str = field(default="Audio", init=False, repr=False)

    def __call__(self):
        return self.pa_type

    def encode_example(self, value) -> dict:
        """Encode example into a format for Arrow."""
        if value is None:
            raise ValueError("value must be provided")
        if isinstance(value, str):
            return {"bytes": None, "path": value}
        elif isinstance(value, Path):
            return {"bytes": None, "path": str(value.absolute())}
        elif isinstance(value, (bytes, bytearray)):
            return {"bytes": value, "path": None}
        elif isinstance(value, dict):
            if "array" in value:
                import soundfile as sf
                buf = BytesIO()
                sf.write(buf, value["array"].astype(np.float32),
                         value["sampling_rate"], format="wav")
                return {"bytes": buf.getvalue(), "path": None}
            elif value.get("path") is not None and os.path.isfile(value["path"]):
                return {"bytes": None, "path": value.get("path")}
            elif value.get("bytes") is not None or value.get("path") is not None:
                return {"bytes": value.get("bytes"), "path": value.get("path")}
        raise ValueError(
            f"An audio sample should have one of \'path\' or \'bytes\' "
            f"but they are missing or None in {value}."
        )

    def decode_example(self, value, token_per_repo_id=None):
        """Decode example audio file into audio data using soundfile."""
        if not self.decode:
            raise RuntimeError(
                "Decoding is disabled for this feature. "
                "Please use Audio(decode=True) instead."
            )

        path = value["path"]
        file_bytes = value["bytes"]
        if path is None and file_bytes is None:
            raise ValueError(
                f"An audio sample should have one of \'path\' or \'bytes\' "
                f"but both are None in {value}."
            )

        import io as _io
        import soundfile as sf

        if file_bytes is not None:
            array, sr = sf.read(
                _io.BytesIO(file_bytes), dtype="float32", always_2d=False
            )
        elif is_local_path(path):
            array, sr = sf.read(path, dtype="float32", always_2d=False)
        else:
            token_per_repo_id = token_per_repo_id or {}
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL
                if source_url.startswith(config.HF_ENDPOINT)
                else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = (
                token_per_repo_id.get(source_url_fields["repo_id"])
                if source_url_fields is not None
                else None
            )
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                array, sr = sf.read(
                    _io.BytesIO(f.read()), dtype="float32", always_2d=False
                )

        if self.sampling_rate and self.sampling_rate != sr:
            try:
                import torchaudio, torch
                resampler = torchaudio.transforms.Resample(sr, self.sampling_rate)
                array = resampler(
                    torch.from_numpy(array).float()
                ).numpy()
            except ImportError:
                import librosa
                array = librosa.resample(
                    array, orig_sr=sr, target_sr=self.sampling_rate
                )
            sr = self.sampling_rate

        return {"path": path, "array": array, "sampling_rate": sr}

    def flatten(self) -> Union["FeatureType", dict[str, "FeatureType"]]:
        from .features import Value
        if self.decode:
            raise ValueError("Cannot flatten a decoded Audio feature.")
        return {
            "bytes": Value("binary"),
            "path": Value("string"),
        }

    def cast_storage(self, storage: Union[pa.StringArray, pa.StructArray]) -> pa.StructArray:
        if pa.types.is_string(storage.type):
            bytes_array = pa.array([None] * len(storage), type=pa.binary())
            storage = pa.StructArray.from_arrays(
                [bytes_array, storage], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_large_binary(storage.type):
            storage = array_cast(storage, pa.binary())
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [storage, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_binary(storage.type):
            path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [storage, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        elif pa.types.is_struct(storage.type) and storage.type.get_all_field_indices("array"):
            storage = pa.array(
                [Audio().encode_example(x) if x is not None else None
                 for x in storage.to_numpy(zero_copy_only=False)]
            )
        elif pa.types.is_struct(storage.type):
            if storage.type.get_field_index("bytes") >= 0:
                bytes_array = storage.field("bytes")
            else:
                bytes_array = pa.array([None] * len(storage), type=pa.binary())
            if storage.type.get_field_index("path") >= 0:
                path_array = storage.field("path")
            else:
                path_array = pa.array([None] * len(storage), type=pa.string())
            storage = pa.StructArray.from_arrays(
                [bytes_array, path_array], ["bytes", "path"], mask=storage.is_null()
            )
        return array_cast(storage, self.pa_type)

    def embed_storage(self, storage, token_per_repo_id=None):
        if token_per_repo_id is None:
            token_per_repo_id = {}

        @no_op_if_value_is_null
        def path_to_bytes(path):
            source_url = path.split("::")[-1]
            pattern = (
                config.HUB_DATASETS_URL
                if source_url.startswith(config.HF_ENDPOINT)
                else config.HUB_DATASETS_HFFS_URL
            )
            source_url_fields = string_to_dict(source_url, pattern)
            token = (
                token_per_repo_id.get(source_url_fields["repo_id"])
                if source_url_fields is not None
                else None
            )
            download_config = DownloadConfig(token=token)
            with xopen(path, "rb", download_config=download_config) as f:
                return f.read()

        bytes_array = pa.array(
            [
                (path_to_bytes(x["path"]) if x["bytes"] is None else x["bytes"])
                if x is not None else None
                for x in storage.to_pylist()
            ],
            type=pa.binary(),
        )
        path_array = pa.array(
            [os.path.basename(path) if path is not None else None
             for path in storage.field("path").to_pylist()],
            type=pa.string(),
        )
        storage = pa.StructArray.from_arrays(
            [bytes_array, path_array], ["bytes", "path"], mask=bytes_array.is_null()
        )
        return array_cast(storage, self.pa_type)
'''

    with open(audio_path, "w") as f:
        f.write(corrected)
    print(f"✅ Auto-repaired {audio_path} (soundfile backend)")

_repair_datasets_audio()
# ── End auto-repair ──────────────────────────────────────────────────

from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import soundfile as sf
import torch
import torchaudio
import webdataset as wds
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from qwen_tts import Qwen3TTSTokenizer

TOKENIZER_SR = 24000
OUTPUT_SR    = 48000

_resamplers = {}
def resample(audio_np, orig_sr, target_sr, device):
    if orig_sr == target_sr:
        return audio_np
    key = (orig_sr, target_sr)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
    t = torch.from_numpy(audio_np).float().to(device)
    if t.dim() == 1: t = t.unsqueeze(0)
    return _resamplers[key](t).squeeze(0).cpu().numpy()

class Assembler:
    def __init__(self, dur_s):
        self.dur  = int(dur_s * OUTPUT_SR)
        self.tail = np.array([], dtype=np.float32)
    def add(self, a):
        buf = np.concatenate([self.tail, a]) if len(self.tail) else a
        chunks = []
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

def enc_flac(a):
    buf = io.BytesIO()
    sf.write(buf, (np.clip(a,-1,1)*32767).astype(np.int16), OUTPUT_SR, format="flac", subtype="PCM_16")
    return buf.getvalue()

def enc_npy(c):
    buf = io.BytesIO(); np.save(buf, c); return buf.getvalue()

def load_tokenizer(device):
    logger.info("Loading tokenizer...")
    return Qwen3TTSTokenizer.from_pretrained(
        "Qwen/Qwen3-TTS-Tokenizer-12Hz", dtype=torch.bfloat16, device_map=device)

@torch.inference_mode()
def tokenize_batch(audios, tokenizer):
    enc = tokenizer.encode(audios=audios, sr=TOKENIZER_SR)
    return [c.cpu().numpy().flatten().astype(np.int32) for c in enc.audio_codes]

def worker(dataset, assembler, device, stage_size, rng, queue, pbar, skipped):
    staging = []
    def flush(force=False):
        if not staging: return
        rng.shuffle(staging)
        cut = len(staging) if force else len(staging) // 2
        b, staging[:] = staging[:cut], staging[cut:]
        if b: queue.put(b)

    for voice in dataset:
        try:
            arr = voice["audio"]["array"]
            sr  = voice["audio"]["sampling_rate"]
            if arr.ndim > 1: arr = arr.mean(-1)
            arr = arr.astype(np.float32)
            arr48 = resample(arr, sr, OUTPUT_SR, device)
        except Exception as e:
            logger.warning(f"Skip: {e}"); skipped[0] += 1; continue

        for c48 in assembler.add(arr48):
            c24 = resample(c48, OUTPUT_SR, TOKENIZER_SR, device)
            staging.append((c24, c48))
            pbar.update(1)

        if len(staging) >= stage_size:
            flush()

    for c48 in assembler.flush():
        c24 = resample(c48, OUTPUT_SR, TOKENIZER_SR, device)
        staging.append((c24, c48))
        pbar.update(1)
    flush(force=True)
    queue.put(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("output_dir")
    ap.add_argument("--split",           default="train")
    ap.add_argument("--dataset-config",  default=None)
    ap.add_argument("--shard-size",      type=int,   default=1000)
    ap.add_argument("--val-percent",     type=float, default=5.0)
    ap.add_argument("--duration",        type=float, default=3.0)
    ap.add_argument("--stage-size",      type=int,   default=500)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--tokenizer-batch", type=int,   default=64)
    ap.add_argument("--queue-size",      type=int,   default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB VRAM")

    tokenizer = load_tokenizer(device)

    out = Path(args.output_dir)
    (out/"train").mkdir(parents=True, exist_ok=True)
    (out/"val").mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset (non-streaming)...")
    kw = {"split": args.split}
    if args.dataset_config: kw["name"] = args.dataset_config
    ds = load_dataset(args.dataset, **kw).shuffle(seed=args.seed)
    logger.info(f"Loaded {len(ds):,} samples")

    tw      = ShardWriter(out/"train", args.shard_size)
    vw      = ShardWriter(out/"val",   args.shard_size)
    rng_stg = np.random.default_rng(args.seed)
    rng_spl = np.random.default_rng(args.seed + 1)
    val_frac= args.val_percent / 100.0
    asm     = Assembler(args.duration)
    gkey    = [0]
    skipped = [0]
    q       = Queue(maxsize=args.queue_size)

    try:
        with tqdm(desc="Processing", unit="chunks") as pbar:
            t = Thread(target=worker,
                       args=(ds, asm, device, args.stage_size, rng_stg, q, pbar, skipped),
                       daemon=True)
            t.start()
            while True:
                batch = q.get()
                if batch is None: break
                for i in range(0, len(batch), args.tokenizer_batch):
                    sub   = batch[i:i+args.tokenizer_batch]
                    codes = tokenize_batch([c24 for c24,_ in sub], tokenizer)
                    for (_, c48), code in zip(sub, codes):
                        key = f"{gkey[0]:08d}"; gkey[0] += 1
                        (vw if rng_spl.random() < val_frac else tw).write(key, enc_flac(c48), enc_npy(code))
            t.join()
    finally:
        tw.close(); vw.close()

    logger.info(f"Done! Train: {tw.count:,} in {tw.num_shards} shards | Val: {vw.count:,} in {vw.num_shards} shards | Skipped: {skipped[0]:,}")

if __name__ == "__main__":
    main()