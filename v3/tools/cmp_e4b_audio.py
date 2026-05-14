#!/usr/bin/env python3
"""Per-stage parity diff harness for the Gemma 4 E4B audio path.

Diffs rvllm's audio helper dumps against the HF reference dumps
written by `gemma4_e4b_audio_hf_dump.py`. Same row-cosine metric
the Gemma vision / Qwen prefill / PLE audits use.

Usage:
  python3 cmp_e4b_audio.py --hf-dir /tmp/e4b_audio_dump \\
                          --rvllm-dir /tmp/rvllm_e4b_audio_dump \\
                          --stages pos_embed,subsample

The rvllm helpers expose RVLLM_E4B_AUDIO_*_DUMP env knobs that, when
set to a directory, drop the matching `audio_*.bin` files at each
sub-stage; this script compares them shape-checked.
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np


SHAPES = {
    "pos_embed":              (13, 1024),
    "subsample":              ("N", 1024),
    "layer0_after_ffn1":      ("N", 1024),
    "layer0_after_attn":      ("N", 1024),
    "layer0_after_lconv":     ("N", 1024),
    "layer0_output":          ("N", 1024),
    "layer11_output":         ("N", 1024),
    "after_output_proj":      ("N", 1536),
    "after_embed_audio":      ("N", 2560),
}

HF_FILE = {
    "pos_embed":              "audio_pos_embed.bin",
    "subsample":              "audio_after_subsample.bin",
    "layer0_after_ffn1":      "audio_layer0_after_ffn1.bin",
    "layer0_after_attn":      "audio_layer0_after_attn.bin",
    "layer0_after_lconv":     "audio_layer0_after_lconv.bin",
    "layer0_output":          "audio_layer0_output.bin",
    "layer11_output":         "audio_layer11_output.bin",
    "after_output_proj":      "audio_after_output_proj.bin",
    "after_embed_audio":      "audio_after_embed_audio.bin",
}


def row_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    num = (a * b).sum(axis=-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-30
    return num / den


def compute_pos_embed_reference(hidden: int = 1024, pos_len: int = 13) -> np.ndarray:
    """Mirror the rvllm Rust formula in
    Gemma4Bringup::audio_pos_embed_upload — used as a sanity floor
    when the rvllm dump is absent."""
    num_t = hidden // 2
    log_inc = math.log(10000.0) / max(num_t - 1, 1)
    inv_t = np.array([math.exp(-k * log_inc) for k in range(num_t)], dtype=np.float32)
    table = np.zeros((pos_len, hidden), dtype=np.float32)
    for i in range(pos_len):
        pid = float(pos_len - 1 - i)
        st = pid * inv_t
        table[i, :num_t] = np.sin(st)
        table[i, num_t:] = np.cos(st)
    return table.astype(np.float16)


def cmp_stage(stage: str, hf_dir: Path, rvllm_dir: Path | None) -> bool:
    hf_path = hf_dir / HF_FILE[stage]
    if not hf_path.exists():
        print(f"  [SKIP] {stage}: HF dump missing ({hf_path})")
        return True
    rows, cols = SHAPES[stage]
    hf = np.fromfile(str(hf_path), dtype=np.float16)
    if isinstance(rows, int):
        hf = hf.reshape(rows, cols)
    else:
        hf = hf.reshape(-1, cols)

    if rvllm_dir is not None:
        rvllm_path = rvllm_dir / HF_FILE[stage]
        if rvllm_path.exists():
            ours = np.fromfile(str(rvllm_path), dtype=np.float16).reshape(hf.shape)
        else:
            print(f"  [SKIP] {stage}: rvllm dump missing ({rvllm_path})")
            return True
    elif stage == "pos_embed":
        ours = compute_pos_embed_reference()
    else:
        print(f"  [SKIP] {stage}: no rvllm dump and no host fallback")
        return True

    if ours.shape != hf.shape:
        print(f"  [FAIL] {stage}: shape {ours.shape} vs HF {hf.shape}")
        return False
    diff = np.abs(ours.astype(np.float32) - hf.astype(np.float32))
    cos = row_cos(ours, hf)
    ok = bool((cos.min() > 0.999) and (diff.max() < 0.1))
    tag = "OK  " if ok else "FAIL"
    print(
        f"  [{tag}] {stage}: shape={hf.shape}  rowcos_min={cos.min():.6f}  "
        f"rowcos_mean={cos.mean():.6f}  max_abs={diff.max():.4f}  "
        f"mean_abs={diff.mean():.4f}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="/tmp/e4b_audio_dump")
    ap.add_argument("--rvllm-dir", default=None)
    ap.add_argument(
        "--stages",
        default="pos_embed,subsample,layer0_after_ffn1,"
        "layer0_after_attn,layer0_after_lconv,layer0_output,"
        "layer11_output,after_output_proj,after_embed_audio",
    )
    args = ap.parse_args()
    stages = args.stages.split(",")
    hf_dir = Path(args.hf_dir)
    rvllm_dir = Path(args.rvllm_dir) if args.rvllm_dir else None

    print(f"HF reference dir: {hf_dir}")
    print(f"rvllm dump dir:   {rvllm_dir if rvllm_dir else '(host fallback for pos_embed)'}")
    all_ok = True
    for s in stages:
        if s not in HF_FILE:
            print(f"  [SKIP] {s}: unknown stage")
            continue
        all_ok &= cmp_stage(s, hf_dir, rvllm_dir)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
