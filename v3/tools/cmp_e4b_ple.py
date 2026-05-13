#!/usr/bin/env python3
"""Row-cosine diff between rvllm PLE dumps and HF E4B reference dumps.

Same diff convention as v3/tools/cmp_g4v_substep.py — read paired
.bin files (f16 row-major, no header), reshape per the expected
[N, D], report per-row cosine, max-abs, and first-divergence row.

Files are matched by filename: any name matching the e4b_*.bin
pattern that exists in BOTH directories is diffed. A summary
table prints at the end.

Usage:
  v3/tools/cmp_e4b_ple.py /tmp/e4b_rvllm_ple /tmp/e4b_hf_ple

Expected file layout:
  e4b_inputs_embeds.bin                 [T, hidden=2560]
  e4b_ple_lookup.bin                    [T, num_layers*ple_dim]
  e4b_ple_context_pre_norm.bin          [T, num_layers*ple_dim]
  e4b_ple_context_post_norm.bin         [T, num_layers, ple_dim]
  e4b_per_layer_inputs.bin              [T, num_layers, ple_dim]
  e4b_layer{L}_residual_in.bin          [T, hidden]
  e4b_layer{L}_after_attn_add.bin       [T, hidden]
  e4b_layer{L}_after_mlp_add.bin        [T, hidden]
  e4b_layer{L}_after_ple_add.bin        [T, hidden]
  e4b_layer{L}_output.bin               [T, hidden]
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def load_f16(path: Path, total_elems_hint: int | None = None) -> np.ndarray:
    """Load a flat f16 buffer. Reshape decision is deferred to the
    caller — diff is row-major and dimension-agnostic past the
    last axis."""
    raw = path.read_bytes()
    if len(raw) % 2 != 0:
        raise ValueError(f"{path}: not f16-aligned ({len(raw)} bytes)")
    arr = np.frombuffer(raw, dtype=np.float16)
    if total_elems_hint is not None and arr.size != total_elems_hint:
        print(f"  WARN {path.name}: size {arr.size} != hint {total_elems_hint}")
    return arr


def row_cosine(
    a: np.ndarray,
    b: np.ndarray,
    last_dim: int,
    a_row_off: int = 0,
    b_row_off: int = 0,
):
    """Returns (cos_per_row, max_abs_per_row, n_rows).
    Reshape a,b to (-1, last_dim), then take the OVERLAP between
    `a[a_row_off:]` and `b[b_row_off:]` so a BOS-prepended rvllm
    dump (T=2) can be diffed against a no-BOS HF dump (T=1) by
    setting a_row_off=1 (rvllm side).
    """
    a2 = a.astype(np.float32).reshape(-1, last_dim)[a_row_off:]
    b2 = b.astype(np.float32).reshape(-1, last_dim)[b_row_off:]
    n = min(a2.shape[0], b2.shape[0])
    if n == 0:
        raise ValueError(f"no overlap rows: a={a2.shape} b={b2.shape}")
    a2 = a2[:n]
    b2 = b2[:n]
    dots = (a2 * b2).sum(axis=-1)
    na = np.sqrt((a2 * a2).sum(axis=-1)) + 1e-30
    nb = np.sqrt((b2 * b2).sum(axis=-1)) + 1e-30
    cos = dots / (na * nb)
    diff = np.abs(a2 - b2)
    max_abs = diff.max(axis=-1)
    return cos, max_abs, n


def infer_last_dim(name: str) -> int:
    """Guess the trailing dimension from the filename so we can
    reshape for row-cosine. ple_dim=256 for the per-layer slice
    files; otherwise the per-row dimension is hidden=2560 or the
    full per-layer-stride 10752."""
    # Per-layer-input files: reshape over ple_dim
    if "_per_layer_inputs" in name or "_context_post_norm" in name:
        return 256
    # Lookup / pre-norm files: full per-layer stride
    if "_ple_lookup" in name or "_ple_context_pre" in name or "_ple_context_pre_norm" in name:
        return 256  # diff per-ple-slice for finer locality
    # QKV concat: row = [Q | K | V], q_dim + 2*kv_dim = 8*256 + 2*2*256 = 3072
    # on E4B sliding (q_dim=2048, kv_dim=512). Use 3072 for sliding,
    # 6144 for global (q_dim=4096, kv_dim=1024). Default the sliding
    # value; user can override --last-dim for global-layer dumps.
    if "_qkv" in name:
        return 3072
    # RoPE tables: [max_pos, rotary_dim/2] — sliding=128, global=64 (partial 0.25)
    if "rope_" in name and "_sliding" in name:
        return 128
    if "rope_" in name and "_global" in name:
        return 64
    # All hidden-size sub-step files
    return 2560


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rvllm_dir", type=Path)
    ap.add_argument("hf_dir", type=Path)
    ap.add_argument("--last-dim", type=int, default=None,
                    help="override inferred last dim for reshape")
    ap.add_argument("--first-rows", type=int, default=8,
                    help="how many rows to dump in the per-file detail block")
    ap.add_argument("--rvllm-row-off", type=int, default=0,
                    help="skip leading rows on the rvllm side")
    ap.add_argument("--hf-row-off", type=int, default=0,
                    help="skip leading rows on the HF side. Note: "
                         "both rvllm and HF dump T=2 when `--prepend-bos` "
                         "(HF default) so the direct row-0-vs-row-0 diff "
                         "works out of the box.")
    args = ap.parse_args()

    rvllm = args.rvllm_dir
    hf = args.hf_dir
    rv_files = {p.name for p in rvllm.glob("e4b_*.bin")}
    hf_files = {p.name for p in hf.glob("e4b_*.bin")}
    common = sorted(rv_files & hf_files)
    rv_only = rv_files - hf_files
    hf_only = hf_files - rv_files

    if rv_only:
        print(f"## rvllm-only files ({len(rv_only)}):")
        for f in sorted(rv_only):
            print(f"  {f}")
    if hf_only:
        print(f"## HF-only files ({len(hf_only)}):")
        for f in sorted(hf_only):
            print(f"  {f}")

    print(f"## diffing {len(common)} common files\n")
    summary = []
    for name in common:
        ld = args.last_dim or infer_last_dim(name)
        a = load_f16(rvllm / name)
        b = load_f16(hf / name)
        try:
            cos, ma, nrows = row_cosine(
                a, b, ld,
                a_row_off=args.rvllm_row_off,
                b_row_off=args.hf_row_off,
            )
        except ValueError as e:
            print(f"{name:60s}  SIZE-MISMATCH  rvllm={a.size} hf={b.size}  ({e})")
            summary.append((name, None, None, None, None))
            continue
        mean_cos = float(cos.mean())
        min_cos = float(cos.min())
        max_ma = float(ma.max())
        bad_idx = int(np.argmin(cos))
        print(f"{name:60s}  rows={nrows:4d} ld={ld:5d}  "
              f"cos: mean={mean_cos:.6f} min={min_cos:.6f}  "
              f"max_abs={max_ma:.3e}  first-bad-row={bad_idx}")
        summary.append((name, nrows, mean_cos, min_cos, max_ma))

    # Compact summary
    print("\n## summary (sorted worst → best by min-cosine)")
    summary.sort(key=lambda x: x[3] if x[3] is not None else 99.0)
    for name, nrows, mean_cos, min_cos, max_ma in summary:
        if mean_cos is None:
            print(f"  SKIP    {name}")
            continue
        marker = "✓" if min_cos > 0.999 else ("≈" if min_cos > 0.95 else "✗")
        print(f"  {marker} {name:60s}  mean={mean_cos:.6f}  min={min_cos:.6f}  max_abs={max_ma:.3e}")


if __name__ == "__main__":
    main()
