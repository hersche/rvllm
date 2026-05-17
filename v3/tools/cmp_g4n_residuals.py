#!/usr/bin/env python3
"""
Compare two G4N_DUMP_DIR directories produced by
`forward_full_to_token` under `G4N_DUMP_DIR=...`. Computes
per-step:
    * row-cosine
    * max abs diff
    * top-K largest-divergence indices
    * NaN/Inf counts

File layout (per the Rust side, gemma4_nvfp4_bring_up.rs::
forward_full_to_token):

    step_00_embed.bf16.bin           # hidden_size u16
    step_NN_attn_out.bf16.bin        # num_q_heads * head_dim_for(NN) u16
    step_NN_post_attn.bf16.bin       # hidden_size u16
    step_NN_post_mlp.bf16.bin        # hidden_size u16
    step_final_norm.bf16.bin         # hidden_size u16
    step_final_logits.f32.bin        # vocab_size f32
    step_final_token.i32.bin         # 1 i32

The reference dir is expected to use the same naming. A skeleton
HF-side dumper that produces a matching layout for the same
checkpoint lives at `dump_g4n_hf_residuals.py` (TODO).

Usage:
    ./cmp_g4n_residuals.py RVLLM_DIR REFERENCE_DIR \
        [--top 10] [--show-only NAME_PATTERN] [--cosine-thresh 0.99]
"""

import argparse
import os
import struct
import sys
from pathlib import Path

import numpy as np


def bf16_bytes_to_f32(b: bytes) -> np.ndarray:
    """Reinterpret raw bf16 bytes as a float32 ndarray (zero-extended
    mantissa). bf16 = top 16 bits of f32, so x_f32 = (x_bf16 as u32) << 16
    when read as bits. Vectorized."""
    if len(b) % 2 != 0:
        raise ValueError(f"bf16 buffer length {len(b)} not multiple of 2")
    u16 = np.frombuffer(b, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_step(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    name = path.name
    if name.endswith(".bf16.bin"):
        return bf16_bytes_to_f32(raw)
    if name.endswith(".f32.bin"):
        return np.frombuffer(raw, dtype=np.float32)
    if name.endswith(".i32.bin"):
        return np.frombuffer(raw, dtype=np.int32)
    raise ValueError(f"unknown dtype suffix in {name}")


def step_sort_key(name: str) -> tuple:
    """Order: step_00_embed → 00,01,…,59 each (attn_out, post_attn,
    post_mlp) → final_norm → final_logits → final_token."""
    if not name.startswith("step_"):
        return (99, name)
    rest = name[len("step_"):].rsplit(".", 2)[0]  # strip suffix
    if rest == "00_embed":
        return (0, 0, 0)
    if rest == "final_norm":
        return (2, 0)
    if rest == "final_logits":
        return (2, 1)
    if rest == "final_token":
        return (2, 2)
    # step_NN_<stage>
    try:
        nn, stage = rest.split("_", 1)
        nn_i = int(nn)
    except (ValueError, IndexError):
        return (99, name)
    stage_order = {"attn_out": 0, "post_attn": 1, "post_mlp": 2}.get(stage, 3)
    return (1, nn_i, stage_order)


def compare_arrays(a: np.ndarray, b: np.ndarray, top_k: int):
    """Return per-step diff metrics."""
    if a.shape != b.shape:
        return {
            "shape_a": a.shape,
            "shape_b": b.shape,
            "shape_mismatch": True,
        }
    # NaN/Inf accounting (only on a — flag if asymmetric).
    nan_a = int(np.isnan(a).sum())
    nan_b = int(np.isnan(b).sum())
    inf_a = int(np.isinf(a).sum())
    inf_b = int(np.isinf(b).sum())
    if nan_a or nan_b or inf_a or inf_b:
        # Replace non-finite with 0 for cosine to avoid NaN
        # propagation; the counts above flag the discrepancy.
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        cos = float("nan")
    else:
        cos = float(np.dot(a.ravel(), b.ravel()) / (norm_a * norm_b))
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    rms = float(np.sqrt(np.mean(diff.astype(np.float64) ** 2)))
    # Top-K divergent indices.
    if top_k > 0 and abs_diff.size > 0:
        k = min(top_k, abs_diff.size)
        idx = np.argpartition(-abs_diff.ravel(), k - 1)[:k]
        idx = idx[np.argsort(-abs_diff.ravel()[idx])]
        top = [(int(i), float(a.ravel()[i]), float(b.ravel()[i]),
                float(abs_diff.ravel()[i])) for i in idx]
    else:
        top = []
    return {
        "shape": a.shape,
        "n": a.size,
        "cos": cos,
        "max_abs": max_abs,
        "rms": rms,
        "nan_a": nan_a, "nan_b": nan_b,
        "inf_a": inf_a, "inf_b": inf_b,
        "top": top,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rvllm_dir", type=Path)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("--top", type=int, default=5,
                    help="Top-K divergent indices per step (0 to disable)")
    ap.add_argument("--show-only", default=None,
                    help="Substring filter on step name")
    ap.add_argument("--cosine-thresh", type=float, default=0.99,
                    help="Highlight steps below this cosine")
    args = ap.parse_args()

    if not args.rvllm_dir.is_dir():
        sys.exit(f"rvllm_dir not a directory: {args.rvllm_dir}")
    if not args.ref_dir.is_dir():
        sys.exit(f"ref_dir not a directory: {args.ref_dir}")

    rvllm_files = sorted(args.rvllm_dir.glob("step_*"),
                         key=lambda p: step_sort_key(p.name))
    ref_files = {p.name: p for p in args.ref_dir.glob("step_*")}

    print(f"# rvllm dir : {args.rvllm_dir}")
    print(f"# ref   dir : {args.ref_dir}")
    print(f"# rvllm steps: {len(rvllm_files)}  /  ref steps: {len(ref_files)}")
    print()
    header = f"{'step':40s}  {'shape':22s}  {'cos':>10s}  {'max_abs':>10s}  {'rms':>10s}"
    print(header)
    print("-" * len(header))

    diverged = []
    for rp in rvllm_files:
        name = rp.name
        if args.show_only and args.show_only not in name:
            continue
        if name not in ref_files:
            print(f"{name:40s}  {'MISSING in ref':22s}")
            continue
        try:
            a = load_step(rp)
            b = load_step(ref_files[name])
        except Exception as e:
            print(f"{name:40s}  load error: {e}")
            continue
        if name.endswith(".i32.bin"):
            # Token id — print directly.
            if a.shape == b.shape and a.size == 1:
                marker = " ✓" if int(a[0]) == int(b[0]) else " ✗"
                print(f"{name:40s}  token rvllm={int(a[0])}  ref={int(b[0])}{marker}")
            else:
                print(f"{name:40s}  shape mismatch  {a.shape} vs {b.shape}")
            continue
        m = compare_arrays(a, b, top_k=args.top)
        if m.get("shape_mismatch"):
            print(f"{name:40s}  SHAPE MISMATCH  {m['shape_a']} vs {m['shape_b']}")
            continue
        cos_s = f"{m['cos']:.6f}"
        flag = " ◀" if m["cos"] < args.cosine_thresh else ""
        print(f"{name:40s}  {str(m['shape']):22s}  {cos_s:>10s}  "
              f"{m['max_abs']:>10.4g}  {m['rms']:>10.4g}{flag}")
        nan = m["nan_a"] + m["nan_b"]
        inf = m["inf_a"] + m["inf_b"]
        if nan or inf:
            print(f"  ⚠  non-finite: nan_a={m['nan_a']} nan_b={m['nan_b']} "
                  f"inf_a={m['inf_a']} inf_b={m['inf_b']}")
        if m["cos"] < args.cosine_thresh:
            diverged.append((name, m))
            if args.top > 0:
                for (i, va, vb, d) in m["top"]:
                    print(f"  top[{i:>8d}]: a={va:+.6g}  b={vb:+.6g}  |Δ|={d:.4g}")

    print()
    if diverged:
        print(f"# {len(diverged)} step(s) below cosine threshold {args.cosine_thresh}")
        print(f"# First divergence: {diverged[0][0]}")
    else:
        print(f"# All compared steps ≥ cosine {args.cosine_thresh} ✓")


if __name__ == "__main__":
    main()
