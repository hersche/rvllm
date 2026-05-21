#!/usr/bin/env python3
"""Element-wise compare of HF-base K/V dumps vs rvllm shadow K/V dumps.

Closes the loop on task #34 (31B spec accept_rate=0 root cause).
The chain:

  1. rvllm side: run gemma-4-31b-nvfp4 with RVLLM_SPEC_DUMP_DIR
     set + the canonical prompt → produces shadow_{k,v}_{sliding,
     global}_src.npy in the dump dir (post-RoPE, post-K-norm,
     pre-NVFP4-quant tensors from the rvllm runtime's drafter
     shadow-population step).

  2. HF side: stop rvllm, run dump_hf_base_kv_31b.py against the
     same dump dir → produces shadow_{k,v}_{sliding,global}_src_hf.npy
     (HF Gemma 4 base's past_key_values for layers 58/59 on the
     same prompt tokens).

  3. THIS tool: element-wise diff each pair. Identifies the per-slot,
     per-head, per-channel element with the largest absolute or
     relative error. The first big-diff position is the root cause
     of the drafter's wrong predictions.

Usage:
    python3 v3/tools/diff_hf_vs_rvllm_kv.py \\
        --rvllm-dump /tmp/rvllm_drafter_dump \\
        --hf-dump    /tmp/hf_base_kv_31b

Expected shape: [T, nkvh, head_dim] f32 for both sides. The
permute in dump_hf_base_kv_31b.py:120 (`K[0].permute(1, 0, 2)`)
ensures the HF dump is in slot-major layout matching rvllm.

Output:
    * Per-tensor RMS of rvllm vs HF
    * Per-tensor RMS of (rvllm - hf)
    * Per-tensor cosine (rvllm vs hf), flattened
    * Per-slot max |rvllm[t] - hf[t]| (top-10 worst slots)
    * Per-(slot, head) max |diff| (top-10)
    * First-divergence detector: smallest t where any |diff| > tol
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_npy(p: Path) -> np.ndarray:
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def cosine_flat(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def summarise(label: str, rvllm: np.ndarray, hf: np.ndarray, tol: float):
    print(f"\n=== {label} ===")
    if rvllm.shape != hf.shape:
        print(f"  shape mismatch: rvllm={rvllm.shape} hf={hf.shape}")
        # Force align by min-shape to still report something.
        T = min(rvllm.shape[0], hf.shape[0])
        rvllm = rvllm[:T]
        hf = hf[:T]
    diff = rvllm.astype(np.float64) - hf.astype(np.float64)
    abs_diff = np.abs(diff)
    print(f"  shape={rvllm.shape}")
    print(f"  rms(rvllm)={rms(rvllm):.6f} rms(hf)={rms(hf):.6f} "
          f"rms(diff)={rms(diff):.6f}")
    print(f"  cosine(flat)={cosine_flat(rvllm, hf):.8f}")
    print(f"  max|diff|={abs_diff.max():.6f}  "
          f"mean|diff|={abs_diff.mean():.6f}")
    # Per-slot |diff| (collapse heads + head_dim).
    if abs_diff.ndim >= 1:
        per_slot = abs_diff.reshape(abs_diff.shape[0], -1).max(axis=1)
        order = np.argsort(per_slot)[::-1]
        top = order[:10]
        print(f"  top-10 worst slots (t, max|diff|):")
        for t in top:
            print(f"    t={int(t):4d}  max|diff|={per_slot[t]:.6f}")
    # First-divergence: smallest t with any |diff| > tol.
    first = -1
    for t in range(rvllm.shape[0]):
        if abs_diff[t].max() > tol:
            first = t
            break
    if first < 0:
        print(f"  ✓ no element exceeds tol={tol}; tensors agree within tolerance")
    else:
        print(f"  ✗ first |diff| > {tol} at slot t={first} "
              f"(max|diff|={abs_diff[first].max():.6f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvllm-dump", required=True, type=Path,
                    help="rvllm dump dir (RVLLM_SPEC_DUMP_DIR contents)")
    ap.add_argument("--hf-dump", required=True, type=Path,
                    help="hf base K/V dump dir (produced by "
                         "dump_hf_base_kv_31b.py)")
    ap.add_argument("--tol", type=float, default=1e-2,
                    help="abs |diff| tolerance for first-divergence")
    args = ap.parse_args()

    # rvllm dumps as flat 1-D arrays (the in-process NPY saver only
    # emits 1-D headers). Reshape to logical [T, nkvh, head_dim] using
    # meta.json so downstream slot/head/channel breakdown works.
    meta_path = args.rvllm_dump / "meta.json"
    if not meta_path.exists():
        print(f"meta.json missing at {meta_path} — cannot reshape rvllm dumps")
        return 2
    meta = json.loads(meta_path.read_text())
    T = int(meta["committed_len"])
    pairs = [
        ("shadow_k_sliding_src", "K sliding (layer 58)",
         (T, int(meta["sliding_num_kv_heads"]), int(meta["sliding_head_dim"]))),
        ("shadow_v_sliding_src", "V sliding (layer 58)",
         (T, int(meta["sliding_num_kv_heads"]), int(meta["sliding_head_dim"]))),
        ("shadow_k_global_src",  "K global  (layer 59)",
         (T, int(meta["full_num_kv_heads"]), int(meta["full_head_dim"]))),
        ("shadow_v_global_src",  "V global  (layer 59)",
         (T, int(meta["full_num_kv_heads"]), int(meta["full_head_dim"]))),
    ]
    missing = []
    for stem, _label, _shape in pairs:
        rvllm_p = args.rvllm_dump / f"{stem}.npy"
        hf_p = args.hf_dump / f"{stem}_hf.npy"
        if not rvllm_p.exists():
            missing.append(("rvllm", str(rvllm_p)))
        if not hf_p.exists():
            missing.append(("hf", str(hf_p)))
    if missing:
        print("missing dump files:")
        for src, p in missing:
            print(f"  ({src}) {p}")
        print("rerun the missing dump step before this tool.")
        return 2

    print(f"rvllm dump: {args.rvllm_dump}")
    print(f"hf dump:    {args.hf_dump}")
    print(f"tol:        {args.tol}")

    for stem, label, shape in pairs:
        rvllm_arr = load_npy(args.rvllm_dump / f"{stem}.npy")
        if rvllm_arr.ndim == 1 and rvllm_arr.size == int(np.prod(shape)):
            rvllm_arr = rvllm_arr.reshape(shape)
        hf_arr = load_npy(args.hf_dump / f"{stem}_hf.npy")
        summarise(label, rvllm_arr, hf_arr, args.tol)

    # Hidden-state cross-check if both sides dumped it.
    bh_r = args.rvllm_dump / "base_hidden_last.npy"
    bh_h = args.hf_dump / "base_hidden_last_hf.npy"
    if bh_r.exists() and bh_h.exists():
        rvllm_arr = load_npy(bh_r)
        hf_arr = load_npy(bh_h)
        summarise("base_hidden_last", rvllm_arr, hf_arr, args.tol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
