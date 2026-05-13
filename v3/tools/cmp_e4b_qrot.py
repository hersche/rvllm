#!/usr/bin/env python3
"""Dequantize rvllm's post-RoPE FP8 Q and diff against an HF reference
computed by replaying HF's apply_rotary_pos_emb on q_normed.

Inputs:
  rvllm_dir/
    e4b_layer{L}_q_fp8_post_rope.bin   u8, [T, num_heads, head_dim]
    e4b_layer{L}_q_scale_post_rope.bin f32, [T, num_heads]
  hf_dir/
    e4b_layer{L}_q_normed.bin          f16, [T, num_heads, head_dim]
    e4b_rope_cos_sliding.bin           f16, [max_pos, head_dim/2]
    e4b_rope_sin_sliding.bin           f16, [max_pos, head_dim/2]

Compares row-cosine between rvllm_q_rot (dequantized) and HF_q_rot
(replay of apply_rotary_pos_emb on q_normed).

This separates the surgical question:
  - bug in RoPE/quantization (mismatch here) vs
  - bug in softmax/PV/O-proj downstream (q_rot matches HF, but attn_out drifts)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


def fp8_e4m3_to_f32(b: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(b.view(np.uint8)).view(torch.float8_e4m3fn)
    return t.to(torch.float32).cpu().numpy()


def rotate_half(x: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvllm-dir", default="/tmp/e4b_rvllm_ple")
    ap.add_argument("--hf-dir", default="/tmp/e4b_hf_ple")
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=256)
    args = ap.parse_args()

    rv = Path(args.rvllm_dir)
    hf = Path(args.hf_dir)
    L = args.layer

    q_fp8 = np.frombuffer((rv / f"e4b_layer{L}_q_fp8_post_rope.bin").read_bytes(), dtype=np.uint8)
    q_scale = np.frombuffer((rv / f"e4b_layer{L}_q_scale_post_rope.bin").read_bytes(), dtype=np.float32)
    q_normed_hf = np.frombuffer((hf / f"e4b_layer{L}_q_normed.bin").read_bytes(), dtype=np.float16).astype(np.float32)
    cos = np.frombuffer((hf / "e4b_rope_cos_sliding.bin").read_bytes(), dtype=np.float16).astype(np.float32)
    sin = np.frombuffer((hf / "e4b_rope_sin_sliding.bin").read_bytes(), dtype=np.float16).astype(np.float32)

    H = args.num_heads
    D = args.head_dim
    half = D // 2

    # Infer T from rvllm fp8 size
    assert q_fp8.size % (H * D) == 0, f"{q_fp8.size} not divisible by H*D={H*D}"
    T = q_fp8.size // (H * D)
    print(f"T={T}, num_heads={H}, head_dim={D}, half={half}")
    print(f"q_scale.size={q_scale.size} (expected {T*H})")
    assert q_scale.size == T * H

    cos = cos.reshape(-1, half)[:T]
    sin = sin.reshape(-1, half)[:T]
    # HF cat-doubles: emb = cat((freqs, freqs), -1). Apply to head_dim.
    cos_full = np.concatenate([cos, cos], axis=-1)  # [T, D]
    sin_full = np.concatenate([sin, sin], axis=-1)

    # Reshape HF q_normed: [T, H, D] (assume same layout as rvllm: head-major within token)
    q_normed_hf = q_normed_hf.reshape(T, H, D)

    # Apply HF rotation: q_rot = q * cos + rotate_half(q) * sin
    cos_bcast = cos_full[:, None, :]  # [T, 1, D]
    sin_bcast = sin_full[:, None, :]
    hf_q_rot = q_normed_hf * cos_bcast + rotate_half(q_normed_hf) * sin_bcast  # [T, H, D]

    # Dequantize rvllm Q: q_fp8 [T,H,D] u8 → f32, then * q_scale[T,H]
    rv_q_fp8_f32 = fp8_e4m3_to_f32(q_fp8).reshape(T, H, D)
    rv_q_scale = q_scale.reshape(T, H)
    rv_q_rot = rv_q_fp8_f32 * rv_q_scale[:, :, None]

    # Diff per (token, head) row
    a = rv_q_rot.reshape(T * H, D)
    b = hf_q_rot.reshape(T * H, D)
    dots = (a * b).sum(-1)
    na = np.sqrt((a * a).sum(-1)) + 1e-30
    nb = np.sqrt((b * b).sum(-1)) + 1e-30
    cos_row = dots / (na * nb)
    max_abs = np.abs(a - b).max(-1)
    print(f"\nrow-cosine (T*H={T*H} rows):")
    print(f"  mean={cos_row.mean():.6f}  min={cos_row.min():.6f}  max_abs={max_abs.max():.3e}")
    worst = int(np.argmin(cos_row))
    t_w, h_w = worst // H, worst % H
    print(f"  worst row: token={t_w} head={h_w}  cos={cos_row[worst]:.6f}  max_abs={max_abs[worst]:.3e}")
    print(f"  rv_q_scale[t={t_w}, h={h_w}]={rv_q_scale[t_w, h_w]:.4f}")
    print(f"  hf_q_rot magnitude[t={t_w}, h={h_w}]: max={np.abs(b.reshape(T,H,D)[t_w, h_w]).max():.4f}")
    print(f"  rv_q_rot magnitude[t={t_w}, h={h_w}]: max={np.abs(a.reshape(T,H,D)[t_w, h_w]).max():.4f}")

    # Per-head summary
    print(f"\nper-head mean cosine:")
    for h in range(H):
        idx = [t * H + h for t in range(T)]
        c = cos_row[idx].mean()
        print(f"  head {h}: cos_mean={c:.6f}")


if __name__ == "__main__":
    main()
