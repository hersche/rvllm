#!/usr/bin/env python3
"""
HF transformers reference dumper that produces a file layout
compatible with `cmp_g4n_residuals.py`.

Goal: load `nvidia/Gemma-4-31B-IT-NVFP4` via HF (dequantizing to
bf16 on the fly), run a single-token forward at position=0 with
BOS, and dump the same per-layer residuals that
`forward_full_to_token` writes under `G4N_DUMP_DIR=...`.

Status: SKELETON. The NVFP4 weight unpacking and the precise
forward hook points still need to be wired against the
checkpoint's exact tensor layout. The framework (file naming,
bf16/f32 dtype conventions, hook strategy) is in place so the
fill-in is mechanical.

Why this matters:
    The Option B native NVFP4 forward in
    rvllm-serve/v3/crates/rvllm-runtime/src/gemma4_nvfp4_bring_up.rs
    runs structurally on 31B (commits #5b1..#5c) but has no
    numerical reference. This script — once completed — supplies
    one, and then `cmp_g4n_residuals.py` does the diff.

Usage (when ready):
    python dump_g4n_hf_residuals.py \
        --model /home/r00t/Gemma-4-31B-IT-NVFP4 \
        --out   /tmp/g4n_hf_dump \
        --token 2 --position 0

TODO (in priority order):
  1. NVFP4 unpack: read `weight` (u8 packed 4-bit) +
     `weight_scale` (e4m3) + `weight_scale_2` (f32) per modelopt
     0.37 W4A16 format. The decode formula:
         w_dequant[i,j] = ((nibble(i,j) decoded to fp4_value)
                            * weight_scale[i, j//16] * weight_scale_2)
  2. Build a `transformers.Gemma4ForCausalLM` with the dequantized
     bf16 weights loaded into the right modules.
  3. Add forward hooks on each decoder block:
       block.input_layernorm  → residual right BEFORE attn
       block.self_attn        → attn_out (= o_proj input dim)
       block.post_attention_layernorm + residual → "post_attn"
       block.mlp + residual    → "post_mlp"
     (Match rvllm's stage labels: attn_out / post_attn / post_mlp.)
  4. Run forward with `input_ids=[token]`, `position_ids=[position]`,
     no past_kv. Pull the hooked tensors, cast to bf16, write as
     `step_NN_<stage>.bf16.bin` matching rvllm's exact naming.
  5. Dump final_norm output (bf16), final logits (f32, post tied
     LM head), and argmax token id (i32) into the same naming.
"""

import argparse
import struct
import sys
from pathlib import Path


def f32_to_bf16_bytes(arr):
    """Round-to-nearest-even narrow of float32 → bf16 as raw u16
    bytes. Returns bytes that match rvllm's `.bf16.bin` files."""
    import numpy as np
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    u32 = arr.view(np.uint32)
    bias = 0x7FFF + ((u32 >> 16) & 1)
    rounded = u32 + bias
    u16 = (rounded >> 16).astype(np.uint16)
    return u16.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True,
                    help="Path to nvidia/Gemma-4-31B-IT-NVFP4 checkpoint")
    ap.add_argument("--out", type=Path, required=True,
                    help="Dump directory (will be created)")
    ap.add_argument("--token", type=int, default=2, help="Input token id (default BOS=2)")
    ap.add_argument("--position", type=int, default=0,
                    help="position_id (default 0)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(
        "ERROR: this dumper is a SKELETON — finish the four TODO\n"
        "steps inside the script body. The framework (paths,\n"
        "bf16 narrow helper, output naming) is wired up; the\n"
        "missing pieces are NVFP4 unpack + HF model build +\n"
        "forward hooks. Run `cmp_g4n_residuals.py` once both\n"
        "sides produce dumps."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
