#!/usr/bin/env python3
"""Standalone dump of Gemma 4 E4B RoPE cos/sin tables.

Writes:
  e4b_rope_cos_sliding.bin / e4b_rope_sin_sliding.bin   [max_pos, head_dim/2]
  e4b_rope_cos_global.bin  / e4b_rope_sin_global.bin    [max_pos, head_dim/2]

The first half of HF's [..., head_dim] cos/sin (with cat-doubling)
equals rvllm's stored frequencies, so we slice.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def write_f16(path: Path, t: torch.Tensor):
    t = t.detach().to("cpu").to(torch.float16).contiguous()
    path.write_bytes(t.numpy().tobytes())
    print(f"  wrote {path.name}: shape={tuple(t.shape)} bytes={path.stat().st_size}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/r00t/gemma4-e4b")
    ap.add_argument("--out-dir", default="/tmp/e4b_hf_ple")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    text_model = (
        getattr(getattr(model, "model", model), "language_model", None)
        or getattr(model, "model", model)
    )
    cfg = text_model.config
    max_pos = int(cfg.max_position_embeddings)
    print(f"max_position_embeddings={max_pos}")

    pos_ids = torch.arange(0, max_pos, device=args.device).unsqueeze(0)
    dummy = torch.zeros(1, 1, device=args.device, dtype=torch.bfloat16)

    rotary = getattr(text_model, "rotary_emb", None)
    if rotary is None:
        print("WARN: no rotary_emb on text_model")
        return

    cos_s, sin_s = rotary(dummy, pos_ids, layer_type="sliding_attention")
    half = cos_s.shape[-1] // 2
    write_f16(out / "e4b_rope_cos_sliding.bin", cos_s[0, :, :half])
    write_f16(out / "e4b_rope_sin_sliding.bin", sin_s[0, :, :half])

    cos_g, sin_g = rotary(dummy, pos_ids, layer_type="full_attention")
    # rvllm only stores the real (rotated) frequencies; HF's
    # proportional rope writes head_dim/2 values where the first
    # rope_angles = int(partial_rotary_factor * head_dim / 2)
    # are real and the rest are zeros. Slice to match.
    cfg = text_model.config
    rope_params = cfg.rope_parameters["full_attention"]
    head_dim_global = getattr(cfg, "global_head_dim", None) or cfg.head_dim
    partial = float(rope_params.get("partial_rotary_factor", 1.0))
    rope_angles = int(partial * head_dim_global // 2)
    print(f"  full_attention: head_dim={head_dim_global} partial={partial} → rope_angles={rope_angles}")
    write_f16(out / "e4b_rope_cos_global.bin", cos_g[0, :, :rope_angles])
    write_f16(out / "e4b_rope_sin_global.bin", sin_g[0, :, :rope_angles])


if __name__ == "__main__":
    main()
