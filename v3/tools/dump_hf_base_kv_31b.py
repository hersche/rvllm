#!/usr/bin/env python3
"""HF base K/V dump at source layers 58/59 for 31B-it.

Codex Round 7 Q10 decisive test. Loads Gemma 4 31B base in
bf16, runs ONE forward over the SAME prompt rvllm dumped from,
captures past_key_values.key_cache[58/59] + value_cache[58/59],
saves in [T, nkvh, hd] layout matching rvllm's shadow dump
files (shadow_k_sliding_src.npy / shadow_v_sliding_src.npy /
shadow_k_global_src.npy / shadow_v_global_src.npy).

After: diff_hf_vs_rvllm_kv.py element-compares.

Usage:
    /home/r00t/.vllm-exp/bin/python3 \\
        v3/tools/dump_hf_base_kv_31b.py \\
        --base       /home/r00t/.vllm/models/gemma-4-31b-it-fp8-block \\
        --rvllm-dump /tmp/rvllm_drafter_dump \\
        --out        /tmp/hf_base_kv_31b

Requires rvllm-serve STOPPED — needs the full 120 GB unified
memory to load 31B base.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",       required=True, type=Path)
    ap.add_argument("--rvllm-dump", required=True, type=Path)
    ap.add_argument("--out",        required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[hf-base-kv] base       = {args.base}")
    print(f"[hf-base-kv] rvllm-dump = {args.rvllm_dump}")
    print(f"[hf-base-kv] out        = {args.out}")

    meta = json.load(open(args.rvllm_dump / "meta.json"))
    print(f"[hf-base-kv] meta = {meta}")
    prompt_tokens = np.load(args.rvllm_dump / "prompt_tokens.npy").tolist()
    print(f"[hf-base-kv] prompt_tokens ({len(prompt_tokens)}) = {prompt_tokens}")
    sliding_src = int(meta["sliding_src_layer"])
    full_src = int(meta["full_src_layer"])
    print(f"[hf-base-kv] sliding_src={sliding_src} full_src={full_src}")

    from transformers import AutoModelForCausalLM

    print(f"[hf-base-kv] loading base (bf16, eager) ...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    base = base.to("cuda")
    base.eval()
    print(f"[hf-base-kv] base loaded in {time.time()-t0:.1f}s")
    print(f"[hf-base-kv] free vram: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB / "
          f"{torch.cuda.mem_get_info()[1]/1e9:.2f} GB")

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device="cuda")
    print(f"[hf-base-kv] running base forward (prompt_len={len(prompt_tokens)}) ...")
    t0 = time.time()
    with torch.no_grad():
        out = base(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    print(f"[hf-base-kv] forward done in {time.time()-t0:.1f}s")

    # Final hidden post final-norm at last prompt position
    final_hidden = out.hidden_states[-1][0, -1]  # [hidden]
    print(f"[hf-base-kv] base_hidden_last shape={tuple(final_hidden.shape)} "
          f"rms={(final_hidden.float()**2).mean().sqrt().item():.4f}")
    np.save(args.out / "base_hidden_last_hf.npy",
            final_hidden.detach().float().cpu().numpy())

    # past_key_values
    pkv = out.past_key_values
    if pkv is None:
        print("[hf-base-kv] FATAL: past_key_values is None")
        sys.exit(1)

    # Cache stores per-layer (K, V). transformers 5.8 DynamicCache uses
    # pkv.layers[li].keys / pkv.layers[li].values (each [bsz, num_kv_heads,
    # seq_len, head_dim]). Older versions have .key_cache / .value_cache.
    if hasattr(pkv, "layers"):
        get_kv = lambda li: (pkv.layers[li].keys, pkv.layers[li].values)
    elif hasattr(pkv, "key_cache"):
        get_kv = lambda li: (pkv.key_cache[li], pkv.value_cache[li])
    else:
        get_kv = lambda li: pkv[li]

    for label, li in [("sliding", sliding_src), ("global", full_src)]:
        K, V = get_kv(li)
        # K, V: [bsz=1, nkvh, T, head_dim]
        K0 = K[0].permute(1, 0, 2).contiguous()  # [T, nkvh, head_dim]
        V0 = V[0].permute(1, 0, 2).contiguous()
        T, nkvh, hd = K0.shape
        print(f"[hf-base-kv] layer {li} ({label}): K shape={tuple(K.shape)} "
              f"→ slot-major [{T},{nkvh},{hd}]")
        for name, t in [("k", K0), ("v", V0)]:
            arr = t.detach().float().cpu().numpy()
            path = args.out / f"shadow_{name}_{label}_src_hf.npy"
            np.save(path, arr)
            rms = (arr**2).mean()**0.5
            print(f"[hf-base-kv]   saved {path.name}: rms={rms:.4f} "
                  f"first8={arr.flatten()[:8].tolist()}")

    manifest = {
        "base_dir": str(args.base),
        "rvllm_dump": str(args.rvllm_dump),
        "meta": meta,
        "files": sorted(p.name for p in args.out.glob("*.npy")),
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[hf-base-kv] manifest written to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
