#!/usr/bin/env python3
"""HF ground-truth accept_per_verify baseline for Gemma 4 31B spec decode.

Codex Q7 (i): proves whether the 31B drafter genuinely fails on
this prompt class (drafter is weak) or whether rvllm's base-side
inputs differ from HF's expectation (impl/quant bug).

Loads Gemma 4 31B BASE + 31B drafter in HF transformers, runs the
official `target.generate(..., assistant_model=...)` path with
greedy temp=0 on the same prompts rvllm fails on (0/324 accept).
HF transformers' MTP candidate generator emits accept metrics —
parse the difference between generated_ids and prompt_ids and
divide by drafted count to get the same `accepted_per_verify`
rvllm reports.

Uses /home/r00t/.vllm-exp/bin/python3 (transformers 5.8.0 with
gemma4_assistant). On Cortex the only 31B base on disk is the
FP8-block compressed-tensors checkpoint; HF will dequant to bf16
at load. That's a *different* precision from rvllm's runtime
NVFP4/FP8 KV cache and exactly the comparison codex Q4 calls out.

Usage:
    /home/r00t/.vllm-exp/bin/python3 \\
        v3/tools/hf_31b_spec_accept_baseline.py \\
        --base   /home/r00t/.vllm/models/gemma-4-31b-it-fp8-block \\
        --assist /home/r00t/gemma-4-31B-it-assistant \\
        --max-new 24
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


PROMPTS = [
    "Was ist die Hauptstadt von Frankreich?",
    "The capital of France is",
    "Hallo, mein Name ist",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base",   required=True, type=Path)
    ap.add_argument("--assist", required=True, type=Path)
    ap.add_argument("--max-new", type=int, default=24)
    args = ap.parse_args()

    print(f"[hf-31b-baseline] base   = {args.base}")
    print(f"[hf-31b-baseline] assist = {args.assist}")
    print(f"[hf-31b-baseline] max_new = {args.max_new}")

    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Tokenizer (shared)
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)

    print(f"[hf-31b-baseline] loading base (bf16 from compressed fp8) ...")
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    base = base.to("cuda")
    base.eval()
    print(f"[hf-31b-baseline] base loaded in {time.time()-t0:.1f}s, "
          f"device={next(base.parameters()).device}")

    print(f"[hf-31b-baseline] loading assist drafter ...")
    t0 = time.time()
    assist = AutoModelForCausalLM.from_pretrained(
        args.assist, local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    assist = assist.to("cuda")
    assist.eval()
    print(f"[hf-31b-baseline] assist loaded in {time.time()-t0:.1f}s")

    print(f"[hf-31b-baseline] free vram: "
          f"{torch.cuda.mem_get_info()[0]/1e9:.2f} GB free / "
          f"{torch.cuda.mem_get_info()[1]/1e9:.2f} GB total")

    results = []
    for prompt in PROMPTS:
        print(f"\n[hf-31b-baseline] ===== prompt: {prompt!r} =====")
        msgs = [{"role": "user", "content": prompt}]
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt"
        ).to("cuda")
        if hasattr(input_ids, "input_ids"):
            input_ids = input_ids["input_ids"]
        print(f"[hf-31b-baseline] prompt_len = {input_ids.shape[-1]}")

        # Baseline: target alone (no spec) — gives the ground-truth
        # tokens we want the drafter to predict.
        print(f"[hf-31b-baseline] target-only generate ...")
        t0 = time.time()
        with torch.no_grad():
            out_target_only = base.generate(
                input_ids,
                max_new_tokens=args.max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        wall_target = time.time() - t0
        target_tokens = out_target_only[0, input_ids.shape[-1]:].tolist()
        target_text = tok.decode(target_tokens, skip_special_tokens=True)
        print(f"[hf-31b-baseline] target tokens ({len(target_tokens)}, "
              f"{wall_target:.2f}s): {target_tokens}")
        print(f"[hf-31b-baseline] target text: {target_text!r}")

        # Spec: target + assistant. HF's MultiTokenPrediction generator
        # is auto-selected when assistant_model is a Gemma4AssistantForCausalLM.
        print(f"[hf-31b-baseline] target + assist generate ...")
        t0 = time.time()
        with torch.no_grad():
            out_spec = base.generate(
                input_ids,
                assistant_model=assist,
                max_new_tokens=args.max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
            )
        wall_spec = time.time() - t0
        spec_tokens = out_spec[0, input_ids.shape[-1]:].tolist()
        spec_text = tok.decode(spec_tokens, skip_special_tokens=True)
        print(f"[hf-31b-baseline] spec tokens ({len(spec_tokens)}, "
              f"{wall_spec:.2f}s): {spec_tokens}")
        print(f"[hf-31b-baseline] spec text: {spec_text!r}")

        # Byte-equality check
        match_prefix_len = 0
        for i in range(min(len(target_tokens), len(spec_tokens))):
            if target_tokens[i] == spec_tokens[i]:
                match_prefix_len += 1
            else:
                break
        agreed = "✓" if (target_tokens == spec_tokens) else "✗"
        print(f"[hf-31b-baseline] {agreed} target == spec: "
              f"prefix match {match_prefix_len}/{len(target_tokens)}")

        results.append({
            "prompt": prompt,
            "prompt_len": int(input_ids.shape[-1]),
            "target_tokens": target_tokens,
            "spec_tokens": spec_tokens,
            "target_text": target_text,
            "spec_text": spec_text,
            "wall_target_s": wall_target,
            "wall_spec_s": wall_spec,
            "speedup": wall_target / wall_spec if wall_spec > 0 else None,
            "match_prefix_len": match_prefix_len,
            "byte_identical": target_tokens == spec_tokens,
        })

    out_path = Path("/tmp/hf_31b_spec_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[hf-31b-baseline] summary written to {out_path}")
    for r in results:
        sp = r["speedup"]
        sp_str = f"{sp:.2f}x" if sp else "n/a"
        print(f"  {r['prompt'][:48]:48s}  "
              f"prefix={r['match_prefix_len']:>3d}/{len(r['target_tokens']):>3d}  "
              f"speedup={sp_str:>7s}  "
              f"byte_id={r['byte_identical']}")


if __name__ == "__main__":
    sys.exit(main() or 0)
