#!/usr/bin/env python3
"""HF reference dump for E4B decode-step localization.

Mirrors rvllm's decode-step probe (RVLLM_E4B_DECODE_DUMP_DIR): runs
HF forward on `prompt + appended_token` (already tokenized) and
dumps the per-layer post-decoder-layer residual at the last
position (where `appended_token` sits, == position `prompt_len`).
That's exactly the same point in the chain rvllm dumps at decode
step 0.

Output:
  e4b_decode_step{N}_layer{L}_output.bin  f16, [hidden]

Usage:
  v3/tools/gemma4_e4b_decode_hf_dump.py \\
    --prompt "The capital of France is" \\
    --appended " Paris" \\
    --step 6 \\
    --out-dir /tmp/e4b_hf_decode

The `--step` arg is the position of the appended token in the
full sequence (= prompt_len) so the filenames line up with rvllm's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_f16(path: Path, t: torch.Tensor):
    t = t.detach().to("cpu").to(torch.float16).contiguous()
    path.write_bytes(t.numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/r00t/gemma4-e4b")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--appended", default=" Paris",
                    help="token text appended to prompt (the first decoded "
                         "token from rvllm)")
    ap.add_argument("--step", type=int, required=True,
                    help="position of the appended token (= prompt_len = "
                         "rvllm's decode step value)")
    ap.add_argument("--out-dir", default="/tmp/e4b_hf_decode")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prepend-bos", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt_ids = tok(args.prompt, add_special_tokens=False).input_ids
    appended_ids = tok(args.appended, add_special_tokens=False).input_ids
    if args.prepend_bos and tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + prompt_ids + appended_ids
    else:
        ids = prompt_ids + appended_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    print(f"full sequence: len={len(ids)} (prompt_len_with_bos={args.step}, appended={appended_ids})")
    print(f"  ids[:20] = {ids[:20]}")

    text_model = (
        getattr(getattr(model, "model", model), "language_model", None)
        or getattr(model, "model", model)
    )

    # Capture each layer's output residual via a forward hook.
    captures = {}
    handles = []

    def make_hook(idx):
        def hook(_module, _inputs, output):
            # Gemma4TextDecoderLayer.forward returns either a tensor or
            # a tuple (hidden_states, *_). Normalize.
            hs = output[0] if isinstance(output, tuple) else output
            captures[idx] = hs.detach().to(torch.float16).cpu()
        return hook

    for L, layer in enumerate(text_model.layers):
        handles.append(layer.register_forward_hook(make_hook(L)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in handles:
        h.remove()

    # Dump per-layer residual at position `args.step` (the appended
    # token's slot). Shape after squeeze: [hidden].
    n_layers = len(text_model.layers)
    for L in range(n_layers):
        hs = captures[L][0, args.step, :]  # [hidden]
        path = out / f"e4b_decode_step{args.step}_layer{L}_output.bin"
        write_f16(path, hs)
    print(f"wrote {n_layers} layer dumps to {out}/")


if __name__ == "__main__":
    main()
