#!/usr/bin/env python3
"""HF transformers reference dump for the Gemma 4 E4B assistant drafter.

Runs the base + assistant locally on a fixed prompt and dumps every
intermediate of the drafter Q-pipeline at layer 0 (plus the base's
shared-KV source layers 22/23). Pair with the rvllm-serve runtime
probes (RVLLM_SPEC_DEBUG_Q_BISECT=1) to find the exact step where
our drafter Q diverges from HF.

Usage:
    /home/r00t/.venv/bin/python v3/tools/dump_hf_drafter_reference.py \\
        --base   /home/r00t/gemma4-e4b \\
        --assist /home/r00t/gemma4-e4b-assistant \\
        --prompt "Hauptstadt von Frankreich?" \\
        --out    /tmp/hf_drafter_dump

Outputs (all f32 .npy, plus a manifest.json):
    base_hidden_last.npy              [hidden=2560]    snapshot for drafter input
    base_argmax_first.npy             []               first decode token
    last_token_embed.npy              [hidden=2560]    pre-scaled embed of last prompt token
    pre_projection_out.npy            [256]            drafter L0 entry residual
    l0_post_input_ln.npy              [256]
    l0_post_q_proj.npy                [1024]
    l0_post_q_norm.npy                [1024]
    l0_post_q_rope.npy                [1024]
    l0_attn_out.npy                   [1024]           softmax(QK^T)V output
    l0_top5_attn_probs.npy            [5]              (slot, prob) tuples
    shadow_k_sliding_src.npy          [ctx, 2, 256]    K at base layer 22 for ctx slots
    shadow_v_sliding_src.npy          [ctx, 2, 256]    V at base layer 22 for ctx slots
    shadow_k_global_src.npy           [ctx, 2, 512]    K at base layer 23
    shadow_v_global_src.npy           [ctx, 2, 512]    V at base layer 23
    manifest.json                                       shapes / dtypes / context

Pair-comparison:
    Each rvllm-serve probe logs (in journalctl, with
    RVLLM_SPEC_DEBUG_Q_BISECT=1) the head8 + RMS at the matching
    stage. Compare manually, or extend with a `cmp_drafter_q.py`
    that reads rvllm dump files once we wire file output on the
    Rust side.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch


def to_f32_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path,
                    help="Base Gemma 4 E4B-it directory")
    ap.add_argument("--assist", required=True, type=Path,
                    help="Gemma 4 E4B-it-assistant drafter directory")
    ap.add_argument("--prompt", default="Hauptstadt von Frankreich?")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[hf-ref] base   = {args.base}")
    print(f"[hf-ref] assist = {args.assist}")
    print(f"[hf-ref] prompt = {args.prompt!r}")

    # Lazy import — transformers boot is slow.
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )

    # Tokenize using base's tokenizer (assistant inherits).
    tok = AutoTokenizer.from_pretrained(args.base, local_files_only=True)
    # Match rvllm-serve's chat-template render: use the same
    # apply_chat_template the production path uses. If the user
    # prompt is bare text, wrap it as a single user turn.
    msgs = [{"role": "user", "content": args.prompt}]
    input_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt"
    )
    # transformers 5.x returns BatchEncoding here in some versions —
    # extract the tensor robustly.
    if hasattr(input_ids, "input_ids"):
        input_ids = input_ids["input_ids"]
    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    print(f"[hf-ref] prompt tokenized to {input_ids.shape[-1]} ids: {input_ids[0].tolist()}")

    # ----------------------------------------------------------
    # Base forward + capture K/V at source layers + final hidden
    # ----------------------------------------------------------
    print("[hf-ref] loading base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="eager",  # so we can hook attention easily
    )
    base.eval()

    # Identify source layers per E4B arch: num_kv_shared_layers=18,
    # num_hidden_layers=42 → shared-KV tail begins at layer 22 (sliding)
    # and layer 23 (full).  We capture the K/V tensors written by these
    # layers' attention modules.
    text_cfg = base.config.text_config if hasattr(base.config, "text_config") else base.config
    n_layers = text_cfg.num_hidden_layers
    n_shared = getattr(text_cfg, "num_kv_shared_layers", 0)
    first_shared = n_layers - n_shared
    layer_types = getattr(text_cfg, "layer_types", None)
    print(f"[hf-ref] base: {n_layers} layers, {n_shared} shared-KV tail layers, first_shared={first_shared}")
    if layer_types:
        print(f"[hf-ref] base: layer types around shared start: {layer_types[first_shared-2:first_shared+4]}")

    # Source layers = LAST sliding + LAST full BEFORE the shared tail.
    # In E4B: 42 layers, last 18 share KV, so tail starts at 24. The
    # sources are typically 22 (sliding) and 23 (full) — i.e. the last
    # layers that actually compute K/V before the tail reuses them.
    sliding_src = None
    full_src = None
    if layer_types:
        for li in range(first_shared - 1, -1, -1):
            if sliding_src is None and layer_types[li] == "sliding_attention":
                sliding_src = li
            if full_src is None and layer_types[li] == "full_attention":
                full_src = li
            if sliding_src is not None and full_src is not None:
                break
    print(f"[hf-ref] base shared-KV source layers (last before tail): "
          f"sliding={sliding_src} full={full_src}")

    # Capture base's K/V at source layers + final hidden via hooks.
    captured = {}

    def make_kv_hook(li: int):
        def hook(module, args, output):
            # Gemma4 attention forward returns (attn_output, attn_weights, past_key_value)
            # We capture pre-output K/V by reading the past_key_value cache after forward.
            return  # KV cache is stored in past_key_values, not in module output
        return hook

    # Better: read past_key_values from output_attentions=True forward
    with torch.no_grad():
        outputs = base(
            input_ids=input_ids.to(base.device),
            use_cache=True,
            return_dict=True,
            output_hidden_states=True,
        )
    print(f"[hf-ref] base forward done. hidden_states len={len(outputs.hidden_states)}")

    # Final hidden (post final_norm)
    # outputs.hidden_states is a tuple of (n_layers+1) tensors [B, T, H]
    # hidden_states[-1] is AFTER final layer (post-final-norm in
    # Gemma4ForCausalLM since model.norm is in model output)
    final_hidden = outputs.hidden_states[-1][0]  # [T, H]
    base_hidden_last = final_hidden[-1]  # [H] at last prompt token
    np.save(args.out / "base_hidden_last.npy", to_f32_np(base_hidden_last))
    print(f"[hf-ref] saved base_hidden_last: shape={tuple(base_hidden_last.shape)} "
          f"rms={(base_hidden_last.float()**2).mean().sqrt().item():.4f}")

    # base argmax at the last prompt position (= base's first decode token)
    logits = base.lm_head(final_hidden)  # [T, V]
    base_argmax = logits[-1].argmax().item()
    np.save(args.out / "base_argmax_first.npy", np.array([base_argmax], dtype=np.int64))
    print(f"[hf-ref] saved base_argmax_first = {base_argmax}")

    # past_key_values: capture K/V at source layers
    pkv = outputs.past_key_values
    if pkv is not None:
        # transformers stores per-layer (key, value). For Cache objects
        # use .key_cache[li] / .value_cache[li].
        if hasattr(pkv, "key_cache"):
            for label, li in [("sliding", sliding_src), ("global", full_src)]:
                if li is None: continue
                k = pkv.key_cache[li][0]  # [n_kv_heads, T, head_dim]
                v = pkv.value_cache[li][0]
                # Transpose to [T, n_kv_heads, head_dim] to match our shadow layout
                k_thd = k.permute(1, 0, 2).contiguous()
                v_thd = v.permute(1, 0, 2).contiguous()
                np.save(args.out / f"shadow_k_{label}_src.npy", to_f32_np(k_thd))
                np.save(args.out / f"shadow_v_{label}_src.npy", to_f32_np(v_thd))
                print(f"[hf-ref] saved shadow K/V {label} src layer {li}: "
                      f"K {tuple(k_thd.shape)} V {tuple(v_thd.shape)}")

    # last_token_embed used by drafter (= embed_tokens(last) * sqrt(hidden))
    # Use get_input_embeddings() to be robust across transformers versions
    # and nested model structures (Gemma4ForCausalLM wraps Gemma4Model).
    embed_tokens = base.get_input_embeddings()
    last_tok = input_ids[0, -1].item()
    last_token_embed_raw = embed_tokens(torch.tensor([last_tok], device=base.device))[0]
    raw_rms = (last_token_embed_raw.float()**2).mean().sqrt().item()
    # Use the base's text-config hidden_size as the embed scale factor.
    h_size = int(text_cfg.hidden_size)
    sqrt_h = float(h_size) ** 0.5
    last_token_embed_scaled = last_token_embed_raw * sqrt_h
    np.save(args.out / "last_token_embed.npy", to_f32_np(last_token_embed_scaled))
    print(f"[hf-ref] saved last_token_embed (id={last_tok}): "
          f"raw_rms={raw_rms:.4f} hidden={h_size} sqrt_h={sqrt_h:.4f} "
          f"scaled_rms={(last_token_embed_scaled.float()**2).mean().sqrt().item():.4f}")

    del base
    torch.cuda.empty_cache()

    # ----------------------------------------------------------
    # Assistant drafter forward — hook every Q-pipeline submodule
    # ----------------------------------------------------------
    print("[hf-ref] loading assistant drafter...")
    try:
        assist = AutoModelForCausalLM.from_pretrained(
            args.assist, local_files_only=True,
            dtype=torch.bfloat16, device_map="cuda",
            trust_remote_code=True,
        )
        assist.eval()
    except Exception as e:
        print(f"[hf-ref] WARNING: cannot load assistant via AutoModel: {e}")
        print(f"[hf-ref] transformers version doesn't know gemma4_assistant.")
        print(f"[hf-ref] We have enough data already (base side); skipping drafter forward.")
        # Write manifest with what we have
        manifest = {
            "prompt": args.prompt,
            "prompt_token_ids": input_ids[0].tolist(),
            "prompt_len": int(input_ids.shape[-1]),
            "base_dir": str(args.base),
            "base_argmax_first": int(base_argmax),
            "sliding_source_layer": sliding_src,
            "full_source_layer": full_src,
            "files": sorted(p.name for p in args.out.glob("*.npy")),
            "drafter_forward_attempted": False,
            "drafter_skipped_reason": str(e),
        }
        with open(args.out / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[hf-ref] manifest written. {len(manifest['files'])} files dumped.")
        return

    # Identify the model.model attribute (Gemma4AssistantModel wrapping
    # decoder layers, embed, norms, pre/post_projection, masked_embedding).
    inner = getattr(assist, "model", assist)
    layer0 = inner.layers[0]

    hooks = []
    captured.clear()

    def grab(name):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach()
        return hook

    # Hooks on the layer-0 submodules.
    hooks.append(layer0.input_layernorm.register_forward_hook(grab("l0_post_input_ln")))
    hooks.append(layer0.self_attn.q_proj.register_forward_hook(grab("l0_post_q_proj")))
    if hasattr(layer0.self_attn, "q_norm"):
        hooks.append(layer0.self_attn.q_norm.register_forward_hook(grab("l0_post_q_norm")))
    # Hook the entire layer-0 self_attn to capture attn_out (before o_proj).
    # In HF Gemma4 attention, attn_out = softmax(QK)V is the o_proj INPUT.
    if hasattr(layer0.self_attn, "o_proj"):
        def grab_o_input(_mod, inp, _out):
            captured["l0_attn_out"] = inp[0].detach()
        hooks.append(layer0.self_attn.o_proj.register_forward_hook(grab_o_input))
    # pre_projection on the inner model.
    if hasattr(inner, "pre_projection"):
        hooks.append(inner.pre_projection.register_forward_hook(grab("pre_projection_out")))

    # Run drafter one step. The Gemma4Assistant.forward needs
    # inputs_embeds + shared_kv_states. For our reference dump we can
    # simulate by calling assist's forward with the prepared inputs.
    # If transformers' Gemma4Assistant requires a specific call shape
    # (input_ids + shared_kv_states), we adapt.
    try:
        with torch.no_grad():
            assist_out = assist(
                input_ids=input_ids[:, -1:].to(assist.device),
                use_cache=False,
                return_dict=True,
            )
        print(f"[hf-ref] drafter forward done (single-token)")
    except Exception as e:
        print(f"[hf-ref] WARNING: drafter forward signature mismatch: {e}")
        print(f"[hf-ref] The drafter likely needs explicit shared_kv_states + "
              f"inputs_embeds. See HF Gemma4AssistantForCausalLM.forward signature.")
        print(f"[hf-ref] Falling back: capture what hooks fired before exception.")

    for h in hooks:
        h.remove()

    for name, t in captured.items():
        # Slice to last position for [B,T,H] tensors (we want position-prompt_len-1)
        if t.dim() == 3:
            t = t[0, -1]
        elif t.dim() == 2:
            t = t[-1] if t.shape[0] > 1 else t[0]
        path = args.out / f"{name}.npy"
        np.save(path, to_f32_np(t))
        rms = (t.float()**2).mean().sqrt().item()
        head8 = t.flatten()[:8].float().cpu().tolist()
        print(f"[hf-ref] saved {name}: shape={tuple(t.shape)} rms={rms:.4f} head8={head8}")

    # Manifest
    manifest = {
        "prompt": args.prompt,
        "prompt_token_ids": input_ids[0].tolist(),
        "prompt_len": int(input_ids.shape[-1]),
        "base_dir": str(args.base),
        "assist_dir": str(args.assist),
        "base_argmax_first": int(base_argmax),
        "sliding_source_layer": sliding_src,
        "full_source_layer": full_src,
        "files": sorted(p.name for p in args.out.glob("*.npy")),
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[hf-ref] manifest written. base_argmax_first={base_argmax}, "
          f"prompt_len={input_ids.shape[-1]}")
    print(f"[hf-ref] dump complete at {args.out}")
    print()
    print("To compare against rvllm-serve runtime probes:")
    print("  1. Start rvllm-serve with RVLLM_SPEC_DEBUG_Q_BISECT=1 +")
    print("     RVLLM_SPEC_DEBUG_CPU_ATTN=1")
    print("  2. POST the SAME prompt at temperature=0, max_tokens=2")
    print("  3. journalctl -u rvllm-serve | grep -E 'spec-qbisect|spec-cpu-attn'")
    print("  4. Compare RMS + head8 stage-by-stage. The first stage that")
    print("     diverges (cosine << 1.0) is where the bug lives.")
    print()
    print("Notes:")
    print(" - HF drafter forward signature may require shared_kv_states +")
    print("   inputs_embeds explicitly. If the single-token call above")
    print("   raised, only pre_projection / input_layernorm hooks fired.")
    print(" - Inspect manifest.files to see what was actually dumped.")


if __name__ == "__main__":
    sys.exit(main() or 0)
