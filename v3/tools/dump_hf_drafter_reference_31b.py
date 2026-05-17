#!/usr/bin/env python3
"""HF transformers reference dump for the Gemma 4 31B-it assistant drafter.

ZERO base load. Reads the rvllm-side dump (RVLLM_SPEC_DUMP_DIR
contents) for base_hidden_last + last_token_embed + shadow K/V,
loads ONLY the 895 MiB 31B drafter via HF transformers, and runs
ONE drafter forward step. Hooks every drafter sub-module to dump
per-layer outputs for comparison against
v3/tools/manual_drafter_reference_31b.py and rvllm's runtime
layer-trace probes.

Requires transformers >= 5.6 with `gemma4_assistant`. On Cortex
the working interpreter is /home/r00t/.vllm-exp/bin/python3
(transformers 5.8.0).

Usage:
    /home/r00t/.vllm-exp/bin/python3 \\
        v3/tools/dump_hf_drafter_reference_31b.py \\
        --assist     /home/r00t/gemma-4-31B-it-assistant \\
        --rvllm-dump /tmp/rvllm_drafter_dump \\
        --out        /tmp/hf_drafter_dump_31b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


def to_f32_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def rms(t: torch.Tensor) -> float:
    return (t.detach().float() ** 2).mean().sqrt().item()


def head8(t: torch.Tensor):
    return t.detach().flatten()[:8].float().cpu().tolist()


def log(label: str, t: torch.Tensor):
    print(f"[hf31b] {label:>34s} shape={tuple(t.shape)} "
          f"rms={rms(t):.4f} head8={head8(t)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assist", required=True, type=Path,
                    help="31B drafter dir (/home/r00t/gemma-4-31B-it-assistant)")
    ap.add_argument("--rvllm-dump", required=True, type=Path,
                    help="rvllm dump dir (RVLLM_SPEC_DUMP_DIR contents)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[hf31b] drafter    = {args.assist}")
    print(f"[hf31b] rvllm-dump = {args.rvllm_dump}")
    print(f"[hf31b] out        = {args.out}")

    # ---- rvllm-side dump ----
    meta = json.load(open(args.rvllm_dump / "meta.json"))
    print(f"[hf31b] meta = {meta}")

    base_hidden_f32 = np.load(args.rvllm_dump / "base_hidden_last.npy")
    last_emb_f32 = np.load(args.rvllm_dump / "last_token_embed.npy")
    K_swa_flat = np.load(args.rvllm_dump / "shadow_k_sliding_src.npy")
    V_swa_flat = np.load(args.rvllm_dump / "shadow_v_sliding_src.npy")
    K_full_flat = np.load(args.rvllm_dump / "shadow_k_global_src.npy")
    V_full_flat = np.load(args.rvllm_dump / "shadow_v_global_src.npy")

    T_committed = int(meta["committed_len"])
    nkvh_s = int(meta["sliding_num_kv_heads"])
    hd_s = int(meta["sliding_head_dim"])
    nkvh_g = int(meta["full_num_kv_heads"])
    hd_g = int(meta["full_head_dim"])
    h_backbone = int(meta["hidden_size"])

    K_swa = K_swa_flat.reshape(T_committed, nkvh_s, hd_s)
    V_swa = V_swa_flat.reshape(T_committed, nkvh_s, hd_s)
    K_full = K_full_flat.reshape(T_committed, nkvh_g, hd_g)
    V_full = V_full_flat.reshape(T_committed, nkvh_g, hd_g)

    print(f"[hf31b] K_swa  shape={K_swa.shape}")
    print(f"[hf31b] K_full shape={K_full.shape}")

    # ---- HF model ----
    from transformers import AutoConfig
    from transformers.models.gemma4_assistant import Gemma4AssistantForCausalLM

    cfg = AutoConfig.from_pretrained(args.assist, local_files_only=True)
    text_cfg = cfg.get_text_config()
    h_drafter = int(text_cfg.hidden_size)
    print(f"[hf31b] drafter hidden={h_drafter} backbone={cfg.backbone_hidden_size} "
          f"use_ordered={cfg.use_ordered_embeddings}")
    assert cfg.backbone_hidden_size == h_backbone, \
        f"meta hidden={h_backbone} vs drafter backbone={cfg.backbone_hidden_size} mismatch"

    print("[hf31b] loading drafter (bf16) ...")
    assist = Gemma4AssistantForCausalLM.from_pretrained(
        args.assist, local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    assist = assist.to("cuda")
    assist.eval()
    device = next(assist.parameters()).device
    print(f"[hf31b] drafter loaded; device={device}")

    # ---- inputs_embeds = cat([last_token_embed, base_hidden]) ----
    last_emb = torch.from_numpy(last_emb_f32).to(device, torch.bfloat16)
    base_hidden = torch.from_numpy(base_hidden_f32).to(device, torch.bfloat16)
    log("last_token_embed (input)", last_emb)
    log("base_hidden_last (input)", base_hidden)
    inputs_embeds = torch.cat([last_emb, base_hidden], dim=-1)\
        .unsqueeze(0).unsqueeze(0)
    # [1, 1, 2*backbone]
    log("inputs_embeds (cat)", inputs_embeds[0, 0])

    # ---- shared_kv_states ----
    # HF Cache.layer shape: [bsz, num_kv_heads, seq_len, head_dim].
    # rvllm dumps are [T, nkvh, hd] → permute + unsqueeze.
    K_swa_t = torch.from_numpy(K_swa).to(device, torch.bfloat16)\
        .permute(1, 0, 2).unsqueeze(0)
    V_swa_t = torch.from_numpy(V_swa).to(device, torch.bfloat16)\
        .permute(1, 0, 2).unsqueeze(0)
    K_full_t = torch.from_numpy(K_full).to(device, torch.bfloat16)\
        .permute(1, 0, 2).unsqueeze(0)
    V_full_t = torch.from_numpy(V_full).to(device, torch.bfloat16)\
        .permute(1, 0, 2).unsqueeze(0)
    shared_kv_states = {
        "sliding_attention": (K_swa_t, V_swa_t),
        "full_attention": (K_full_t, V_full_t),
    }
    print(f"[hf31b] shared_kv_states sliding K {tuple(K_swa_t.shape)} "
          f"V {tuple(V_swa_t.shape)}")
    print(f"[hf31b] shared_kv_states full    K {tuple(K_full_t.shape)} "
          f"V {tuple(V_full_t.shape)}")

    # ---- position_ids ----
    # HF candidate_generator.py:1370 sets position_ids = [[len-1]]
    # ONCE and reuses across all drafter iterations.
    position_ids = torch.tensor(
        [[T_committed - 1]], dtype=torch.long, device=device)
    print(f"[hf31b] position_ids = {position_ids.tolist()}")

    # ---- Hook every drafter sub-module ----
    captured: dict[str, torch.Tensor] = {}

    def grab(name: str):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[name] = t.detach()
        return hook

    inner = assist.model
    n_layers = len(inner.layers)
    print(f"[hf31b] drafter has {n_layers} layers")

    assist.pre_projection.register_forward_hook(grab("pre_projection_out"))
    for li in range(n_layers):
        layer = inner.layers[li]
        layer.input_layernorm.register_forward_hook(grab(f"L{li}_input_ln"))
        if hasattr(layer.self_attn, "q_proj"):
            layer.self_attn.q_proj.register_forward_hook(grab(f"L{li}_q_proj"))
        if hasattr(layer.self_attn, "q_norm"):
            layer.self_attn.q_norm.register_forward_hook(grab(f"L{li}_q_norm"))
        def make_grab_input(key: str):
            def h(_m, inp, _out):
                captured[key] = inp[0].detach()
            return h
        layer.self_attn.o_proj.register_forward_hook(
            make_grab_input(f"L{li}_attn_out_preo"))
        layer.self_attn.o_proj.register_forward_hook(grab(f"L{li}_o_proj"))
        layer.post_attention_layernorm.register_forward_hook(grab(f"L{li}_post_attn_ln"))
        layer.pre_feedforward_layernorm.register_forward_hook(grab(f"L{li}_pre_ff_ln"))
        layer.mlp.down_proj.register_forward_hook(grab(f"L{li}_mlp_down"))
        layer.post_feedforward_layernorm.register_forward_hook(grab(f"L{li}_post_ff_ln"))

    inner.norm.register_forward_hook(grab("final_norm"))
    assist.lm_head.register_forward_hook(grab("logits"))

    # ---- Forward ----
    print("[hf31b] running drafter forward ...")
    with torch.no_grad():
        out = assist(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=None,
            shared_kv_states=shared_kv_states,
            use_cache=False,
        )

    logits = out.logits if hasattr(out, "logits") else captured.get("logits")
    top5_list = None
    argmax_tok = None
    if logits is not None:
        last_logits = logits[0, -1] if logits.dim() == 3 else logits.flatten()
        argmax_tok = int(last_logits.argmax())
        top5 = torch.topk(last_logits, 5)
        top5_list = [(int(top5.indices[i]), float(top5.values[i])) for i in range(5)]
        print(f"[hf31b] DRAFTER argmax token: {argmax_tok}  "
              f"(logit={float(last_logits[argmax_tok]):.4f})")
        print(f"[hf31b] DRAFTER top-5: {top5_list}")

    # Save all captured tensors (slice [B,T,*] -> [*] at last seq pos)
    for name, t in captured.items():
        if t.dim() == 3:
            t = t[0, -1]
        elif t.dim() == 2 and t.shape[0] > 1:
            t = t[-1]
        elif t.dim() == 2:
            t = t[0]
        path = args.out / f"{name}.npy"
        np.save(path, to_f32_np(t))
        log(f"saved {name}", t)

    manifest = {
        "drafter_dir": str(args.assist),
        "rvllm_dump": str(args.rvllm_dump),
        "meta": meta,
        "n_layers": n_layers,
        "argmax_token": argmax_tok,
        "top5": top5_list,
        "files": sorted(p.name for p in args.out.glob("*.npy")),
    }
    with open(args.out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[hf31b] manifest written to {args.out / 'manifest.json'}")


if __name__ == "__main__":
    sys.exit(main() or 0)
