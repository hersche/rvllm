#!/usr/bin/env python3
"""Manual Gemma 4 E4B assistant-drafter Q-pipeline reference.

Loads drafter weights from safetensors directly and runs the Q-side of
layer 0 in PyTorch on CPU/GPU. Logs RMS + head8 at every stage so we
can byte-compare against rvllm-serve's [spec-qbisect] probe lines.

Inputs come from a prior `dump_hf_drafter_reference.py --out DIR` run:
    DIR/base_hidden_last.npy    [hidden=2560]
    DIR/last_token_embed.npy    [hidden=2560]   (raw embed × sqrt(2560))

Outputs RMS + head8 for:
    pre_projection_in (concat)
    pre_projection_out (after [256, 5120] @ x)
    L0 post_input_ln
    L0 post_q_proj
    L0 post_q_norm
    L0 post_q_rope

Usage:
    /home/r00t/.venv/bin/python v3/tools/manual_drafter_reference.py \\
        --assist /home/r00t/gemma4-e4b-assistant \\
        --base-dump /tmp/hf_drafter_dump \\
        --position 13     # prompt_len - 1 for "Hauptstadt von Frankreich?"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def rms(t: torch.Tensor) -> float:
    return (t.float() ** 2).mean().sqrt().item()


def head8(t: torch.Tensor):
    return t.flatten()[:8].float().cpu().tolist()


def log(label: str, t: torch.Tensor):
    print(f"[manual-ref] {label:>32s} shape={tuple(t.shape)} rms={rms(t):.4f} head8={head8(t)}")


def gemma_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma 4 RMSNorm: x / sqrt(mean(x^2) + eps) * gamma (NO +1)."""
    var = x.float().pow(2).mean(dim=-1, keepdim=True)
    x_n = x.float() * torch.rsqrt(var + eps)
    return (x_n * gamma.float()).to(x.dtype)


def gemma_partial_rope(x: torch.Tensor, position: int, rope_theta: float,
                       rotary_dim: int, head_dim: int) -> torch.Tensor:
    """Partial split-half NeoX RoPE matching our kernel:
       lo' = lo*cos - hi*sin ;  hi' = lo*sin + hi*cos
       rotates first rotary_dim elements (across paired halves) of each head.
    """
    nheads = x.shape[0]
    out = x.clone()
    half_rotary = rotary_dim // 2
    half_head = head_dim // 2
    # Build inv_freq for half_rotary (sliding/global theta)
    inv_freq = 1.0 / (rope_theta ** (
        torch.arange(0, half_rotary, dtype=torch.float32) / half_rotary
    ))
    freqs = position * inv_freq  # shape [half_rotary]
    cos = freqs.cos()
    sin = freqs.sin()
    for h in range(nheads):
        for i in range(half_rotary):
            lo = x[h, i].float()
            hi = x[h, i + half_head].float()
            out[h, i] = (lo * cos[i] - hi * sin[i]).to(x.dtype)
            out[h, i + half_head] = (lo * sin[i] + hi * cos[i]).to(x.dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assist", required=True, type=Path)
    ap.add_argument("--base-dump", required=True, type=Path)
    ap.add_argument("--position", required=True, type=int)
    ap.add_argument("--layer", default=0, type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[manual-ref] device={device}")

    # Load drafter safetensors.
    state = load_file(args.assist / "model.safetensors", device=device)
    cfg = json.load(open(args.assist / "config.json"))
    text_cfg = cfg["text_config"]
    hidden = text_cfg["hidden_size"]              # 256
    backbone = cfg["backbone_hidden_size"]         # 2560
    num_heads = text_cfg["num_attention_heads"]    # 4
    eps = text_cfg["rms_norm_eps"]                 # 1e-6
    layer_types = text_cfg["layer_types"]
    rope_params = text_cfg["rope_parameters"]
    eff_hd = text_cfg["head_dim"] if layer_types[args.layer] == "sliding_attention" \
        else text_cfg["global_head_dim"]
    is_sliding = layer_types[args.layer] == "sliding_attention"
    if is_sliding:
        rope_theta = rope_params["sliding_attention"]["rope_theta"]
        partial = rope_params["sliding_attention"].get("partial_rotary_factor", 1.0)
    else:
        rope_theta = rope_params["full_attention"]["rope_theta"]
        partial = rope_params["full_attention"].get("partial_rotary_factor", 1.0)
    rotary_dim = int(eff_hd * partial)
    print(f"[manual-ref] hidden={hidden} backbone={backbone} num_heads={num_heads}")
    print(f"[manual-ref] layer={args.layer} is_sliding={is_sliding} "
          f"eff_hd={eff_hd} rope_theta={rope_theta} rotary_dim={rotary_dim}")

    # Inputs (from base dump).
    last_emb = torch.from_numpy(np.load(args.base_dump / "last_token_embed.npy")).to(device, torch.bfloat16)
    base_hidden = torch.from_numpy(np.load(args.base_dump / "base_hidden_last.npy")).to(device, torch.bfloat16)
    log("last_token_embed", last_emb)
    log("base_hidden_last", base_hidden)

    # pre_projection input concat: [inputs_embeds, hidden_states]
    pre_in = torch.cat([last_emb, base_hidden], dim=0)
    log("pre_projection_in (concat)", pre_in)

    # pre_projection.weight [256, 5120]; out = x @ W^T (PyTorch convention)
    W_pre = state["pre_projection.weight"]   # [256, 5120] bf16
    pre_out = (pre_in.float() @ W_pre.float().T).to(torch.bfloat16)
    log("pre_projection_out", pre_out)

    # ----- Layer L = args.layer -----
    L = args.layer
    pref = f"model.layers.{L}"
    W_input_ln = state[f"{pref}.input_layernorm.weight"]
    W_q_proj   = state[f"{pref}.self_attn.q_proj.weight"]  # [q_rows, 256]
    W_q_norm   = state[f"{pref}.self_attn.q_norm.weight"]  # [eff_hd]
    log(f"L{L} input_ln_gamma", W_input_ln)
    log(f"L{L} q_norm_gamma", W_q_norm)

    # Step 1: input_layernorm on hidden=256 vector.
    h = pre_out
    h_norm = gemma_rmsnorm(h, W_input_ln, eps)
    log(f"L{L} post_input_ln", h_norm)

    # Step 2: q_proj. W_q_proj is [q_rows, hidden] bf16.
    q_rows = num_heads * eff_hd
    assert W_q_proj.shape == (q_rows, hidden), f"got {W_q_proj.shape} want ({q_rows},{hidden})"
    q = (h_norm.float() @ W_q_proj.float().T).to(torch.bfloat16)
    log(f"L{L} post_q_proj", q)

    # Step 3: q_norm per head (treat as num_heads "rows" of eff_hd dim).
    q_heads = q.view(num_heads, eff_hd)
    q_normed = gemma_rmsnorm(q_heads, W_q_norm, eps)
    log(f"L{L} post_q_norm", q_normed)

    # Step 4: partial split-half NeoX RoPE.
    q_roped = gemma_partial_rope(q_normed, args.position, rope_theta, rotary_dim, eff_hd)
    log(f"L{L} post_q_rope", q_roped)


if __name__ == "__main__":
    sys.exit(main() or 0)
