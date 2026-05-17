#!/usr/bin/env python3
"""Manual Gemma 4 31B-it assistant-drafter full-forward reference.

Adapted from manual_drafter_reference.py (E4B) for the 31B drafter.
Loads drafter weights from safetensors directly and runs the FULL
forward (pre_projection + 4 layers + final_norm + tied full-vocab
LM head + argmax) in PyTorch on CPU/GPU. Logs RMS + head8 at every
stage so we can byte-compare against rvllm-serve's
[layer-trace] probe lines (RVLLM_GEMMA4_SPEC_LAYER_TRACE=1).

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
import os
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
    ap.add_argument("--assist", required=True, type=Path,
                    help="Drafter dir (e.g. /home/r00t/gemma-4-31B-it-assistant)")
    ap.add_argument("--base-dump", required=True, type=Path,
                    help="rvllm dump dir (RVLLM_SPEC_DUMP_DIR)")
    ap.add_argument("--position", type=int, default=None,
                    help="Q RoPE position; default = meta.committed_len - 1")
    ap.add_argument("--layer", default=0, type=int)
    args = ap.parse_args()

    # Load rvllm dump meta
    meta_path = args.base_dump / "meta.json"
    if meta_path.exists():
        meta = json.load(open(meta_path))
        print(f"[manual-ref-31b] meta = {meta}")
        if args.position is None:
            args.position = int(meta["committed_len"]) - 1
            print(f"[manual-ref-31b] auto position = committed_len-1 = {args.position}")
    else:
        meta = None
        if args.position is None:
            raise SystemExit("--position required when meta.json missing")

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

    # ----- Cross-attn (matching FA-2 f16io kernel semantics) -----
    # rvllm dumps shadow K/V FLAT 1-D (length = committed_len * nkvh
    # * head_dim). Reshape using meta.
    if meta is None:
        raise SystemExit("meta.json required to reshape shadow K/V")
    T_committed = int(meta["committed_len"])
    if is_sliding:
        nkvh_s = int(meta["sliding_num_kv_heads"])
        hd_s = int(meta["sliding_head_dim"])
        K_flat = np.load(args.base_dump / "shadow_k_sliding_src.npy")
        V_flat = np.load(args.base_dump / "shadow_v_sliding_src.npy")
        K = torch.from_numpy(K_flat.reshape(T_committed, nkvh_s, hd_s)).to(device, torch.bfloat16)
        V = torch.from_numpy(V_flat.reshape(T_committed, nkvh_s, hd_s)).to(device, torch.bfloat16)
    else:
        nkvh_g = int(meta["full_num_kv_heads"])
        hd_g = int(meta["full_head_dim"])
        K_flat = np.load(args.base_dump / "shadow_k_global_src.npy")
        V_flat = np.load(args.base_dump / "shadow_v_global_src.npy")
        K = torch.from_numpy(K_flat.reshape(T_committed, nkvh_g, hd_g)).to(device, torch.bfloat16)
        V = torch.from_numpy(V_flat.reshape(T_committed, nkvh_g, hd_g)).to(device, torch.bfloat16)
    n_kv_heads = K.shape[1]
    print(f"[manual-ref] K/V loaded: T={K.shape[0]} n_kv_heads={n_kv_heads} head_dim={K.shape[2]}")

    # GQA: each query head h reads kv head h // (num_heads / n_kv_heads).
    gqa_factor = num_heads // n_kv_heads
    # HF Gemma4 attention scaling = 1.0 (modeling_gemma4.py:1178);
    # rvllm defaults to 1/sqrt(d_k). Toggle here to compare.
    scale = float(os.environ.get("MANUAL_ATTN_SCALE", 1.0 / (eff_hd ** 0.5)))
    T = K.shape[0]
    attn_out = torch.zeros(num_heads, eff_hd, dtype=torch.float32, device=device)
    for h in range(num_heads):
        kv_h = h // gqa_factor
        scores = torch.einsum("d,td->t", q_roped[h].float(), K[:, kv_h, :].float()) * scale
        probs = scores.softmax(dim=-1)
        attn_out[h] = torch.einsum("t,td->d", probs, V[:, kv_h, :].float())
    attn_out_bf16 = attn_out.to(torch.bfloat16)
    log(f"L{L} attn_out", attn_out_bf16)
    # Show top-5 attention probabilities for head 0 to confirm peakedness
    h = 0
    kv_h = h // gqa_factor
    scores_h0 = torch.einsum("d,td->t", q_roped[0].float(), K[:, kv_h, :].float()) * scale
    probs_h0 = scores_h0.softmax(dim=-1)
    top5 = torch.topk(probs_h0, 5)
    print(f"[manual-ref] L{L} h=0 top-5 attn probs: {[(int(top5.indices[i]), float(top5.values[i])) for i in range(5)]}")

    # o_proj: [hidden=256, q_rows=1024] bf16
    W_o = state[f"{pref}.self_attn.o_proj.weight"]
    assert W_o.shape == (hidden, q_rows), f"o_proj shape {W_o.shape} want ({hidden},{q_rows})"
    attn_flat = attn_out_bf16.reshape(-1)
    proj_pre_norm = (attn_flat.float() @ W_o.float().T).to(torch.bfloat16)
    log(f"L{L} post_o_proj", proj_pre_norm)

    # post_attention_layernorm
    W_post_attn = state[f"{pref}.post_attention_layernorm.weight"]
    proj_normed = gemma_rmsnorm(proj_pre_norm, W_post_attn, eps)
    log(f"L{L} post_attn_layernorm", proj_normed)

    # residual_1 add: residual + post_attn_norm(o_proj(attn_out))
    residual_1 = pre_out
    hidden_after_attn = residual_1.float() + proj_normed.float()
    hidden_after_attn = hidden_after_attn.to(torch.bfloat16)
    log(f"L{L} after_attn_finisher", hidden_after_attn)

    # ----- MLP -----
    W_pre_ff = state[f"{pref}.pre_feedforward_layernorm.weight"]
    W_post_ff = state[f"{pref}.post_feedforward_layernorm.weight"]
    W_gate = state[f"{pref}.mlp.gate_proj.weight"]
    W_up = state[f"{pref}.mlp.up_proj.weight"]
    W_down = state[f"{pref}.mlp.down_proj.weight"]
    layer_scalar = state[f"{pref}.layer_scalar"].item()
    print(f"[manual-ref] L{L} layer_scalar = {layer_scalar:.6f}")

    residual_2 = hidden_after_attn
    h_ff = gemma_rmsnorm(hidden_after_attn, W_pre_ff, eps)
    log(f"L{L} post_pre_ff_norm", h_ff)

    gate = (h_ff.float() @ W_gate.float().T)
    up = (h_ff.float() @ W_up.float().T)
    # Gemma 4 uses gelu_pytorch_tanh
    gelu_gate = torch.nn.functional.gelu(gate, approximate="tanh")
    silu_out = gelu_gate * up
    mlp_out = (silu_out @ W_down.float().T).to(torch.bfloat16)
    log(f"L{L} mlp_out", mlp_out)

    mlp_normed = gemma_rmsnorm(mlp_out, W_post_ff, eps)
    log(f"L{L} post_ff_layernorm", mlp_normed)

    hidden_after_mlp = residual_2.float() + mlp_normed.float()
    hidden_after_mlp = hidden_after_mlp.to(torch.bfloat16)
    log(f"L{L} after_residual_2 (pre_scalar)", hidden_after_mlp)

    # layer_scalar (multiply)
    hidden_post_scalar = (hidden_after_mlp.float() * layer_scalar).to(torch.bfloat16)
    log(f"L{L} after_layer_scalar (= L{L+1} input)", hidden_post_scalar)

    # ----- Run remaining layers 1..3 to get final drafter token -----
    h_cur = hidden_post_scalar
    for L_next in range(1, text_cfg["num_hidden_layers"]):
        pref_n = f"model.layers.{L_next}"
        is_sliding_n = layer_types[L_next] == "sliding_attention"
        if is_sliding_n:
            rope_theta_n = rope_params["sliding_attention"]["rope_theta"]
            partial_n = rope_params["sliding_attention"].get("partial_rotary_factor", 1.0)
            eff_hd_n = text_cfg["head_dim"]
            nkvh_s = int(meta["sliding_num_kv_heads"])
            hd_s = int(meta["sliding_head_dim"])
            K_n = torch.from_numpy(
                np.load(args.base_dump / "shadow_k_sliding_src.npy").reshape(
                    T_committed, nkvh_s, hd_s)).to(device, torch.bfloat16)
            V_n = torch.from_numpy(
                np.load(args.base_dump / "shadow_v_sliding_src.npy").reshape(
                    T_committed, nkvh_s, hd_s)).to(device, torch.bfloat16)
        else:
            rope_theta_n = rope_params["full_attention"]["rope_theta"]
            partial_n = rope_params["full_attention"].get("partial_rotary_factor", 1.0)
            eff_hd_n = text_cfg["global_head_dim"]
            nkvh_g = int(meta["full_num_kv_heads"])
            hd_g = int(meta["full_head_dim"])
            K_n = torch.from_numpy(
                np.load(args.base_dump / "shadow_k_global_src.npy").reshape(
                    T_committed, nkvh_g, hd_g)).to(device, torch.bfloat16)
            V_n = torch.from_numpy(
                np.load(args.base_dump / "shadow_v_global_src.npy").reshape(
                    T_committed, nkvh_g, hd_g)).to(device, torch.bfloat16)
        rotary_dim_n = int(eff_hd_n * partial_n)
        n_kv_heads_n = K_n.shape[1]
        gqa_factor_n = num_heads // n_kv_heads_n
        # Honor MANUAL_ATTN_SCALE env across ALL layers, not just L0.
        scale_n = float(os.environ.get("MANUAL_ATTN_SCALE", 1.0 / (eff_hd_n ** 0.5)))
        q_rows_n = num_heads * eff_hd_n

        # Layer L_next forward
        W_in_ln = state[f"{pref_n}.input_layernorm.weight"]
        W_q = state[f"{pref_n}.self_attn.q_proj.weight"]
        W_qn = state[f"{pref_n}.self_attn.q_norm.weight"]
        W_op = state[f"{pref_n}.self_attn.o_proj.weight"]
        W_pal = state[f"{pref_n}.post_attention_layernorm.weight"]
        W_pfl = state[f"{pref_n}.pre_feedforward_layernorm.weight"]
        W_pol = state[f"{pref_n}.post_feedforward_layernorm.weight"]
        W_g = state[f"{pref_n}.mlp.gate_proj.weight"]
        W_u = state[f"{pref_n}.mlp.up_proj.weight"]
        W_d = state[f"{pref_n}.mlp.down_proj.weight"]
        ls_n = state[f"{pref_n}.layer_scalar"].item()

        residual_in = h_cur
        h_norm_n = gemma_rmsnorm(h_cur, W_in_ln, eps)
        log(f"L{L_next} post_input_ln", h_norm_n)
        q_n = (h_norm_n.float() @ W_q.float().T).to(torch.bfloat16)
        log(f"L{L_next} post_q_proj", q_n)
        q_n_heads = q_n.view(num_heads, eff_hd_n)
        q_n_normed = gemma_rmsnorm(q_n_heads, W_qn, eps)
        log(f"L{L_next} post_q_norm", q_n_normed)
        q_n_roped = gemma_partial_rope(q_n_normed, args.position, rope_theta_n, rotary_dim_n, eff_hd_n)
        log(f"L{L_next} post_q_rope", q_n_roped)
        # cross-attn
        attn_n = torch.zeros(num_heads, eff_hd_n, dtype=torch.float32, device=device)
        for h_i in range(num_heads):
            kv_h_i = h_i // gqa_factor_n
            sc = torch.einsum("d,td->t", q_n_roped[h_i].float(), K_n[:, kv_h_i, :].float()) * scale_n
            pr = sc.softmax(dim=-1)
            attn_n[h_i] = torch.einsum("t,td->d", pr, V_n[:, kv_h_i, :].float())
        attn_n_bf16 = attn_n.to(torch.bfloat16).reshape(-1)
        log(f"L{L_next} attn_out", attn_n_bf16)
        proj_pre_n = (attn_n_bf16.float() @ W_op.float().T).to(torch.bfloat16)
        log(f"L{L_next} post_o_proj", proj_pre_n)
        proj_norm_n = gemma_rmsnorm(proj_pre_n, W_pal, eps)
        log(f"L{L_next} post_attn_layernorm", proj_norm_n)
        h_after_attn = (residual_in.float() + proj_norm_n.float()).to(torch.bfloat16)
        log(f"L{L_next} after_attn_finisher", h_after_attn)
        # MLP
        h_ff_in = gemma_rmsnorm(h_after_attn, W_pfl, eps)
        log(f"L{L_next} post_pre_ff_norm", h_ff_in)
        gate_n = (h_ff_in.float() @ W_g.float().T)
        up_n = (h_ff_in.float() @ W_u.float().T)
        gelu_gate_n = torch.nn.functional.gelu(gate_n, approximate="tanh")
        silu_out_n = gelu_gate_n * up_n
        mlp_out_n = (silu_out_n @ W_d.float().T).to(torch.bfloat16)
        log(f"L{L_next} mlp_out", mlp_out_n)
        mlp_normed_n = gemma_rmsnorm(mlp_out_n, W_pol, eps)
        log(f"L{L_next} post_ff_layernorm", mlp_normed_n)
        h_after_mlp = (h_after_attn.float() + mlp_normed_n.float()).to(torch.bfloat16)
        log(f"L{L_next} after_residual_2 (pre_scalar)", h_after_mlp)
        h_cur = (h_after_mlp.float() * ls_n).to(torch.bfloat16)
        log(f"L{L_next} after_layer_scalar", h_cur)

    # final_norm
    W_final = state["model.norm.weight"]
    h_final = gemma_rmsnorm(h_cur, W_final, eps)
    log("final_norm output (drafter_hidden)", h_final)

    # LM head — branch on use_ordered_embeddings
    use_ordered = bool(cfg.get("use_ordered_embeddings", True))
    W_embed = state["model.embed_tokens.weight"]              # [vocab, hidden]
    vocab = text_cfg["vocab_size"]
    if use_ordered:
        # MaskedEmbedder path (E4B-style)
        W_centroids = state["masked_embedding.centroids.weight"]
        token_ordering = state["masked_embedding.token_ordering"].long()
        n_cent = cfg["num_centroids"]
        top_k = cfg["centroid_intermediate_top_k"]
        per_centroid = vocab // n_cent
        cent_logits = (h_final.float() @ W_centroids.float().T)
        top_cent = torch.topk(cent_logits, top_k).indices
        cand_ids = []
        for c in top_cent.tolist():
            cand_ids.extend(token_ordering[c * per_centroid:(c+1) * per_centroid].tolist())
        cand_ids = torch.tensor([c for c in cand_ids if 0 <= c < vocab],
                                dtype=torch.long, device=device)
        cand_logits = (h_final.float() @ W_embed[cand_ids].float().T)
        argmax_idx = int(cand_logits.argmax())
        argmax_tok = int(cand_ids[argmax_idx])
        print(f"[manual-ref-31b] MaskedEmbedder argmax token: {argmax_tok}  "
              f"(logit={cand_logits[argmax_idx]:.4f})")
        top5_idx = torch.topk(cand_logits, 5).indices.tolist()
        print(f"[manual-ref-31b] top-5: "
              f"{[(int(cand_ids[i]), float(cand_logits[i])) for i in top5_idx]}")
    else:
        # Full-vocab tied LM head (31B path)
        logits = (h_final.float() @ W_embed.float().T)  # [vocab]
        argmax_tok = int(logits.argmax())
        top5 = torch.topk(logits, 5)
        print(f"[manual-ref-31b] Full-vocab argmax token: {argmax_tok}  "
              f"(logit={logits[argmax_tok]:.4f})")
        print(f"[manual-ref-31b] top-5 token,logit: "
              f"{[(int(top5.indices[i]), float(top5.values[i])) for i in range(5)]}")


if __name__ == "__main__":
    sys.exit(main() or 0)
