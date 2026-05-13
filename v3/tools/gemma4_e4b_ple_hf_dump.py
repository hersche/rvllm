#!/usr/bin/env python3
"""HF Gemma 4 E4B-it reference-state dump for PLE diff harness.

Loads the model via `transformers`, runs a fixed short prompt, and
writes per-step f16 buffers that rvllm's `precompute_ple` +
`exec_layer` PLE injection are supposed to match.

Files written (all f16 little-endian, row-major, no header):

  e4b_token_ids.bin                          [T]                 i32
  e4b_inputs_embeds.bin                      [T, hidden=2560]    f16
    (post-embed-lookup × √hidden — what rvllm `residual_ptr` carries
     after EmbeddingGather and BEFORE the bf16 widen / PLE precompute)
  e4b_ple_lookup_pre_scale.bin               [T, num_layers * ple_dim]  f16
    (output of `get_per_layer_inputs` BEFORE the × √ple_dim scale)
  e4b_ple_lookup.bin                         [T, num_layers * ple_dim]  f16
    (output of `get_per_layer_inputs` AFTER  the × √ple_dim scale —
     this is what rvllm's loader bakes into embed_tokens_per_layer)
  e4b_ple_context_pre_norm.bin               [T, num_layers * ple_dim]  f16
    (after linear · model_projection × scale_proj, BEFORE RMSNorm)
  e4b_ple_context_post_norm.bin              [T, num_layers, ple_dim]  f16
    (after the shared-γ RMSNorm; reshape view)
  e4b_per_layer_inputs.bin                   [T, num_layers, ple_dim]  f16
    (final combine: (lookup + context_normed) × scale_input)

For each layer L in `--dump-layers` (default 0,1,5,11,41):

  e4b_layer{L}_residual_in.bin               [T, hidden]  f16
    (residual at start of layer L's forward)
  e4b_layer{L}_after_attn_add.bin            [T, hidden]  f16
    (HF: `hidden_states = residual + self.post_attention_layernorm(
     self.self_attn(self.input_layernorm(residual)))` — NO scalar
     yet, this is the input to the MLP residual sum)
  e4b_layer{L}_after_mlp_add.bin             [T, hidden]  f16
    (HF: after MLP residual add — NO scalar yet, this is the input
     to the PLE block)
  e4b_layer{L}_after_ple_add.bin             [T, hidden]  f16
    (HF: residual + PLE contribution — NO scalar yet)
  e4b_layer{L}_output.bin                    [T, hidden]  f16
    (HF: × layer_scalar — what feeds into layer L+1 as residual_in)

For diffing: rvllm writes the same names under
RVLLM_E4B_PLE_DUMP_DIR (todo: add in next commit). `cmp_e4b_ple.py`
(also todo) does row-wise cosine + max_abs per file and reports
first-divergence.

Run on GPU (~20 GiB free required):

  /home/r00t/.venv/bin/python3 v3/tools/gemma4_e4b_ple_hf_dump.py \
      --model-dir /home/r00t/gemma4-e4b \
      --prompt 'Wer bist du?' \
      --out-dir /tmp/e4b_hf_ple \
      --device cuda \
      --dump-layers 0,5,11,41

Or CPU (much slower but no GPU contention):

  ... --device cpu
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gemma4 import modeling_gemma4 as g4


def write_f16(path: Path, t: torch.Tensor):
    arr = t.detach().to(torch.float16).cpu().contiguous().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  wrote {path.name:<44} shape={tuple(arr.shape)} {arr.dtype}")


def write_i32(path: Path, t: torch.Tensor):
    arr = t.detach().to(torch.int32).cpu().contiguous().numpy()
    with open(path, "wb") as f:
        f.write(arr.tobytes())
    print(f"  wrote {path.name:<44} shape={tuple(arr.shape)} {arr.dtype}")


def patch_text_model(model: g4.Gemma4TextModel, out_dir: Path):
    """Capture the PLE precompute substeps inside the
    `project_per_layer_inputs` method by replacing it on the
    instance."""
    orig_project = model.project_per_layer_inputs

    def hooked_project(inputs_embeds, per_layer_inputs):
        write_f16(out_dir / "e4b_inputs_embeds.bin", inputs_embeds[0])
        write_f16(out_dir / "e4b_ple_lookup.bin", per_layer_inputs[0].reshape(
            per_layer_inputs.shape[1], -1
        ))
        # Replay the four steps of project_per_layer_inputs verbatim.
        plp = model.per_layer_model_projection(inputs_embeds)
        write_f16(
            out_dir / "e4b_ple_context_pre_scale.bin",
            plp[0].reshape(plp.shape[1], -1),
        )
        plp = plp * model.per_layer_model_projection_scale
        write_f16(
            out_dir / "e4b_ple_context_pre_norm.bin",
            plp[0].reshape(plp.shape[1], -1),
        )
        plp = plp.reshape(
            *inputs_embeds.shape[:-1],
            model.config.num_hidden_layers,
            model.hidden_size_per_layer_input,
        )
        plp = model.per_layer_projection_norm(plp)
        write_f16(out_dir / "e4b_ple_context_post_norm.bin", plp[0])
        combined = (plp + per_layer_inputs) * model.per_layer_input_scale
        write_f16(out_dir / "e4b_per_layer_inputs.bin", combined[0])
        return combined

    model.project_per_layer_inputs = hooked_project


def patch_decoder_layer(layer: g4.Gemma4TextDecoderLayer, layer_idx: int,
                         out_dir: Path):
    """Hook one decoder layer's forward to dump residual at each
    HF-canonical sub-step. We replicate the forward body verbatim
    so the hook captures intermediate values without depending on
    HF internal hook hygiene."""
    orig_forward = layer.forward

    def hooked_forward(
        hidden_states,
        per_layer_input=None,
        shared_kv_states=None,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        **kwargs,
    ):
        # Residual at start of layer.
        write_f16(
            out_dir / f"e4b_layer{layer_idx}_residual_in.bin",
            hidden_states[0],
        )
        # ── Attention residual sum (no scalar yet) ──
        residual = hidden_states
        hs = layer.input_layernorm(hidden_states)
        # HF input_ln checkpoint: matches rvllm's scratch.delta_f16
        # state right after RmsnormInplaceLaunch.
        write_f16(out_dir / f"e4b_layer{layer_idx}_input_ln.bin", hs[0])
        # HF qkv checkpoint: post-Q/K/V projection, layout concat
        # [Q (q_dim) | K (kv_dim) | V (kv_dim)] to match rvllm's
        # scratch.q_out f16 buffer after Bf16ToF16SatLaunch.
        sa = layer.self_attn
        q_full = sa.q_proj(hs)
        k_full = sa.k_proj(hs)
        v_full = sa.v_proj(hs)
        qkv = torch.cat([q_full, k_full, v_full], dim=-1)
        write_f16(out_dir / f"e4b_layer{layer_idx}_qkv.bin", qkv[0])
        # Per-head q_norm / k_norm in HF Gemma4Attention. Reshape Q
        # and K to [B, S, num_heads, head_dim] then apply the
        # head_dim-shape γ broadcast. Write flat [S, num_heads*head_dim]
        # to match rvllm's contiguous q_normed / k_normed layout.
        b, s, _ = q_full.shape
        num_h = sa.config.num_attention_heads
        num_kvh = sa.config.num_key_value_heads
        head_dim = sa.head_dim if hasattr(sa, "head_dim") else (q_full.shape[-1] // num_h)
        q_reshaped = q_full.view(b, s, num_h, head_dim)
        k_reshaped = k_full.view(b, s, num_kvh, head_dim)
        q_normed = sa.q_norm(q_reshaped)
        k_normed = sa.k_norm(k_reshaped)
        write_f16(
            out_dir / f"e4b_layer{layer_idx}_q_normed.bin",
            q_normed[0].reshape(s, num_h * head_dim),
        )
        write_f16(
            out_dir / f"e4b_layer{layer_idx}_k_normed.bin",
            k_normed[0].reshape(s, num_kvh * head_dim),
        )
        # Manually replay HF's attention math up to (but not
        # including) o_proj, so we can dump attn_out and compare
        # against rvllm's scratch.attn_out [T, q_dim] f16.
        try:
            import math as _math
            from transformers.models.gemma4.modeling_gemma4 import (
                apply_rotary_pos_emb,
            )
            cos, sin = position_embeddings
            # This transformers version's apply_rotary_pos_emb takes
            # ONE tensor at a time: f(x, cos, sin, unsqueeze_dim=1).
            # Our q_normed/k_normed are [B, S, H, D]; HF expects
            # [B, H, S, D] for RoPE with unsqueeze_dim=1.
            q_for_rope = q_normed.transpose(1, 2)  # [B, H, S, D]
            k_for_rope = k_normed.transpose(1, 2)  # [B, Hkv, S, D]
            q_rot = apply_rotary_pos_emb(q_for_rope, cos, sin)
            k_rot = apply_rotary_pos_emb(k_for_rope, cos, sin)
            # GQA expand: repeat each KV head num_h/num_kvh times.
            n_rep = num_h // num_kvh
            if n_rep > 1:
                k_rot_exp = k_rot.repeat_interleave(n_rep, dim=1)
                v_exp = v_full.view(b, s, num_kvh, head_dim).transpose(1, 2).repeat_interleave(n_rep, dim=1)
            else:
                k_rot_exp = k_rot
                v_exp = v_full.view(b, s, num_kvh, head_dim).transpose(1, 2)
            scale = 1.0 / _math.sqrt(head_dim)
            # Causal attention: [B, H, S, D] @ [B, H, D, S]
            scores = torch.einsum("bhsd,bhtd->bhst", q_rot.float(), k_rot_exp.float()) * scale
            mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=-1)
            attn = torch.einsum("bhst,bhtd->bhsd", probs, v_exp.float())
            attn = attn.transpose(1, 2).contiguous().view(b, s, num_h * head_dim).to(q_normed.dtype)
            write_f16(out_dir / f"e4b_layer{layer_idx}_attn_out.bin", attn[0])
        except Exception as _e:
            print(f"  attn_out manual replay failed: {_e}")
        hs, _ = layer.self_attn(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        # rvllm's scratch.attn_out is the attention output BEFORE
        # o_proj. HF's self_attn returns post-o_proj output, so we
        # need to back it out. attn_pre_o = self_attn output ×
        # o_proj.weight.pinv()? Easier: re-run via the attention
        # sub-modules. For first localization, write only the
        # post-o_proj output (= what HF returns) — rvllm's "attn_out"
        # is BEFORE o_proj; comparing requires HF re-execution that's
        # non-trivial. Skip until we have a stronger reason.
        hs = layer.post_attention_layernorm(hs)
        hs = residual + hs
        write_f16(out_dir / f"e4b_layer{layer_idx}_after_attn_add.bin", hs[0])
        # ── MLP residual sum (no scalar yet) ──
        residual = hs
        hs = layer.pre_feedforward_layernorm(hs)
        hs = layer.mlp(hs)
        # E4B has enable_moe_block=False — single MLP path. Don't
        # bother dumping the MoE branch.
        hs = layer.post_feedforward_layernorm(hs)
        hs = residual + hs
        write_f16(out_dir / f"e4b_layer{layer_idx}_after_mlp_add.bin", hs[0])
        # ── PLE block (E4B only) ──
        if layer.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hs
            hs = layer.per_layer_input_gate(hs)
            hs = layer.act_fn(hs)
            hs = hs * per_layer_input
            hs = layer.per_layer_projection(hs)
            hs = layer.post_per_layer_input_norm(hs)
            hs = residual + hs
            write_f16(out_dir / f"e4b_layer{layer_idx}_after_ple_add.bin", hs[0])
        # ── Final layer_scalar ──
        hs = hs * layer.layer_scalar
        write_f16(out_dir / f"e4b_layer{layer_idx}_output.bin", hs[0])
        return hs

    layer.forward = hooked_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/home/r00t/gemma4-e4b")
    ap.add_argument("--prompt", default="Wer bist du? Antworte in einem Satz.")
    ap.add_argument("--out-dir", default="/tmp/e4b_hf_ple")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--dump-layers", default="0,1,5,11,41",
                    help="comma-separated layer indices to dump")
    ap.add_argument("--prepend-bos", action="store_true", default=True,
                    help="prepend bos_token_id to match rvllm-serve's "
                         "/v1/completions tokenization (default ON)")
    ap.add_argument("--no-prepend-bos", dest="prepend_bos",
                    action="store_false",
                    help="don't prepend BOS (matches raw `tok(prompt)`)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_layers = sorted(int(x) for x in args.dump_layers.split(","))

    print(f"loading tokenizer + model from {args.model_dir} ({args.dtype} on {args.device})")
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # The text model is at model.model.language_model for the
    # Gemma4ForConditionalGeneration top-level wrapper. Locate it.
    text_model = (
        getattr(getattr(model, "model", model), "language_model", None)
        or getattr(model, "model", model)
    )
    print(f"text model class: {type(text_model).__name__}")
    print(f"num_hidden_layers={text_model.config.num_hidden_layers}, "
          f"hidden_size={text_model.config.hidden_size}, "
          f"ple_dim={getattr(text_model, 'hidden_size_per_layer_input', None)}")

    # Dump full RoPE cos/sin tables for sliding + global variants.
    # rvllm writes [max_pos, head_dim] f16 row-major; HF's
    # Gemma4RotaryEmbedding.forward returns (cos, sin) shaped
    # [batch, seq_len, head_dim]; call with position_ids = arange
    # to materialize the full table.
    try:
        max_pos = int(text_model.config.max_position_embeddings)
        head_dim_sliding = int(getattr(text_model.config, "head_dim", 256))
        rotary_sliding = getattr(text_model, "rotary_emb_local", None) or getattr(text_model, "rotary_emb", None)
        rotary_global = getattr(text_model, "rotary_emb", None)
        pos_ids = torch.arange(0, max_pos, device=args.device).unsqueeze(0)
        dummy_x = torch.zeros(1, 1, device=args.device, dtype=dtype_map[args.dtype])
        # rvllm stores [max_pos, head_dim/2] (single-copy, no
        # cat-doubling). HF returns [batch, seq, head_dim] with
        # `cat((freqs, freqs), dim=-1)` so the first `half` columns
        # are the canonical freqs. Slice to match rvllm.
        if rotary_sliding is not None:
            cos_s, sin_s = rotary_sliding(dummy_x, pos_ids)
            half_s = cos_s.shape[-1] // 2
            write_f16(out / "e4b_rope_cos_sliding.bin", cos_s[0, :, :half_s].to(torch.float16))
            write_f16(out / "e4b_rope_sin_sliding.bin", sin_s[0, :, :half_s].to(torch.float16))
            print(f"  dumped sliding cos/sin: shape={tuple(cos_s.shape)} half={half_s}")
        if rotary_global is not None and rotary_global is not rotary_sliding:
            cos_g, sin_g = rotary_global(dummy_x, pos_ids)
            half_g = cos_g.shape[-1] // 2
            write_f16(out / "e4b_rope_cos_global.bin", cos_g[0, :, :half_g].to(torch.float16))
            write_f16(out / "e4b_rope_sin_global.bin", sin_g[0, :, :half_g].to(torch.float16))
            print(f"  dumped global cos/sin: shape={tuple(cos_g.shape)} half={half_g}")
    except Exception as e:
        print(f"  WARN: rope table dump skipped: {e}")

    # Plain text encoder hooks.
    patch_text_model(text_model, out)
    for L in dump_layers:
        patch_decoder_layer(text_model.layers[L], L, out)

    # Tokenize + forward. `--prepend-bos` makes the input match
    # rvllm-serve's /v1/completions which prepends bos_token_id
    # (=2 on Gemma 4) automatically. With BOS, T=2 for "Hi" so
    # the per-row diff against rvllm's dump is direct (no offset).
    raw_ids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
    if args.prepend_bos and tok.bos_token_id is not None:
        bos = torch.tensor([[tok.bos_token_id]], dtype=raw_ids.dtype)
        raw_ids = torch.cat([bos, raw_ids], dim=1)
    inputs = {"input_ids": raw_ids.to(args.device)}
    print(f"tokenized: shape={tuple(inputs['input_ids'].shape)} ids={inputs['input_ids'][0].tolist()[:32]}")
    write_i32(out / "e4b_token_ids.bin", inputs["input_ids"][0])

    with torch.no_grad():
        out_obj = model(**inputs)
    if hasattr(out_obj, "logits"):
        write_f16(out / "e4b_final_logits.bin", out_obj.logits[0])

    print(f"done — dumps under {out}")


if __name__ == "__main__":
    main()
