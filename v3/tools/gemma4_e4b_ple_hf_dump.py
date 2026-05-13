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
        hs, _ = layer.self_attn(
            hidden_states=hs,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
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

    # Plain text encoder hooks.
    patch_text_model(text_model, out)
    for L in dump_layers:
        patch_decoder_layer(text_model.layers[L], L, out)

    # Tokenize + forward
    inputs = tok(args.prompt, return_tensors="pt").to(args.device)
    print(f"tokenized: shape={tuple(inputs.input_ids.shape)} ids={inputs.input_ids[0].tolist()[:32]}")
    write_i32(out / "e4b_token_ids.bin", inputs.input_ids[0])

    with torch.no_grad():
        out_obj = model(**inputs)
    if hasattr(out_obj, "logits"):
        write_f16(out / "e4b_final_logits.bin", out_obj.logits[0])

    print(f"done — dumps under {out}")


if __name__ == "__main__":
    main()
