# Qwen 3.5 27B Dense — Bring-up Plan

Sibling family to the existing Qwen 3.6 35B-A3B (MoE). The dense
27B checkpoint at `/home/r00t/qwen36-27b-fp8` is the canonical
target. This plan tracks the multi-week phased bring-up.

## On-disk shape (already verified by `Qwen35Arch::from_dir`)

* `model_type == "qwen3_5"` (top-level + text_config).
* 64 hidden layers, hidden=5120, intermediate=17408 (dense MLP).
* num_attention_heads=24, num_key_value_heads=4, head_dim=256
  → GQA ratio = 6.
* `attn_output_gate=true` (q_proj output width = 2 × heads × hd
  = 12288; second half is the output gate, post-sigmoid mixed
  with the attention output before o_proj).
* `layer_types`: 3 × `linear_attention` followed by
  `full_attention`, repeating 16 times. 48 linear + 16 full.
* FP8 e4m3 with BF16 `weight_scale_inv` blockwise scales,
  block size 128 (same format as Qwen 3.6).
* RoPE `rope_theta = 5_000_000` (read from
  `text_config.rope_parameters.rope_theta`).
* Vocab 248320 (shared Qwen tokenizer).
* MTP head (1 layer) present — speculative-decoding head, parked
  for now.
* Vision tower: Qwen3-VL ViT (27 blocks, hidden=1152,
  intermediate=4304, num_heads=16, patch_size=16,
  spatial_merge_size=2, out_hidden_size=5120). Identical shape
  to the Qwen 3.6 ViT — pure reuse.

Per-layer weight names (full-attn layer 3, sample):

```
layers.3.input_layernorm.weight                    BF16 [5120]
layers.3.self_attn.q_proj.weight                   FP8  [12288, 5120]
layers.3.self_attn.q_proj.weight_scale_inv         BF16 [96, 40]
layers.3.self_attn.k_proj.weight                   FP8  [1024, 5120]
layers.3.self_attn.k_proj.weight_scale_inv         BF16 [8, 40]
layers.3.self_attn.v_proj.weight                   FP8  [1024, 5120]
layers.3.self_attn.v_proj.weight_scale_inv         BF16 [8, 40]
layers.3.self_attn.o_proj.weight                   FP8  [5120, 6144]
layers.3.self_attn.o_proj.weight_scale_inv         BF16 [40, 48]
layers.3.self_attn.q_norm.weight                   BF16 [256]
layers.3.self_attn.k_norm.weight                   BF16 [256]
layers.3.post_attention_layernorm.weight           BF16 [5120]
layers.3.mlp.gate_proj.weight                      FP8  [17408, 5120]
layers.3.mlp.up_proj.weight                        FP8  [17408, 5120]
layers.3.mlp.down_proj.weight                      FP8  [5120, 17408]
layers.3.mlp.{gate,up,down}_proj.weight_scale_inv  BF16 (block 128)
```

Per-layer weight names (linear-attn layer 0):

```
layers.0.linear_attn.in_proj_qkv.weight   FP8  [10240, 5120]
layers.0.linear_attn.in_proj_z.weight     FP8  [6144,  5120]
layers.0.linear_attn.in_proj_a.weight     BF16 [48, 5120]
layers.0.linear_attn.in_proj_b.weight     BF16 [48, 5120]
layers.0.linear_attn.conv1d.weight        BF16 [10240, 1, 4]
layers.0.linear_attn.A_log                BF16 [48]
layers.0.linear_attn.dt_bias              BF16 [48]
layers.0.linear_attn.norm.weight          BF16 [128]
layers.0.linear_attn.out_proj.weight      FP8  [5120, 6144]
+ same dense MLP as full-attn layers
```

`outside.safetensors` carries `embed_tokens` / `lm_head` (both
BF16, [248320, 5120]), `model.norm.weight`, the 27 ViT blocks
(BF16, no FP8), patch_embed, pos_embed, and the patch merger.

## Phase plan

### Phase 0 — Foundation (DONE)
* `qwen35_arch.rs` with `Qwen35Arch::from_dir`.
* `ModelFamily::Qwen35` + `VisionArch::Qwen35` enum entries.
* Family resolver dispatch (auto-detects before Qwen 3.6).
* `Qwen35Bringup::load` stub — validates arch + logs summary.
* cuda_worker intercept emits a typed error on every request.
* Exhaustive match arms across rvllm-serve (handlers, tokenize,
  cuda_worker, main).
* Branch: `rusty_sm121_qwen36_26b` (legacy name; the model itself
  is 27B dense). Forked from `rusty_sm121_mistral @ 6061b8f`.

### Phase 1 — FP8 weight loader (NEXT)
* `qwen35_weights.rs` — `Qwen35LoadedModel` + linear-attn /
  full-attn / dense-MLP per-layer struct. Sibling of
  `qwen36_weights` minus the MoE expert tensors.
* `qwen35_load.rs` — safetensors index walker that handles the
  `layers-{i}.safetensors` per-layer shards plus
  `outside.safetensors`. Reuses `Fp8LinearWeight` from the loader.
* Inventory validator: 64 layers × correct kind, vision blocks,
  embed/head, MTP optional.

### Phase 2 — Decoder forward (text-only)
* Engine state: arena, KV cache (matching `layer_types`),
  scratch buffers, runtime flags.
* Full-attn layer: RMSNorm → fused QKV (FP8 GEMV at decode, FP8
  GEMM at prefill) → q_norm/k_norm → RoPE → KV-write → attention
  (FA-2 BF16 KV) → o_proj with output-gate split → residual.
* Linear-attn layer: in_proj_qkv + in_proj_z + in_proj_a/b →
  conv1d → Gated DeltaNet (reuse Qwen 3.6 kernels) → out_proj.
* Dense MLP: gate_proj + up_proj + silu_mul + down_proj (steal
  the Mistral 3.5 dense MLP code path; identical shape).
* Embed + final RMSNorm + lm_head.
* Smoke: short-prompt greedy → first sane German/English token.

### Phase 3 — Vision tower
* Port the Qwen 3.6 ViT forward (27 blocks identical shape).
* Patch merger (2×2 spatial). Output [num_tokens, 5120] BF16.
* Image preprocessing reuses Qwen pipeline.

### Phase 4 — Splice + serve
* `Qwen35Bringup::generate_with_vision_slots` mirroring
  `Mistral35Bringup::generate_with_vision_slots` — device-resident
  splice region, slot-aware row routing.
* cuda_worker dispatch: replace the Phase 0 error sink with the
  real engine.

### Phase 5 — Perf
* FA-decode + batched prefill + fused QKV/gate_up gemv (port
  from Mistral 3.5; identical shapes for the GEMV side).
* W4A16-equivalent for FP8 (mistral35 V7 with cp.async is
  reusable since the block-scale layout matches).
* Codex round on the new arch.

## Markers + invariants

* `model_type=qwen3_5` is the strict gate; `attn_output_gate=true`
  + dense MLP (`num_experts` absent) discriminates from Qwen 3.6.
* Branch name `rusty_sm121_qwen36_26b` is historical — the model
  is actually Qwen 3.5 27B dense. Kept as-is at the user's request.
