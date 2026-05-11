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

### Phase 2 — Decoder forward (text-only) — multi-session

Breaking down into measurable sub-phases:

  **Phase 2a — Forward-kernel registration**
  Reuse `Qwen36OutsideKernels` directly. The Qwen 3.6 kernel set
  is shape-generic; the only Qwen-3.6-only entries are the MoE
  router / topk / shared-gate / indirect-FP8-GEMV kernels which
  Qwen 3.5 simply doesn't dispatch. Loading those kernels anyway
  costs ~0 (PTX modules cached, no GPU memory until launched).

  Concrete deliverable: refactor `Qwen36Bringup::load`'s inline
  PTX-loading block into a `pub fn load_outside_kernels(kernels:
  &KernelLoader) -> Qwen36OutsideKernels`, then call it from
  `Qwen35Bringup::load`. Risk: medium — touches a working
  ~500-LOC block in qwen36_bring_up. Mitigation: extract is pure
  motion + compile-tested.

  **Phase 2b — Scratch + KV cache + RoPE tables**
  * `Mistral35Scratch`-style struct with per-decode-token buffers
    (h_residual, h_work, q_out, k_out, v_out, attn_out, o_out,
    gate_out, up_out, silu_mid, down_out, logits, token_out).
  * KV cache: BF16 [max_pos, n_kv_heads, head_dim] per FULL-attn
    layer (16 total at 27B). Linear-attn layers carry SSM state
    instead (per-head `[head_dim_v, head_dim_v]` covariance);
    Qwen 3.6 already allocates this — reuse helper.
  * RoPE tables (cos/sin): `rotary_dim = head_dim *
    partial_rotary_factor = 256 * 0.25 = 64`. Build the
    `[max_pos, rotary_dim/2]` cos/sin tables host-side from
    `rope_theta = 1e7`; upload as F16Weight. Mirrors Qwen 3.6's
    pattern in `qwen36_bring_up::Qwen36RopeTables`.

  **Phase 2c-A — Outside-only smoke forward (DONE — 9164ccf)**
    HTTP-reachable forward that drives
      embed → final-RMSNorm-with-FP8-quant → cuBLASLt fp8_gemm
        → argmax_f16 → DtoH
    skipping all 64 transformer layers. Output token is structurally
    valid but semantically meaningless. Proves the kernel-load +
    scratch + cuBLASLt + HTTP roundtrip end-to-end.

  **Phase 2c-B — Per-layer forward (NEXT)**

  Kernel inventory needed for the dispatch loop. Every entry is
  already a `kernels/*.ptx` file (loaded today by Qwen 3.6); add a
  field+load line per kernel into `Qwen35OutsideKernels`:

    Full-attn (16 layers) + outside path adds:
      rmsnorm_inplace_f16          — input_layernorm, post_attn_ln,
                                       q_norm, k_norm.
      fp8_gemv_blockwise_*          — q_proj, k_proj, v_proj, o_proj
                                       (blockwise = block-128
                                       scale_inv layout).
      split_q_gate_f16              — split q_proj's interleaved
                                       [n_heads, 2*head_dim] output
                                       into q + gate halves.
      fused_rope_qwen_partial_f16kv — partial RoPE (rotary_dim=64
                                       of head_dim=256) + KV-write
                                       fused.
      flash_attention               — FA-2 decode with F16 KV.
      sigmoid_mul_f16               — applies the q_gate to attn out
                                       before o_proj.

    Linear-attn (48 layers) adds:
      conv_state_advance_f16        — conv1d state slide + history
                                       assembly.
      causal_conv1d_f16             — depthwise causal 1D conv.
      qwen_linear_alpha_beta_f16    — alpha/beta scalars from
                                       in_proj_a/b.
      qwen_linear_silu_l2_gqa_f16   — silu + Q/K L2 norm + GQA expand
                                       + V silu-pack.
      gated_delta_rule_decode_f16   — Gated-DeltaNet decode-step
                                       (one-launch state update +
                                       readout).
      qwen_linear_rmsnorm_gated_f16 — per-v-head RMSNormGated with
                                       silu(z) gate.

    Dense MLP (every layer) adds:
      fp8_gemv_blockwise_wpr_native_f16in_dual_silu — fused
                                       (gate FP8 GEMV) + (up FP8
                                       GEMV) + silu_mul → one
                                       launch per layer for the
                                       gate+up halves.
      fp8_gemv_blockwise_wpr_native_f16in           — single FP8
                                       GEMV for down_proj (blockwise
                                       scale).
                                       Note: Qwen 3.6 only loads the
                                       dual/dual_silu/indirect
                                       variants; the single-blockwise
                                       wpr_native_f16in needs adding
                                       OR the down_proj can route
                                       through a non-fused dual call
                                       with the second weight zeroed
                                       (waste). Best to add the
                                       single-GEMV PTX entry.

  Per-layer dispatch (`forward_qwen35_decode(token, position)`):
    h_residual ← embed_gather(token)
    for layer in 0..64:
      h_work ← rmsnorm(h_residual, input_layernorm)
      attn_out ← match layer_types[layer]:
        Full        → full_attn(h_work, layer, position)
        Linear      → linear_attn(h_work, layer)
      h_residual += attn_out                          # post-attn residual
      h_work ← rmsnorm(h_residual, post_attn_ln)
      h_residual += dense_mlp(h_work, layer)          # post-mlp residual
    h_work ← rmsnorm(h_residual, final_norm)
    logits ← lm_head FP8 GEMM(h_work)
    return argmax(logits)

  **Phase 2c-C — generate + prefill loop**
    Mirrors Mistral 3.5 / Qwen 3.6's `generate_with_prompt` —
    HtoD prompt tokens, per-position forward to seed KV cache +
    linear-attn state, then decode loop.

  Down-projection note: Qwen 3.6 doesn't ship a single-output
  blockwise FP8 GEMV (only dual / dual_silu / indirect); for
  Qwen 3.5's `mlp.down_proj` and the full-attn `o_proj` /
  `linear_attn.out_proj` we route through cuBLASLt's
  `fp8_gemm_blockwise` at M=1 (the same path Qwen 3.6 / Gemma 4
  use for blockwise FP8 GEMM elsewhere). Adding a dedicated
  single-GEMV PTX is a Phase 5 perf opt.

  Verified PTX availability (Phase 2c-B prerequisites):
  ✓ rmsnorm_inplace_f16
  ✓ split_q_gate_f16
  ✓ fused_rope_qwen_partial_f16kv
  ✓ flash_attention   (FA-2 decode kernel)
  ✓ sigmoid_mul_f16
  ✓ conv_state_advance_f16
  ✓ causal_conv1d_f16
  ✓ qwen_linear_alpha_beta_f16
  ✓ qwen_linear_silu_l2_gqa_f16
  ✓ gated_delta_rule_decode_f16
  ✓ qwen_linear_rmsnorm_gated_f16
  ✓ fp8_gemv_blockwise_wpr_native_f16in_dual_silu
  ✗ fp8_gemv_blockwise_wpr_native_f16in (single)
       → cuBLASLt fp8_gemm_blockwise fallback for now;
         dedicated PTX in Phase 5 perf pass.

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
