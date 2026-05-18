# CLAUDE.md — rvllm-serve

Repo-local guidance. Pair with `~/CLAUDE.md` (system-wide context:
services, profiles, brain ecosystem, model paths) and
`llm_instructions_sm121.md` (algorithmic picture of the NVFP4
pipeline). This file covers what's specific to working **inside
this tree**.

## Active branch

`rusty_sm121_vision` — the only branch you should push to.

Forked from `rusty_sm121_inference_server` (now the merge target /
fallback). The older `rusty_sm121_nvfp4` and `rusty_sm121` branches
are frozen — do not cherry-pick from them.

## Layout

```
v3/                        Cargo workspace (all binaries + libraries)
  crates/
    rvllm-core/            errors, IDs, RvllmError
    rvllm-mem/              CUDA arena (HBM + Unified for GB10)
    rvllm-cutlass/         cuBLASLt + CUTLASS SM120 FP8 GEMM bindings
    rvllm-attention/       FA2/FA3 wrappers, paged decode launchers
    rvllm-fused/           fused kernels (RoPE+pack, RMSNorm+QKV, …)
    rvllm-metadata/        config.json + tokenizer config
    rvllm-graph/           CUDA Graph capture/replay
    rvllm-sampling/        sampling + repetition penalty
    rvllm-loader/          weight loading (fp8-block, NVFP4)
    rvllm-runtime/         engine, scheduler, bring-up (Llama,
                           Qwen3-VL, Gemma4)
    rvllm-serve/           OpenAI HTTP server (axum), tokenize,
                           chat templates, vision-aware admission
kernels/                    raw .cu sources + sm_121 PTX output
  build.sh                  rebuilds PTX + manifest.json (ALWAYS
                            re-run after a kernel edit)
  build_cutlass_sm120_so.sh CUTLASS .so build (use sm_121a on GB10,
                            sm_120a on RTX 5090 / 6000 Blackwell)
  sm_121/                   per-arch PTX + manifest + libcutlass.so
```

## Build commands

```bash
cd v3 && cargo build --release --bin rvllm-server --features cuda,gb10
bash kernels/build.sh sm_121
```

**Both `cuda` and `gb10` features are required** on GB10. Without
`gb10` the bring-up takes the FA3-SM90 path and crashes with
`Fa3SoMissing`.

After every kernel edit, **re-run `kernels/build.sh sm_121`** —
otherwise the next service start picks the stale PTX. Manifest drift
between binary `RVLLM_BUILD_REVISION` and `manifest.json` is a real
class of bug; the rule is: always rebuild PTX as the LAST step
after the final commit in a chain.

## Runtime profile + service

Active profile lives at `/home/r00t/.rvllm/active-profile.env`
(symlink). Available profiles in `/home/r00t/.rvllm/profiles/`:

- `mobile-31b-rvllm.env`         — Gemma 4 31B fp8-block (default)
- `mobile-31b-rvllm-nvfp4.env`   — Gemma 4 31B NVFP4 KV
- `mobile-qwen-rvllm.env`        — Qwen 3.6 35B-A3B fp8 + vision (F16 KV)
- `mobile-qwen-rvllm-nvfp4.env`  — Qwen 3.6 35B-A3B fp8 + vision + NVFP4 KV (validated 2026-05-15: text + ball.png coherent on hardware; 3.5× KV mem reduction)
- `mobile-qwen35-rvllm.env`      — Qwen 3.5 27B dense (F16 KV) + Qwen3-VL vision
- `mobile-qwen35-rvllm-nvfp4.env` — Qwen 3.5 27B dense NVFP4 KV + Qwen3-VL vision (validated 2026-05-14: text + ball.png coherent on hardware)
- `mobile-e4b-rvllm.env`         — Gemma 4 E4B-it bf16->f16 + native audio (F16 KV)
- `mobile-e4b-rvllm-nvfp4.env`   — Gemma 4 E4B-it NVFP4 KV (validated 2026-05-15: text + SigLIP vision + audio all coherent; needs `RVLLM_NVFP4_HADAMARD=0`)
- `mobile-mistral35-rvllm.env`   — Mistral 3.5 128B NVFP4 + Pixtral vision
- `combo-*`, `creative-*`, `work-*` — Rusty mode-switch variants

**Per-family debug-flag gates.** Some families have opt-in debug
knobs that ALTER the forward path (single-token cap, KV bypass,
RoPE-position override, layer dumps). They MUST be off in production
profiles or the model returns capped/incorrect output. Each gate
also rejects stale debug envs at startup so a stale debug-session
env cannot leak into prod:

- Mistral 3.5 (`mistral35`): `RVLLM_DEBUG_MISTRAL35=1` is the
  umbrella gate. `RVLLM_SMOKE_MAX_NEW`, `RVLLM_KV_BYPASS`,
  `RVLLM_SMOKE_ROPE_POS_OVERRIDE`, `RVLLM_SMOKE_FULL_DUMP`,
  `RVLLM_SMOKE_ATTN_NO_PAST`, `RVLLM_BOUNDARY_DUMP*`,
  `RVLLM_SMOKE_LAYER_RMS`, `RVLLM_SMOKE_DUMP_DIR`,
  `RVLLM_SMOKE_NO_RESTORE`, `RVLLM_SMOKE_SINGLE` are inert without
  the gate. Production profile must omit ALL of these. See
  `v3/crates/rvllm-runtime/src/mistral35_bring_up.rs::validate_no_stale_debug_envs`.

Switch + restart:
```bash
sudo ln -sfn /home/r00t/.rvllm/profiles/<profile>.env \
             /home/r00t/.rvllm/active-profile.env
sudo systemctl restart rvllm-serve
```

## NVFP4 KV cache coverage

| Family | Module | NVFP4 KV status |
|---|---|---|
| Gemma 4 31B | `gemma4_bring_up.rs` | wired; production NVFP4 profile (`mobile-31b-rvllm-nvfp4.env`). Hadamard rotation + per-token Q scale + amax6 V policy. |
| Gemma 4 E4B-it | `gemma4_bring_up.rs` | wired (`mobile-e4b-rvllm-nvfp4.env`). Validated 2026-05-15 with the full E4B modality stack — text + SigLIP vision (256 ViT tokens spliced, "orangefarbene kreisförmige Sonne" on ball.png) all coherent on the NVFP4 KV path. Vision/audio splice and NVFP4 attention are orthogonal code paths, so the combo required no new wiring. Family-specific knob: `RVLLM_NVFP4_HADAMARD=0` (E4B's smaller hidden=2560 + GQA-4 distribution makes Hadamard rotation degrade quality — opposite of 31B where it's required). |
| Mistral 3.5 128B | `mistral35_bring_up.rs` | NVFP4 weights + NVFP4 KV (active production profile). |
| Qwen 3.5 27B dense | `qwen35_bring_up.rs` | **wired 2026-05-14** (`mobile-qwen35-rvllm-nvfp4.env`). Five-step landing: KV allocator + dtype field, Qwen-specific `fused_rope_qwen_partial_nvfp4kv` kernel (NeoX partial RoPE with rotary_dim=64, amax6 V policy, no Hadamard), kernel load + Q-side scratch, decode dispatch (per-head FA-2 NVFP4, no GQA cap), prefill via per-token decode fallback. Validated on hardware: text + Qwen3-VL vision. F16 path bit-identical when `RVLLM_NVFP4_KV` is unset. |
| Qwen 3.6 35B-A3B | `qwen36_bring_up.rs` | **wired 2026-05-15** (`mobile-qwen-rvllm-nvfp4.env`). 4-commit port (~384 LOC, 0dd5d98..d31b8ae): layout plumbing, kernel load, decode dispatch, prefill fallback. Both kernels (`fused_rope_qwen_partial_nvfp4kv` + `flash_attention_2_decode_nvfp4kv_kernel`) reused as-is from the Qwen 3.5 work — only the dispatch wiring is per-family. Per-head decode handles GQA=8 without split-decode. Batched prefill flips to per-token loop on Nvfp4; unified-NVFP4-prefill (PTX exists) is a follow-up. Validated on hardware: text (German ghost joke 31 tok in 1.07s) + Qwen3-VL vision (same caption as F16 baseline). KV memory at 4096 ctx: 20 MiB packed + 2.5 MiB scales vs 80 MiB F16 (3.5× reduction). F16 path bit-identical when the gate is off. |

## Speculative decoding — Gemma 4 family (state as of 2026-05-17)

Active branch: `rusty_sm121_qwen36_26b` (head `2693894`). Not
yet merged into `rusty_sm121_vision`.

### E4B-it spec — production, 2.46× faster than non-spec

`mobile-e4b-rvllm-spec.env` (or any profile with
`RVLLM_GEMMA4_SPEC_DECODE=1` +
`RVLLM_GEMMA4_DRAFTER_DIR=/home/r00t/gemma4-e4b-assistant`,
`RVLLM_GEMMA4_SPEC_K=4`). Greedy temp=0 only (typical-acceptance
parked, see commit message of `689fba3`). Hardware-validated:

* 3 factual prompts byte-identical to non-spec E4B
  ("Die Hauptstadt von Frankreich ist **Paris**.", etc.).
* 80-token counting prompt: non-spec ~11.0 s vs spec ~4.47 s,
  `accepted_per_verify=1.5`, `drafter_avg_ms=6.9`,
  `verify_avg_ms=62.6`, `prefill_one_avg_ms=67.5`.
* Drafter forward uses the centroid masked-embedding head
  (drafter ships `use_ordered_embeddings=true`).
* ZeroClaw config (`~/workspace/data/zeroclaw/config.toml`)
  is tied to this stack: `default_model=gemma-4-e4b-it`,
  `default_temperature=0.0` (spec session loop is greedy-only).

Key e4b commits (top → bottom):
* `689fba3` — fix accept_len>0 branch: classical bonus was being
  committed (prefill_one + shadow KV) but never pushed to emitted
  / on_token. After commit, push the bonus. THIS is what unlocked
  the speedup.
* `dea6f54` — drop sqrt(hidden_size) embed-half scale (the
  earlier `apply_pre_projection_embed_scale` call dragged the
  drafter into degenerate prediction mode for E4B).
* `6d254b2` — defensive `force_common_prefix_override` swap-at-top
  in `run_generate` + worker disarms all spec hook atomics on
  non-spec branch.

### 31B-it spec — wired, drafter accept_rate=0, root cause LIKELY V-cache on `attention_k_eq_v=true` global layers

Profile: `mobile-31b-rvllm-spec.env` (drafter dir
`/home/r00t/gemma-4-31B-it-assistant`, K=4, perf trace ON).
What works:

* Resolver `Gemma4Arch::assistant_shared_kv_sources()` returns
  `(58, 59)` for 31B (last sliding + last full across full layer
  range; E4B path unchanged → still `(22, 23)`). Tests cover
  Some(0), None, E4B, one-type-missing variants.
* Loader: `use_ordered_embeddings` parsed at top-level config;
  centroid + token_ordering tensors are `Option<u64>` and only
  required when the flag is true. 31B drafter ships 48 tensors
  (no centroid tensors) and loads cleanly (895.5 MiB resident,
  hidden=1024, 4 layers).
* Runtime: when `use_ordered_embeddings=false` the drafter LM
  head branches to a full-vocab tied path:
  `f16_gemm_f32(workspace.hidden, drafter.top.embed_tokens,
   workspace.gemm_f32, 1, vocab, hidden)` + `ArgmaxLaunch{
   num_tokens=1, vocab}` over the f32 logits →
  `workspace.out_token_id`. `max_projection` includes
  `vocab_size` when the flag is off so `gemm_f32` fits 1 MiB of
  f32 logits.
* Hardware: 31B spec produces **correct output** via the
  verify-fallback path ("Die Hauptstadt von Frankreich ist
  **Paris**."), but `accepted_per_verify = 0.0` always.
  Drafter wall: ~28 ms/iter (vs E4B's ~7 ms).

What we ruled out (all `accept_rate=0` regardless):
* Source-pair: tried `(58,59) (52,53) (58,53) (52,59) (4,5)`
  via `RVLLM_GEMMA4_SPEC_SOURCE_PAIR=<s>,<f>`. None help.
* Embed-half scale: tried both with + without
  `apply_pre_projection_embed_scale(sqrt(backbone))` via
  `RVLLM_GEMMA4_SPEC_EMBED_SCALE_31B`. With scale: drafter is
  random varied tokens. Without scale: drafter is degenerate
  (always token 2313). NEITHER matches. The scale knob is in
  fact WRONG (codex Q1(b) confirms HF's only embed scale is
  baked into the loader's `target_model_input_embeddings`, which
  rvllm already applies via `Gemma embedding scale: sqrt(H)` at
  load time). The knob double-scales — **default it OFF and
  retire when the real fix lands**.
* Q-bisect probe (`RVLLM_SPEC_DEBUG_Q_BISECT=1`): layer-0 Q-side
  healthy (post_q_norm RMS ≈ 1.02, post_q_proj max ~95).
* Layer-trace (`RVLLM_GEMMA4_SPEC_LAYER_TRACE=1`): per-drafter-
  layer `attn_out(pre_oproj)` RMS ~0.5 across ALL 4 layers
  (varies only ~0.02 between k-steps), `post_residual1` RMS 7-9
  (residual dominates by 10×+), `post_mlp` RMS 1.4-2.0,
  `post_final_norm` RMS 5.3 with near-constant direction across
  iters → LM-head argmax stuck in a narrow vocab region.
* Trained `layer_scalar` values (non-1.0:
  31B [0.146, 0.578, 0.613, 0.520], E4B [0.032, 0.205, 0.359,
  0.164]) are correctly applied via `scale_inplace`.
* BC=16 flash-attn at head_dim=512 + GQA=8: kernel's
  `kv_head = head_idx / (num_heads / num_kv_heads)` handles
  ratio 8 generically; smem at D=512/BC=16 fits sm_121's
  ~64 KiB budget.
* My full-vocab LM head: exact same kernel pair the BASE LM head
  uses; argmax kernel correctly handles `vocab > blockDim.x`.

### Root-cause hypothesis (codex Round 2, 2026-05-17)

For base global layers with `attention_k_eq_v=true` (Gemma 4 31B
layer 59 is one such layer), HF's reference at
`models/gemma4/modeling_gemma4.py:1203-1207` and `1241-1254`:

1. computes `key_states` via `k_proj`,
2. sets `value_states = key_states` (= K **before** any K-norm
   or RoPE),
3. applies V's own `value_norm` to `value_states` SEPARATELY,
4. writes BOTH `K_post_norm_rope` AND `V_post_value_norm` to the
   KV cache.

K and V in the cache are therefore **NOT byte-identical** even
though they share a single source projection. rvllm's base path
likely writes V=K bit-identical (or skips V entirely and aliases
at read time). When the drafter cross-attends to the FULL source
layer 59 via `populate_shadow_kv_range_from_base`, it reads
`v_cache = k_cache + half_bytes` — wrong content → wrong V →
degenerate drafter output. The cross-attn output is small (~RMS
0.5) because the K↔V cancellation in softmax(QK)·V never produces
the right semantic.

This explains ALL the observed symptoms:
* drafter forward runs structurally, no crashes
* attn_out RMS is non-zero but tiny + near-constant
* drafter LM-head input direction is fixed → argmax stuck
* changing source pair doesn't help (the bug is per-layer V
  content, not which layer to read)
* changing embed scale doesn't help (the bug is downstream of
  the attn stage)

The E4B base has `attention_k_eq_v=true` on its global layers
too, but the E4B drafter cross-attends to layers (22, 23) — both
INSIDE the non-shared prefix; we have not independently verified
whether E4B's V-shadow at layer 23 is HF-faithful (it might be
the same bug, just compensating in E4B because the drafter is
smaller and the centroid-masked LM head clips to top-K).

### Round 3 (2026-05-17) — V-cache hypothesis disproved + new evidence

Read-only investigation refuted the Round 2 V-cache hypothesis:

* HF `models/gemma4/modeling_gemma4.py:1198` defines `v_norm =
  Gemma4RMSNorm(head_dim, eps, with_scale=False)` — **V-norm IS
  parameter-free**. Not gamma-applied as Round 2 implied. rvllm's
  `fused_qkv_rmsnorm.cu` V-head branch sets `use_gamma=false` →
  matches HF. K_cache and V_cache differ in BOTH frameworks (K
  has gamma+RoPE, V has parameter-free norm only).
* `fused_rope_partial_nvfp4kv.cu` writes K and V to NVFP4
  cache independently, with separate scale policies. No k_eq_v
  shortcut. V-cache content is HF-faithful.

So V-cache is correct. The bug is elsewhere.

NEW evidence from this session (all gates leave production
untouched, env-gated probes only, no code changes):

* **KV-dtype invariance**: FP8-KV (`mobile-31b-rvllm-fp8kv-spec.env`)
  vs NVFP4-KV (`mobile-31b-rvllm-spec.env`) produce BIT-IDENTICAL
  drafter outputs:
  ```
  iter=0  drafts=[2021, 3050, 506, 1638]   base_argmax_k=[229912, 506, 3890, 3890]
  iter=2  drafts=[3946, 14423, 241113, 237009] base_argmax_k=[3564, …]
  ```
  Same in both runs. So either KV quant noise is below the
  LM-head argmax threshold, or the drafter cross-attn
  contribution is too small to register at argmax level.
* **Cross-attn scale insensitivity**: `RVLLM_SPEC_FA_SCALE=mtp`
  (scale=1.0) vs `stable` (1/sqrt(head_dim)) — iter 0 IDENTICAL
  drafts, iter 1+ differs but accept_rate stays 0.0.
* **Source-pair matrix**: 5 candidates × 2 scale states × 2 KV
  dtypes = 20 cells. Every cell accept_rate=0.0.

The drafter is generating tokens that are FORMALLY VARYING
(non-degenerate after embed-scale ON; degenerate when off) but
NEVER matching the base's predictions. The drafter forward is
running structurally but its prediction is qualitatively wrong.

Remaining hypothesis space:
1. Drafter weight upload has a subtle layout issue specific to
   31B's larger dims (q_proj [16384,1024] full layer, embed_tokens
   [262144,1024]) that the e4b path doesn't exercise.
2. Drafter `pre_projection` input layout (the cat[embed, hidden]
   order or per-half magnitudes) doesn't match what the 31B
   drafter was trained against.
3. The drafter's predictions DO carry useful signal but the
   LM-head argmax over full vocab=262144 picks the wrong token
   because of magnitude/scale issues unique to the
   use_ordered_embeddings=false path.

Codex Q7's surgical step (HF-vs-rvllm side-by-side dump at
`pre_projection_in` and `run_drafter_pre_projection` output) is
the next decisive move. Requires loading 31B base + drafter in
HF transformers Python, capturing the tensor at iter 0 for the
same prompt, and comparing byte-for-byte with rvllm's dumps.

Production state: rvllm-serve currently STOPPED (user instructed
"do not restart any services" for development); pre-session
state had e4b-spec live with the full brain stack. Switch back
when ready: `sudo ln -sfn /home/r00t/.rvllm/profiles/
mobile-e4b-rvllm-spec.env /home/r00t/.rvllm/active-profile.env
&& sudo systemctl start rvllm-serve vllm-embedding zeroclaw
whisper-fast chatterbox`.

New file: `~/.rvllm/profiles/mobile-31b-rvllm-fp8kv-spec.env`
(31B fp8-block weights + FP8 KV + spec) created for the
KV-dtype-invariance test; kept on disk as a baseline for future
work.

### Round 4 (2026-05-17) — Q content effectively ignored, scale arg works

New evidence from this session (commit `12970cd` adds the
`RVLLM_SPEC_ZERO_Q` knob in the batched session loop):

* `RVLLM_SPEC_ZERO_Q=1` (zero workspace.q immediately before
  cross-attn) → drafts BIT-IDENTICAL to non-zero Q.
* `RVLLM_SPEC_Q_ROPE_MODE` ∈ {current, pos0, no_rope} → all three
  produce IDENTICAL drafts (pre-existing knob, same result).
* `RVLLM_GEMMA4_SPEC_LAYER_TRACE` cross-check: attn_out RMS DOES
  differ between Q-zeroed (0.61) and Q-normal (0.51) — the
  cross-attn IS consuming Q. The difference is just too small
  to flip the LM-head argmax over 262144 vocab.

Codex Round 4 (cited HF source for every claim) ruled out most
hypotheses and pinned ONE concrete HF mismatch: HF Gemma4 sets
`self.scaling = 1.0` at `modeling_gemma4.py:1178`, NOT
`1/sqrt(head_dim)`. rvllm defaults to "stable" (1/sqrt(d_k)) for
drafter cross-attn — a 16× (sliding) / 22.6× (global) QK-logit
difference vs HF.

Hardware test of `RVLLM_SPEC_FA_SCALE=mtp` (scale=1.0) vs default
`stable` (1/sqrt(d_k)) — confirmed scale arg IS reaching kernel:

| Site | stable | mtp |
|---|---|---|
| L0 attn_out RMS | 0.61 | 0.68 |
| L0 attn_out head8[0] | -0.142 | -0.521 |
| L3 attn_out RMS | 0.52 | 0.65 |
| post_final_norm RMS | 6.17 | 6.18 |
| post_final_norm max | 89.81 | 90.62 |
| post_final_norm direction head8[2] | 42.56 | 43.38 |
| drafter draft[0] | 2021 | 2021 |

So scale matters at attn_out level (RMS +10–25%, direction
shifts 3×) BUT the drafter forward chain (residual_1 + post-attn-
norm + MLP + post_ff_norm + residual_2 + layer_scalar + final_norm)
SMOOTHS the difference out so post_final_norm direction is
nearly identical. LM-head argmax over 262144 vocab picks the
same token.

The drafter forward chain is essentially robust to cross-attn
output variations. Either:
1. rvllm is missing an AMPLIFICATION step in the drafter chain
   that HF applies (e.g. a per-token scale tied to drafter's own
   embed_tokens magnitude, or a different residual blend).
2. Or HF's trained layer_scalar values + cross-attn output IN HF
   have a magnitude relationship that produces meaningful
   differences at post_final_norm; ours has different magnitude
   relationship somewhere upstream.

Next decisive step (codex Round 4 Q5): adapt
`v3/tools/manual_drafter_reference.py` (E4B masked-head template,
already exists) for the 31B full-vocab head, feed it
rvllm-dumped base_hidden_last + last_token_embed + shadow K/V,
and dump T1..T11 in PyTorch. Compare against rvllm's
layer-trace probes. Where they FIRST diverge identifies the
bug. Avoids loading 70 GB HF base.

Active env knobs (all default OFF, env-gated):
* `RVLLM_GEMMA4_SPEC_SOURCE_PAIR=<s>,<f>`
* `RVLLM_GEMMA4_SPEC_DRAFT_TRACE=1`
* `RVLLM_GEMMA4_SPEC_LAYER_TRACE=1`
* `RVLLM_SPEC_DEBUG_Q_BISECT=1`
* `RVLLM_SPEC_Q_ROPE_MODE` ∈ {current, pos0, no_rope}
* `RVLLM_SPEC_ZERO_Q=1` (batched session loop ONLY, since
  commit `12970cd`)
* `RVLLM_SPEC_FA_SCALE` ∈ {stable=1/sqrt(d_k), mtp=1.0}
* `RVLLM_GEMMA4_SPEC_EMBED_SCALE_31B` (DEPRECATED, double-scales)

### Active diagnostic env knobs (env-gated, all default OFF)

* `RVLLM_GEMMA4_SPEC_SOURCE_PAIR=<sliding>,<full>` — override the
  resolver's source-pair choice. Validates both indices land on
  the right layer types; panics with a clear message otherwise.
* `RVLLM_GEMMA4_SPEC_DRAFT_TRACE=1` — print one
  `[draft-trace] iter=N next_base_seed=X drafts=[...]
   base_argmax_k=[...] accept_len=N` line per iter in the
  batched session loop.
* `RVLLM_GEMMA4_SPEC_LAYER_TRACE=1` — per-drafter-layer
  rms/max/head8 dump of `attn_out(pre_oproj)`,
  `post_residual1(hidden)`, `post_mlp(hidden)`, plus a
  `post_final_norm(hidden)` dump at the LM-head input.
* `RVLLM_SPEC_DEBUG_Q_BISECT=1` (pre-existing) — Q-side stage
  probes in the drafter's first layer.
* `RVLLM_GEMMA4_SPEC_EMBED_SCALE_31B=1` — **deprecated, leave
  off**. Applies `apply_pre_projection_embed_scale(sqrt(backbone))`
  on the embed half. Per codex Round 2 this is double-scaling on
  top of the loader's pre-applied sqrt(hidden_size). To be
  removed once the V-cache fix lands.

### Codex Round 2 answers (cited HF lines)

For the full Q&A see the prompt + response in the conversation
preceding commit `2693894`. Key citations:

* Drafter input contract:
  `transformers/generation/candidate_generator.py:1357-1379`,
  `transformers/models/gemma4_assistant/modeling_gemma4_assistant.py:123-126,169-188`.
* No embed scale at the assistant; base loader's embed scale is
  the only one (rvllm already applies it). HF
  `models/gemma4/modeling_gemma4.py:1579-1582`.
* Drafter LM head is the tied full-vocab path; no softcap, no
  extra norm.
  `gemma4_assistant.py:110-126,185-188`.
* Source pair for 31B is (58, 59).
  `models/gemma4/modeling_gemma4.py:1182-1188,1251-1254`.
* **V-cache write for `attention_k_eq_v=true`**:
  `models/gemma4/modeling_gemma4.py:1203-1207` (`value_states =
  key_states` before V-norm), then V-norm + cache update at
  `:1241-1254`.

### Documents on disk

* This file (rvllm-serve/CLAUDE.md) carries the lasting picture.
* The conversation history holds the draft-trace evidence and
  the codex round-2 verbatim reply.

## Native multimodal vision (Qwen3-VL + Gemma4 + Pixtral)

Three vision towers now run end-to-end as native Rust+CUDA inside
this process — Qwen3-VL (Qwen 3.5 / 3.6), Gemma 4 SigLIP (31B / E4B),
and Mistral 3.5 Pixtral — no Python sidecar. `image_url` parts on
`/v1/chat/completions` go straight from the OpenAI handler through
the same process to the GPU. Verified end-to-end on `/tmp/ball.png`
for all three families.

## Native audio (Gemma 4 E4B-it)

E4B-it has a native 12-layer chunked-attention audio encoder + a
`POST /v1/audio/transcriptions` (whisper-compat multipart) endpoint.
Audio path in summary: symphonia decode + rubato resample to 16 kHz
mono f32 -> log-mel (CPU) -> subsample Conv2d stack -> 12 encoder
blocks (FFN + chunked attention + LightConv1D + FFN + norms) ->
output_proj (1024 -> 1536) -> embed_audio_projection (1536 -> 2560)
-> splice into prefill residual at AudioSlot positions. Real-speech
transcription verified on user-recorded mp3/ogg/m4a inputs:
"Das ist ein Test.", "I am a human.", "Wie ist das Wetter heute?"
all transcribed correctly. Audio module is in
`v3/crates/rvllm-runtime/src/{audio_preprocess,gemma4_bring_up,gemma4_audio_forward}.rs`
and the endpoint at `v3/crates/rvllm-serve/src/openai/transcriptions.rs`.

Per-request flow:
1. **Admission** (`crates/rvllm-serve/src/openai/handlers.rs ::
   collect_vision_items`): fetches each `image_url` (data: URI or
   http(s)) under bounded caps —
   `RVLLM_VISION_MAX_IMAGES` (default 8),
   `RVLLM_VISION_MAX_TOTAL_BYTES` (64 MiB),
   `RVLLM_VISION_MAX_TOTAL_TOKENS` (8192). Per-fetch hard timeout
   5 s + 20 MiB cap. Literal `<|image|>` / `<|image_pad|>` markers
   in user/assistant text are rejected at admission so they cannot
   collide with the post-render token-id splice scan.
2. **Tokenize** (`crates/rvllm-serve/src/tokenize.rs ::
   render_chat_with_vision`): renders the chat template, then
   expands each image-pad token (Qwen `248056`, Gemma `258880`) to
   `vision_items[i].num_tokens` copies and emits
   `VisionSlot{token_start, num_tokens, vision_item_idx}`.
3. **GPU pre-pass** (`crates/rvllm-serve/src/cuda_worker.rs`):
   per-image `Qwen36Bringup::forward_qwen_vision` /
   `Gemma4Bringup::forward_gemma_vision`. Each loop checks
   `req.cancelled` per image. Output stays device-side
   (`VisionForwardOutput.data`).
4. **Splice** in the prefill embed step:
   `crates/rvllm-runtime/src/qwen36_bring_up.rs ::
   forward_qwen36_decode` / `gemma4_bring_up.rs ::
   run_generate` copy each output into `residual_ptr` at
   `slot.token_start * row_bytes` after `EmbeddingGatherLaunch` and
   before `F16ToBf16Launch`. Vision-bearing requests force
   `common_prefix_len = 0` and the chunked-prefill batch path
   (clean error on F16-KV).

Architecture dispatch: `VisionArch` (router.rs) is resolved at
startup from `Qwen36Arch::from_dir(model_dir)` —
`Some(_) → Qwen36`, else `Gemma4`. `forward_gemma_vision` is gated
by `#[cfg(feature = "cuda")]`; the default/mock build does not pull
in cudarc.

### Vision kernel set (sm_121, all f16)

In `kernels/`: `vit_pos_emb_lookup_2d`, `vit_pos_embed_interp`
(Qwen bilinear), `vit_rotary_2d` (Qwen cat-trick),
`vit_rotary_gemma4_2d` (Gemma per-chunk), `vit_avgpool` +
`_to_f32`, `vit_standardize` + `_f32_to_f16`, `extract_head`,
`scatter_heads`, `transpose_heads_v`, `transpose_2d`,
`gelu_tanh_mul`, `silu_mul`, `scale_inplace`,
`softmax_row_f32_to_f16`, `vector_add`, `vnorm`. cuBLASLt:
`f16_gemm_f32_batched_strided` + `bf16_gemm_f32_batched_strided`.

**INVARIANT** in the batched-strided wrappers
(`crates/rvllm-cutlass/src/cublaslt.rs`): cuBLAS internally swaps
a/b inside `cublasLtMatmul`, so the wrappers apply caller
`stride_a` to internal `layout_b` (which holds `a_*16`) and vice
versa — verified, **do not "fix" the swap**.

bf16 sibling kernels + `rmsnorm_inplace_bf16_gbf16` are committed
but NOT wired into the forward — **Phase 3 is formally deferred**
(2026-05-05). f16 path delivers correct vision output on real
images for both models; bf16 marginal gain (~0.003 mean cos +
1 outlier row) does not justify the debug cost.

Per-sub-step debug tooling for any future bf16 attempt:
- rvllm side: gate `RVLLM_GEMMA4_VIT_SUBSTEP_BLK=<idx>` (alongside
  `RVLLM_GEMMA4_VIT_DUMP_DIR`) → dumps every sub-step in that
  block as `g4v_blk{B}_{step}.bin`.
- HF reference: `v3/tools/gemma_vision_substep_hf_dump.py`.
- Diff harness: `v3/tools/cmp_g4v_substep.py` with first-divergence
  pointer. Full replay recipe in `v3/GEMMA_VISION_AUDIT.md`.

### Correctness methodology

Layer-by-layer f16 dumps via `RVLLM_QWEN36_VIT_*_DUMP` /
`RVLLM_GEMMA4_VIT_DUMP_DIR` vs HF reference dumps, compared
row-cosine. Qwen tower is byte-faithful per layer (cos = 0.9999).
Gemma is byte-faithful through block 13 and drifts to mean
cos = 0.9974 by block 26 / 0.9969 at post_projection — pure f16
compound + sqrt(1152) ≈ 33.94 saturation in the pooler. The
pooler→standardize bridge runs in f32 explicitly to recover the
saturated rows (`crates/rvllm-runtime/src/gemma4_bring_up.rs`,
commit b2969c6).

### E2E smoke (re-run after any vision-touching change)

```bash
B64=$(base64 -w0 /tmp/ball.png)
curl -s http://127.0.0.1:8010/v1/chat/completions -d '{
  "model":"<gemma-4-31b-it|qwen3-6-35b-a3b>",
  "messages":[{"role":"user","content":[
    {"type":"image_url","image_url":{"url":"data:image/png;base64,'$B64'"}},
    {"type":"text","text":"Was zeigt das Bild?"}]}],
  "max_tokens":80,"temperature":0.2}'
# Qwen   → "Das Bild zeigt einen orangefarbenen Ball."
# Gemma  → "Das Bild zeigt einen orangefarbenen Kreis auf einem hellblauen Hintergrund."
```

Reserved-marker rejection (must return 400):
```bash
curl -s http://127.0.0.1:8010/v1/chat/completions -d \
  '{"model":"...","messages":[{"role":"user","content":"hi <|image|> bye"}],"max_tokens":5}'
# → error.code = "reserved_marker_in_text"
```

## Qwen 3.6 batched prefill (Phases 4b/5/6/7) — production default

Status (head `aac7220`): all transformer-stack batched-prefill
phases are GREEN and **default-ON** in production. Per-request
1.77×–2.69× TTFT improvement at N = 22 / 293 vs the legacy
token-major path.

Five env-gates control the path; each is ON by default and can be
opted out individually with `=0`:

| Env | What it batches | Files |
|---|---|---|
| `RVLLM_QWEN36_BATCH_LINEAR_PREFILL` | linear-attn (Gated-DeltaNet) over N tokens, one launch per layer | `kernels/gated_delta_rule_prefill_f16.cu`, `kernels/conv_state_advance_batched_f16.cu` |
| `RVLLM_QWEN36_BATCH_FULL_PREFILL` | full-attn causal-prefill via existing `flash_attention_2_f16kv_kernel` + cast f16↔f32 | reuse |
| `RVLLM_QWEN36_BATCH_MOE_PREFILL` | router GEMV + topk-softmax batched over N | `kernels/router_gemv_batched_f16_to_f32.cu`, `kernels/topk_softmax_batched_f32.cu` |
| `RVLLM_QWEN36_BATCH_MOE_ROUTED_FFN` | per-row indirect FP8 GEMVs (8 k-rounds × 3 launches per layer) | `kernels/fp8_gemv_blockwise_wpr_native_f16in_*_indirect_batched_topk.cu`, `kernels/scaled_add_f16_to_f32_devw_batched_topk.cu` |
| `RVLLM_QWEN36_BATCH_MOE_SHARED` | shared-expert batched + final residual | `kernels/shared_gate_dot_sigmoid_f16_batched.cu`, `kernels/scaled_add_f16_to_f32_devw_batched.cu` |

Outer-loop deletion (Phase 7) is implicit: when the gates are on,
`forward_qwen36_decode_cancellable` runs strictly layer-major and
skips the legacy `for tok_local in 0..num_tokens` chain.

### Audit
Per-(layer, phase, tok) hidden-state dumps via
`RVLLM_QWEN36_DUMP_DIR=/tmp/dump`; `v3/tools/cmp_qwen36_prefill_layers.py`
diffs two dump dirs row-wise. With all five gates on vs token-major
default: all 40 layers × N tokens × 2 phases produce
`cos = 1.000000` / `max_abs = 0.0` (byte-equivalent, 1782 rows on
the N = 22 canary).

### Bench
`RVLLM_QWEN36_TIMING=1` logs per-prefill `[qwen36-timing]` lines
with prompt_tokens + prefill_ms + per-gate state. Headline:
* N = 22:  449 ms → 254 ms (1.77×, −43%)
* N = 293: 6836 ms → 2539 ms (2.69×, −63%)

The win scales with prompt length because per-token launch
overhead dominates for the legacy path.

### Production rollout invariant — round-26 / 27 race fixes
* `pos_cl_region` / `context_lens` / `positions` per-token slot
  ids live in DRAM, populated once per request by
  `qwen_fill_pos_slots_i32` on `self.stream`. The legacy
  `Region::copy_from_host` path — which was a sync
  `cuMemcpyHtoD_v2` on the legacy default stream and raced with
  the non-blocking compute stream — is replaced. Both token-major
  and layer-major branches use the device-fill path now.
* The token-major path was non-deterministic across runs prior to
  the round-26 fix; repeated greedy canaries used to alternate
  between two valid German jokes. Post-fix the same canary is
  byte-stable.

### Phase 8 — TODO
Decode-step CUDA Graph capture is parked as Phase 8 in
`v3/QWEN_BATCHED_PREFILL_PLAN.md`. Codex round-28 picked a
workspace-based factoring (`decode_step_launch_only` reading
device pointers, no per-call arena allocation, no sync DtoH inside
the captured body) over capturing the current allocation-heavy
forward. Realistic scope 500–1000 LOC, separate session. Not
blocking the prefill rollout above.

## Other docs in this tree

- `llm_instructions_sm121.md` — NVFP4 + FP8 algorithmic picture
- `best_configs_sm121.md` — sweep-validated NVFP4 quality knobs
- `parameters_for_nvfp4_sm121.md` — env-knob reference
- `fp8_gemm_debug_spec.md` — per-shape FP8 GEMM correctness notes
- `v3/GEMMA4_SPEC.md` / `GEMMA4_IMPLEMENTATION.md` — Gemma 4 layer
  shapes, weight names, KV variation
- `v3/GEMMA_VISION_AUDIT.md` — layer-by-layer Gemma ViT cosine
  audit + bf16-wiring debug plan
- `v3/GB10_SPEC.md` — GB10 hardware quirks (sm_121 caveats, FA3
  unavailability)
- `CONTRIBUTING.md` — upstream workflow

## Known pitfalls

- **`cargo` cwd**: every cargo invocation must be from `v3/`, not
  the repo root. Otherwise: `could not find Cargo.toml`.
- **PTX manifest drift**: rebuild `kernels/build.sh sm_121` AFTER
  the final commit in a chain, including manifest-only commits.
- **F16-KV + vision**: vision-bearing requests force the
  chunked-prefill batch path; F16-KV is incompatible there and
  raises a clean error. Use FP8 or NVFP4 KV for vision profiles.
- **Manager-wide systemd env**: stale
  `systemctl set-environment RVLLM_*=…` entries leak into
  rvllm-serve even when unit + profile are clean. Check
  `systemctl show-environment` and `systemctl unset-environment`
  if a behaviour persists across config changes.
- **bf16 vision forward**: kernels are committed, wiring is not.
  Don't enable until the per-sub-step debug plan in
  `v3/GEMMA_VISION_AUDIT.md` is run through.

## Option B (Gemma 4 31B NVFP4) — deferred-work register

Active branch `rusty_sm121_qwen36_26b`. Floor + Stream-5a/b
landed (`bc89650..32d0b0e`). #5f-PRIME and #5g scaffolds at
`fac2114` and `49a4ff0`. 60-layer single-token forward +
device-resident residual + scaled-add kernel + Option B routed
through `cuda_worker` text-only at 0.31 s/token. All 14 ondisk
smokes pass; argmax stable. Remaining streams from the codex
review:

| Stream | Status | Next concrete step |
|---|---|---|
| #5f-PRIME (unified NVFP4 prefill) | Kernel handle loaded; not wired. | Inline launcher mirroring `PagedPrefillNvfp4Launcher::launch_nvfp4kv_unified_sm121` (~200 LOC) + batched M>1 MLP path (~300-500 LOC). Routes `forward_prompt_to_token` through unified prefill for N>1. |
| Stream-6a (BaseKvSource trait) | `Gemma4Nvfp4Bringup::drafter_base_kv_view` returns a `DrafterBaseKvView`. Production drafter not yet consuming it. | Define `BaseKvSource` trait, implement for production `KvCache` + `Gemma4Nvfp4KvState`. Parameterize `populate_shadow_kv*_from_base` (gemma4_bring_up.rs:4583, 5742). Flip the `cuda_worker.rs` `spec_decode=1` rejection for `Gemma4Nvfp4`. |
| Stream-6b (Hadamard drafter Q rotation) | Off on Option B (sign tables not allocated). | If Hadamard later turns on for Option B (production NVFP4 profile uses it), the drafter's Q must be rotated by the same per-layer R before cross-attn via `hadamard_rotate_f16_kernel`. Per-layer sign vectors live in production's `Gemma4LayerScratch.hadamard_signs_k`; Option B needs the same allocation + production-side rotate/unrotate (gemma4_bring_up.rs:4790-4860, :8985). Blocked by Stream-6a. |
| Stream-7 (vision real splice) | Admission rejects vision for Gemma4Nvfp4 (commit `71cdcac`). Real splice deferred. | Attach Gemma ViT weights to `Gemma4Nvfp4LoadedModel` (loader change in `gemma4_nvfp4_load.rs` + new vision field). The ViT is weight-format-agnostic, runs in bf16. Splice output rows into Option B's device residual buffer between `embed_one_token_to_device` and the layer loop. Needs the multi-token device path from #5f-PRIME OR per-token-loop with vision-slot indexing. |
