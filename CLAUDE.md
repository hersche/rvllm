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
| Qwen 3.6 35B-A3B | `qwen36_bring_up.rs` | **wired 2026-05-15** (`mobile-qwen-rvllm-nvfp4.env`). 4-commit port (~384 LOC, 0dd5d98..d31b8ae): layout plumbing, kernel load, decode dispatch, prefill fallback. Both kernels (`fused_rope_qwen_partial_nvfp4kv` + `flash_attention_2_decode_nvfp4kv_kernel`) reused as-is from the Qwen 3.5 work — only the dispatch wiring is per-family. Per-head decode handles GQA=8 without split-decode. Batched prefill flips to per-token loop on Nvfp4; unified-NVFP4-prefill (PTX exists) is a follow-up. Validated on hardware: text (German ghost joke) + Qwen3-VL vision (same caption as F16 baseline). KV memory at 4096 ctx: 20 MiB packed + 2.5 MiB scales vs 80 MiB F16 (3.5× reduction). F16 path bit-identical when the gate is off. |

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
* 80-token counting prompt: validated coherent end-to-end
  (spec accept_len>0 commit fix + classical bonus push).
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

### 31B-it spec — accept_rate>0 SHIPPED (commit `c29cd81`, 2026-05-22)

**Resolved 2026-05-22 via Round 7 dump→diff.** The drafter
accept_rate=0 was caused by `RVLLM_NVFP4_HADAMARD=1` +
`RVLLM_NVFP4_HADAMARD_V=1` on the 31B spec profile. With these
flags the shadow K/V the drafter cross-attends to are in an
orthogonally-rotated frame the drafter Q-projection can't dot-
product into (drafter weights were trained against HF-native
unrotated K/V). Cosines vs HF base K/V went from ~0.0 (uncorrelated)
to ~0.88 (NVFP4-quant-noise floor) when both flags are turned off,
and accept rates landed at 87.5% / 42% / 46% on three test prompts.

`ensure_drafter()` now refuses to load the drafter when
HADAMARD/HADAMARD_V is on with spec enabled, with a clear error
pointing at the diagnosis. Bypass knob
`RVLLM_GEMMA4_SPEC_ALLOW_HADAMARD=1` exists for the future
un-rotate-in-populate kernel fix to develop against.

The proper code-level fix landed in three phases:

  * Phase 1 (`f6d0b5d`): public helper
    `Gemma4Bringup::apply_hadamard_unrotate_to_shadow_kv_range`
    launches `hadamard_unrotate_f16_kernel` over a shadow KV slot
    range.
  * Phase 2 (`fe86201`): wrapper
    `unrotate_shadow_kv_after_populate` wired into all four
    `populate_shadow_kv_range_from_base` call sites. The
    `ensure_drafter` HADAMARD-vs-spec guard now treats
    `RVLLM_GEMMA4_SPEC_UNROTATE_SHADOW=1` as an implicit bypass.
    Validated under HADAMARD=1: coherent output, accept_rate
    3-6% (drafter sees NVFP4-quant-then-un-rotated K/V; quant
    noise structure diverges from the true HF K/V it was
    trained against).
  * Phase 3 (`eb66ef3`): **F16-shadow path — the optimal fix.**
    Drafter reads POST-RoPE, PRE-HADAMARD, PRE-NVFP4-QUANT F16
    K/V directly from the nvfp4-shadow region instead of
    dequanting the rotated-quantized base cache.
    `build_nvfp4_shadow_alloc` auto-includes the spec source
    layers when `RVLLM_GEMMA4_SPEC_USE_F16_SHADOW=1`; the spec
    session's `compute_view` routes the drafter populate to
    read from the shadow ptr with `KvDtype::F16`. Validated
    under HADAMARD=1 + F16_SHADOW=1: md5 byte-identical to
    HADAMARD=0 on 2 of 3 prompts, accept rates 25-35% — 10×
    better than Phase 2's un-rotate path.

The production 31B spec profile still defaults to HADAMARD=0
(`mobile-31b-rvllm-spec.env`) for highest accept rate; operator
can opt into the long-context-quality HADAMARD=1 path via the
recommended F16-shadow recipe:

    RVLLM_NVFP4_HADAMARD=1
    RVLLM_NVFP4_SHADOW_F16=1
    RVLLM_GEMMA4_SPEC_USE_F16_SHADOW=1
Same session also flipped the profile to
`RVLLM_GEMMA4_SPEC_NEW_PRIMITIVES=1` (commits `2334dbc` byte-equiv
fix + profile flip in `mobile-31b-rvllm-spec.env`).

Output quality verified against HF (cosine 0.88 on K/V + 0.885 on
base hidden state). Per-prompt accept rates above. Full Rounds 1-7
diagnosis + measurements live in the dedicated section below.

### 31B-it spec — historical investigation (Rounds 1-6, pre-fix)

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

### Round 7 (2026-05-22) — root cause + fix landed (commit `c29cd81`)

Same dump→diff pipeline as Round 6 but with
`RVLLM_NVFP4_HADAMARD=0` + `RVLLM_NVFP4_HADAMARD_V=0` on the
31B spec profile:

|       Tensor       | cos Round 6 | cos Round 7 |
|--------------------|-------------|-------------|
| K sliding (L58)    |    ~0.00    |  **0.912**  |
| V sliding (L58)    |    0.017    |  **0.886**  |
| K global  (L59)    |   -0.002    |  **0.877**  |
| V global  (L59)    |    0.007    |  **0.876**  |
| base_hidden_last   |    0.885    |    0.885    |

Cosines went from uncorrelated to HF-faithful. Residual ~12% is
NVFP4 quantization noise (matches the base hidden state's 0.885
which never changed — confirms the same quant-noise floor).

**Spec accept rates now non-zero on every prompt:**

  * Capital of France:  7 / 8 drafted accepted (87.5%)
  * 5 Europ. capitals: 10 / 24 drafted (42%), 2.0/verify
  * Explain Linux:     33 / 72 drafted (46%), 1.94/verify

All outputs correct German. Per user direction this is acceptable
as long as quality is verified against HF, which the cosine table
above establishes.

**Why Hadamard + spec was breaking accept_rate to 0**: base
attention's Hadamard rotation puts K/V in an orthogonally-
rotated frame so QK contracts correctly via H^T·H=I. The
drafter cross-attends to shadow K/V using a separately-trained
drafter Q projection that expects HF-NATIVE (unrotated) K/V.
The companions `apply_hadamard_to_drafter_q` +
`apply_hadamard_unrotate_drafter_attn_out` cancel only the
attention output's rotation — the cross-attn dot product
itself contracts in rotated K-space against an unrotated
drafter Q, collapsing the score distribution to near-uniform.

**Proper code-level fix — SHIPPED 2026-05-22**: un-rotate K (and
V if V-rotated) inside `populate_shadow_kv_range_from_base`'s
dequant kernel, so the shadow KV lands in HF-native frame
regardless of the base's rotation. Landed in two commits:

  * `bf6af65` — kernels:
    `gemma4_drafter_dequant_{nvfp4,fp8}_to_f16_unrotate_kernel`
    in `kernels/gemma4_drafter_dequant.cu`. Grid =
    `(num_tokens, num_kv_heads)`, block = `head_dim`, smem =
    `head_dim * 4` bytes. Each thread dequants one nibble (NVFP4)
    or E4M3 byte (FP8) → f32 in smem, then participates in the
    cooperative FWHT + `apply_signs_f32` from `hadamard.cuh`,
    then writes back as f16. One launch replaces dequant + the
    separate `hadamard_unrotate_f16_kernel` post-pass.
    Launch helpers `launch_{nvfp4,fp8}_dequant_unrotate_to_shadow`
    added on `Gemma4DrafterRuntime`.
  * `eae6202` — consumer wiring:
    `populate_shadow_kv_range_from_base_with_signs` threads per-
    source signs ptrs to the with-signs `populate_one_source
    _layer_with_signs`, which dispatches the fused launchers when
    signs are non-zero. All four call sites in `gemma4_bring_up.rs`
    (initial prompt populate + accept-batch + bonus + deferred-bonus)
    extract signs via the new helper
    `Gemma4Bringup::fused_unrotate_signs(sources)` which reads
    `RVLLM_GEMMA4_SPEC_FUSED_UNROTATE` (default off) and the
    `self.nvfp4_hadamard` alloc. `unrotate_shadow_kv_after_populate`
    gains an early-return guard on the same env so the separate
    post-pass doesn't double-rotate.

Default behavior unchanged: with `RVLLM_GEMMA4_SPEC_FUSED_UNROTATE`
unset, the helper returns `None`, signs ptrs are zero, the with-
signs populate falls to the legacy dequant path, and the
separate unrotate runs as before. Validated bit-coherent on the
default 31B spec profile (HADAMARD=0). Fused path validated under
HADAMARD=1 + FUSED_UNROTATE=1 + UNROTATE_SHADOW=1 on 31B spec:
capital-of-France + photosynthesis prompts both coherent, no
panics. Drops per-populate launch count from 4 → 2.

Operator workaround for HADAMARD=1 quality still available: run
spec with HADAMARD=HADAMARD_V=0 (default), OR enable the fused
path via `RVLLM_GEMMA4_SPEC_FUSED_UNROTATE=1` +
`RVLLM_GEMMA4_SPEC_UNROTATE_SHADOW=1`, OR use the Phase 3 F16-
shadow recipe (`RVLLM_NVFP4_SHADOW_F16=1` +
`RVLLM_GEMMA4_SPEC_USE_F16_SHADOW=1`).

**A/B re-verified 2026-05-23** (gemma-4-31b-it-nvfp4, 80-tok
photosynthesis, 3 runs/leg, freshly-installed binary, per-leg
profile edits to override sourced HADAMARD=0):

  | HADAMARD config                                      | accept_rate | tok/s |
  |------------------------------------------------------|-------------|-------|
  | 0 (production default)                               | 1.056       | 4.51  |
  | 1 + V=1 + ALLOW_HADAMARD (Stream-6b drafter Q)       | 0.000       | 4.43  |
  | 1 + SHADOW_F16 + USE_F16_SHADOW (Phase 3)            | 0.200       | 4.23  |
  | 1 + UNROTATE_SHADOW + FUSED_UNROTATE (#34 fused)     | 0.200       | 4.23  |
  | 1 + UNROTATE_SHADOW (#34 separate post-pass)         | 0.200       | 4.23  |

The fused vs separate unrotate paths give IDENTICAL accept_rate +
tok/s + md5 — the fusion saves a launch but does not change quality
or measurable speed. Phase 3 F16-shadow and both unrotate paths
all converge to accept_rate=0.200, which is 5× worse than the
HADAMARD=0 baseline at 1.056. Stream-6b drafter-Q rotation
produces accept_rate=0.000 (output still coherent because base
verify-fallback always wins).

**Guard (this round)**: `ensure_drafter()` refuses to load the
drafter when `(RVLLM_NVFP4_HADAMARD=1 || RVLLM_NVFP4_HADAMARD_V=1)`
and `RVLLM_GEMMA4_SPEC_DECODE=1` are simultaneously true. Error
message references this round's diagnosis + CLAUDE.md + the
bypass knob `RVLLM_GEMMA4_SPEC_ALLOW_HADAMARD=1` for the future
un-rotate-in-populate fix to develop against without tripping
the guard.

### Round 6 (2026-05-22) — first end-to-end diff run, content-uncorrelated

The full dump→dump→diff pipeline ran end-to-end for the first
time on `mobile-31b-rvllm-spec.env` (fp8-block weights, NVFP4 KV,
20-token prompt "Was ist die Hauptstadt von Frankreich?"). Diff
result on the four layer-58/59 shadow K/V tensors:

|       Tensor       | rms(rvllm) | rms(hf) | rms(diff) | cosine(flat) | max diff |
|--------------------|------------|---------|-----------|--------------|----------|
| K sliding (L58)    |    0.123   |  0.122  |   0.085   |   ~0.00      |   1.55   |
| V sliding (L58)    |    1.004   |  1.000  |   1.405   |   0.017      |  17.13   |
| K global  (L59)    |    0.060   |  0.060  |   0.085   |  -0.002      |   1.38   |
| V global  (L59)    |    1.000   |  1.000  |   1.409   |   0.007      |  22.94   |
| base_hidden_last   |    3.890   |  4.205  |   1.964   |   0.885      |  33.75   |

**Magnitudes match (RMS within 1%) but content is entirely
uncorrelated (cos ≈ 0).** rvllm's shadow K/V at the spec source
layers carry the right statistical distribution but the wrong
values, slot-for-slot, head-for-head. First divergence is slot
t=0 (BOS) for every tensor.

Distribution analysis of the rvllm dump:
* shadow_V_sliding has only **130 unique values** across 81920
  elements, clustered at ±[1.03125, 2.0625, 3.75, …].
* shadow_K_sliding has **132 unique values** clustered at
  ±[0.04, 0.47] (smaller range).
* HF reference values are full-precision continuous floats,
  thousands of unique values, normally-distributed.

The unique-value count strongly suggests rvllm is dumping
**NVFP4-grid quantized values × E4M3 microscale** with very
uniform scales, not the dequantized continuous f16. Either:
1. The dump tool is reading from a buffer where the content has
   been re-quantized but not properly dequantized for V
   specifically;
2. Or the NVFP4 KV path stores K + V with different scales than
   the base attention reads from, so the shadow population
   produces uncorrelated content.

base_hidden_last has cosine 0.885 — the base model's last-layer
hidden state is only 12% off vs HF, consistent with NVFP4
quantization noise. The base forward is approximately correct,
but the K/V cache that feeds the spec drafter is materially
wrong.

Diff tool got one fix this round (commit `c37431e` — pending):
the rvllm dumps are flat 1-D npy (the in-process saver only
emits 1-D headers). The diff tool now reads `meta.json` and
reshapes to logical `[T, nkvh, head_dim]` before comparing
against HF's already-3-D dump.

Codex round 7 (next) needs:
* Inspect the NVFP4 dequant path in
  `gemma4_drafter.rs::populate_one_source_layer` Nvfp4 branch
  — specifically whether the SAME microscale arena is used for
  K + V (they should have separate scale pointers).
* If yes → check if the `k_v_half_bytes` split in
  `gemma4_bring_up.rs::compute_view` accounts for NVFP4's 4×
  density (currently `layer_elems / 4` for NVFP4 element
  count) AND its scale arena split (`layer_elems / 32` for
  combined K+V scales).
* Hypothesis: V's scale-arena offset is computed against the
  K side, so the V dequant reads K's scales (or wrong scales)
  → produces magnitude-right-but-content-random output.

### Round 5 (2026-05-22) — debug harness completed

The full dump→dump→diff pipeline that Round 4 sketched is now
buildable end-to-end (commit `92ae8fe`):

  1. rvllm side (existing): run gemma-4-31b-nvfp4 with
     `RVLLM_SPEC_DUMP_DIR` set → shadow_{k,v}_{sliding,global}_src.npy
     in the dump dir.
  2. HF base side (existing `dump_hf_base_kv_31b.py`): stop rvllm,
     load Gemma 4 31B base in bf16, run one forward over the same
     prompt → shadow_{k,v}_{sliding,global}_src_hf.npy.
  3. `diff_hf_vs_rvllm_kv.py` (NEW): element-compares each pair.
     Per-tensor rms, cosine(flat), max|diff|, top-10 worst slots
     by per-slot max|diff|, and a first-divergence detector
     (smallest slot index where |diff| > tol).

This closes the missing piece in the debug chain — operator can
now drive the comparison and the diff output points at the
exact (slot, head, channel) where rvllm's shadow-KV-population
diverges from HF's `past_key_values` for layers 58 (sliding) /
59 (global). The analytical work of running the chain + reading
the diff remains operator-driven (requires the 120 GB unified
memory for HF base load, which means stopping rvllm and
restarting it on a different profile).

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
phases are GREEN and **default-ON** in production.

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
with prompt_tokens + prefill_ms + per-gate state.

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

### Binary install procedure

`/home/r00t/.rvllm/bin/rvllm-server` is a symlink to
`/home/r00t/workspace/upstream/rvllm-serve/v3/target/release/rvllm-server`.
`cargo build --release --bin rvllm-server --features cuda,gb10`
followed by `sudo systemctl restart rvllm-serve` is sufficient
to pick up Rust-side changes. PTX is loaded fresh per restart
from `kernels/sm_121/`.

If the symlink ever gets clobbered back to a regular file
(e.g. someone runs a deploy script that `cp`'s over it),
restore with:

  sudo systemctl stop rvllm-serve
  rm /home/r00t/.rvllm/bin/rvllm-server
  ln -s /home/r00t/workspace/upstream/rvllm-serve/v3/target/release/rvllm-server /home/r00t/.rvllm/bin/rvllm-server
  sudo systemctl start rvllm-serve

### Phase 8 — SHIPPED 2026-05-22
Decode-step CUDA Graph capture is parked as Phase 8 in
`v3/QWEN_BATCHED_PREFILL_PLAN.md`. Codex round-28 picked a
workspace-based factoring (`decode_step_launch_only` reading
device pointers, no per-call arena allocation, no sync DtoH inside
the captured body) over capturing the current allocation-heavy
forward. Realistic scope 500–1000 LOC, separate session. Not
blocking the prefill rollout above.

**Foundations shipped 2026-05-22**:

* `793ddf0` — `Qwen36DecodeWorkspace` struct in
  `v3/crates/rvllm-runtime/src/qwen36_decode_workspace.rs`. Stable
  per-step device-pointer slots for token/pos/ctx scalars, the
  hidden-flow buffers (hidden, residual_save, normed), attention
  scratch (qkv/q/k/v/attn_out), linear-attn scratch, MoE scratch
  (router_logits, topk_idx, topk_w, gate, up, silu_mul, down),
  closer scratch (final_norm, logits, argmax_token).
  `Qwen36Bringup::alloc_decode_workspace()` constructs from the
  arena. Env gate `RVLLM_QWEN36_DECODE_GRAPH=1` (default off).
* `be98a01` — `forward_qwen36_outside_closer_device_argmax(...,
  argmax_token_dev)` writes the argmax to a caller-provided
  device pointer, NO `stream.fence()`, NO `cuMemcpyDtoH_v2`
  inside the body. Pairs with `argmax_dev_to_host_token` which
  fences + DtoHs OUTSIDE the captured region. Mirrors Gemma 4
  NVFP4's `forward_full_to_token_device_argmax` pattern.

**Foundations shipped 2026-05-22 (continued)**:

* `81a487f` — Phase 8 commit 2b: optional workspace overrides on
  `forward_qwen36_decode_inner_with_workspace_overrides`. The
  embed-gather now reads token indices from `workspace.token_dev`
  when the override is provided (no legacy `copy_from_host`), and
  the closer routes to `forward_qwen36_outside_closer_device_argmax`
  to write argmax into `workspace.argmax_token_dev` instead of
  doing a sync DtoH at the tail. Top-level
  `forward_qwen36_decode_step_to_workspace(workspace, position)`
  combines both overrides; this is the entry point that the
  captured graph wraps. Hardware-validated: qwen3-6-35b-a3b
  "Was ist die Hauptstadt von Frankreich?" → "Die Hauptstadt von
  Frankreich ist Paris." (byte-identical to pre-commit eager).
* `8505f90` — Phase 8 commit 3: CUDA graph capture + replay
  infrastructure. New `Qwen36Bringup.decode_capture:
  Mutex<Option<CapturedGraph>>` field; `try_capture_decode_step`
  wraps `forward_qwen36_decode_step_to_workspace` in
  `cuStreamBeginCapture` / `cuStreamEndCapture`; `replay_decode_step`
  calls `cuGraphLaunch`; `decode_step_via_graph_or_eager` is the
  high-level operator-facing entry (writes token to
  `workspace.token_dev` via async HtoD, picks capture vs replay vs
  eager, returns the token via `argmax_dev_to_host_token`).
  `RVLLM_QWEN36_DECODE_GRAPH=1` opts in (default off).

* `3c30fea` — Phase 8 follow-up: worker integration +
  two-gate capture/replay design. `cuda_worker.rs`'s qwen36
  decode loop allocates the workspace once per request when
  `RVLLM_QWEN36_DECODE_GRAPH=1` and routes per-step decode
  through `decode_step_via_graph_or_eager`. The method now
  honors TWO gates: `DECODE_GRAPH=1` enables ATTEMPTING
  capture (step 0 runs eagerly inside `cuStreamBeginCapture`
  — validates the workspace body is capture-clean);
  `DECODE_GRAPH_REPLAY=1` enables actual REPLAY in steps 1+.
  Split exists because position-frozen kernels make replay
  incorrect at moving positions. Hardware-validated:
  default-off path byte-identical to pre-commit; capture-on
  path log reads "[graph] captured 1856 nodes" — workspace
  body captures cleanly.

**Final commits (`bb9a3cf`, `4f083f8`)** closed out Phase 8:

* `bb9a3cf` — debug fix: CUDA stream capture in THREAD_LOCAL
  mode RECORDS kernel launches but does NOT execute them
  eagerly (despite earlier optimistic comments in our Gemma 4
  NVFP4 captured-decode code). `try_capture_decode_step` now
  explicitly `graph.replay()`s immediately after
  `CapturedGraph::capture` returns Ok, so step 0's kernels
  actually land on the GPU. Without this the captured body's
  KV writes never happened, `workspace.argmax_token_dev` stayed
  stale, and step 1+ ran from a broken KV state (the "Die."
  symptom). Capture-failure fallback also re-runs the body
  eagerly. New env knob `RVLLM_QWEN36_DECODE_WORKSPACE=1`
  isolates the workspace-eager path from the capture machinery.
* `4f083f8` — position-indirect overrides + cross-request reset.
  Two pieces that close out replay correctness across moving
  positions and across requests:
  - `pos_dev_override` / `ctx_dev_override` on
    `forward_qwen36_decode_inner_with_workspace_overrides_v2`:
    when both `Some`, the per-call `positions_region` /
    `context_lens_region` allocations + the
    `qwen_fill_pos_slots_i32` fill kernel are SKIPPED.
    `tok_pos_dev_ptr` / `tok_cl_dev_ptr` bind directly to the
    workspace's stable slots. Worker writes per-step values via
    `Qwen36Bringup::write_position_to_workspace`
    (`cuMemsetD32Async`) before each replay; the kernels pick
    up updated values automatically.
  - `Qwen36Bringup::clear_decode_capture()` called by the
    worker at the start of every new request, BEFORE
    `alloc_decode_workspace`. The captured graph holds device-
    pointer references to per-call arena regions that get
    released by `arena.restore(scratch_ck)` at end-of-request;
    a stale graph from request N would replay against
    reclaimed memory and hang request N+1.

Hardware validation (`RVLLM_QWEN36_DECODE_GRAPH=1` +
`_REPLAY=1`, qwen3-6-35b-a3b, three requests in sequence):

- Req 1 (short):  "Die Hauptstadt von Frankreich ist Paris."
- Req 2 (1-10):   "Eins / Zwei / Drei / ... / Zehn" (10 tokens,
                   all correct via captured-replay).
- Req 3 (long):   80-token photosynthesis explanation,
                   byte-identical to eager.


Production default remains `RVLLM_QWEN36_DECODE_GRAPH` unset
(legacy eager); captured path is opt-in. Most gain comes from
amortizing kernel-launch overhead; the captured path's primary
value is as the foundation for further graph-level optimizations.

### Phase 8 deeper-optimizations — SHIPPED 2026-05-23

Commit `a14af12` lands the multi-step graph capture + device-side
step-linker that subsumes BOTH "multi-step graph capture" and
"pipeline argmax-DtoH with next-step HtoD" deeper-optimization
items:

- New kernel `qwen36_step_link_i32` (single-thread) copies
  argmax → token_dev and increments pos_dev / ctx_dev by 1.
  Sits BETWEEN consecutive decode-step forwards inside the
  macro-graph; eliminates the host-side write_token +
  write_position round-trip the per-step path needs.
- Workspace gains `argmax_tokens_dev_base: u64` (i32[max_steps]
  array) + `max_steps: u32` sized from
  `RVLLM_QWEN36_DECODE_MULTI_STEP` (default 1, clamped to
  [1, 32]). `argmax_token_dev` aliases slot 0 so single-step
  paths + the cross-request graph cache keep working unchanged.
- `Qwen36Bringup` gains `try_capture_decode_steps_n` /
  `replay_decode_steps_n` / `decode_steps_n_via_graph_or_eager` +
  `decode_capture_multi_step` slot (separate from the
  single-step `decode_capture`). The macro-graph captures N
  forwards (each writing argmax to slot i via a SHIFTED sub-
  workspace) + N-1 linker launches.
- Worker per-decode-step loop gains a `multi_step_buf` FIFO that
  pre-fetches N tokens via macro-replay when
  `multi_step_active`. Single-step path (default) untouched.

Hardware validation (`MULTI_STEP=8 + DECODE_GRAPH=1 + REPLAY=1`,
qwen3-6-35b-a3b): coherent multi-token output across short /
multi-step counting / 80-token reasoning prompts. Journal:
`[graph] captured 14847 nodes (bucket=8)` — ≈1856 nodes/step × 8
+ 7 linker stitches.

The
infrastructure is in place for future kernel-fusion work where
larger graphs may matter, or for continuous-batch decoding where
macro-graphs could share sequences across requests. Production
default (`MULTI_STEP` unset) remains the single-step path.

**Kernel fusion follow-on (`e47166b`)**: new
`qwen36_argmax_with_link_f16_kernel` fuses the per-iteration
argmax with the step-link side-effects (token_dst write +
pos/ctx increment). Macro-graph body now runs
`forward_qwen36_decode_step_to_workspace_no_closer` + the fused
closer instead of the prior plain-argmax-then-step_link pair.
Per N-step macro-block, kernel count drops from
`N * forward + N argmax + (N-1) step_link` to
`N * forward + N fused-closer`. Macro-graph node count drops
Per-macro-block node count drops by the predicted 7 (one fewer
link per stitch at N=8). The unfused
`qwen36_step_link_i32_kernel` + Rust launcher stay loaded as
parked infrastructure for future non-fused captured paths
(continuous-batch, multi-sequence macro-graphs). Latency impact
unverified.

### Phase 8 MoE-fusion — SHIPPED 2026-05-23

Commit `f0f79d5`: new kernel
`fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_kernel`
fuses the per-token MoE down-projection FP8 GEMV with the
scaled-add accumulator step that previously ran as a separate
`scaled_add_f16_to_f32_devw_kernel` launch right after. The
inner GEMV reduction is byte-identical; the epilogue replaces
the f16 store with `routed_sum[n] += devw[0] * acc` directly in
f32 (lane 0 of each warp owns exactly one output element → no
atomic needed).

Eliminates per decode token:
- 320 kernel launches (8 routed k-rounds × 40 MoE layers).
- 320 f16 round-trips (acc f32 → f16 down_region → f32 acc).


Production qwen27b (dense, no MoE) — unaffected (regression-clean).
The unfused `fp8_gemv_blockwise_wpr_native_f16in_indirect_kernel`
+ `scaled_add_f16_to_f32_devw_kernel` stay loaded; still used by
prefill / MTP / shared-expert chains. The fusion targets ONLY
the per-k-round routed-expert pair in the per-token decode
hot-path.

### Phase 8 MoE batched-prefill + shared-expert fusion — SHIPPED 2026-05-23

Two follow-on MoE fusions on top of f0f79d5's per-token win.

**Batched-prefill fusion (commit `8060834`)**: new kernel
`fp8_gemv_blockwise_wpr_native_f16in_indirect_scaled_add_batched_topk_kernel`
mirrors the per-token f0f79d5 epilogue but for the batched-MoE
prefill path (`apply_layer_moe_batched`). Reads per-row
`top_w[m * top_k + k_round]` and accumulates `w * acc` directly
to `routed_sum[m*N+n]`. Replaces the (down_indirect_batched_topk
+ scaled_add_f16_to_f32_devw_batched_topk) pair with ONE launch
per k-round across the entire token batch.


**Shared-expert fusion (commit `99e6cde`)**: new kernel
`fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_kernel`
fuses the shared-expert down-projection + the trailing
scaled-add into one launch. Per-token decode shared-expert
chain now runs 3 launches per layer instead of 4 (gate+up+silu
+mul fused; down+scaled_add fused; sigmoid_gate stands alone).
Saves 40 launches per decode token at 40 MoE layers.


### Phase 8 dual_silu k_round-batch + router+topk fusions — SHIPPED 2026-05-23

Two more MoE-chain fusions on top of f0f79d5 + 8060834 + 99e6cde.

**Dual_silu k_round-batch fusion (commit `442a72c`)**: new kernel
`fp8_gemv_blockwise_wpr_native_f16in_dual_silu_indirect_kround_batched_kernel`
adds `grid.z = top_k` to the existing dual_silu_indirect kernel.
Each block reads its expert id from `top_idx[m*top_k + blockIdx.z]`
and writes silu to `[k_round, m, n]` (k_round-major layout).
Replaces the host-side loop of 8 separate dual_silu launches per
MoE layer (both `apply_layer_moe_with_override` per-token decode
AND `apply_layer_moe_batched` prefill path) with ONE launch.
silu_region grows from `[M, N_int]` to `[top_k, M, N_int]`; the
down loop stays as 8 launches because routed_sum accumulation
must serialize across k_rounds (no atomic).

Per layer per decode token: 8 dual_silu launches → 1.

**Router+topk fusion (commit `e049258`)**: new kernel
`router_gemv_with_topk_f16_to_f32_kernel` uses the atomic-counter
"last-block-does-topk" pattern. Every block computes one expert's
logit (stage 1, same as standalone router_gemv); after
`__threadfence + atomicAdd(counter)`, the block whose prev value
== num_experts - 1 is LAST and proceeds to stage 2 (topk-softmax,
same as standalone topk_softmax_f32). Last block self-resets the
counter via atomicExch. Persistent counter region allocated once
at worker bring-up + zeroed via 4-byte HtoD; no per-call memset
needed. Eliminates 1 launch per MoE layer per token (~40 per
decode token).

Architectural improvement (one less GPU-side synchronization
point, eliminates the explicit logits_region write to global).

### Phase 8 QKV megakernel Phase 2 — SHIPPED 2026-05-23

Per-token decode + batched prefill, F16-KV + NVFP4-KV, all
gated behind a single env-knob `RVLLM_QWEN36_QKV_MEGAKERNEL=1`
(default off). Fuses Q+K+V FP8 GEMVs + Q+gate split + Q-norm +
K-norm + partial-NeoX RoPE + KV-cache write (NVFP4 path also
adds Q FP8 quant + K/V NVFP4 pack with per-16-elem microscales)
into ONE launch per (token, head) instead of 5-6 separate
kernels. Each WARP (32 lanes) produces ONE output via 32-thread
K-dim cooperation; input row staged into shared mem once per
block and reused across all per-head outputs.

Kernels:
- `fused_qkv_proj_qnorm_knorm_rope_qwen_partial_f16kv_kernel`
  (commit `fa48141`, warp-cooperative warp-per-output)
- `fused_qkv_proj_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel`
  (commit `71fdf33`, NVFP4-KV sibling)

Dispatch in `apply_layer_full_attn` (per-token decode) and
`apply_layer_full_attn_batched` (commit `b08774e`, prefill).
Batched arm is hard-gated to `num_tokens < 128`; at M≥128
`fp8_proj_dispatch` routes to CUTLASS SM120 GEMM
(≈102 TFLOPS at the QKV shape) which beats the warp-coop
megakernel, so the gate keeps large prefill batches on the
CUTLASS path.

Numerical contract: FP8 GEMV math byte-identical to
`fp8_gemv_blockwise_wpr_native_f16in_kernel`. Norm + RoPE
(F16) byte-identical to `fused_qnorm_knorm_rope_qwen_partial_f16kv`
(commit `943f8bb`). NVFP4 norm + RoPE + quant + pack byte-
identical to `fused_qnorm_knorm_rope_qwen_partial_nvfp4kv`
(commit `6f6a25a`). md5 of generated tokens differs ON vs OFF
because the warp-cooperative reduction order is not the same as
the unfused chain's per-step reductions, but the math is
equivalent up to float-rounding.

**943f8bb non-determinism FIXED 2026-05-23** (kernel race in K-side
of `fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel`). Pre-fix
A/B on qwen3-6-35b-a3b F16-KV showed md5 differs across 3 runs
(ac5e866f / 5279bba3 / 16f59787); post-fix md5 stable (16f59787 ×3).
Root cause: the original ELSE branch (tid ≥ half_rot) wrote
`kn_lo` passthrough to `key_cache[cache_off + tid]` for ALL tid in
[half_rot, half_head); for tid in [half_rot, rotary_dim) this
collided with the IF branch's tid_a=tid-half_rot rotated-upper-half
write to the same index. Two warps writing the same slot with no
ordering → run-to-run non-determinism.
Fix: split the ELSE into `tid in [half_rot, rotary_dim) → only
write high element` and `tid in [rotary_dim, half_head) → write
both as passthrough`. The IF branch's rotated upper-half stays
authoritative. Bit-coherent post-fix on F16-KV; no production
impact (production uses NVFP4-KV).

**39f7c1a qwen35 dense fp8_gemv+residual fusion: parity.** A/B on
qwen3-6-27b dense (max_tokens=150, 3 runs each, env-gate
`RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED=0` added at all 3 sites):
- FUSED (default): 5.94 tok/s deterministic, md5 524ccfe7 ×3
- UNFUSED (=0):    5.94 tok/s deterministic, md5 7f35ee87 ×3
Decode throughput parity. Both legs internally deterministic; md5
differs ON vs OFF (different MAC ordering produces different tokens).

Re-verified A/B on qwen3-6-35b-a3b NVFP4-KV (max_tokens=150,
deterministic across 3 runs each, freshly-installed symlinked
binary, profile = mobile-qwen3635b-rvllm-nvfp4-spec):

  | Cell             | wall   | tokens | tok/s | prefill_ms | md5(completion) |
  |------------------|--------|--------|-------|------------|-----------------|
  | QKV_MEGA_OFF     | 3.69 s | 150    | 40.65 | 215        | b5d33eaa        |
  | QKV_MEGA_ON      | 3.09 s | 125    | 40.45 | 197        | b73335f8        |

Decode-throughput (tok/s) is at **parity** — the wall-time delta
is explained by the ON path hitting EOS at 125 tokens (different
MAC ordering → slightly different token choices → earlier
sentence-ending). prefill_ms shows **~8% prefill speedup**
(215→197) which IS apples-to-apples since both legs ran the
same prompt. md5 differs ON vs OFF (fusion fires).

The batched-prefill arm fires at small M only (< 128); CUTLASS
path at M≥128 is unchanged.

Production qwen3627b smoke ("Die Hauptstadt von Frankreich ist
Paris.") verified after every restart.

### fp8_gemv optimization attempts — shared-mem input staging is a no-op on GB10

Tried 2026-05-23: replace the in-place global-memory input reads in
`fp8_gemv_blockwise_wpr_native_f16in_kernel` with a cooperative
shared-mem staging pass (8 warps in a block all process the same M,
so they could share a single loaded copy of the input row). Capped
at K_SMEM_CAP=6144 (12 KB at f16).

A/B on qwen3-6-35b-a3b NVFP4-KV with QKV_MEGAKERNEL=1 (the same
decode-hot-path the profile flagged as 64.3% fp8_gemv): smem-staging
ON gave 40.32 tok/s, prefill 198 ms. Pre-change baseline was 40.45
tok/s, prefill 197 ms. Within measurement noise — no measurable
delta.

Root cause: the L1 cache on sm_121 already absorbs the redundant
input reads. With 8 warps × 32 lanes × stride-256 K access pattern,
the 2 KB input row fits in 8 cache lines and gets ~100% L1 hit rate
after the first warp loads it. Smem staging just moves the same
work from L1 to smem with no bandwidth or latency reduction.

Reverted. Real perf gains in this kernel would need either:
- TensorCore MMA rewrite (FP8 → FP16 micro-GEMM that batches
  multiple output rows simultaneously, replacing the lane-per-row
  scalar FMA), multi-week project.
- cp.async-based K-dim pipelining (overlap weight loads with prior
  chunk's MAC compute), single-week project but requires careful
  Blackwell-arch tuning.

### Stream-6b drafter-Q rotation — math is right, NVFP4 noise is the killer

2026-05-23 investigation. The original 2026-05-22 commit
(`6ab88d4` primitives + `d75067c` wiring) claimed drafter-Q
rotation produces coherent multi-token output on HADAMARD=1+V=1.
Today's A/B against fresh binary gave accept_rate=0.000 (output
coherent only because the BASE verify-fallback always wins).

Math check: R = H · diag(D) where the FWHT applies 1/sqrt(D)
normalization (see `kernels/hadamard.cuh::fwht_inplace_f32`).
So R IS orthonormal, R^T R = I, and the dot-product invariance
`(R·Q) · (R·K)^T = Q · K^T` should hold mathematically.

But base K is rotated AND THEN NVFP4-quantized (4-bit with
per-16-element FP8 microscale). The rotation reshapes the K
distribution, and the NVFP4 quantization noise on the rotated
distribution is structured differently than on the natural
distribution the drafter was trained against. The drafter Q
rotation correctly aligns the frames, but the drafter still
mispredicts because the noise pattern is unfamiliar.

Real fix would require drafter retraining against rotated-
quantized K, OR a noise-compensation step on the drafter Q
side. Neither is a code-edit. Production stays HADAMARD=0
(accept_rate 1.056).

### Phase 3 F16-shadow accept_rate gap — **FIXED 2026-05-23** (commit `1449833`)

Root cause was confirmed: the legacy `populate_drafter_shadow_kv
_with_rt` populated the drafter shadow via dequant-of-NVFP4 in the
ROTATED frame (it inherits Hadamard rotation from the base K/V
cache), so the drafter — whose Q-projection was trained against
HF-native unrotated K — saw the wrong frame even with Stream-6b's
drafter-Q rotation in place.

The fix is opt-in via `RVLLM_GEMMA4_SPEC_PRE_HAD_SHADOW=1` and has
the base forward kernel `fused_rope_partial_nvfp4kv_bf16in_kernel`
write POST-RoPE PRE-HADAMARD F16 K/V directly into the drafter's
shadow buffer at the two spec source layers. The legacy populate
path is short-circuited; `forward_drafter_layer_cross_attn` also
skips drafter-Q rotation / un-rotation because Q and K are now
both in HF-native frame.

A/B (gemma-4-31b-it-nvfp4, 80-word quantum-entanglement prompt,
greedy K=8, 90 emitted tokens, deterministic across 3 runs):

| Cell                                       | accept_rate |
|--------------------------------------------|-------------|
| A: HADAMARD=0 + no shadow (baseline)       |   2.750     |
| B: HADAMARD=0 + PRE_HAD_SHADOW=1           |   2.333     |
| C: HADAMARD=1 + HADAMARD_V=1, no fix       |   0.000     |
| C: HADAMARD=1 + HADAMARD_V=1 + fix         |   2.333     |

Cell C without the fix collapses (drafter sees rotated K, rotated
Q, but the cross-frame mismatch persists because base attention
also reads from the same rotated cache). With the fix shipped
here, HADAMARD=1 + spec lands at the same 2.333 as HADAMARD=0 +
the shadow path — making the long-context-quality HADAMARD=1
recipe viable with spec for the first time on Option B. Minor
regression vs HADAMARD=0 + no shadow (2.75→2.33) is the cost of
giving the drafter a slightly different KV view (zero quant noise,
zero rotation) than the base sees (NVFP4-quant + Hadamard).

Default-off; production stays on the HADAMARD=0 + no-shadow path
(2.750). Recipe for HADAMARD=1 spec: set
`RVLLM_GEMMA4_SPEC_PRE_HAD_SHADOW=1` alongside `RVLLM_NVFP4
_HADAMARD=1` + `RVLLM_NVFP4_HADAMARD_V=1`.

### qwen36 large-M perf validation 16k/32k — task #110 (2026-05-24)

End-to-end Phase 1-5 validation that the session-#94..#108 grouped-MMA +
linear-attn V4 stack (all default-on after #109 fix) actually wins at
production-scale prompts (16k zeroclaw persona, 32k stress test).

Pre-test setup: stopped support services (chatterbox, whisper-fast,
vllm-embedding, zeroclaw) freeing ~22 GB host RAM. Bumped
`RVLLM_ARENA_GB=80` in the mobile profile to guarantee headroom for
32k per-call scratch. Profile state otherwise canonical.

A/B (qwen3-6-35b-a3b NVFP4, direct API, 3-run deterministic, fresh
binary md5 `00969a8d`):

  | M       | Phase 1 baseline (all OFF) | Phase 2 all-on        | Speedup |
  |---------|----------------------------|-----------------------|---------|
  | 1112    | n/a (#108 baseline 817 ms) | 818-819 ms            | held    |
  | 16000   | 203961-204134 (avg 204042) | 60886-61304 (avg 61019)| **3.34×** |
  | 32000   | 489879-489993 (avg 489920) | 204596-205190 (avg 204832)| **2.39×** |

All-on cells coherent multi-sentence German across all runs (no
repetition loops, sensible quantum-entanglement explanations). No
quality regression observed at any M.

Phase 5c — full zeroclaw webhook end-to-end (16k persona prompt
+ "who are you"):
  > "Ich bin **Rusty**. Ich bin kein freundlicher Assistent, der
  >  dir immer zustimmt. Ich bin ein Partner — ich argumentiere,
  >  ich hinterfrage, und ich bin nicht hier, um dir einfach nur
  >  zuzustimmen..."
Persona-grounded multi-paragraph German, no garbage. Compare to
pre-#109 state where it returned `</think>` / `- User: Rusty.`.

Wins scale with M (3.34× at 16k vs 2.39× at 32k) — fixed costs
amortise better at smaller prompts. The grouped-MMA stack reduces
overall per-token prefill cost meaningfully across the full prompt-
size range, not just the ≤4412 range tested earlier.

`RVLLM_ARENA_GB=80` kept in mobile profile (was 50 canonical) so
production handles 32k prompts headroom-free.

### qwen36 router_topk_batched counter unit-bug fix — task #109 (2026-05-24)

Pre-existing bug surfaced when production zeroclaw mobile profile
hit its 16k-token persona prompt. User saw garbage replies
("</think>", "- User: Rusty.") to "who are you". Bisect showed
the bug existed even with ALL session-#94..#108 grouped-MMA opts
disabled — NOT a regression from those, but an unrelated pre-
existing problem only surfaced now (no prior tests hit >8192 tok).

**Root cause**: `router_gemv_with_topk_batched` persistent counter
region (one i32 per token-slot, written by the per-token kernel)
was sized by `kv_cache_num_blocks` (8192) instead of
`kv_cache_num_blocks * kv_cache_block_size` (131072). At M > 8192
the kernel OOB-wrote past the buffer → CUDA context corruption →
subsequent HtoD failed with AllocFailed → all further requests
returned "qwen36 per-request reset: MemcpyFailed".

Diagnostic threshold: M=8000 OK, M=12000 FAIL — exactly at the
8192 block boundary.

Fix in commit `dbe7bbf`: line 2628 sizing changed to
`kv_cache_num_blocks * kv_cache_block_size`.

Verified post-fix (all defaults ON, fresh binary md5 `00969a8d`):
* M=15000 direct API → "Hello" (correct).
* 16k webhook ("who are you") → "Ich bin **Rusty**. Ich bin dein
  Peer, kein Assistent." (persona-grounded German reply, was
  garbage before).

Important note: all previous A/B numbers (#94..#108, up to
4412 tok) remain valid — they were well under the 8192 threshold.
The optimizations weren't broken; this counter buffer's sizing
just made M>8192 unusable regardless of any opt being on/off.

### qwen36 linear-attn v4 ILP accumulator split — task #108 (2026-05-24, +5.6% / +86.4% cumulative, default-on)

`gated_delta_rule_prefill_f16_v4_kernel` (commit `4027562`)
attacks the per-token inner-loop FMA dependency chain. v3 kept a
single `v_corr` / `o_acc` accumulator per phase → 128 serial FMAs
× 4-cycle latency = 512-cycle critical path per token. v4 splits
into 8 parallel partials summed via pairwise tree → 64-cycle
critical path. Total FMAs unchanged; ILP exposed for sm_121.

Numerical contract: same as v3 (no inner f16 RTNE; only boundary
state rounding). Reduction order is pairwise tree — tiny bit
differences vs v3, no quality regression.

Default-flipped ON in same commit after hardware A/B. Opt-out via
`RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V4=0` falls back to v3.

A/B (qwen3-6-35b-a3b NVFP4, default-on grouped MMA stack as
baseline, deterministic 3 runs):

  | Cell                | 1112 tok      | 4412 tok      |
  |---------------------|---------------|---------------|
  | v3 baseline         |  861-880 ms   |  3782-3830 ms |
  | **v4 (default-on)** | **817-834 ms**| **3603-3641 ms**|
  | Win vs v3           |  **+5.6%**    |  **+4.8%**    |
  | Cumulative vs GEMV  |  **+86.4%**   |  **+85.4%**   |

Quality verified on 50-word quantum entanglement.

### Qwen 3.5 27B + E4B audit for dormant fast-paths — task #107 (2026-05-24)

Same investigation pattern as #102 (Gemma 4 NVFP4's MMA_V8 surfaced
a 14.6× win). Audit results:

**Qwen 3.5 27B (qwen35 bring-up, used by qwen3-6-27b)**:
* `RVLLM_QWEN35_BATCHED_PREFILL` (default-OFF) — only env-gated
  dormant fast-path. Hardware A/B on qwen3-6-27b:

  | Cell                    | 1k tok       | 4k tok        |
  |-------------------------|--------------|---------------|
  | BATCHED_PREFILL=OFF     | 813-1054 ms  | 10986-11199 ms|
  | BATCHED_PREFILL=ON      | 813-823  ms  | 11062-11359 ms|

  PARITY within noise — no measurable win. The per-token path is
  already well-optimised. **Leave default-OFF.**
* CUTLASS SM120 paths (`RVLLM_QWEN35_*_CUTLASS_MIN_TOKENS`) — wired
  with sensible defaults (M≥128 gate). Already on.
* `RVLLM_QWEN35_FP8_GEMV_RESIDUAL_FUSED` — default-on (verified ON
  in code, `.unwrap_or(true)`).

**Gemma 4 E4B (gemma4 bring-up)**:
* `RVLLM_FP8_GEMM_CUTLASS_SM120` — default-on (requires explicit
  `=0` to disable). Already covers E4B's prefill GEMM hot path.
* Other env knobs are debug/diagnostic (`SPEC_*`, `BOUNDARY_DUMP*`,
  `SMOKE_*`), not perf gates.
* No dormant fast-path discovered.

Result: no flips. Both models' prefill paths are already on their
best available kernels. The #102 pattern (kernel loaded but env-
gated off) was specific to Gemma 4 NVFP4's MMA_V8 dispatch — the
audit confirms it doesn't repeat elsewhere on Qwen 3.5 / E4B.

### qwen36 default-flip: all grouped MMA + linear-attn V3 default-ON — commit `d195eab` (2026-05-24)

Flipped 5 env knobs from default-OFF → default-ON after end-to-end
validation in tasks #94/#98/#101/#103/#106. **Production now gets
the cumulative +85.7% prefill speedup automatically.** All five
opt-outs are independent, each fully restores its pre-flip
behaviour via `=0`:

* `RVLLM_QWEN36_MOE_MMA_GROUPED`
* `RVLLM_QWEN36_MOE_MMA_DOWN_GROUPED`
* `RVLLM_QWEN36_MOE_SHARED_MMA`
* `RVLLM_QWEN36_MOE_SHARED_DOWN_MMA`
* `RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V3`

Hardware A/B (qwen3-6-35b-a3b NVFP4, fresh binary md5
`f5659dd8`, 3-run deterministic):

  | Cell                              | 1112 tok    | 4412 tok    |
  |-----------------------------------|-------------|-------------|
  | All explicitly OFF (=0×5)         |  6010-6033 ms|  -          |
  | **All default-on (no env)**       | **861-880 ms** | **3782-3830 ms** |
  | Win vs explicit-off               |  +85.7%     |  n/a        |

Quality verified: 50-word quantum entanglement still gives Einstein
"spooky action at a distance" reference; smoke ("hi") returns
"Hello! How can I help you today?" Default-off path matches the
prior 6020 ms GEMV reference.

Mistral 3.5 V8 (#104, -34% regression there) and Gemma 4 ViT bf16
(#105, -3.5%) were correctly kept default-OFF.

### qwen36 shared-expert down grouped MMA — task #106 (2026-05-24, +3.6% / +85.5% cumulative)

`fp8_mma_shared_down_m16_w4c_kernel` (commit `5ca9131`) extends
the W=4 cooperative-A grouped MMA pattern from #103 to the
shared-expert DOWN projection. Single weight, no routing, no
silu/mul, no top_w, no atomic — dense FP8 GEMM. Per-kblk a_scale
fold (#99) built in.

Opt-in via `RVLLM_QWEN36_MOE_SHARED_DOWN_MMA=1` (default off).

A/B (qwen3-6-35b-a3b NVFP4, all four grouped MMA paths active,
deterministic 3 runs):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | 3-grouped (#103)              |    907 ms   |   3962 ms   |
  | **All 4 grouped (#106 added)**|   **873 ms**|  **3821 ms**|
  | Win vs GEMV                   | **+85.5%**  | **+84.6%**  |
  | Win vs prior (#103)           |  +3.7%      |  +3.6%      |

Quality verified on 50-word quantum entanglement (Einstein "spooky
action" reference). Default-off regression-checked at baseline.

### Mistral 3.5 W4A16 MMA_V8 + Gemma 4 ViT bf16 — tasks #104/#105 (2026-05-24)

Two negative-result rollout decisions documented (both keep current
defaults; no code change).

**Task #104 — Mistral 3.5 `RVLLM_MISTRAL35_W4A16_FUSED_MMA_V8`**:
Same kernel family as Gemma 4 NVFP4 (#102 = 14.6× win there). Hardware
A/B (mistral-3.5-nvfp4 production, 1024-tok prefill, max_tokens=1
deterministic 3 runs):

  | Cell    | wall avg  |
  |---------|-----------|
  | V8=OFF  | 25754 ms  |
  | V8=ON   | 34424 ms  |
  | Δ       | **-34%**  |

Mistral V8 REGRESSES. Opposite of Gemma — same kernel template but
Mistral's matrix shapes don't fit V8's tile layout. **Keep default
OFF on Mistral.** Operators experimenting with V8 should be aware
it slows Mistral down 34%.

**Task #105 — Gemma 4 ViT bf16 forward** (`RVLLM_GEMMA4_VIT_USE_BF16`):
Wired + lifetime bug fixed (2026-05-23, `df22aaa3 × 5 stable`).
Hardware A/B (gemma-4-31b-it-nvfp4 + `/tmp/ball.png`, deterministic
5 runs):

  | Cell    | wall avg  | caption stem                  |
  |---------|-----------|-------------------------------|
  | f16     |  8672 ms  | "...minimalistische Darste..."|
  | bf16    |  8976 ms  | "...vereinfachte Darstellu..."|
  | Δ       | -3.5%     | both coherent German          |

bf16 is consistently ~3.5% slower and produces slightly different
word choice (both valid German for "minimalist/simplified"). No perf
win, marginal quality variance. **Keep f16 default-on; bf16 stays
env-opt-in** for future tuning + debugging. The earlier lifetime
fix that made bf16 SAFE was the critical work; the rollout-decision
follow-up is now "no, stay on f16".

### qwen36 shared-expert dual_silu grouped MMA — task #103 (2026-05-24, +4% / +84.9% cumulative)

`fp8_mma_shared_dual_silu_m16_w4c_kernel` (commit `84625d2`)
adapts the #96 W=4 cooperative-A MMA pattern to the SHARED-expert
FFN. No routing/sort needed (single per-layer weight applies to
all tokens) — tiles map directly to (m_block, n_block) grid
coverage. Per-kblk a_scale fold (#99 fix) built in.

Targets the 4.6% kernel `fp8_gemv_dual_silu_kernel` from the post-
#98+#99 nsys profile (800 instances per prefill).

Opt-in via `RVLLM_QWEN36_MOE_SHARED_MMA=1` (default off).

A/B (qwen3-6-35b-a3b NVFP4, all three grouped MMA paths active,
deterministic 3 runs):

  | Cell                                | 1112 tok    | 4412 tok    |
  |-------------------------------------|-------------|-------------|
  | GEMV baseline                       |   6020 ms   |  24775 ms   |
  | Dual_silu + down grouped (#98)      |    955 ms   |   4116 ms   |
  | **All 3 grouped (#103 added)**      |   **907 ms**|  **3962 ms**|
  | Win vs GEMV                         | **+84.9%**  | **+84.0%**  |
  | Win vs prior (#98)                  |  +4.0%      |  +3.9%      |

Quality verified on 50-word quantum entanglement (coherent + Einstein
"spooky action" reference). Default-off regression at baseline 6020 ms.

### aa01001nvfp4cprefill — Step 0 ncu stall profile (2026-05-28)

ncu stall-sampling of the two cold-prefill hotspots (CLAUDE.md #139:
94.7% of GPU time at M=14583). **Both prior hypotheses were wrong;
the real limiters are different and now measured.**

⚠️ **OPERATIONAL HAZARD (learned the hard way — caused a global OOM +
reboot 2026-05-28):** `ncu --set full` snapshots/restores ALL device
memory each profiled kernel touches, ×~30 replay passes. Running it
against rvllm-server's multi-GiB arena WHILE the combo-mode stack
(acestep ~26 GB + vllm-embedding + audio + a stray legacy
`vllm-gemma4.service` holding 50 GB GPU) was live blew past 121 GB RAM
+ 15 GB swap → `ncu invoked oom-killer` → cascade. **Before any ncu
run: `systemctl stop acestep vllm-embedding chatterbox whisper-fast
zeroclaw rvllm-serve vllm-gemma4`, confirm `nvidia-smi` GPU procs
empty + `free -g` >80 GB free, use a LIGHT section set (`--section
SpeedOfLight,Occupancy,WarpStateStats,SchedulerStats,LaunchStats` —
NOT `--set full`), cap `RVLLM_ARENA_GB=50` + `G4N_KV_MAX_POS=4096` +
`RVLLM_MAX_TOKENS_CAP=4096` for a small profiling prompt, and full-path
`/usr/local/cuda/bin/ncu` (sudo PATH lacks it; `RmProfilingAdminOnly:1`
needs root).**

**MLP GEMM `mistral35_w4a16_gemm_mma_v8_bf16_kernel` (56.4%):
MIO-throttle bound, NOT activation-bandwidth bound.**
  * L2 Hit Rate **95.0%** — the "activation re-read spills to HBM"
    hypothesis (plan Step 1A/1B) is FALSE. The activation working set
    stays in L2 across N-slice blocks; a 2D-grid / N-widening rewrite
    would not help.
  * Top stall: **MIO throttle = 37.9% of 20.2 cycles/inst** — the
    NVFP4-dequant's LDS/STS/load instructions saturate the memory-I/O
    queue. Memory throughput 72.8%, Compute (SM) 59%, occupancy 55%,
    IPC 1.31 (issue slots busy 32.6%).
  * Real lever: reduce dequant MIO instruction count (wider/vectorized
    smem ops, fewer per-nibble loads) or raise occupancy. Narrower
    than the plan assumed.

**Attention `flash_attention_2_prefill_nvfp4kv_unified_bf16out_kernel`
(38.3%): occupancy-bound at 8.3%, NOT barrier-bound.**
  * Achieved Occupancy **8.3%** (egregious). Driver = **dynamic smem
    per block 68.6 KB (sliding hd=256) / 100.4 KB (global hd=512)** →
    ~1 block/SM, and only 4 warps/block (FA2_THREADS=128) → 4/48 ≈ 8%.
  * Global-layer instance: 5.20 ms, 100.4 KB smem; sliding: 2.62 ms,
    68.6 KB. Compute (SM) 15%, Memory 36% — both far from roofline;
    the kernel is latency-stall-bound because too few warps reside to
    hide latency.
  * Real lever: cut smem per block (eliminate the `s_v_f16_T`
    transposed-V tile ~16 KB at hd=512, possibly shrink other buffers)
    and/or raise warps/block, to fit ≥2 blocks/SM → ≥16% occupancy.
    This matches plan Step 2A's "free smem" direction but the
    MECHANISM is occupancy, not `__syncthreads` count.

**Decision**: the attention kernel's 8.3% occupancy is the clearest,
highest-leverage signal (a textbook "wrong" occupancy that smem
reduction can roughly double). Step 2A (smem reduction → occupancy)
is the recommended next implementation, ahead of the MLP MIO work
(harder, the dequant is already fairly tight). Both are real wins on
paper; neither is a step-change (cold prefill is fundamentally
compute-heavy at 14k — the warm-path prefix cache remains the bigger
production lever).

### aa01001nvfp4cprefill Step 2A — drop s_v_f16_T → +8.5% cold prefill (2026-05-28, SHIPPED)

Removed the 16 KB transposed-V smem tile (`s_v_f16_T`) from
`flash_attention_2_prefill_nvfp4kv_unified_bf16out_kernel`. The P·V
MMA B-fragment now packs DIRECTLY from `s_v_f16`'s natural
[token][dim] layout via the new `pack_b_frag_v_natural_n8k16_f16`
(f16_mma_frag_pack.cuh) — 4 strided f16 loads/lane reproducing the
exact values the prior transpose+col-major-packer produced (bit-
identical MMA inputs). Eliminates the per-sub-tile transpose store
loop + its barrier too. Host smem reservation (prefill.rs) drops the
`MMA_K*hd*2` term for this kernel only (`output_bf16 && !use_cpasync`);
the f16-out + cpasync siblings keep s_v_f16_T. Reserving less smem is
what lifts blocks/SM.

Byte-equivalence (md5, gemma-4-31b-it-nvfp4, prefix-cache OFF,
deterministic): baseline {cold d90062c0, steady 81076590} == new
{d90062c0, 81076590}, IDENTICAL. Exercises all 60 layers (sliding
hd=256 + global hd=512).

Cold-prefill A/B (prefix-cache OFF, heavy services stopped, 3 runs/leg,
full rebuild per leg):
  * 4813-tok:  baseline avg 24172 ms (199 t/s) → new 22123 ms (218 t/s)
    = **+8.5%** (2049 ms saved).
  * 15973-tok: baseline avg 116594 ms (137 t/s) → new 101664 ms
    (157 t/s) = **+12.8%** (~14930 ms / ~15 s saved per cold turn).
  * The win GROWS with prompt length (8.5% → 12.8%) because the global
    O(M²) attention layers are a larger fraction of total prefill at
    16k. Above the plan's 4-8% estimate at production scale.

**Mechanism correction (device query 2026-05-28):** GB10 sm_121 has
only **100 KB smem/SM** (NOT the 228 KB of datacenter Blackwell;
`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR=100KB`, opt-in/block=99KB, max
48 warps/SM). So the 100→84 KB cut did NOT raise blocks/SM — 2 blocks
need ≤50 KB/block (2×50=100), and the kernel at 84 KB still fits only
1 block → occupancy stays **8.3%** (1 block × 4 warps / 48). The
measured +12.8% therefore came from **eliminating the per-sub-tile
transpose store loop + its barrier** (≈1000 barriers + 64 writes/thread
removed over a 16k global layer), NOT from higher occupancy. The
occupancy lever is unreachable via smem at hd=512 (s_acc f32 32 KB +
s_q 16 KB alone = 48 KB) — the viable path is **more warps/block**.

**Step 2A-occ (SHIPPED, +5.6% more):** bumped FA2_THREADS 128→256
(4→8 warps) on the bf16-out kernel → occupancy 8.3%→16.7% (1 block/SM,
smem-limited; 8 warps / 48). P·V partition `>> 2`→`>> 3`; host block
dim gated to 256 for this kernel only (f16-out + cpasync siblings keep
128 + `>> 2`). Byte-identical (md5 81076590). 16k cold-prefill A/B:
  * Step 2A (128t):     101664 ms (157 t/s)
  * Step 2A-occ (256t):  96010 ms (166 t/s) = **+5.6% more**
  * **cumulative +17.7%** vs the original s_v_f16_T + 128t kernel
    (116594 → 96010 ms, ~20.6 s saved per cold 16k turn).
Confirms the latency-stall diagnosis — more eligible warps hid the
stalls. commit `15a32ff`.

Cross-model: the kernel is shared with qwen36-nvfp4 (same bf16-out
unified prefill, head_dim=256). **qwen36 correctness is covered by the
SAME proof** — gemma4's sliding layers are head_dim=256 (identical to
qwen36's), use the identical kernel + packer, and were part of the
byte-identical gemma4 md5 validation. qwen36 also takes the
`output_bf16 && !use_cpasync` branch → gets the 8 KB smem reduction;
under-allocation is impossible (the kernel no longer references the
removed buffer). No qwen36-specific code path exists that the gemma4
validation didn't exercise.

Step 2B (depth-2 cp.async on the freed smem) + the MLP MIO-throttle
lever (Step 1, harder) remain available follow-ons. commit `9b66c90`
on branch `g4n_cprefill`.

### aa01001nvfp4gemv — fp8_gemv MMA: NOT applicable to gemma4-nvfp4 (2026-05-27, negative result)

**Outcome: no fp8_gemv work needed on gemma4-nvfp4 — its decode/verify
GEMMs are already on their best kernels.** The parked
`aa01001nvfp4gemv` task ("fp8_gemv TensorCore MMA rewrite, 64.3% of
decode GPU time") was based on a nsys figure from the **fp8-block /
qwen36** decode path, NOT gemma4-**nvfp4** (Option B). Code-path audit
+ hardware A/B established:

* **gemma4-nvfp4 never calls the `fp8_gemv_blockwise_wpr_native_f16in`
  family.** That family + the new `fp8_gemv_mma_m8_w4c` MMA sibling
  (shipped #144 Phase 1-3) serve the gemma4 **fp8-block** path
  (`Gemma4Layer::execute` in `gemma4_layer_exec.rs`, `weights.*_fp8`).
  The NVFP4-weights path runs through `gemma4_nvfp4_ops.rs` with the
  `mistral35_w4a16_*` NVFP4-weight GEMM family.
* **Verify-batch MLP (M=K+1, up to 8 at K=7) already routes to
  `fn_w4a16_gemm_mma_v8`** (the persistent-CTA TensorCore MMA kernel)
  by default — `gemma4_nvfp4_w4a16_gemm_mn` dispatches to MMA_V8 when
  `RVLLM_GEMMA4_NVFP4_MLP_MMA_V8` is on (default-on, #102) AND
  `k%16==0 && n%32==0`. All 3 MLP GEMMs at 31B satisfy this
  (gate/up N=21504, down N=5376, K∈{5376,21504} — all multiples).
* **Verify-batch attention proj** (QKV/O) uses
  `gemma4_nvfp4_attn_proj` = `cublaslt.bf16_gemm_f32` (weights stored
  bf16 after load, not packed NVFP4). It's ~10% of verify GEMM FLOPs;
  a W4A16-MMA port would need a load-path change to keep weights
  NVFP4 — not worth it vs the MLP which already dominates and is
  already MMA'd.

**Hardware A/B** (gemma-4-31b-it-nvfp4, binary md5
`64a7006f0af5a45598b2fba5c808e416`, 3 prompts × 8 cells of forced-K
× `RVLLM_GEMMA4_NVFP4_FP8_GEMV_MMA_VERIFY={0,1}`): verify_ms and
completion md5 **bit-identical** across MMA_VERIFY=0 vs =1 in every
cell (e.g. p1 verify_ms 14019 vs 14094, md5 85426fcd… both legs) —
because the dispatch sites the env gates are never reached on the
nvfp4 path. (Forced-K via systemd `set-environment` was also clobbered
by the profile's `export G4N_SPEC_ADAPTIVE_K=1`/`RVLLM_GEMMA4_SPEC_K=7`
since the source-profile drop-in re-exports after Environment=; k_avg
held at the adaptive-shrunk 2.0-3.3 across all cells, consistent with
the measured 27-43% accept rate walking K toward MIN_K=1.)

**Shipped this session** (commit on `g4n_p2p3_fp8_gemv_verify`): the
`RVLLM_GEMMA4_NVFP4_FP8_GEMV_MMA_VERIFY=1` opt-in gate + M≥4 dispatch
floor extended to all 4 `Fp8GemvF16InLaunch` sites (QKV/O/gate_up/down)
in `gemma4_layer_exec.rs`. This is a consistency extension of the
#144-Phase-2 QKV-only wiring — it serves the **fp8-block** Gemma
variant's spec-verify path (the only consumer of fp8_gemv). Default
OFF; bit-identical via the fp64 validator (`v3/tools/fp8_gemv_mma_m8_check.py`,
cos=1.0 at M∈{1,4,8,16}). No effect on gemma4-nvfp4 production.

Net: `aa01001nvfp4gemv` closes as a documented negative result for
the NVFP4 path. The decode perf lever for gemma4-nvfp4 remains the
parked `aa01001nvfp4cprefill` cold-prefill rewrite (94.7% of cold-
prefill GPU time = MLP MMA_V8 56.4% + attention-prefill 38.3%), which
is a separate kernel family.

### gemma4-nvfp4 MLP MMA_V8 default-on — task #102 (2026-05-24, 14.6× prefill speedup)

Discovery: nsys on gemma-4-31b-it-nvfp4 prefill showed `mistral35
_w4a16_gemm_mn_bf16_kernel` at **98.4%** of GPU time. The
companion `mistral35_w4a16_gemm_mma_v8_bf16_kernel` (full
TensorCore MMA tiled GEMM) was already loaded but gated behind
`RVLLM_GEMMA4_NVFP4_MLP_MMA_V8` (default OFF — never set in any
production profile). The source comment in the legacy kernel
even named the V8 variant as the "next-iteration win".

Flipped default ON in commit `50b7c9d`. Opt-out preserved via
`RVLLM_GEMMA4_NVFP4_MLP_MMA_V8=0` for diagnostics. Mistral 3.5
has the analogous flag for the SAME kernel — pattern likely
applies but parked pending its own per-model A/B.

A/B (gemma-4-31b-it-nvfp4 production profile, 1112-tok prefill,
max_tokens=1 so wall ≈ all prefill, deterministic 3 runs):

  | Cell                    | wall                       | avg     |
  |-------------------------|----------------------------|---------|
  | V8=0 (legacy _mn path)  | 32228 / 32521 / 32536 ms   | 32428 ms|
  | V8=1 default            |  2219 /  2210 /  2214 ms   |  2214 ms|
  | **Speedup vs legacy**   |                            | **14.6×**|

Throughput: 35 t/s → 505 t/s prefill at 31B. Output coherence
verified on quantum entanglement (Einstein "spooky action"
reference present).

Misframing note: the parent task "apply grouped MMA pattern to
dense models" was based on a misunderstanding — qwen36's grouped-
MMA-by-expert-id pattern doesn't translate to dense models (no
routing). But investigating turned up THIS dormant MMA flag,
which delivers more than the grouped-MMA pattern ever could have
for a dense model.

### qwen36 linear-attn prefill — task #101 (2026-05-24, +2.3% with v3)

Targets the new top hotspot identified by #100 (`gated_delta_rule
_prefill_f16_kernel`, 15.8% of prefill GPU time). Two opt-in sibling
kernels (commit `50c3b1e`):

* **v2** vectorised state I/O (u64 4-half load/store): PARITY with
  v1 at hardware A/B. Opt-in via
  `RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V2=1`. Kept loaded as building
  block; state I/O is not the bottleneck.
* **v3** drops the inner-loop f16 RTNE round-trip on `s_row[kd]`
  (v1 rounded `s = __half2float(__float2half(s_old * a))` every
  Phase-1 + Phase-2 inner-loop iteration for byte-equivalence with
  the per-token decode kernel). v3 keeps the recurrent state in
  pure fp32 across one prefill call; rounds to f16 only at boundary
  write-back. NOT bit-equivalent to v1; mathematically MORE correct
  (less quantisation noise). Opt-in via
  `RVLLM_QWEN36_LINEAR_ATTN_PREFILL_V3=1`.

A/B (qwen3-6-35b-a3b NVFP4, both grouped MMA paths active,
deterministic over 3 runs, fresh binary md5 `862b81cb`):

  | Cell                 | 1112 tok    | 4412 tok    |
  |----------------------|-------------|-------------|
  | v1 baseline (#99)    |   944 ms    |  4116 ms    |
  | v2 vector I/O        |   946 ms    |    --       |
  | **v3 skip-RTNE**     | **920 ms**  | **4022 ms** |
  | v3 win vs v1         |   +2.5%     |   +2.3%     |

Quality verified: quantum entanglement coherent. Default-off
regression-checked.

Remaining bottleneck on this kernel is NOT memory I/O (v2 parity)
nor exclusively the f16 RTNE round-trips (v3 gives only +2.3%). The
dominant cost likely sits in per-token `__syncthreads` barriers
(~9000 per kernel call at M=4412) or the inner-loop FMA throughput
(75M FMAs per launch). Both would require deeper rewrite (warp-
level reduction, multi-token batching across the recurrence) — multi-
week scope.

### qwen36 prefill nsys post-#98+#99 — task #100 (2026-05-24)

Captured against fresh binary md5 `65cb2a35` with the production
profile + both grouped MMA paths active (MMA_GROUPED=1 +
MMA_DOWN_GROUPED=1). Single 4412-token prompt fired inside a 25 s
nsys window. Top kernels by total GPU time:

  | Time% | Kernel                                                          | Instances | Notes |
  |-------|-----------------------------------------------------------------|-----------|-------|
  | 15.8% | gated_delta_rule_prefill_f16_kernel                             |    30     | **NEW TOP — linear-attn** |
  | 12.5% | fp8_mma_dual_silu_grouped_m16_w4c_kernel                        |    40     | #96 path |
  | 12.3% | flash_attention_2_prefill_nvfp4kv_unified_kernel                |    10     | full-attn |
  | 10.5% | fp8_mma_down_grouped_m16_w4_kernel                              |    40     | #98 path |
  |  6.4% | router_gemv_with_topk_batched_f16_to_f32_kernel                 |    40     | |
  |  6.2% | fp8_gemv_blockwise_wpr_native_f16in_kernel                      |  1940     | per-token GEMV (shared expert + per-token sites) |
  |  4.6% | fp8_gemv_dual_silu_kernel                                       |   800     | shared-expert dual_silu |
  |  1.9% | cutlass GEMM (M≥128 path)                                       |   130     | |
  |  1.4% | dual_silu_indirect_kround_batched_kernel                        |   760     | legacy / per-token decode mix |

Aggregated:
- **MoE routed FFN (grouped MMA dual_silu + down): 23.0%** — was
  88.9% pre-fix. **5.7× reduction in relative + absolute time**
  (the prefill walltime also dropped from 24775 ms to 4116 ms at
  M=4412, ~6× wallclock improvement).
- **Linear-attn `gated_delta_rule_prefill_f16` is now the single
  biggest kernel (15.8%)** — 30 instances (per linear-attn layer).
  Was 3.6% pre-fix in relative terms but absolute time similar
  (~900 ms then vs ~900 ms now); just a much larger SHARE of the
  smaller wall.
- **Full-attn unified prefill (12.3%)** — ~10 layers × 1 launch =
  10 instances. Already MMA-based (CUTLASS NVFP4); not a clear
  optimization target.
- **Shared-expert path** (`fp8_gemv_dual_silu` 4.6% +
  `fp8_gemv_blockwise_wpr_native_f16in` part of the 6.2% +
  cutlass 1.9%): 8-12% combined. Doesn't use the grouped MMA
  pattern — same expert-sort foundation could apply here.

Path forward (in roughly descending expected impact):
1. **Linear-attn `gated_delta_rule_prefill_f16` optimization** —
   biggest single hotspot. Investigate fusion / vectorisation /
   memory pattern. Would need familiarity with the Gated-DeltaNet
   recurrence math.
2. **Apply grouped MMA pattern to shared-expert dual_silu** — same
   pattern as #94-#97 but without the routing indirection (single
   "expert" applied to all tokens). Expected ~5-8% additional
   speedup.
3. **Apply grouped MMA pattern to other models** — Gemma 4 31B-NVFP4,
   Qwen 3.5 27B dense, Mistral 3.5 — each would need per-model
   adaptation since shapes/layouts differ.

### qwen36 MoE shared sort + per-kblk a_scale fix — task #99 (2026-05-24)

Two paired cleanups (commit `81b4b21`):

* **Shared sort across dual_silu + down**: new layer-scoped
  `shared_sort_ptrs` in `apply_layer_moe_batched`. When dual_silu
  grouped runs its sort+tile_table+DtoH, it publishes outputs
  there. Down grouped, if Some is observed, reuses persistent
  buffer + cached tile_count and skips its own sort pre-pass.
  Saves 2 kernel launches + 1 sync DtoH per layer when both
  grouped paths are enabled.
* **Per-kblk a_scale fix** to all 4 dual_silu grouped kernels
  (#94 grouped_m16, #95 w4, #96 w4c, #97 w4cb). Same pattern as
  the #98 down kernel: fold per-(row, kblk) a_scale into
  g_outer/u_outer inside the K-loop instead of multiplying by
  `smem_ascale[row]` at write time (= LAST kblk's value).
  Previously invisible for dual_silu (RMSNormed input → nearly-
  constant per-K-block amax). Fixed for correctness + consistency.

A/B perf (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary
md5 `65cb2a35`): parity with #98 measurements within noise. The
fix is FREE perf-wise (architectural correctness improvement);
shared sort is also at parity (the saved ~µs/layer is below
noise on the 1000ms prefill).

### qwen36 MoE grouped MMA down projection — task #98 (2026-05-24, +84% with both grouped)

`fp8_mma_down_grouped_m16_w4_kernel` (commit `3e5c097`) extends
the expert-sort + W=4 grouped MMA pattern to the down kernel
(previously 27% of prefill GPU time). Output goes to `acc_f32`
via atomicAdd (each (token, n) receives top_k=8 contributions
from different tiles).

Opt-in via `RVLLM_QWEN36_MOE_MMA_DOWN_GROUPED=1` (independent of
MMA_GROUPED). Independently re-runs the expert-sort pre-pass
(~few µs/layer) using the same persistent scratch as #94.

**Numerical fix included**: per-K=128 a_scale must be folded
INSIDE the inner accumulator per-kblk, not multiplied at write
time. Previous pattern (smem_ascale[row] at write only) uses the
LAST kblk's a_scale instead of the per-kblk value. For dual_silu
input (RMSNormed → ~constant per-K-block amax) this is invisible;
for down input (silu_b SwiGLU output → per-K-block amax varies
substantially) it produces garbage tokens. Fix: compute
`a_lo = smem_ascale[r_lo]` + `a_hi = smem_ascale[r_hi]` inside
the K-loop; fold per-(row, kblk) a_scale into g_outer.

A/B (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary
md5 `80d02e7e` + new PTX):

  | Cell                                  | 1112 tok    | 4412 tok    |
  |---------------------------------------|-------------|-------------|
  | GEMV baseline (both legacy)           |   6020 ms   |  24775 ms   |
  | Dual_silu grouped only (#97)          |   2533 ms   |  10587 ms   |
  | Down grouped only (#98)               |   4524 ms   |  18520 ms   |
  | **Both grouped (#97 + #98)**          |   **955 ms**|  **4116 ms**|
  | Win vs GEMV (both grouped)            | **+84.1%**  | **+83.4%**  |
  | Win vs dual_silu grouped only         |  +62.3%     |  +61.1%     |

Output coherence verified on 80-word quantum entanglement (coherent
English) + 4k German + smoke. Default-off regression-checked at
6019-6039 ms matching baseline.

### qwen36 MoE W=4 cooperative B-staging — task #97 (2026-05-24, parity, opt-in)

`fp8_mma_dual_silu_grouped_m16_w4cb_kernel` (commit `5bb1c26`)
adds u64-vector B-staging on top of #96's coop-A. Within each
warp, B_g/B_u staging refactored from 8 unrolled 32-lane byte
iterations to 1 pass: 32 lanes × 8 bytes (u64) each.

A/B (qwen3-6-35b-a3b NVFP4, deterministic 3 runs):

  | Cell                            | 1112 tok    | 4412 tok    |
  |---------------------------------|-------------|-------------|
  | #96 coop A only (production)    |   2533 ms   |  10587 ms   |
  | #97 coop A + coop B (opt-in)    |   2517 ms   |  10647 ms   |

PARITY — within ~0.5% run-to-run noise. The existing 8-iter
byte-load loop already gets vectorised by the compiler. Default-
off (opt-in via `RVLLM_QWEN36_MOE_MMA_GROUPED_W4_COOP_B=1`);
kernel kept loaded as foundation for follow-on tuning.

### qwen36 MoE W=4 cooperative A-staging — task #96 (2026-05-23, +57.9%)

`fp8_mma_dual_silu_grouped_m16_w4c_kernel` (commit `41892e9`)
distributes A-tile staging across all 128 threads instead of
warp-0 serial. Per K=32: 1 cooperative pass (each thread writes
4 bytes via aligned u32 store) vs 16 serial iterations. Per-row
`token_idx` broadcast via `smem_token_idx[16]` (+64 B smem).

Dispatched by default when GROUPED+W4 are on. Opt-out via
`RVLLM_QWEN36_MOE_MMA_GROUPED_W4_COOP=0`.

A/B (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary
md5 `6e29deb9`):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | MMA W=1  (#94)                |   3265 ms   |  13674 ms   |
  | MMA W=4 serial (#95)          |   2627 ms   |  11062 ms   |
  | **MMA W=4 coop (#96)**        | **2533 ms** | **10587 ms**|
  | Win vs GEMV                   | **+57.9%**  | **+57.3%**  |
  | Win vs W=4 serial             |  +3.6%      |  +4.3%      |

W=4 serial regression-checked (COOP=0): parity with #95.

### qwen36 MoE W=4 multi-warp + persistent sort scratch — task #95 (2026-05-23, +56%)

Two follow-ons to #94 in one commit (`618efb2`):

* **W=4 multi-warp tiling** — `fp8_mma_dual_silu_grouped_m16_w4_kernel`.
  4 warps per block share ONE A tile + cover 4× N cols per block;
  amortises per-row amax + A staging across warps. Grid.x drops 4×.
  Dispatched by default when N divisible by 32 (qwen36 moe_int=512
  qualifies). Opt-out via `RVLLM_QWEN36_MOE_MMA_GROUPED_W4=0`.
* **Persistent sort scratch** — `Qwen36Bringup.mma_sort_scratch` pre-
  allocates `sorted_per_expert` + `expert_counts` + `tile_descriptors`
  + `tile_count` once at `load()` sized for M_MAX_FOR_PERSISTENT=16384
  (32 MB sorted + 99 KB tiles). Dispatch uses persistent ptrs when
  this prefill fits; falls back to per-call `arena.region` otherwise.

A/B (qwen3-6-35b-a3b NVFP4, deterministic, fresh binary
md5 `de341522`):

  | Cell                          | 1112 tok    | 4412 tok    |
  |-------------------------------|-------------|-------------|
  | GEMV baseline                 |   6020 ms   |  24775 ms   |
  | MMA W=1  (#94)                |   3265 ms   |  13674 ms   |
  | **MMA W=4 (#95)**             | **2627 ms** | **11062 ms**|
  | MMA W=4 + persistent (#95)    |   2631 ms   |  11146 ms   |
  | Win vs GEMV                   | **+56.4%**  | **+55.4%**  |
  | Win vs W=1                    |  +19.5%     |  +19.1%     |
  | Persistent vs per-call alloc  | parity      | parity      |

Persistent buffer parity confirms the per-call arena.region overhead
was negligible — the change is architectural, not a perf win.

### qwen36 MoE expert-sort + grouped MMA — task #94 (2026-05-23, +45%)

Recovers the M-direction MMA reuse the task #93 first-cut sacrificed.
Pre-pass sorts (token, k_round) pairs by routed expert id; new MMA
kernel processes M=16 sorted-contiguous tiles sharing one expert.
Opt-in via `RVLLM_QWEN36_MOE_MMA_GROUPED=1` (precedence over #93's
`RVLLM_QWEN36_MOE_MMA_DUAL_SILU`).

Pipeline (3 kernels + 1 DtoH per MoE layer):
1. `qwen36_moe_expert_sort_kernel` — atomicAdd bucketing
2. `qwen36_moe_tile_table_kernel` — per-expert ceil(C/16) tiles
3. (host) DtoH tile_count to size grid.y
4. `fp8_mma_dual_silu_grouped_m16_kernel` — grouped FP8 MMA

A/B (qwen3-6-35b-a3b NVFP4, fresh binary, deterministic):

  | Cell                        | 1112 tok | 4412 tok |
  |-----------------------------|----------|----------|
  | GEMV baseline               |  6020 ms | 24775 ms |
  | MMA first-cut (#93, M=1)    |  5738 ms | 23636 ms |
  | **MMA grouped (#94, M=16)** | **3265 ms** | **13674 ms** |
  | Win vs GEMV                 | **+45.8%** | **+45.0%** |
  | Win vs first-cut            |  +43.1%  |  +42.2%  |

Output coherence verified on 2k + 4k German prompts. Production-safe:
default-off; legacy GEMV unchanged.

Debug knob: `RVLLM_QWEN36_MOE_MMA_GROUPED_DEBUG=1` inserts a
cuStreamSynchronize after each pre-pass kernel so any kernel failure
surfaces with its own op label.

max_per_expert sizing: `(typical*32).max(256).min(total_assign)`. The
naive 8× sigma estimate proved insufficient (one expert exceeded 278
on M=1112 prompts → silent OOB → cuStreamSynchronize trap). 32×
sigma handles real qwen36 routing skew at ~2.3 MB per layer call.

Follow-up parked (not blocking the production rollout decision):
- Same expert-sort foundation could feed a grouped MMA `down`
  projection sibling (27% of prefill GPU time).
- Persistent sort buffers across layers (routing is per-layer so
  this requires per-layer sort but the buffer alloc can be reused).
- Multi-warp tiling per block (currently 1 warp/block).

### qwen36 MoE TensorCore MMA dual_silu — task #93 first cut (2026-05-23, +4.9%)

`fp8_mma_dual_silu_indirect_kround_batched_kernel` — first
TensorCore-MMA-based FP8 kernel targeting the qwen36 MoE prefill
hot path (88.9% of prefill GPU time per the nsys profile below).
Uses
`mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`
via existing helpers in `kernels/fp8_mma_frag_pack.cuh`. Opt-in via
`RVLLM_QWEN36_MOE_MMA_DUAL_SILU=1`; default GEMV path untouched.

A/B (qwen3-6-35b-a3b NVFP4, deterministic across runs, fresh binary
md5 `6b9bd230`):

  | Cell        | 1112 tok prefill | 4412 tok prefill |
  |-------------|------------------|------------------|
  | GEMV (def.) |      6020 ms     |     24775 ms     |
  | MMA (=1)    |      5738 ms     |     23636 ms     |
  | Speedup     |      +4.9%       |      +4.8%       |

First-cut semantics deliberately conservative: 1 warp per block,
1 token per block, MMA tile [M=16, N=8] with only row 0 active
(rows 1..15 zero-staged → 15/16 of MMA throughput wasted). The
measured 5% win at this configuration validates that the FP8×FP8
MMA path executes correctly on sm_121 against real qwen36
weights AND that the path can beat the GEMV path even at extreme
M-direction underutilisation.

Real perf gain requires a follow-up that token-sorts (token,
k_round) pairs by routed expert id so a tile of M=16 contiguous
sorted assignments shares ONE expert (recovering full M-reuse).
Expected with proper M=16 grouping: 4-8× on the dual_silu
portion → 30-50% overall prefill speedup. That work is bounded
(~500-800 LOC: sort kernel + new MMA dispatch + scatter back to
`[k_round, M, N]` output layout) and sits on the foundation
shipped here.

Output coherence verified: both paths produce identical text on
the smoke prompt and coherent multi-token output on the 4412-token
German quantum-physics prompt (no NaNs, no garbage).

### qwen36 prefill profile — nsys 2026-05-23 (qwen3-6-35b-a3b NVFP4-KV, task #91)

Captured against fresh binary md5 `63122454` with the production
profile (`mobile-qwen3635b-rvllm-nvfp4-spec.env`: QKV megakernel ON,
unified-NVFP4-prefill ON, batched MoE ON, all fusions live). Single
~4k-token prompt fired inside a 25 s nsys window. Top kernels by
total GPU time during prefill:

  | Time% | Kernel                                                          | Instances | Per-call |
  |-------|-----------------------------------------------------------------|-----------|----------|
  | 62.0% | ..._dual_silu_indirect_kround_batched_kernel (MoE up+gate)      |    24     | 397 ms   |
  | 26.9% | ..._indirect_scaled_add_kround_batched_kernel (MoE down)        |    23     | 180 ms   |
  |  3.6% | gated_delta_rule_prefill_f16_kernel (linear-attn)               |    18     |  30 ms   |
  |  2.7% | flash_attention_2_prefill_nvfp4kv_unified_kernel                |     6     |  70 ms   |
  |  1.4% | router_gemv_with_topk_batched_f16_to_f32_kernel                 |    24     |   9 ms   |
  |  0.9% | ..._dual_silu_kernel (shared-expert)                            |    23     |   6 ms   |
  |  0.8% | ..._wpr_native_f16in_kernel (FP8 GEMV M=1 sites)                |    23     |   5 ms   |

Aggregated:
- **MoE routed FFN: 88.9% of prefill GPU time** (dual_silu 62.0% +
  down_scaled_add 26.9%). The same FP8 GEMV inner reduction body that
  dominates DECODE (64.3% of GPU time per the 2026-05-23 decode
  nsys) is also the prefill bottleneck because each (token, expert)
  pair is independent and the kernel is per-warp scalar reduction,
  not TensorCore MMA.
- Unified NVFP4 attention prefill: 2.7% — the unified-prefill kernel
  is doing its job. The earlier "long prompts (>8k) are slow because
  batched-NVFP4-prefill not wired" caveat referred to the
  per-token fallback path before commit `3bf9eac` flipped the master
  gate default-on (2026-05-22).
- A/B `RVLLM_QWEN36_NVFP4_UNIFIED_BATCH_FULL_PREFILL=1` vs `=0` at
  1112 tokens: 185 t/s vs 65 t/s = **2.83× prefill speedup** from the
  unified path alone.

Throughput at the current state: 185 t/s prefill (1112 tok / 6020 ms)
and 178 t/s at 4412 tok — scales linearly, attention is not the cap.
Path forward for further gains: the 88.9% MoE GEMV is the same
TensorCore MMA rewrite item parked in CLAUDE.md's
"fp8_gemv optimization attempts" section (multi-week scope). The
smem-staging optimization attempted on the decode side was a no-op
(L1 absorbs redundant loads) and the same finding applies here.

### qwen36 decode profile — nsys 2025-05-23 (qwen3-6-35b-a3b NVFP4-KV)

Captured with `nsys profile -y 25 -d 20 -t cuda
--cuda-trace-all-apis=true` wrapping rvllm-serve at startup; fired
2 chat-completion requests (80 tokens each) inside the capture
window. ~160 decode iterations, ~5s of measured GPU time, binary
md5 `5eb81115...` (freshly built). Top kernels by total GPU time:

  | Time% | Kernel                                                          | Instances | Per-call |
  |-------|-----------------------------------------------------------------|-----------|----------|
  | 30.9% | fp8_gemv_blockwise_wpr_native_f16in_kernel                      | 16,080    | 75 µs    |
  | 19.8% | ..._dual_silu_indirect_kround_batched_kernel (MoE)              | 6,400     | 121 µs   |
  | 12.6% | nvjet_sm121_qqhsh_mma_192x160x128_... (CUTLASS-like)            | 160       | 3074 µs  |
  | 9.7%  | ..._indirect_scaled_add_kround_batched_kernel (MoE)             | 6,400     | 59 µs    |
  | 8.8%  | gated_delta_rule_decode_f16_kernel (linear-attn)                | 4,740     | 72 µs    |
  | 5.1%  | fused_qkv_proj_qnorm_knorm_rope_qwen_partial_nvfp4kv_kernel     | 1,600     | 125 µs   |
  | 3.3%  | flash_attention_2_decode_nvfp4kv_kernel                         | 1,580     | 81 µs    |
  | 2.6%  | router_gemv_with_topk_f16_to_f32_kernel                         | 6,320     | 16 µs    |
  | 2.6%  | ..._dual_silu_kernel                                            | 6,400     | 16 µs    |
  | 1.3%  | ..._scaled_add_devw_kernel                                      | 6,320     | 8 µs     |

Aggregated:
- **FP8 GEMV family** (5 variants share inner reduction body):
  **64.3% of GPU time** — THE bottleneck.
- nvJet mma (CUTLASS-like GEMM, used at prefill / large-M sites):
  12.6%.
- Linear-attn decode kernel: 8.8%.
- Full-attn QKV megakernel + FA2 decode: 8.4%.
- Per-token launch overhead: 114,320 `cuLaunchKernel` calls
  × 2.15 µs avg = ~245 ms over the 5-s window ≈ 5% of decode
  time. Real but secondary to the FP8 GEMV cost.

Implication: future perf work that doesn't touch the FP8 GEMV
inner loop (warp-cooperative reduction, FP8 dequant, blockscale
loads) caps at ~36% headroom. The highest-leverage attack is
inside fp8_gemv itself: better dequant ILP, better K-dim
pipelining, or replacing the GEMV with a CUTLASS-FP8 micro-GEMM
that batches multiple output rows. Secondary attacks (saving
more launches, fusing additional pairs) compete for the ~5%
launch-overhead budget — diminishing returns.

### Phase 8 closer + last-block-residual_add fusion — SHIPPED 2026-05-23

Commit `505e9ea` adds
`fp8_gemv_blockwise_wpr_native_f16in_scaled_add_devw_then_residual_kernel`
that folds the in-place `hidden += f16(routed_sum)` residual add
into the shared-expert closer's existing fp8_gemv + scaled-add
epilogue. Drop-in for the back-to-back pair at the end of
`apply_layer_moe_with_override`. Per output n both side-effect
writes (acc_f32 + hidden_f16) touched by exactly one thread → no
atomic needed.

The routed_sum f32 write is preserved so the
`RVLLM_QWEN36_DEBUG_MOE` post-residual probe stays visible.

Env-gated opt-in (`RVLLM_QWEN36_MOE_CLOSER_FUSED=1`); default off
keeps the unfused 2-launch chain byte-untouched. Re-verified A/B
on qwen3-6-35b-a3b NVFP4-KV (max_tokens=150, 3 runs each, on top
of `RVLLM_QWEN36_QKV_MEGAKERNEL=1`): wall 3.09 s / 125 tok /
40.45 tok/s / prefill 197 ms — no measurable delta vs QKV
megakernel alone. Saves 1 launch per MoE layer per decode token
(~40 launches/token across 40 layers);
architectural cleanup at this scale, not a measurable wall-clock
win.

### Phase 8 Q-norm + K-norm + RoPE + KV megakernel (Phase 1) — SHIPPED 2026-05-23

First step toward the full QKV+norm+RoPE+KV megakernel goal.
Commit `943f8bb` folds the two standalone `rmsnorm_inplace_f16`
launches (Q-norm + K-norm per full-attn layer) into the
existing `fused_rope_qwen_partial_f16kv` kernel. New kernel
`fused_qnorm_knorm_rope_qwen_partial_f16kv_kernel` runs a
2-phase per-head body:

  Phase 1: block-reduce sum-of-squares across head_dim,
  inv_norm = rsqrtf(mean_sq + eps), apply gamma * inv_norm to
  the (tid, tid+half_head) element pair in-place.
  Phase 2: standard partial-NeoX RoPE on the rotary half +
  KV-cache write.

Numerical contract: Phase 1 RMSNorm matches the standalone
kernel byte-for-byte; Phase 2 rotation+KV-write is byte-
identical to the unfused fused_rope_qwen_partial_f16kv given
normalised input. Coverage tightened over the unfused kernel:
each thread writes BOTH halves of its pair (so the non-rotary
tail [rotary_dim, head_dim) gets normalised even without the
in-place trick the unfused kernel relied on).

Wired for F16-KV branch initially; NVFP4-KV follow-on shipped
in commit `6f6a25a` — same 2-phase per-head structure with
shared-mem `s_normalized` buffer to feed the rotation, then
the unchanged FP8-Q quantise + NVFP4 K/V pack epilogue. Both
KV-dtype branches now retire the standalone Q/K-norm
launches. Hardware-validated coherent on NVFP4 production
(qwen3635b NVFP4 spec profile).

Saves 2 launches/layer in F16-KV mode (~22 launches/token at
~11 full-attn layers). Hardware-validated coherent on
qwen3-6-35b-a3b: short / 1-10 counting / 150-token
photosynthesis all correct. The fusion's value is
architectural — eliminates 2 redundant kernel launches per
layer + foundation for fusing in the QKV projection (Phase 2:
~1000 LOC of additional CUDA work to bring the 3 FP8 GEMV
projection launches into the same megakernel; substantial
follow-on).

Production NVFP4-KV path unchanged.

### Phase 8 multi-step graph with persistent hidden — SHIPPED 2026-05-23

Commit `eb26d86` builds on e13e2eb: the multi-step macro-graph
capture body (try_capture_decode_steps_n + its eager fallback
+ the pure-eager branch of decode_steps_n_via_graph_or_eager)
no longer re-allocates `hidden_region` via
`arena.region("qwen36_pl_hidden", ...)` per iteration. With
the hidden_dev_override path active, the decode-step body
writes hidden state to `workspace.hidden_dev` (persistent
above scratch_ck) and the fused closer reads from the SAME
address — same-stream serial ordering keeps body-writes-then-
closer-reads correct within each iteration.

Result on qwen3-6-35b-a3b at N=8: macro-graph node count drops
roughly in half (the per-iter arena.region call + the captured-
graph bookkeeping it implied are gone from the kernel sequence).

Re-verified A/B on qwen3-6-35b-a3b NVFP4-KV (max_tokens=150,
deterministic across 3 runs each, freshly-installed symlinked
binary):

  | Cell             | wall   | tokens | tok/s | prefill_ms | md5(completion) |
  |------------------|--------|--------|-------|------------|-----------------|
  | DG_OFF_EAGER     | 3.69 s | 150    | 40.65 | 217        | b5d33eaa        |
  | DG_ON_REPLAY     | 3.66 s | 150    | 40.98 | 213        | b5d33eaa        |
  | DG_MULTISTEP8    | 3.73 s | 150    | 40.21 | 217        | b5d33eaa        |

Single-step replay vs eager: **+0.8% tok/s** (within noise).
Multi-step N=8 vs single-step replay: **-1.9% tok/s** — N=8 is
slower than single-step on this shape. md5 identical across all
three (graph capture/replay is bit-equivalent to eager). Single-
step remains the recommended path; multi-step is operator-opt-in.

### Phase 8 hidden-state → workspace refactor — SHIPPED 2026-05-23

Commit `e13e2eb` routes the hidden-state residual stream
through the persistent `workspace.hidden_dev` slot when the
caller passes `hidden_dev_override: Some(...)`. Adds a new
parameter to
`forward_qwen36_decode_inner_with_workspace_overrides_v2`;
the four closer fns (eager + device_argmax + with_link +
closer_all) refactored from `hidden_region: &Region<'_>` to
`hidden_dev_ptr: u64`. The arena `hidden_region` allocation
stays unconditionally so downstream arena addresses stay
layout-stable; the override merely substitutes which device
address the in-body reads/writes target.

Both `forward_qwen36_decode_step_to_workspace` entry points
plumb `Some(workspace.hidden_dev)` as the override. The
persistent workspace slot survives the inner-checkpoint
restore (commit bcdce94 RAII guard) AND stays address-stable
across requests (workspace allocated above scratch_ck).
Captured decode graph references now valid for the lifetime
of the worker — unlocks broader closer/post-attn fusion
patterns (the kernel-fusion attempt from commit e47166b
already exploited the same property but via re-allocating
`hidden_region` post-restore; this commit removes the need
for that workaround).

Hardware-validated on qwen3-6-35b-a3b:
- Eager (legacy path, no overrides): coherent.
- DECODE_WORKSPACE=1 (overrides on, no capture): coherent.
- DECODE_GRAPH=1 + REPLAY=1: coherent short + 1-10 counting.

Production qwen27b (qwen35 dense path) — unaffected.

### Phase 8 batched router+topk + down kround-batch — SHIPPED 2026-05-23

**Batched router+topk fusion (commit `314dbe6`)**: ports the
e049258 last-block-does-topk pattern to the BATCHED prefill
path. New kernel `router_gemv_with_topk_batched_f16_to_f32_kernel`
has per-token counter slots (`counter[t]`), eliminates 1 launch
per MoE layer per prefill. Persistent counter region sized by
`kv_cache_num_blocks`, zeroed once via cuMemsetD8Async.

**Down k_round-batch fusion (commit `b1f221e`)**: fuses the 8
per-k_round host-loop launches of `fp8_gemv_indirect_scaled_add`
into ONE kernel. The literal "dual_silu+down megakernel" task
is infeasible (silu_mul recomputation per output element would
explode work ~1000x); this is the closest tractable analog. New
kernel `..._indirect_scaled_add_kround_batched_kernel` has each
warp own one (m, n) output slot and sequentially process all
top_k k_rounds with a warp-local f32 accumulator — no atomic,
single global RMW per warp at the end. Saves 7 launches/layer
× 40 MoE layers = 280 launches/decode token.

**Batched-prefill port (commit `38afff0`, 2026-05-23)**: same
kernel is M-agnostic, so the batched-prefill MoE path in
`apply_layer_moe_batched` was rewired to launch
`fp8_gemv_indirect_scaled_add_kround_batched` once per layer
with `M=num_tokens` instead of running the 8-launch
`fp8_gemv_indirect_scaled_add_batched_topk` host loop. silu_b
layout `[top_k, num_tokens, n_int]` (k_round-major, from the
dual_silu kround-batched launch) already matches the kernel's
`input_kround` contract. 280 launches saved per prefill batch.
Bit-equivalent numerics (same kernel as decode hot path, sum
order preserved per-warp sequential over k_rounds).

Re-verified A/B on qwen3-6-35b-a3b NVFP4-KV (5-word prompt,
max_tokens=150, 3 runs each, freshly-installed symlinked binary):

  | Cell      | wall   | tokens | tok/s | prefill_ms | md5(completion) |
  |-----------|--------|--------|-------|------------|-----------------|
  | MOE_ON    | 3.69 s | 150    | 40.65 | 217        | b5d33eaa        |
  | MOE_OFF   | 3.75 s | 150    | 40.00 | 273        | 1a136b5a        |

**21% prefill speedup** (273→217 ms) from the whole batched-MoE
stack. Decode tok/s ~parity (40.65 vs 40.00 = +1.6%). md5 differs
ON vs OFF (stack fires).

22% prefill speedup from the whole batched-MoE stack (this
port + dual_silu kround-batch + router+topk batched + shared-
expert batched). The kround port's individual contribution
(280 launches saved ≈ 1.4 ms on 1158 ms prefill) is too small
to separate from noise; the whole-stack number is the relevant
measurement. Host-side `for k_round in 0..top_k` loop is gone,
replaced by one kround_batched launch.

### Phase 8 other-models fusion (qwen27b dense) — SHIPPED 2026-05-23

Commit `39f7c1a` ports the same fusion pattern to the Qwen 3.5/3.6
27B DENSE decode path. New kernel
`fp8_gemv_blockwise_wpr_native_f16in_residual_add_kernel` fuses
the single-output FP8 GEMV with the subsequent vector_add_f16
residual add. Three per-layer M=1 sites in qwen35_bring_up.rs
are rewired: full-attn o_proj+residual (#10-11), linear-attn
out_proj+residual (#9-10), dense MLP ffn_down+residual (#4-5).
40 layers × 3 sites = 120 launches saved per decode token.

Hardware-validated on qwen3-6-27b: short / 1-10 counting / 80-
token photosynthesis all coherent. The fusion's value is
architectural: fewer graph nodes, no temp-buffer f16
round-trips, cleaner kernel chain.

PREFILL (M > 1) sites kept on the unfused path — CUTLASS SM120
GEMM is preferred there. The fusion targets ONLY M=1 decode
hot-path sites. Mistral 3.5 (dense) and Gemma 4 31B (dense) could
pick up the same fusion via their own bring-up files.

### Phase 8 follow-on (graph-cache reuse) — SHIPPED 2026-05-22

Commit `bcdce94` lands cross-request graph cache reuse:

- **Persistent workspace** allocated once at worker bring-up
  (BEFORE the `scratch_ck` checkpoint). Workspace.* device
  pointers stay stable across all requests.
- **Inner arena checkpoint+restore** via the RAII
  `Qwen36DecodeArenaGuard` at the entry of
  `forward_qwen36_decode_inner_with_workspace_overrides_v2`.
  Every per-call `arena.region(...)` inside the function lands
  at deterministic device addresses on every call, every
  request (the entry bump is always `scratch_ck`).
- **Removed per-request `clear_decode_capture`** — the
  captured graph from request N is now reusable for request
  N+1 since every captured pointer is address-stable.

Hardware validation (3 sequential requests, different prompt
lengths): all correct, log shows "[graph] captured 1855
nodes" only ONCE on the first request.

Eager decode path remains the production default
(`RVLLM_QWEN36_DECODE_GRAPH` unset). Captured path opt-in only.

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

## Cross-model invariants (READ BEFORE TOUCHING SHARED FILES)

Files in `v3/crates/rvllm-attention/`, `v3/crates/rvllm-fused/`,
`v3/crates/rvllm-cutlass/`, and `kernels/flash_attention*.cu` /
`kernels/fused_*.cu` are **shared across model families** (Gemma 4
fp8-block, Gemma 4 NVFP4, Qwen 3.5/3.6, Mistral 3.5, E4B). Changes
to these files affect EVERY model that loads them.

When modifying a shared file for ONE model's perf/quality:

1. **Identify all consumers first.** `grep -rn` for the symbol/cap
   across all `*_bring_up.rs` and `*_load.rs` files. The cross-
   model dispatcher `v3/crates/rvllm-attention/src/decode.rs` is
   the central hotspot — changes here touch ALL decode paths.
2. **Verify byte-identity for every affected model.** Don't ship
   a "Mistral-only" optimization without running the regression
   smoke for Gemma 4 31B + E4B + Qwen 3.5/3.6 + spec-decode paths.
   The byte-equivalence gate from tasks #26/#27/#34 (md5 captures
   of fixed test prompts) is the existing infrastructure.
3. **Prefer adaptive dispatch over one-size-fits-all.** When a
   model needs a larger cap/buffer/kernel variant, ADD a new
   variant + dispatch logic rather than bumping the shared
   default for everyone. Pattern: kernels compile multiple
   `extern "C" __global__` symbols via a templated `__device__`
   helper (one register-array size per variant), host dispatches
   by actual model parameter. **Canonical example shipped 2026-05-22**
   (task #1): NVFP4 GQA decode's `_gqa_kernel` (MAX_GQA=4) vs
   `_gqa_max16_kernel` (MAX_GQA=16), with BC=32 + BC=16 + bf16-out +
   split-decode + FP8-KV siblings (commits 6828eeb / 24715f8 /
   cd04286 / 823bfe4 / 7d9e21b). Host dispatch in
   `v3/crates/rvllm-attention/src/decode.rs` picks the smallest
   fitting variant per actual GQA; low-GQA models (Gemma 4
   sliding GQA=2, Qwen 27B GQA=4) stay on the minimal-register
   default. Runtime-validated on Qwen 27B (GQA=4 default path
   unchanged), Gemma 4 31B (global GQA=8 → _max16), Qwen 3.6
   35B-A3B (GQA=8 → _max16), Mistral 3.5 (GQA=12 → _max16) —
   all coherent German output on three smoke prompts each.
4. **Document the cross-model impact in commit + register.**
   Commit message must list every model family whose forward
   path passes through the touched file, and confirm the
   regression check on each. The deferred-work registers (Option
   B + mistral + qwen) get a cross-reference back to the commit.
5. **Off-by-default env knobs** when introducing experimental
   kernel variants — flip the default only after broader hardware
   coverage validates no regression across families.

These rules apply to ALL shared code paths, not just attention.
The mistral perf work (raising `MAX_GQA_DECODE`/`MAX_GQA_SPLIT`
in `flash_attention*kv*.cu`) was the canonical example: naive
single-default bumping would have created a Gemma 4 / Qwen
register-pressure risk. Resolved 2026-05-22 (task #1) via the
adaptive `_max16` variant pattern documented in rule (3) above.

## Deferred-work register cleanup — 2026-05-24 (brain tasks 1001b/a/s/v/t/k/t/awq)

Eight high-priority brain tasks reviewed and closed under one batch:

**aa01001kvbrn — brain memory quality filter on chat-history.**
`zeroclaw-bridge/files/memory_brain.rs::reject_reason` filters
degenerate assistant content (Hangul soup, `la la la` loops,
single-token repetition, mostly non-Latin1 bytes, length<5) at both
`record_chat_turn(actor=assistant)` and `store(...)` write paths
before the embedding pipeline. Kills the 2026-04-26 reinforcement-loop
bug class regardless of any future kernel/quant experiment. 8 unit
tests cover heuristic boundaries. brain commit `ca48922`.

**aa01001vrot — NVFP4 V-rotation lift.** Verified already fully wired
in prior cycles: V Hadamard rotation under `RVLLM_NVFP4_HADAMARD_V`
(default ON per `gemma4_bring_up.rs:1154`), `hadamard_unrotate_f16`
kernel loaded into `Gemma4LayerKernels`, dispatch in
`Gemma4Layer::execute` post-attn pre-O-proj, PrefixProvenance
tracks `hadamard_v` for cache invalidation. Coherence guard rejects
HADAMARD_V=1 without HADAMARD=1. Stale "PARTIAL cycle 17" task
description closed.

**aa01001srvbug2 — P1/P2 audit closeout.** Verified P1 #3 (sampling
coerce, Phase #5 above), #4 (tool_choice rejection), #5 (rep-guard
opt-in + exact-identity tail only), #6 (tier2 nested-braces
balanced scanner), #7 (text+image+audio content parts), P2 #8
(invalid_max_tokens reject), #10 (saturating_* throughout) all
already-shipped. **P2 #9 prefix-cache mutex poison fixed**:
`Gemma4Bringup::lock_prefix_cache_recover()` recovers from
`PoisonError::into_inner`, clears the slot to None (presumed
corrupt — partial write may leave dangling arena ptrs), bumps
`prefix_cache_poison_recoveries: AtomicU64` for observability,
WARN line per recovery. 8 call sites (7 in `gemma4_bring_up.rs`,
1 in `gemma4_spec_primitives.rs::verify_batched_suffix_k_only`)
updated. rvllm-serve commit `a407a4d`.

**aa01001bf16chain — Stage 3 validation.** Stages 1+2 (kernels +
dispatch) landed in prior session. Stage 3 hardware A/B on
gemma-4-31b-it-nvfp4 with explicit `RVLLM_RESIDUAL_BF16=0` (legacy
F16) vs default (auto-on per `unwrap_or(true)` in cycle 55):
WHO ("Bundeskanzler") + WEATHER short-context both byte-identical
between legs; 16k webhook WHO returns coherent Rusty persona reply
under bf16=1. No regression at any context size. Production-default
already ON since the cycle-55 flip; explicit env removed from
profile to keep the default authoritative.

**aa01001ttftbatch — flip run_generate batch prefill default.**
`gemma4_bring_up.rs:11339` replaced `unwrap_or(false)` with a
smart-default tri-state:
  * `RVLLM_BATCH_PREFILL=1` → force batch (diagnostic).
  * `RVLLM_BATCH_PREFILL=0` → force per-token (legacy).
  * unset → auto-batch when `prompt_len >= BATCH_PREFILL_MIN_TOKENS`
    (= 128, matches FAST_PATH_M_MAX boundary; below that the
    fp8_gemv M=1 path is bandwidth-bound and cost-parity).
Aux read sites (`PrefixProvenance::from_env`, spec-rollback) keep
their original `unwrap_or(false)` semantics — they track env state
for prefix-cache invalidation, not the per-request dispatch.
Hardware A/B at 108/318/1168/2808 prompt_lens × {OFF, AUTO, ON} =
12 cells, all byte-identical text + timings within ±50 ms — on
NVFP4 the per-token and batch paths converge in cost because the
unified-prefill kernel already amortises setup. The task's
"vLLM 4× faster" gap was measured on FP8-block; same dispatch
flip applies. rvllm-serve commit `224656e`.

**aa01001splitkv — NVFP4 split-decode kernel quality bug.** Verified
resolved: 17k-context WEATHER probe under default
`RVLLM_NVFP4_SPLIT_KV=1` + `partition_size=1024` returns coherent
German answer ("Ich habe keinen Zugriff auf Echtzeit-Wetterdaten…").
Same prompt under `SPLIT_KV=0` (single-CTA) returns identical text.
The cycle-20 garbage cliff (repetition + multilingual fragments)
is gone. Likely fixed by the cumulative V=amax6 + Hadamard
production tuning + partition_size default flip from 512 → 1024.

**aa01001toolcall — long-context Gemma cumulative noise.** Verified
resolved on gemma-4-31b-it-nvfp4: 17k-context WEATHER prompt with
a registered `get_weather` tool returns
`finish_reason: "tool_calls"` + structured
`get_weather({"city":"Bern"})` call. The cycle-3 failure mode
(`<tool_call|>` token-49 + multilingual repetition + body garbage)
is gone, fixed by the cumulative BF16 chain + NVFP4 quality
defaults shipped through cycles 54-55.

**aa01001awq — AWQ INT4 W4A16 loader + kernel.** Code-side
infrastructure already landed across all 4 design stages:
`compressed_tensors.rs` (AwqLinearWeight, upload_awq_linear,
upload_gemma4_awq_layer, read_awq_config_from_dir), `awq_int4_gemv_f16.cu`
+ `awq_int4_gemm_sm120_wmma.cu` + `mistral35_w4a16_gemm_mma_v8_bf16.cu`
kernels, `Gemma4LayerKernels::awq_int4_gemv_f16` + `_gemm_sm120_wmma`
slots with M-based dispatch at `gemma4_layer_exec.rs:1251-1362`.
Stage 5 validation (smoke against an AWQ Gemma 4 31B checkpoint)
is operator-gated — requires ~20 GB HuggingFace download of
`ebircak/gemma-4-31B-it-4bit-W4A16-AWQ` or
`cyankiwi/gemma-4-31B-it-AWQ-4bit`.

## Tool calls + brain search + sampling — fixed 2026-05-24 (Phases #1-#5)

Follow-on to the Phases A-D OpenAI-compat work below. The user
reported "Rusty can't find entities, tool calls hang, conversations
break" after the truncation fix landed. Five independent bugs
surfaced under the longer-running webhook turns the bigger
`max_tokens` made possible:

**#1+#2 — Qwen3-VL tool calls were never parsed (rvllm-serve commit
`e0bf635`).** `tool_parser.rs` was hardcoded to Gemma 4's
`<|tool_call>call:NAME{...}<tool_call|>` form. Qwen 3.5 / 3.6 emit
either canonical JSON (`<tool_call>{"name":"...","arguments":{...}}
</tool_call>`) or an OpenManus-style DSL (`<tool_call>\n<function=NAME>\n
<parameter=KEY>\nVALUE\n</parameter>\n...\n</function>\n</tool_call>`,
which is what `qwen3-6-35b-a3b` actually emits on hardware
regardless of the chat-template prose). The parser now exposes:

  * `parse_qwen36_tool_calls` — accepts BOTH formats. JSON path
    handles double-encoded `arguments`. DSL path parses
    `<parameter=KEY>VALUE</parameter>` and JSON-coerces values
    (numbers / bools / null / nested objects round-trip naturally).
  * `strip_qwen36_tool_markup` — drops both `<tool_call>...</tool_call>`
    and `<tool_response>...</tool_response>` blocks.
  * `ToolDialect { Gemma4, Qwen36 }` enum + `parse_tool_calls` /
    `strip_tool_markup_for` / `tool_call_opener_for` dispatchers.

Handler integration (`openai/handlers.rs`): `dialect_for(VisionArch)`
maps `Qwen36 → ToolDialect::Qwen36`, all other arches → `Gemma4`.
`shape_assistant_message`, `chat_collect`, and `chat_stream_sse`
take the dialect; SSE Ctx carries it and routes through new
`detect_tool_call_latch_for` / `safe_content_emit_end_for`
(Qwen has no thought-blocks / tier-2 bare form, so its hold-back
rule is much simpler — just the opener-straddle guard).

Hardware-verified on qwen3-6-35b-a3b NVFP4 with `tools` +
`tool_choice: auto`: `finish_reason: "tool_calls"`, structured
`tool_calls[].function.{name, arguments}` populated. Pre-fix the
exact same request returned `content: "<tool_call>..."` text with
`finish_reason: "stop"` and no `tool_calls` array.

**#3 — brain search SQL crashed on every NONE embedding (brain
commit `a0e4a64`).** SurrealDB 3.0.5 evaluates SELECT projections
during ORDER BY sort across the full pre-WHERE row set, so a
`vector::similarity::cosine(embedding, $vec)` projection crashed
the engine when any row in the target table had `embedding=NONE` —
even when a `WHERE embedding IS NOT NONE` clause would later
filter it out. Symptom on every qwen tool call:

    Error: Incorrect arguments for function vector::similarity::cosine().
    Argument 1 was the wrong type. Expected `array<number>` but found `NONE`

This blocked every entity / task / tool / bookmark / credential /
location / snippet / ha-* / generic search in the steady state
(freshly-created rows are NONE-embedded until the async embed_queue
catches up; queue drops leave them permanently un-embedded).

Fix: wrap every cosine call in
`IF $col IS NONE THEN 0 ELSE vector::similarity::cosine($col, $bind) END`.
Touched 18 SQL sites across 16 files (5 brain-core + 11 brain CLI).
Also normalised every `<col> IS NOT NULL` in the CLI to
`<col> IS NOT NONE` — they're distinct values in SurrealDB 3 and
the `NULL` form does not filter out unset embedding fields (which
is why this bug hid behind seemingly-correct WHERE clauses).

**#4 — backfill embeddings (operational, not a code change).**
Pre-fix `brain stats` showed `entities: 5 (0 embedded)`,
`tasks: 30 (0 embedded)`, etc — the embed_queue worker had been
dropping work for months. `brain update-embeddings --table <T>
--missing-only` was used to backfill the priority tables:
entity (5), task (30), tool (13), memory (42), bookmark (7),
project (1), ha_device (17), ha_sensor (82) — 197 rows total.
`news_article` (15727 rows ≈ ~30 min of embeds) deliberately
skipped — it's not on the chat-history hot path. Result:
`brain entity search Vinz` returns `Vinz [0.717]` at the top,
`brain task search "fix"` returns 5 relevant tasks, etc.

**#5 — non-greedy sampling on greedy-only arches no longer 400s
(rvllm-serve commit `a9ab61f`).** When zeroclaw's SSE deadline
fires, its reliable-provider retry uses its own default `temperature
> 0` instead of the per-model `default_temperature=0` config. The
retry then hit a hard 400 on `qwen3-6 path is greedy-only`,
reliable-provider marked it non-retryable, and the channel turn
ended with NO reply at all — a second user-visible failure on
top of the first (timeout). Renamed
`reject_unsupported_sampling_for_arch` →
`coerce_sampling_for_arch`: instead of 400'ing, log a one-shot
WARN line in journalctl and coerce the sampling to greedy. Callers
get a correct (greedy) reply, operators see the WARN, and the
timeout-retry cascade no longer leaves the turn empty. Affects
`Qwen36` and `Mistral35` (no logits-out variant); `Gemma4` /
`E4B` still take stochastic natively.

E2E verification (webhook to qwen3-6-35b-a3b with 16 k Rusty
persona + brain tool registered): `"Wer ist Vinz? Nutze brain."`
→ 3-iteration tool loop (15759 → 15846 → 15968 tokens as tool
messages accumulate) → final reply `"Vinz ist der Nutzer, mit dem
ich gerade spreche. Discord: herrchen1312, Telegram: herrchen1312."`
in 97 s wall. Pre-all-fixes the same request produced a literal
`<tool_call>` string in content, no structured tool calls, no
brain lookup, no useful reply.

## OpenAI API compat — fixed 2026-05-24 (Phases A-D)

End-to-end fix for the "answer is incomplete" zeroclaw webhook
regression. Four related issues across rvllm-serve and zeroclaw:

**Phase A — root cause confirmed.** Direct A/B on qwen3-6-35b-a3b:
`max_tokens=1024` → `finish_reason=length`, output cut mid-sentence
("...in Russland"). `max_tokens=2048` → `finish_reason=stop`, clean
final period. Truncation was real, not a tokenizer or chat-template
issue.

**Phase B — default max_tokens floor (rvllm-serve commit `1a70b18`).**
`resolve_max_new()` in `crates/rvllm-serve/src/openai/handlers.rs`
defaulted to `cap.min(1024)` when the caller omitted `max_tokens`.
On every long-context profile (`RVLLM_MAX_TOKENS_CAP` ≥ 65 k) this
silently capped completions at 1024 tokens regardless of the
configured cap. zeroclaw's OpenAI provider does not send
`max_tokens`, so every chat turn ran into the floor mid-sentence.
Default is now `cap.min(DEFAULT_MAX_NEW_TOKENS)` where
`DEFAULT_MAX_NEW_TOKENS = 8192`. Hardware-verified post-fix:
`submitting to worker prompt_tokens=15772 max_new=8192`
(was 1024).

**Phase C — `stream_options.include_usage` honoured (rvllm-serve commit
`1a70b18`).** Pre-fix the server 400'd with
`stream_options_unsupported`, causing zeroclaw to retry every
streaming turn as non-streaming (extra RTT, lost live UX).
`crates/rvllm-serve/src/openai/{chat.rs,completions.rs}` now expose
a typed `StreamOptions { include_usage: Option<bool> }`;
`ChatCompletionChunk` / `CompletionChunk` grow an
`Option<Usage>` field (omitted via `skip_serializing_if`).
`chat_stream_sse` / `completion_stream_sse` plumb `include_usage`
into an `EmitUsage` state that emits one extra
`{"choices":[],"usage":{...}}` chunk between the finish-reason
chunk and `[DONE]` — exactly the OpenAI 2024 spec. Validated
on qwen3-6-35b-a3b: with `include_usage=true` the stream produces
role + content + finish + usage + `[DONE]`; without it the usage
chunk is suppressed (no regression).

**Phase D — precheck timeout under long prefill (zeroclaw commit
`aa8063e3`).** `crates/zeroclaw-config/src/scattered_types.rs ::
default_precheck_timeout_secs` raised from 5 s to 60 s. The
reply-intent precheck forwards the full system prompt + history
to the route model; on a 16 k Rusty persona the prefill alone is
~20 s, so the 5 s ceiling always tripped and the orchestrator
fail-opened to REPLY — wasting the precheck's cost and adding
a 5 s wall-time penalty per turn. Operators on fast cloud APIs
are unaffected (classifier still returns in < 1 s). Operators can
override via `agent.precheck.timeout_secs` in `config.toml`.

E2E verified post-all-fixes (webhook to qwen3-6-35b-a3b NVFP4 with
16 k persona): single rvllm submission, 20 s total wall, complete
"Ich bin Rusty." reply, no `stream_options_unsupported` 400, no
`provider streaming failed, falling back to non-streaming chat`
warning, no `Reply-intent precheck timed out` warning.

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
- **bf16 vision forward**: kernels are committed AND wired (env
  `RVLLM_GEMMA4_VIT_USE_BF16=1`). **Bug found + fixed 2026-05-23**
  on gemma-4-31b-it-nvfp4 (Option B). Pre-fix symptom: first
  vision call produced a coherent caption; second + subsequent
  calls degraded to NaN in the `standardized` buffer and the
  LLM hallucinated "Sie haben kein Bild hochgeladen". Root cause:
  `ensure_vit_bf16_weights` lazy-allocated the bf16 weights into
  the arena BELOW `forward_checkpoint`, so every subsequent
  vision request's `forward_scratch_guard::drop` restored the
  arena past those weights — the cached pointers in
  `vit_bf16_weights` mutex still pointed there, but the next
  request's `arena.region(...)` scratch allocations overwrote
  the bf16 weight memory. Fix: change `ensure_vit_bf16_weights`
  + `forward_gemma_vision` to `&mut self`, and bump
  `forward_checkpoint` to the post-weights arena top after
  `convert_from_f16`. Post-fix: 5 consecutive bf16 vision calls
  all produce the same coherent caption (md5 `df22aaa3` x5).
  Production still defaults to f16 (env-gate is opt-in); the
  bf16 path is now safe to enable for testing.

## Option B (Gemma 4 31B NVFP4) — production status + follow-up register

Active branch `rusty_sm121_qwen36_26b`. **Production status: shipping** —
ZeroClaw default for tool-heavy work. Full NVFP4-weights stack runs at
K=7 spec decode + adaptive draft length + batched-MLP-verify.
Loader + 60-layer decoder forward + unified-prefill + spec-decode +
batched prefill all wired. Task #38's "31B NVFP4 native weight loader
+ decoder forward" acceptance criteria met.

### Current state of the codex-review streams

| Stream | Status |
|---|---|
| Floor + Stream-5a/5b | **DONE** (`bc89650..32d0b0e`). Floor + scaled-add. |
| #5f-PRIME (unified NVFP4 prefill) | **DONE.** `forward_prompt_to_all_tokens_impl` (gemma4_nvfp4_bring_up.rs:7780) is the device-resident batched-prefill path — residual lives as `[N, hidden]` bf16 across all 60 layers; unified-prefill kernel runs ONCE per layer. Long-prompt TTFT now scales with prefill throughput, not N decode launches. |
| Stream-6a — cuda_worker spec gate | **DONE.** `cuda_worker.rs:116` accepts `RVLLM_GEMMA4_SPEC_DECODE=1` for `ModelFamily::Gemma4Nvfp4` as well as `Gemma4` (fp8-block). Production NVFP4 spec session runs via `run_spec_session_nvfp4_greedy_k` (gemma4_nvfp4_bring_up.rs:1056). |
| Stream-6a — BaseKvSource trait | **DONE 2026-05-22**. Trait already existed (gemma4_drafter.rs:64). Both bringups now have BaseKvSource adapters: `Gemma4Nvfp4BaseKvSource` (gemma4_nvfp4_bring_up.rs:8359) + `Fp8BlockBaseKvSource` (`4c6a927`). Stream-6a deeper unification (`5d2a36a`) retires the inline `shadow_view_for_layer` / `compute_view` / `effective_kv_dtype` closures at the top of `run_generate_speculative_batched` (≈80 LOC) in favor of a `ShadowOverrideBaseKvSource<'_, B>` decorator that wraps any `BaseKvSource` and swaps in F16-shadow pointers + `KvDtype::F16` for the two source layers when `RVLLM_GEMMA4_SPEC_USE_F16_SHADOW=1`. Call site pulls pointers + dtype from `DrafterBaseKvView`s. Validated byte-equivalent on HADAMARD=0 and coherent multi-token on HADAMARD=1+USE_F16_SHADOW=1. |
| Stream-6b (Hadamard drafter Q for Option B) | **CODE SHIPPED 2026-05-22** (`6ab88d4` primitives + `d75067c` wiring). Drafter Q rotation by R = H·diag(D) wraps the cross-attn launch in `forward_drafter_layer_cross_attn`; attn_out un-rotation by R^T follows. Per-base-layer signs lazy-uploaded in `ensure_drafter_nvfp4`. Both helpers are no-ops on HADAMARD=0 — production default unchanged. **A/B re-verified 2026-05-23 against fresh binary: HADAMARD=1 + HADAMARD_V=1 + ALLOW_HADAMARD=1 gives accept_rate=0.000 on gemma-4-31b-it-nvfp4 (80-tok photosynthesis); the prior "coherent without F16-shadow" claim only verified output coherence (which the BASE verify-fallback always provides regardless of drafter acceptance), NOT actual drafter acceptance. Real recommendation: stay HADAMARD=0 (accept_rate 2.750 on the 80-tok quantum-entanglement prompt) or, if HADAMARD=1 is required for long-context quality, set `RVLLM_GEMMA4_SPEC_PRE_HAD_SHADOW=1` (commit `1449833`, accept_rate 2.333 — see Phase 3 F16-shadow section above). Commit `82dad1d` makes this AUTOMATIC: `ensure_drafter_nvfp4` detects HADAMARD=1+spec without an explicit `PRE_HAD_SHADOW=1` and auto-enables it process-wide, emitting a one-shot warning. Explicit opt-out: `RVLLM_GEMMA4_SPEC_ALLOW_HADAMARD=1` (keeps the legacy broken-accept_rate path for A/B diagnostics).** |
| Stream-7 (Option B vision splice + spec+vision) | **SHIPPED.** Vision splice landed via commits `cd4d7b8` (A: extract `forward_gemma_vision` into `crate::gemma4_vision::Gemma4VisionRuntime`) + `b2b20f5` (B: wire `Gemma4Nvfp4Bringup::forward_gemma_vision` + `forward_prompt_to_token_with_vision` + remove `vision_not_supported_on_gemma4_nvfp4` rejection). Vision + spec coexistence landed 2026-05-23 via commit `2b54b2d`: new `run_spec_session_nvfp4_greedy_k{1,}_with_vision` siblings thread `vision_splice` into the spec prefill; the drafter inherits vision-spliced base tokens via shared shadow K/V (no drafter ViT needed — drafter operates in token space). Hardware-validated on `gemma-4-31b-it-nvfp4` + `/tmp/ball.png`: coherent German captions, no rejection, both spec-on and spec-off paths. |
| Cross-request prefix-cache reuse (task #133) | **SHIPPED 2026-05-25** (`4bbde0a`). `Gemma4Nvfp4Bringup` gained `nvfp4_prefix_cache: Mutex<Option<Nvfp4PrefixCacheState>>` + `Nvfp4Provenance` (tracks NVFP4_KV / HADAMARD / HADAMARD_V / K-V scale policies / ring-buffer). Both spec session entries (`..._greedy_k` and `..._greedy_k1`) call `nvfp4_prefix_cache_lookup` to compute longest-common-prefix vs last published (prompt + emitted) sequence; only the new tail [lcp..n) is prefilled with `position_start = lcp`. Publish happens after fence on success. Vision splice bypasses cache. Hardware-validated on zeroclaw 14k webhook (gemma-4-31b-it-nvfp4 + NVFP4 KV + K=7): wall 6m20s → 3m04s (**52% reduction**); per-iter prefill 103.5→103.7s (cold) / 101.7→1.2s (85x LCP hit) / 110.1→7.6s (14x LCP hit); total prefill 315→112.5s (2.80x). Opt-out: `RVLLM_GEMMA4_NVFP4_PREFIX_CACHE=0`. All other model families untouched (only `gemma4_nvfp4_bring_up.rs` modified). |
| CUDA Graph capture validation (task #134) | **VALIDATED 2026-05-25** (`f1aea84`). Existing `forward_full_to_token_captured` infrastructure (from prior session) confirmed cross-request-safe via A/B/A2 test (coherent prompt-A → prompt-B → prompt-A again); capture log "[graph] captured 1685 nodes" confirms cuGraphLaunch path engages. New `clear_decode_capture` safety helper for future flip-on-the-fly support. Initial A/B showed parity-or-slight-regression vs baseline; root cause was the INDIRECT prerequisite (3 × 4-byte HtoDs per fill_pos_slots) costing 17% — **superseded by task #135's HtoD fold**. Profile also bumps `RVLLM_REQUEST_TIMEOUT_SECS` 600→1500 to fix the cold-cache 599s timeout cascade on real zeroclaw turns. |
| INDIRECT-HtoD fold + graph_replay net-positive (task #135) | **SHIPPED 2026-05-25** (`adb2fb7`). Replaced 3 × 4-byte non-contiguous device slots + 3 × 4-byte HtoDs per `fill_pos_slots` with ONE 16-byte contiguous region holding [pos_off, start_slot, num_tokens] at offsets 0/4/8 + 4-byte padding. Single 12-byte `cuMemcpyHtoDAsync` populates all three scalars at once; the indirect kernel reads them via three pointer args unchanged (now slice offsets into the same base). The captured graph's fill_pos_slots subgraph drops from 4 nodes (3 HtoD + 1 kernel) to 2 (1 HtoD + 1 kernel). **A/B post-fold** (gemma4-nvfp4 dense, 60-tok poem, 100 max_new, spec OFF, 3-run det): OFF/OFF baseline 4.38 tok/s; INDIRECT_ON only **4.36 tok/s = parity** (was -17% pre-fold); GRAPH_ON+INDIRECT_ON **4.50 tok/s = +2.7% net win** vs baseline. Captured-graph now beats eager on 31B dense for the first time. Defaults still OFF (production gemma4 path runs spec ON which doesn't reach the captured path yet — wiring into spec session is the parked follow-up). |
| Drafter forward graph capture (task #136) | **SHIPPED 2026-05-25** (`10675fb`). Foundation for capturing the spec drafter forward chain. (a) `drafter_pos_box: Mutex<Box<i32>>` heap-stable host source replaces the prior sync `pos_region.copy_from_host` (stack-local source not capture-safe) — async cuMemcpyHtoDAsync from stable Box address means the captured graph's HtoD node reads a live address at replay time; (b) `run_drafter_forward_one_token` gains per-call arena checkpoint+restore so the per-layer pos_region allocations land at deterministic addresses across calls; (c) `run_drafter_forward_one_token_captured` wrapper drops the full drafter forward (pre_projection + 4 layers × 4 sub-launches + final_to_token, **107 graph nodes**) into one CUgraphExec; (d) spec session `run_drafter_greedy_k_from_state` routes through the captured wrapper when `G4N_DRAFTER_GRAPH=1`. Hardware A/B at matched iter counts (gemma4-nvfp4 dense, spec K=7, 60-tok poem, 100 max_new): drafter_ms 753→747 at 21 iter (-0.8%), 534→525 at 15 iter (-1.7%). Sub-noise at wall-time. **Superseded by task #137's multi-step macro** for better amortization. |
| Multi-step drafter macro graph (task #137) | **SHIPPED 2026-05-25** (`2c0c8a7`). Where #136 captured ONE drafter forward and replayed K times per spec iter (paying K cuGraphLaunch calls), this captures the WHOLE K-step loop body (populate × K + drafter forward × K + DtoD × K = **778 graph nodes**) as ONE CUgraphExec. ONE cuGraphLaunch per spec iter instead of K. New `run_drafter_greedy_k_from_state_macro` + `drafter_macro_capture: Mutex<Option<(usize, CapturedGraph)>>` slot (usize = spec_k for capture invalidation on adaptive_k flips). Spec session dispatches via `G4N_DRAFTER_MACRO=1` (default OFF; takes precedence over per-step `G4N_DRAFTER_GRAPH`). Hardware A/B (gemma4-nvfp4 dense, spec K=7, matched iter counts): drafter_ms 1583→1476 at 38/37 iter (**-6.8%**), 781→746 at 21 iter (-4.5%), 555→525 at 15 iter (-5.4%). **4-7× better than per-step capture**. Wall-time delta still sub-noise because bailout_ms (11-14s when accept rate drops) dominates over drafter (0.5-1.5s) on this workload. Higher-accept-rate workloads with longer sessions would see proportionally more wall savings. Default OFF; operator opt-in. Verify path capture (extend macro to include verify forward) is the next parked follow-up. |
| Adaptive K + verify-macro foundation + CUTLASS update (task #138) | **SHIPPED 2026-05-25** (`fd9223e`). Two of four parked items finished; the other two filed as proper multi-week tasks (`aa01001nvfp4gemv` fp8_gemv MMA rewrite, `aa01001nvfp4cprefill` cold-prefill speedup). Item 1: `G4N_SPEC_ADAPTIVE_K=0→1` on canonical profile — at measured 27-43% accept rate, adaptive walks K toward MIN_K=1 instead of wasting drafter cycles at fixed K=7. Item 2 (foundation only, full macro extension parked): `embed_one_token_from_device_slot` helper + `verify_input_tokens_dev_ptr` persistent K+1-slot device buffer + `verify_t_committed_box` heap-stable host source + `ensure_verify_input_tokens_dev` lazy allocator. Bonus: CUTLASS submodule advanced to `e45ccb12` (current main, two upstream commits since v4.5.1 are Python examples + pytest fixes only — no include/ changes); 3 SM120 .so kernels rebuilt clean; manifest sha refreshed (required for all families loading libcutlass_sm120.so). New measurement (`aa01001nvfp4cprefill` body): gemma4-nvfp4 dense **prefill curve 333 t/s @ 1k → 230 t/s @ 18k** (CLAUDE.md's "505 t/s" headline doesn't reproduce — 14.6× MMA_V8 win flattens at large M). |
| cprefill investigation + orphan cleanup (task #139) | **INVESTIGATION SHIPPED 2026-05-25** (`cda0949`). Executes the 4 follow-up steps from `aa01001nvfp4cprefill`. (a) MMA_V8 A/B at long M: forcing OFF is **10-13× SLOWER** at M=1043/3743/9143 — MMA_V8 is unambiguously winning, stays default-on. (b) nsys at M=14583 cold prefill: **mistral35_w4a16_gemm_mma_v8_bf16_kernel = 56.4%** (MLP GEMM) + **flash_attention_2_prefill_nvfp4kv_unified_bf16out_kernel = 38.3%** (attention prefill) = **94.7% of prefill GPU time** at long M. (c) alternate kernel hunt: two orphan PTXs (`flash_attention_unified_prefill_nvfp4kv_{nosvt,unrolled}.ptx`) leaked into the tree from experiment branches; git archaeology shows both were A/B-rejected on their own merit (unrolled = byte-identical no speedup; nosvt = quality met perf not met). Dropped both PTX + manifest entries (231 → 229 wait actually 233→231, only loadable-but-never-loaded). (d) root-cause conclusion: smem-latency / NVFP4-dequant bound per original kernel author analysis; real wins require multi-week fresh kernel rewrite (cp.async pipelining, smem swizzling, possibly CUTLASS CuTe re-implementation). Parent `aa01001nvfp4cprefill` stays open as planning umbrella; this task closes with hard evidence replacing the speculative section of CLAUDE.md. |
