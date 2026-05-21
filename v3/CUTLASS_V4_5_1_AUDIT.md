# CUTLASS v4.5.1 audit + perf A/B (2026-05-21)

Submodule: `ae6bccf3` → `982cb9e7` (v4.5.1, 8 commits).

## Hardware A/B summary

All measurements: 3 runs each, greedy temp=0, GB10 sm_121.

### Mistral 3.5 NVFP4 (634-tok Linux prompt, 30 decode)

| CUTLASS | wall (avg) | md5 | delta |
|---|---|---|---|
| ae6bccf3 (v4.4.x) | 33.73s | `75d36e68` | baseline |
| 982cb9e7 (v4.5.1) | 33.73s | `75d36e68` | **0%** |

### Qwen 3.6 35B-A3B (1057-char prompt, 30 decode)

| CUTLASS | wall (avg) | md5 |
|---|---|---|
| ae6bccf3 (v4.4.x) | 2.148s | `5dd028e5` |
| 982cb9e7 (v4.5.1) | 2.156s | `5dd028e5` |
| delta | +0.4% (noise) | — |

### Qwen 3.6 35B-A3B (21k-char prompt, 5 decode — prefill-heavy)

| CUTLASS | wall (avg) | md5 |
|---|---|---|
| ae6bccf3 (v4.4.x) | 31.29s | `e1552489` |
| 982cb9e7 (v4.5.1) | 31.41s | `e1552489` |
| delta | +0.4% (noise) | — |

### Tile-shape sweep (mistral35, v4.5.1)

| TILE_M/N/K | wall | md5 |
|---|---|---|
| 128/128/128 (default) | 33.73s | 75d36e68 |
| 128/256/128 | 33.80s | 75d36e68 |
| 256/128/128 | 33.81s | 75d36e68 |

→ Tile config has no effect because the NVFP4 GEMM kernel is **not on
mistral35's hot path** (see Why-not section below).

## Where v4.5.1 actually applies in our code

CUTLASS .so symbol → consumer:

| Symbol | Consumer | Hot? |
|---|---|---|
| `cutlass_fp8_gemm_blockscale_sm120` | gemma4 FP8-block prefill, qwen35/36 prefill | yes |
| `cutlass_fp8_gemm_blockscale_sm120_prep_sfa/b` | siblings of above | yes |
| `cutlass_nvfp4_gemm_sm120` (and friends) | none (built, uncalled) | **no** |

## Why no measurable perf gain

The two v4.5.1 changes most likely to move perf:

1. **PTX `cvt.rn.bf16x2.e4m3x2`** in `numeric_conversion.h`. Used when
   converting e4m3 → bf16 in CUTLASS epilogues. Our SM120 FP8 GEMM has
   `ElementD = half_t` (f16, not bf16) — the new cvt does NOT apply to
   our path. The legacy `cvt.rn.f16x2.e4m3x2` still runs.

2. **SM120 blockscaled MMA fixes** ("added missing MXFP8MMAOP and
   MXF8F6F4MMAOP for sm120"). These are MX-scaled MMA variants; our
   FP8 GEMM uses regular E4M3 (not MX-FP8). The NVFP4 GEMM does use
   MX-style block-scaled MMAs but it's uncalled by any runtime
   (mistral35 routes through cublasLt BF16 GEMM after a custom W4A16
   dequant — see `mistral35_bring_up.rs:2249` "ROOT CAUSE" comment).

3. **SM100 F8F6F4 trait template fix** is a CUTLASS-internal cleanup
   (rewrite `MMA_Traits<…>` partial specialization to use type-
   templated SM100_MMA_F8F6F4_SS). Transitive only — no SM120 builder
   changed signature.

4. **Example 93 paged GQA** is **SM100a-only** (uses TMEM). Not
   portable to sm_121.

## Validations performed

- ✓ Submodule bumped, builds clean (`build_cutlass_sm120_so.sh sm_121a`)
- ✓ Manifest updated to match new `.so` sha256
- ✓ Service starts on mistral profile + qwen profile
- ✓ Canonical regression: mistral 634-tok prompt → md5 75d36e68
- ✓ Qwen long-prefill regression: byte-identical output across v4.4.x/v4.5.1
- ✓ Tile-shape sweep: no kernel change picks up tile config (confirms
  CUTLASS NVFP4 GEMM kernel is dormant on mistral hot path)

## Future kernel-touching opportunities (not in this commit)

1. **Move Gemma 4 FP8 + Qwen FP8 output from f16 → bf16**. Would
   enable the `cvt.rn.bf16x2.e4m3x2` PTX instruction in the epilogue.
   Requires downstream consumer changes (the Rust caller currently
   reads f16 output). Estimate: 0.5-1 day, possible 1-5% epilogue
   speedup — but epilogue is a small share of total kernel time.

2. **Wire the dormant NVFP4 GEMM into mistral35 prefill**. The kernel
   exists (`cutlass_nvfp4_gemm_sm120`) but isn't called — the
   historical bug-hunt comment at mistral35_bring_up.rs:2249 records
   why. Re-investigation could unlock W4A16 prefill amortization.
   Estimate: multi-day. High risk, high upside.

3. **Hand-port example 93's GQA layout** to sm_121 without TMEM.
   Use shared memory + cooperative warps instead. Multi-day kernel
   engineering. Reference value only; speculative payoff.

## Net result

v4.5.1 is a clean version bump with zero regression and zero
measurable perf delta on any current production workload. The
audit's value is establishing this baseline definitively — future
CUTLASS-touching work can build from v4.5.1 without worrying about
known-good vs new-known-good.
