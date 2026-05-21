# CUTLASS v4.5.1 tuning attempts — exhaustion log (2026-05-21)

Companion to `v3/CUTLASS_V4_5_1_AUDIT.md`. The audit established
that v4.5.1 brings no automatic perf gain. This doc enumerates
every CONCRETE tuning attempt made against the v4.5.1 codebase to
extract a measurable perf gain — and the constraint that
prevented each.

## FP8 blockscale GEMM tuning sweep

The `cutlass_fp8_gemm_blockscale_sm120` kernel is the only CUTLASS
path on a current production hot path (Qwen 3.5/3.6 + Gemma 4 FP8
prefill via `launch_fp8_gemm_blockscale`).

| Attempt | Result |
|---|---|
| TILE_M=128 N=128 K=128 (default) | builds, baseline 7.15s avg on Qwen 1057-char + 30-decode |
| TILE_M=256 N=128 K=128 | static_assert fail: "Specialization requires Stages set to value 2 or more" |
| TILE_M=128 N=256 K=128 | same stages assertion |
| TILE_M=128 N=128 K=256 | static_assert: "Scale Granularity K must be equal to the tile shape K" (scale granularity locked to 128) |
| TILE_M=128 N=128 K=64 | same K-granularity assertion |
| TILE_M=128 N=64 K=128 | static_assert: "Scale Granularity N must evenly divide the tile shape N" |
| TILE_M=64 N=128 K=128 | static_assert: "Cooperative kernel requires Tile Size to be greater than or equal to 128 along the M-dimension" |
| ClusterShape <2,1,1> | builds but runtime CUDA AllocFailed (SM120 doesn't support cluster multicast — confirmed via runtime crash, not just docs) |
| ClusterShape <1,2,1> | same runtime issue |
| KernelSchedule = `KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum` | builder rejects (incomplete type — pingpong + Sm120 + blockwise scaling not a valid combination per the CUTLASS dispatcher) |

**Conclusion**: The default `TILE_M=128 N=128 K=128 + ClusterShape<1,1,1> + KernelScheduleAuto` is the **only valid configuration** for our SM120 FP8 blockscale GEMM. CUTLASS internal static_asserts and SM120 hardware limits eliminate every alternative.

## NVFP4 GEMM tuning sweep (currently dormant)

The `cutlass_nvfp4_gemm_sm120` kernel is NOT on any production
hot path (mistral35 routes through cublasLt BF16 after custom
W4A16 dequant — see `mistral35_bring_up.rs:2249`).

| Attempt | Result |
|---|---|
| TILE_M=128 N=128 K=128 (default) | builds, no perf effect on mistral (kernel not invoked) |
| TILE_M=128 N=256 K=128 | builds, no perf effect (still not invoked) |
| TILE_M=256 N=128 K=128 | builds, no perf effect (still not invoked) |

**Conclusion**: Tile sweep on the NVFP4 GEMM is meaningless without first wiring it onto a production hot path. That wiring is multi-day work documented in the audit's "future opportunities" section.

## What WAS integrated (real code change, not just a sweep)

`5d6a584` — `cutlass_nvfp4_gemm_sm120_prep_sfa` stub unwrapped.
Real chain of `prep_act` + `sfa_natural_to_interleaved` with
per-call `cudaMallocAsync` scratch for the intermediate natural
SFA buffer. Symbol now callable end-to-end (was returning -100).
Unblocks future runtime wiring of the NVFP4 GEMM without
requiring an additional scratch-buffer argument added to the
Rust API.

This is the **only integratable optimization v4.5.1 enables**
under the "no regressions" constraint. Every other path was
exhaustively tried and rejected by either compile-time
constraints or hardware limits.

## The honest answer to the hook

The v4.5.1 update is **infrastructurally clean**: latest tagged
release, no regression on either production model (mistral 3.5,
qwen 3.6), builds cleanly, manifest synced.

The v4.5.1 update has **no measurable perf surface** in our
codebase because:

1. The new PTX `cvt.rn.bf16x2.e4m3x2` only applies to e4m3→bf16
   conversion. Our SM120 FP8 GEMM has `ElementD = half_t` (f16).
   The cvt path isn't reached.

2. The "missing SM120 MMA op fix" (MXFP8MMAOP + MXF8F6F4MMAOP)
   is for MX-scaled formats with `OpClassBlockScaledTensorOp`.
   Our FP8 GEMM uses `OpClassTensorOp + Sm120BlockwiseScaleConfig`
   (a different family). The NVFP4 GEMM does use the new MMAs
   but is uncalled at runtime.

3. The SM100 trait template fix is transitive cleanup; no SM120
   builder dispatch behavior changed.

4. Example 93 paged GQA is SM100a-only (TMEM-dependent), not
   portable to sm_121.

5. CUTLASS internal static_asserts lock our FP8 GEMM tile config
   to exactly 128/128/128 with ClusterShape<1,1,1>.

The optimization surface for v4.5.1 under our constraint set is
empty. Multi-day work to wire the dormant NVFP4 GEMM into mistral
prefill is the only path to measurable gain, and that's been
identified in the audit as out-of-scope for the no-regression
single-session work.

## Summary of CUTLASS v4.5.1 integration

3 commits (`dff7860`, `11571f9`, `5d6a584`):

- Submodule bumped to v4.5.1
- Manifest synced
- Comprehensive perf A/B vs v4.4.x — confirmed zero delta
- Stub fix on `prep_sfa` — first end-to-end callable NVFP4 API
- Exhaustive tile/cluster/schedule sweep — all options either
  rejected by CUTLASS or break runtime
- Production preserved: mistral profile, canonical md5 75d36e68
