# Smem reduction analysis — `flash_attention_unified_prefill_nvfp4kv.cu`

Branch: `perf/smem-svt-260520` (off `rusty_sm121_qwen36_26b`)
Target: 68 KiB → ≤32 KiB to unlock 2+ CTA/SM on GB10 sm_121.

## Current smem layout (head_dim=256, BLOCK_M=16, tile_size=32, MMA_K=16)

| Buffer | Size | KB | Purpose |
|---|---|---|---|
| `s_q_f16` | BLOCK_M × head_dim × 2 | 8 | Dequanted Q |
| `s_k_f16` | tile_size × head_dim × 2 | 16 | K tile row-major |
| `s_v_f16` | tile_size × head_dim × 2 | 16 | V tile row-major |
| `s_v_f16_T` | MMA_K × head_dim × 2 | 8 | V transposed for P·V pack_b |
| `s_acc` | BLOCK_M × head_dim × 4 | 16 | f32 O accumulator |
| `s_s` | BLOCK_M × max(MMA_K, tile_size) × 4 | 2 | Softmax output |
| `s_p_f16` | BLOCK_M × MMA_K × 2 | 0.5 | P cast to f16 for MMA |
| `s_q_scale` | BLOCK_M × 4 | 0.06 | Per-row Q scale |
| `s_m, s_l, s_alpha` | 3 × BLOCK_M × 4 | 0.2 | Online softmax state |
| **Total** | | **~67 KB** | |

At head_dim=512 (Gemma 4 global layers), buffers scale 2× → **~134 KB**.

## Reduction candidates evaluated

### 1. Eliminate `s_v_f16_T` via `ldmatrix.x4.trans` ✅ technically viable
- Saves **8 KB** at head_dim=256 / **16 KB** at head_dim=512.
- Requires writing PTX inline-asm helper for `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` and per-lane register-pattern verification vs `mma.sync.aligned.m16n8k16.row.col`.
- Effort: ~1 day kernel surgery + byte-equivalence A/B.
- New total at head_dim=256: ~59 KB. **Still > 32 KB.**

### 2. Halve `s_acc` to f16 ❌ rejected
- Online-softmax rescaling does `acc *= alpha` where `alpha = exp(m_old - m_new)` and can grow exponentially across tiles.
- f16 accumulator would lose all precision after ~3-4 rescales; output quality cliff.
- 16 KB savings not worth the regression risk.

### 3. Reduce `BLOCK_M` from 16 to 8 ⚠️ partial
- Saves ~5 KB (s_q, s_acc, s_p, s_s halve).
- New total at head_dim=256: ~62 KB.
- Doubles grid size in M-dim, doubling K/V reload cost. Likely net-neutral or slower.

### 4. Halve `s_k_f16` + `s_v_f16` via tile_size=16 ⚠️ partial
- Saves **16 KB**. New total: ~51 KB. Still > 32 KB.
- Doubles tile-loop iteration count → more sync/dequant overhead per K tile.

### 5. Stream-dequant K/V per MMA k-step ✅ technically viable, deep
- Eliminate the explicit K/V smem tile altogether; dequant directly into MMA-A/B fragments per k-step (16 elements at a time).
- Saves **32 KB** (full `s_k_f16` + `s_v_f16`).
- Requires rewriting the tile loop as a streaming pipeline, double-buffered dequant + MMA, careful warp scheduling.
- Effort: ~3-5 days kernel surgery + careful occupancy/throughput tuning.
- New total at head_dim=256: ~35 KB. **Borderline at the target.**

### 6. Combine #1 + #5 ✅ the only path to ≤32 KB
- Eliminate `s_v_f16_T` (#1) AND stream-dequant K/V (#5).
- Saves 40 KB total. New total at head_dim=256: ~27 KB. **Hits target.**
- Effort: full kernel rewrite, ~1 week with profiler in the loop.

## Negative results from prior experiments (same target)

| Experiment | Branch | Outcome |
|---|---|---|
| Lift `#pragma unroll 1` on MMA loops | `kernel_smem_qwen35_experiment` (commit `bacb77f`) | Steady-state perf identical (9.33s vs 9.33s, byte-equiv output). Kernel NOT MMA-instruction-bound. |

## Conclusion

The 68 KiB → ≤32 KiB target requires combined reduction #1 + #5, a multi-day rewrite needing Nsight Compute roofline + occupancy analysis in the loop. No single-edit change can achieve it.

**Realistic incremental milestone**: candidate #1 alone (8 KB savings + ~14% occupancy headroom at head_dim=256) is the highest perf-per-effort move, but does not unlock 2 CTA/SM by itself.

**Next-session prep**:
1. Set up Nsight Compute roofline capture on a representative prefill (qwen35 16k prompt).
2. Confirm whether the kernel is bound by smem latency (NVFP4 dequant pipe), smem capacity (occupancy=1), MMA throughput, or DRAM (block_tables/KV cache fetch).
3. Pick the dominant bound and target that first.
