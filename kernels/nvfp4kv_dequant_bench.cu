// Task #143 P2 Phase 1 — isolated NVFP4 K-tile dequant micro-benchmark
// for the gemma4-nvfp4 prefill cold-path bottleneck.
//
// Per the nsys profile in task #139, `flash_attention_2_prefill_nvfp4kv_
// unified_bf16out_kernel` is 38.3% of cold-prefill GPU time at M=14k
// (= 9.5s of 25s total). The original kernel-author analysis pinned the
// bottleneck to per-tile NVFP4 K/V dequant via `cvt.rn.f16x2.e2m1x2`
// + smem store, not the MMA itself. cp.async-staged double-buffered
// load+dequant is the canonical sm_120+ remediation: while warp N
// dequants tile T into smem buffer A, an async load streams tile T+1
// from HBM into smem buffer B, overlapping HBM latency with compute.
//
// This file ships TWO standalone kernels with byte-identical output:
//   * `nvfp4kv_dequant_sync_kernel`  — synchronous baseline. Per K
//     iteration: `__ldg(packed)` from HBM → register → dequant via
//     `unpack16_nvfp4_to_f16_fast` → smem store. Models the load
//     pattern in flash_attention_unified_prefill_nvfp4kv_bf16out.cu's
//     K-tile staging.
//   * `nvfp4kv_dequant_cpasync_kernel` — cp.async-staged. Per K
//     iteration: `cp.async` the NEXT tile from HBM directly into smem
//     (skipping the register stage), `cp.async.commit_group` +
//     `cp.async.wait_group(1)` to keep one tile in flight. Dequant
//     happens on the CURRENT tile while N+1 is loading.
//
// Both kernels write the same final smem layout, then accumulate an
// XOR of all output halves to a single device u32 so the optimizer
// cannot dead-code the dequant loop (the compute is what we're
// timing, not the store).
//
// The Python harness drives a (block-count × iteration-count) sweep
// + cudaEvent timing; PASS criterion is XOR equality between the two
// variants (proves byte-equivalent semantics). Throughput is reported
// in NVFP4 elements/second so downstream sessions can decide whether
// to integrate cp.async into the full attention kernel.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "nvfp4_utils.cuh"

// Each "tile" = 16 NVFP4 elements packed in 8 bytes (1 × u64) + 1
// E4M3 microscale.  Per iteration each thread dequants its own tile
// → 16 f16 in smem.  Sweep `K_TILES` iterations of `BLOCK_SIZE`
// threads to amplify the throughput signal.

#define TILE_BYTES 8       // u64 packed nvfp4 (16 elements)
#define TILE_HALFS 16

extern "C"
__global__ void nvfp4kv_dequant_sync_kernel(
    const unsigned char* __restrict__ k_packed,   // [num_tiles_total, 8]
    const float*         __restrict__ k_scales,   // [num_tiles_total]
    unsigned int*        __restrict__ xor_accum,  // [1]
    int                                k_tiles_per_block,
    int                                block_tiles_stride
) {
    extern __shared__ __half s_buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    long long base_tile = (long long)bid * block_tiles_stride;

    unsigned int local_xor = 0u;
    for (int iter = 0; iter < k_tiles_per_block; ++iter) {
        long long tile_idx = base_tile + (long long)iter * blockDim.x + tid;
        // Synchronous load.
        unsigned long long packed = __ldg(
            reinterpret_cast<const unsigned long long*>(
                k_packed + tile_idx * TILE_BYTES));
        float scale = __ldg(&k_scales[tile_idx]);

        __half local[TILE_HALFS];
        rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(packed, scale, local);

        // Store to smem (forces use; mimics the attention kernel
        // staging pattern that the rest of the inner loop reads).
        __half* s_dst = s_buf + tid * TILE_HALFS;
        #pragma unroll
        for (int j = 0; j < TILE_HALFS; ++j) {
            s_dst[j] = local[j];
            unsigned short bits;
            asm("mov.b16 %0, %1;" : "=h"(bits) : "h"(*reinterpret_cast<unsigned short*>(&local[j])));
            local_xor ^= (unsigned int)bits;
        }
    }

    // Reduce per-block xor into the global accumulator (1 atomicXor
    // per block; cost negligible vs the dequant loop).
    if (tid == 0) {
        // warp reduce first to amortize
    }
    unsigned int warp_xor = local_xor;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_xor ^= __shfl_xor_sync(0xffffffff, warp_xor, offset);
    }
    if ((tid & 31) == 0) {
        atomicXor(xor_accum, warp_xor);
    }
}

// cp.async-staged variant. Pipelined depth=2: at any moment, tile N
// is being dequanted from smem buf[N%2] while tile N+1 is in flight
// from HBM to smem buf[(N+1)%2].
//
// The cp.async instruction requires 4/8/16-byte aligned loads at
// known offsets and was introduced in sm_80 (Ampere). It's stable on
// sm_120/sm_121 (Blackwell consumer) — the same path the in-tree
// `mistral35_w4a16_gemm_mma_v8_bf16.cu` already uses for w4a16 weight
// staging. We use 8-byte cp.async per thread (one nvfp4 tile).
//
// Smem layout: 2 × [BLOCK_SIZE, 16] f16 = 2 × 4 KB for BLOCK_SIZE=128
// = 8 KB total (plus the actual sync-kernel's [BLOCK_SIZE, 16] output
// staging area which we keep separate so byte-equivalence with the
// sync kernel is preserved).
extern "C"
__global__ void nvfp4kv_dequant_cpasync_kernel(
    const unsigned char* __restrict__ k_packed,
    const float*         __restrict__ k_scales,
    unsigned int*        __restrict__ xor_accum,
    int                                k_tiles_per_block,
    int                                block_tiles_stride
) {
    extern __shared__ unsigned char s_raw[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Double-buffered staging: 2 × [BLOCK_SIZE, TILE_BYTES] for packed
    // + 2 × [BLOCK_SIZE] for scales + 1 × [BLOCK_SIZE, TILE_HALFS] for
    // the final dequant output (sync-equivalent layout).
    unsigned char* s_packed_a = s_raw;
    unsigned char* s_packed_b = s_packed_a + blockDim.x * TILE_BYTES;
    float*         s_scale_a  = reinterpret_cast<float*>(
                                  s_packed_b + blockDim.x * TILE_BYTES);
    float*         s_scale_b  = s_scale_a + blockDim.x;
    __half*        s_out      = reinterpret_cast<__half*>(
                                  s_scale_b + blockDim.x);

    long long base_tile = (long long)bid * block_tiles_stride;

    auto issue_load = [&](int iter, unsigned char* dst_packed, float* dst_scale) {
        if (iter >= k_tiles_per_block) return;
        long long tile_idx = base_tile + (long long)iter * blockDim.x + tid;
        const unsigned char* src_packed = k_packed + tile_idx * TILE_BYTES;
        unsigned char* my_dst_packed = dst_packed + tid * TILE_BYTES;
#if __CUDA_ARCH__ >= 800
        unsigned int dst_smem = __cvta_generic_to_shared(my_dst_packed);
        asm volatile(
            "cp.async.ca.shared.global [%0], [%1], 8;"
            :: "r"(dst_smem), "l"(src_packed)
        );
#else
        unsigned long long packed = *reinterpret_cast<const unsigned long long*>(src_packed);
        *reinterpret_cast<unsigned long long*>(my_dst_packed) = packed;
#endif
        // scale via direct HtoS (still synchronous; 4 bytes is fine
        // through L2 + L1 since the dequant uses it on the SAME pass).
        dst_scale[tid] = __ldg(&k_scales[tile_idx]);
    };

#if __CUDA_ARCH__ >= 800
    auto commit  = []() { asm volatile("cp.async.commit_group;"); };
    auto wait_one = []() { asm volatile("cp.async.wait_group 0;"); };
#else
    auto commit  = []() { __syncthreads(); };
    auto wait_one = []() { __syncthreads(); };
#endif

    // Prime pipeline: issue load 0 → commit → wait → ready to consume.
    issue_load(0, s_packed_a, s_scale_a);
    commit();
    wait_one();
    __syncthreads();

    unsigned int local_xor = 0u;
    for (int iter = 0; iter < k_tiles_per_block; ++iter) {
        // Issue NEXT tile (iter+1) into the OTHER buffer.
        unsigned char* nxt_packed = (iter & 1) ? s_packed_a : s_packed_b;
        float*         nxt_scale  = (iter & 1) ? s_scale_a  : s_scale_b;
        if (iter + 1 < k_tiles_per_block) {
            issue_load(iter + 1, nxt_packed, nxt_scale);
            commit();
        }

        // Consume CURRENT tile from THIS buffer.
        unsigned char* cur_packed = (iter & 1) ? s_packed_b : s_packed_a;
        float*         cur_scale  = (iter & 1) ? s_scale_b  : s_scale_a;
        unsigned long long packed = *reinterpret_cast<unsigned long long*>(
            cur_packed + tid * TILE_BYTES);
        float scale = cur_scale[tid];

        __half local[TILE_HALFS];
        rvllm_nvfp4::unpack16_nvfp4_to_f16_fast(packed, scale, local);

        __half* s_dst = s_out + tid * TILE_HALFS;
        #pragma unroll
        for (int j = 0; j < TILE_HALFS; ++j) {
            s_dst[j] = local[j];
            unsigned short bits = *reinterpret_cast<unsigned short*>(&local[j]);
            local_xor ^= (unsigned int)bits;
        }

        // Wait for the NEXT tile to be ready before the next iter
        // consumes it. wait_group(0) drains all in-flight; we only
        // ever have one outstanding so this is the right call.
        if (iter + 1 < k_tiles_per_block) {
            wait_one();
            __syncthreads();
        }
    }

    unsigned int warp_xor = local_xor;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        warp_xor ^= __shfl_xor_sync(0xffffffff, warp_xor, offset);
    }
    if ((tid & 31) == 0) {
        atomicXor(xor_accum, warp_xor);
    }
}
