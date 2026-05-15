// Gemma 4 E4B assistant-drafter MaskedEmbedder kernel.
//
// One fused single-CTA kernel that mirrors HF's
// `Gemma4AssistantMaskedEmbedder.forward` for the L=1 / B=1 case used by
// speculative decoding. Inputs / outputs match the Rust launcher in
// `gemma4_drafter.rs::launch_masked_embedder_argmax_f16`.
//
// Algorithm (HF reference at modeling_gemma4_assistant.py:42-87):
//
//   1. centroid_logits = hidden @ centroids^T          // [num_centroids]
//   2. top_k_indices   = topk(centroid_logits, top_k)  // [top_k]
//   3. for each selected centroid c, gather `per_centroid` token ids
//      from `token_ordering[c * per_centroid : (c+1) * per_centroid]`.
//   4. selected_logits = hidden @ lm_head[selected_token_ids]^T
//      // [top_k * per_centroid]   — typically 4096 dot products.
//   5. argmax over the candidate set (HF's scatter to a full-vocab
//      mask_value is skipped here — for greedy decoding we only need
//      the argmax token id, so we save the 256 KiB scatter buffer).
//
// Shape constants on E4B / assistant:
//   hidden_size   = 256
//   num_centroids = 2048
//   top_k         = 32
//   per_centroid  = vocab / num_centroids = 262144 / 2048 = 128
//   vocab         = 262144
//
// Launch:
//   grid  = (1, 1, 1)
//   block = (256, 1, 1)
//   smem  = sizeof(float) * hidden_size      // hidden as f32
//         + sizeof(float) * num_centroids    // centroid logits f32
//         + sizeof(int)   * top_k            // selected centroid ids
//         + sizeof(float) * num_warps        // per-warp best logit
//         + sizeof(int)   * num_warps        // per-warp best token id
//
// On E4B (256 threads = 8 warps): smem = 1024 + 8192 + 128 + 32 + 32 =
// 9408 bytes — comfortably under the 48 KiB default cuLaunchKernel cap,
// no `cuFuncSetAttribute` dance needed.
//
// Phase 3's top-k is serialized on thread 0 (32 passes over 2048
// centroid logits = ~65 K ops). Phase 4 dominates wall time; serial
// top-k overhead is in the microsecond range and not worth a parallel
// implementation.

#include <cuda_fp16.h>
#include <cstdint>

// Commit 28: optional sparse output. When `out_sparse_ids` /
// `out_sparse_logits` are non-null, writes the full top_k *
// per_centroid candidate table (~4096 entries on E4B / assistant)
// to host-allocated buffers. Host then samples with temperature
// for typical-acceptance spec-decode. Greedy argmax path is
// unchanged when these pointers are null.
extern "C" __global__ void gemma4_masked_embedder_argmax_f16_kernel(
    const __half*  __restrict__ hidden,
    const __half*  __restrict__ centroids,
    const int64_t* __restrict__ token_ordering,
    const __half*  __restrict__ lm_head,
    int hidden_size,
    int n_centroids,
    int top_k,
    int per_centroid,
    int vocab,
    int*   __restrict__ out_token_id,
    float* __restrict__ out_logit,
    int*   __restrict__ out_sparse_ids,
    float* __restrict__ out_sparse_logits
) {
    extern __shared__ float smem_buf[];

    float* hidden_f32       = smem_buf;
    float* centroid_logits  = hidden_f32 + hidden_size;
    int*   top_centroid_ids = reinterpret_cast<int*>(centroid_logits + n_centroids);
    float* warp_best_logits = reinterpret_cast<float*>(top_centroid_ids + top_k);
    // num_warps = blockDim.x / 32; layout pads to 8 entries which is
    // the supported maximum on this kernel (block size = 256).
    int*   warp_best_ids    = reinterpret_cast<int*>(warp_best_logits + 8);

    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp_id   = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    // === Phase 0: initialize sparse outputs (only if requested) ===========
    // Sparse slots that the main loop never visits (c<0 / out-of-vocab)
    // keep these sentinels so the host sampler can mask them.
    if (out_sparse_ids != nullptr) {
        const int total_sparse = top_k * per_centroid;
        for (int i = tid; i < total_sparse; i += blockDim.x) {
            out_sparse_ids[i]    = -1;
            out_sparse_logits[i] = -INFINITY;
        }
        __syncthreads();
    }

    // === Phase 1: hidden f16 → smem f32 ====================================
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        hidden_f32[i] = __half2float(hidden[i]);
    }
    __syncthreads();

    // === Phase 2: centroid_logits[c] = hidden · centroids[c] ===============
    // Each warp handles one centroid per outer iteration; loop until all
    // `n_centroids` rows are covered.
    for (int c = warp_id; c < n_centroids; c += num_warps) {
        float acc = 0.0f;
        const __half* row = centroids + (size_t)c * hidden_size;
        for (int k = lane; k < hidden_size; k += 32) {
            acc += hidden_f32[k] * __half2float(row[k]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0) centroid_logits[c] = acc;
    }
    __syncthreads();

    // === Phase 3: top-k centroids (serial argmax on thread 0) ==============
    // Mark each selected slot with -INFINITY so the next pass skips it.
    if (tid == 0) {
        for (int k = 0; k < top_k; k++) {
            float best     = -INFINITY;
            int   best_idx = -1;
            for (int c = 0; c < n_centroids; c++) {
                float v = centroid_logits[c];
                if (v > best) {
                    best     = v;
                    best_idx = c;
                }
            }
            top_centroid_ids[k] = best_idx;
            if (best_idx >= 0) {
                centroid_logits[best_idx] = -INFINITY;
            }
        }
    }
    __syncthreads();

    // === Phase 4: sparse logits over top_k × per_centroid + argmax =========
    // Each warp picks dot product index `idx` from a strided loop. lane 0
    // accumulates this warp's local best (token_id, logit). After the loop,
    // each warp stages its best into smem; thread 0 reduces across warps.
    const int total = top_k * per_centroid;
    float my_best    = -INFINITY;
    int   my_best_id = -1;

    for (int idx = warp_id; idx < total; idx += num_warps) {
        const int centroid_slot = idx / per_centroid;
        const int sub           = idx % per_centroid;
        const int c             = top_centroid_ids[centroid_slot];
        // Defensive: if top-k pulled fewer than top_k valid centroids
        // (e.g. n_centroids < top_k — unreachable on E4B), skip.
        if (c < 0) continue;
        const int64_t tok64 = token_ordering[(int64_t)c * per_centroid + sub];
        if (tok64 < 0 || tok64 >= (int64_t)vocab) continue;
        const int token_id = (int)tok64;

        float acc = 0.0f;
        const __half* row = lm_head + (int64_t)token_id * hidden_size;
        for (int k = lane; k < hidden_size; k += 32) {
            acc += hidden_f32[k] * __half2float(row[k]);
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_xor_sync(0xFFFFFFFFu, acc, off);
        }
        if (lane == 0 && acc > my_best) {
            my_best    = acc;
            my_best_id = token_id;
        }
        // Commit 28: write sparse candidate (id, logit) for the
        // host-side typical-acceptance sampler. Lane-0 only — `acc`
        // post-shuffle is identical across the warp's lanes for
        // this one (idx, token_id), so a single write per warp
        // is correct. Slot index `idx` in the flat [top_k *
        // per_centroid] output buffer; the lane filtering for
        // `c < 0` / out-of-vocab below either skips the write
        // (sentinel -1 / -INFINITY) so the host sampler can mask
        // them out.
        if (lane == 0 && out_sparse_ids != nullptr) {
            out_sparse_ids[idx]    = token_id;
            out_sparse_logits[idx] = acc;
        }
    }

    if (lane == 0) {
        warp_best_logits[warp_id] = my_best;
        warp_best_ids[warp_id]    = my_best_id;
    }
    __syncthreads();

    // === Phase 5: cross-warp reduction (single thread) =====================
    if (tid == 0) {
        float best     = -INFINITY;
        int   best_id  = -1;
        for (int w = 0; w < num_warps; w++) {
            float v = warp_best_logits[w];
            if (v > best) {
                best    = v;
                best_id = warp_best_ids[w];
            }
        }
        if (out_token_id) out_token_id[0] = best_id;
        if (out_logit)    out_logit[0]    = best;
    }
}
