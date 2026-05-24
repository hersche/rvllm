// g4n_slot_mapping_mod_sliding_i32: ringbuf0 Phase 4 — derive a
// per-sliding-layer slot_mapping by taking (slot_mapping[t] %
// sliding_window). Output buffer is consumed by sliding-attention
// layers under RVLLM_KV_RING_BUFFER=1; global-attention layers
// continue to read the unwrapped slot_mapping. Companion to the
// rope-side wrap (`fused_rope_partial_*kv_kernel`'s `sliding_window`
// arg) and the block_tables ring pattern. All three pieces wrap by
// the same modulus so prefill writes + decode writes + attention
// reads land on identical physical slots.
//
// Launch: same grid/block layout as g4n_fill_pos_slots_i32:
//   grid  = (ceil(num_tokens / block), 1, 1)
//   block = (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256)
g4n_slot_mapping_mod_sliding_i32_kernel(
    const int* __restrict__ slot_mapping_in,    // [num_tokens] i32 source
    int*       __restrict__ slot_mapping_out,   // [num_tokens] i32 wrapped
    int sliding_window,
    int num_tokens
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    if (sliding_window <= 0) {
        slot_mapping_out[t] = slot_mapping_in[t];
    } else {
        int s = slot_mapping_in[t];
        // `s` is non-negative in all production paths (start_slot >= 0,
        // t >= 0). Defensive: if a future caller passes negative, fall
        // back to unwrapped to surface the bug rather than silently
        // produce the wrong physical slot.
        if (s < 0) {
            slot_mapping_out[t] = s;
        } else {
            slot_mapping_out[t] = s % sliding_window;
        }
    }
}
