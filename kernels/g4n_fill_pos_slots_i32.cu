// g4n_fill_pos_slots_i32: device-side fill of Option B's per-token
// metadata for the NVFP4 forward path (Gemma 4 31B native).
//
// Background: the legacy Option B path wrote `positions[0]`,
// `slot_mapping[0]`, and `context_lens[0]` via three SYNCHRONOUS
// `cuMemcpyHtoD_v2` calls on the implicit default stream, while
// every kernel in the forward chain runs on a NON-blocking stream
// (`self.stream`). The default-stream-vs-non-blocking-stream
// ordering is NOT guaranteed by CUDA; the same race that produced
// non-deterministic "Schule" / "Supermarkt" tokens on Qwen 3.6
// applies here.
//
// This kernel runs on the same stream as the RoPE+KV-write and FA2
// decode/prefill kernels, so the writes are stream-ordered. ABI is
// generic over the two Option B modes:
//
//   Mode A (mini cos/sin tables, per-launch 1-row, current default):
//     `position_offset = 0`, kernel writes `positions[t] = t`
//     (the kernel reads cos_table[positions[t] * half_rotary + freq],
//     and the per-launch tables have rows for actual positions).
//
//   Mode B (persistent f16 RoPE tables — lands in the next commit):
//     `position_offset = start_slot`, kernel writes
//     `positions[t] = start_slot + t` so the kernel indexes the
//     correct absolute row.
//
// slot_mapping and context_lens are independent of mode:
//
//   slot_mapping[t] = start_slot + t   (where to write K/V in cache)
//   context_lens[t] = start_slot + t + 1
//
// `q_scale_ptr` is set once at KV-state allocation and is NOT
// touched here — it is per-bringup, not per-forward.
//
// Launch: grid = (ceil(num_tokens / block), 1, 1), block = (256,
// 1, 1). Single launch per forward call; the prior 3x sync HtoD
// was 3 launches on the wrong stream.

extern "C" __global__ void __launch_bounds__(256)
g4n_fill_pos_slots_i32_kernel(
    int* __restrict__ positions,        // [num_tokens] i32
    int* __restrict__ slot_mapping,     // [num_tokens] i32
    int* __restrict__ context_lens,     // [num_tokens] i32
    int position_offset,
    int start_slot,
    int num_tokens
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tokens) return;
    positions[t]    = position_offset + t;
    slot_mapping[t] = start_slot + t;
    context_lens[t] = start_slot + t + 1;
}
