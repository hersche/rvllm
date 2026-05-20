// g4n_fill_pos_slots_i32_indirect: device-pointer variant of
// `g4n_fill_pos_slots_i32_kernel`. Identical math; differs only in
// that `position_offset`, `start_slot`, and `num_tokens` are read
// from device int* slots instead of inlined into the kernel arg
// buffer at launch time.
//
// Why: `cuStreamBeginCapture` records the literal scalar values
// passed to a kernel at capture time. To replay a captured forward
// against a NEW token position (the decode loop's per-iter use
// case), the per-iter scalars must live behind a stable device
// pointer that the caller updates via async HtoD between replays.
//
// Layout:
//   positions[t]    = position_offset + t
//   slot_mapping[t] = start_slot + t
//   context_lens[t] = start_slot + t + 1
//
// Caller must:
//   1. Allocate stable 4-byte device slots for each of the three
//      scalar inputs (lifetime ≥ the captured graph's lifetime).
//   2. Update those slots via `cuMemcpyHtoDAsync_v2` on the same
//      stream right before each `cuGraphLaunch` replay.
//   3. Bound num_tokens to the captured `kv.max_query_tokens` —
//      this kernel does NOT re-check.
//
// Launch: grid = (ceil(num_tokens_at_capture / block), 1, 1), block =
// (256, 1, 1). num_tokens_at_capture is what the GRID was sized for;
// at replay time `*num_tokens_ptr` must be ≤ that value (the kernel
// silently no-ops threads with t >= *num_tokens_ptr).

extern "C" __global__ void __launch_bounds__(256)
g4n_fill_pos_slots_i32_indirect_kernel(
    int* __restrict__ positions,        // [num_tokens] i32
    int* __restrict__ slot_mapping,     // [num_tokens] i32
    int* __restrict__ context_lens,     // [num_tokens] i32
    const int* __restrict__ position_offset_ptr,
    const int* __restrict__ start_slot_ptr,
    const int* __restrict__ num_tokens_ptr
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int n = *num_tokens_ptr;
    if (t >= n) return;
    int pos_off = *position_offset_ptr;
    int sslot   = *start_slot_ptr;
    positions[t]    = pos_off + t;
    slot_mapping[t] = sslot + t;
    context_lens[t] = sslot + t + 1;
}
