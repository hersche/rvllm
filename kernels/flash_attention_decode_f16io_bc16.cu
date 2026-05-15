// BC=16 variant of `flash_attention.cu` for the Gemma 4 E4B
// assistant-drafter cross-attention against the global-source K/V
// shadow at `head_dim=512`.
//
// On GB10 sm_121 the per-CTA dynamic shared-memory ceiling is
// ~100 KiB. The compile-time BC=32 used by the production
// `flash_attention.ptx` exceeds that at `head_dim=512` (2 * 32 *
// 512 * 4 + ... ≈ 128 KiB) and `cuFuncSetAttribute` rejects the
// launch. Halving the K/V tile to BC=16 brings the same workload
// down to ~64 KiB, which fits.
//
// We compile the SAME kernel body with `FA2_BC=16` by forcing the
// macro before the #include. The compiled PTX module
// (`flash_attention_decode_f16io_bc16.ptx`) carries the entire
// kernel set with BC=16, but the runtime only loads the f16io
// decode entry by name. Other kernels in this PTX are unused.
//
// The symbol name stays
// `flash_attention_2_decode_f16io_kernel` — different module,
// same kernel name, resolved per-module by the loader.

#define FA2_BC 16
#include "flash_attention.cu"
