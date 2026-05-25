#!/usr/bin/env python3
"""Task #143 P2 Phase 1 — isolated NVFP4 K-tile dequant micro-benchmark
harness.

Loads kernels/sm_121/nvfp4kv_dequant_bench.ptx and times the two
variants exposed there:
  * `nvfp4kv_dequant_sync_kernel`     — synchronous __ldg + dequant.
  * `nvfp4kv_dequant_cpasync_kernel`  — cp.async-staged depth=2.

The benchmark sweeps `k_tiles_per_block ∈ {64, 256, 1024}` × 1024
blocks. For each (variant, k_tiles) cell we run 5 iterations after
2 warmups and report:
  * mean kernel ms
  * dequant throughput (NVFP4 elements / second)
  * the per-block xor accumulator (must match between variants —
    proves byte-equivalence)

Pass condition:
  1. Both kernels finish without CUDA error.
  2. XOR accumulators match exactly (kernel outputs are byte-identical
     on the dequant pass).
  3. cp.async kernel's throughput is reported. Headline number is
     informational — the goal of Phase 1 is the technique landing +
     correctness, not a default-on flip.

Run: ./v3/tools/nvfp4kv_dequant_bench.py [--arch sm_121]
"""

import argparse
import pathlib
import sys

import numpy as np
from cuda.bindings import driver as drv


REPO = pathlib.Path(__file__).resolve().parents[2]


def CHECK(ret, label):
    if isinstance(ret, tuple):
        err, *rest = ret
        if err != drv.CUresult.CUDA_SUCCESS:
            sys.exit(f"CUDA error in {label}: {err}")
        return rest[0] if len(rest) == 1 else rest
    if ret != drv.CUresult.CUDA_SUCCESS:
        sys.exit(f"CUDA error in {label}: {ret}")
    return None


def event_create():
    return CHECK(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DEFAULT), "evt create")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="sm_121")
    ap.add_argument("--blocks", type=int, default=1024)
    args = ap.parse_args()

    arch = args.arch
    ptx = REPO / "kernels" / arch / "nvfp4kv_dequant_bench.ptx"
    if not ptx.exists():
        sys.exit(f"missing PTX (run kernels/build.sh {arch}): {ptx}")

    CHECK(drv.cuInit(0), "cuInit")
    dev = CHECK(drv.cuDeviceGet(0), "cuDeviceGet")
    ctx = CHECK(drv.cuDevicePrimaryCtxRetain(dev), "ctxRetain")
    CHECK(drv.cuCtxSetCurrent(ctx), "ctxSetCurrent")

    mod = CHECK(drv.cuModuleLoadData(ptx.read_bytes() + b"\0"), "load ptx")
    fn_sync    = CHECK(drv.cuModuleGetFunction(
        mod, b"nvfp4kv_dequant_sync_kernel"),    "get sync")
    fn_cpasync = CHECK(drv.cuModuleGetFunction(
        mod, b"nvfp4kv_dequant_cpasync_kernel"), "get cpasync")

    BLOCK = 128
    blocks = args.blocks
    TILE_BYTES = 8

    rng = np.random.default_rng(2026)

    fail = False
    for k_tiles in (64, 256, 1024):
        block_tiles_stride = k_tiles * BLOCK
        total_tiles = blocks * block_tiles_stride
        packed_bytes = total_tiles * TILE_BYTES
        scale_bytes  = total_tiles * 4

        # Random packed nvfp4 bytes + random per-tile scale.
        packed_host = rng.integers(0, 256, size=packed_bytes,
                                    dtype=np.uint8)
        scale_host  = (rng.standard_normal(total_tiles).astype(np.float32)
                       * 0.5 + 1.0)

        d_packed = CHECK(drv.cuMemAlloc(packed_bytes), "alloc packed")
        d_scale  = CHECK(drv.cuMemAlloc(scale_bytes),  "alloc scale")
        d_xor    = CHECK(drv.cuMemAlloc(4),            "alloc xor")
        CHECK(drv.cuMemcpyHtoD(d_packed, packed_host.ctypes.data, packed_bytes),
              "H2D packed")
        CHECK(drv.cuMemcpyHtoD(d_scale,  scale_host.ctypes.data,  scale_bytes),
              "H2D scale")

        results = {}
        for name, fn, smem in (
            ("sync",    fn_sync,
             BLOCK * 16 * 2),  # [BLOCK, 16] f16
            ("cpasync", fn_cpasync,
             # 2× [BLOCK, 8] packed + 2× [BLOCK] scale + [BLOCK, 16] out
             2 * BLOCK * TILE_BYTES + 2 * BLOCK * 4 + BLOCK * 16 * 2),
        ):
            # 2 warmups, 5 timed iterations.
            ev_a = event_create()
            ev_b = event_create()
            ts = []
            for it in range(7):
                CHECK(drv.cuMemsetD8(d_xor, 0, 4), "zero xor")
                if it >= 2:
                    CHECK(drv.cuEventRecord(ev_a, 0), "record a")
                args_arr = [
                    np.array([int(d_packed)], dtype=np.uint64),
                    np.array([int(d_scale)],  dtype=np.uint64),
                    np.array([int(d_xor)],    dtype=np.uint64),
                    np.array([k_tiles],            dtype=np.int32),
                    np.array([block_tiles_stride], dtype=np.int32),
                ]
                pp = np.array([a.ctypes.data for a in args_arr], dtype=np.uint64)
                CHECK(drv.cuLaunchKernel(fn, blocks, 1, 1, BLOCK, 1, 1,
                                         smem, 0, pp.ctypes.data, 0),
                      f"launch {name}")
                if it >= 2:
                    CHECK(drv.cuEventRecord(ev_b, 0), "record b")
                    CHECK(drv.cuEventSynchronize(ev_b), "sync b")
                    ms = CHECK(drv.cuEventElapsedTime(ev_a, ev_b), "elapsed")
                    ts.append(float(ms))
                else:
                    CHECK(drv.cuCtxSynchronize(), "sync warm")

            xor_host = np.zeros(1, dtype=np.uint32)
            CHECK(drv.cuMemcpyDtoH(xor_host.ctypes.data, d_xor, 4), "D2H xor")
            mean_ms = float(np.mean(ts))
            min_ms  = float(np.min(ts))
            elems   = float(total_tiles * 16)
            tput_geps = elems / (mean_ms / 1e3) / 1e9
            results[name] = {
                "ms_mean": mean_ms,
                "ms_min":  min_ms,
                "xor":     int(xor_host[0]),
                "tput_geps": tput_geps,
            }

        for d in (d_packed, d_scale, d_xor):
            CHECK(drv.cuMemFree(d), "free")

        s = results["sync"]
        c = results["cpasync"]
        xor_ok = s["xor"] == c["xor"]
        speedup = s["ms_mean"] / c["ms_mean"] if c["ms_mean"] > 0 else 0.0

        print(f"k_tiles={k_tiles:4d} | "
              f"sync    ms_mean={s['ms_mean']:8.3f} min={s['ms_min']:8.3f} "
              f"tput={s['tput_geps']:7.2f} Gelem/s xor=0x{s['xor']:08x}")
        print(f"             | "
              f"cpasync ms_mean={c['ms_mean']:8.3f} min={c['ms_min']:8.3f} "
              f"tput={c['tput_geps']:7.2f} Gelem/s xor=0x{c['xor']:08x} "
              f"| speedup={speedup:.3f}× | xor_match={xor_ok}")
        if not xor_ok:
            print(f"  XOR mismatch — kernel outputs DIVERGE. FAIL.")
            fail = True

    print()
    if fail:
        sys.exit(1)
    print("Correctness OK across all k_tiles. cp.async speedup reported.")


if __name__ == "__main__":
    main()
