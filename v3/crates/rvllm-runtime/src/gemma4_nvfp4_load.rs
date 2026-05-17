//! Phase 2 of the nvidia/Gemma-4-31B-IT-NVFP4 bring-up: device
//! upload for the NVFP4 MLP weights + retained-bf16 attention
//! weights into an HbmArena. Wiring into the runtime bring-up
//! happens in Phase 3.
//!
//! Mirrors `mistral35_load::upload_nvfp4_linear` but for the
//! Gemma checkpoint's tensor names (`weight` / `weight_scale_2` /
//! `input_scale` instead of `weight_packed` / `weight_global_scale`
//! / dynamic per-token activation amax).
//!
//! `quant_config.input_activations.dynamic = false` means the
//! checkpoint ships a per-tensor static input scale. The runtime
//! consumes it via `Gemma4Nvfp4LinearLoaded::input_scale_ptr`;
//! the activation quantization kernel must read this scalar
//! rather than computing a fresh amax at request time.

#![cfg(feature = "cuda")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_loader::gemma4_arch::Gemma4Arch;
use rvllm_loader::gemma4_nvfp4_weights::{
    Gemma4Nvfp4LinearLoaded, Gemma4Nvfp4MlpKind,
};
use rvllm_loader::mistral35_weights::Nvfp4LinearShape;
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use rvllm_mem::HbmArena;

/// Memory-mapped shard pool. Duplicates the Mistral-side
/// `ShardPool` so Phase 2 doesn't have to refactor that file; a
/// Phase 5 cleanup can promote both to a shared helper.
pub(crate) struct Gemma4Nvfp4ShardPool {
    pub mmaps: Vec<memmap2::Mmap>,
    pub tensors: BTreeMap<String, (usize, TensorEntry)>,
    pub model_dir: PathBuf,
}

impl Gemma4Nvfp4ShardPool {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut mmaps = Vec::with_capacity(idx.shards.len());
        let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
        for (shard_idx, shard_path) in idx.shards.iter().enumerate() {
            let f = std::fs::File::open(shard_path).map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: shard_path.clone(),
                source,
            })?;
            let mmap = unsafe { memmap2::Mmap::map(&f) }.map_err(|source| RvllmError::Io {
                err: rvllm_core::IoError::from(&source),
                path: shard_path.clone(),
                source,
            })?;
            let header = ShardHeader::parse(shard_path, &mmap)?;
            for (name, entry) in header.tensors.into_iter() {
                tensors.insert(name, (shard_idx, entry));
            }
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, tensors, model_dir: model_dir.to_path_buf() })
    }

    pub fn must_get(&self, name: &str) -> Result<(usize, &TensorEntry)> {
        match self.tensors.get(name) {
            Some((si, e)) => Ok((*si, e)),
            None => Err(RvllmError::Loader {
                err: LoaderError::MissingTensor { name: name.to_string() },
                ctx: LoaderCtx {
                    path: self.model_dir.clone(),
                    tensor: Some(name.to_string()),
                },
                bt: std::backtrace::Backtrace::capture(),
            }),
        }
    }

    pub fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let mm = &self.mmaps[si];
        let start = e.file_offset as usize;
        &mm[start..start + e.nbytes as usize]
    }
}

/// Upload one Gemma 4 NVFP4 MLP linear. Reads `weight`,
/// `weight_scale`, `weight_scale_2`, `input_scale` from the pool,
/// validates byte-count vs `shape`, and stages each into the
/// arena. Does NOT run the CUTLASS SFB transform — the default
/// W4A16 GEMV path reads the natural [N, K/16] layout directly,
/// same as Mistral's recipe (RVLLM_W4A16_GEMV opt-in).
///
/// `base` is the layer-relative path prefix, e.g.
/// `"model.language_model.layers.0.mlp.gate_proj"`. The four
/// tensor names are derived by suffixing `.weight`, `.weight_scale`,
/// `.weight_scale_2`, `.input_scale`.
///
/// `weight_scale_2` semantics match LLMCompressor / compressed-tensors:
/// the on-disk value is the ENCODE scale; the GEMM epilogue's
/// `alpha_ptr` is the DECODE scale = `1 / weight_scale_2`. This
/// helper writes the decode form into the device scalar so callers
/// can pass it straight to CUTLASS without a host-side reciprocal.
pub fn upload_gemma4_nvfp4_linear(
    arena: &HbmArena<'_>,
    pool: &Gemma4Nvfp4ShardPool,
    base: &str,
    shape: Nvfp4LinearShape,
) -> Result<Gemma4Nvfp4LinearLoaded> {
    // (1) `.weight`  U8 [N, K/2]
    let packed_name = format!("{base}.weight");
    let (psi, pe) = pool.must_get(&packed_name)?;
    if pe.dtype != DType::U8 {
        return Err(corrupt(format!(
            "{packed_name}: expected U8, got {:?}",
            pe.dtype
        )));
    }
    if pe.shape != [shape.n, shape.packed_cols()] {
        return Err(corrupt(format!(
            "{packed_name}: expected shape {:?}, got {:?}",
            [shape.n, shape.packed_cols()], pe.shape
        )));
    }
    let packed_raw = pool.bytes_of(psi, pe);
    let packed_region = arena.region("gemma4_nvfp4_w_packed", packed_raw.len(), 16)?;
    unsafe { packed_region.copy_from_host(packed_raw)? };
    let packed_ptr = packed_region.device_ptr();
    let packed_bytes = packed_raw.len();

    // (2) `.weight_scale`  E4M3 [N, K/16]
    let scale_name = format!("{base}.weight_scale");
    let (ssi, se) = pool.must_get(&scale_name)?;
    if se.dtype != DType::Fp8E4M3 {
        return Err(corrupt(format!(
            "{scale_name}: expected Fp8E4M3, got {:?}", se.dtype
        )));
    }
    if se.shape != [shape.n, shape.scale_cols()] {
        return Err(corrupt(format!(
            "{scale_name}: expected shape {:?}, got {:?}",
            [shape.n, shape.scale_cols()], se.shape
        )));
    }
    let scale_raw = pool.bytes_of(ssi, se);
    let nat_region = arena.region("gemma4_nvfp4_w_sfb_natural", scale_raw.len(), 16)?;
    unsafe { nat_region.copy_from_host(scale_raw)? };
    let sfb_natural_ptr = nat_region.device_ptr();

    // (3) `.weight_scale_2`  F32 scalar  → device alpha = 1/ws2
    let ws2_name = format!("{base}.weight_scale_2");
    let (gsi, ge) = pool.must_get(&ws2_name)?;
    if ge.dtype != DType::F32 {
        return Err(corrupt(format!(
            "{ws2_name}: expected F32, got {:?}", ge.dtype
        )));
    }
    if !(ge.shape.is_empty() || ge.shape == [1]) {
        return Err(corrupt(format!(
            "{ws2_name}: expected scalar shape ([] or [1]), got {:?}", ge.shape
        )));
    }
    let ws2_raw = pool.bytes_of(gsi, ge);
    if ws2_raw.len() != 4 {
        return Err(corrupt(format!(
            "{ws2_name}: expected 4 bytes (F32 scalar), got {}", ws2_raw.len()
        )));
    }
    let ws2_f32 = f32::from_le_bytes([ws2_raw[0], ws2_raw[1], ws2_raw[2], ws2_raw[3]]);
    let alpha_f32 = if ws2_f32.is_finite() && ws2_f32 != 0.0 {
        1.0_f32 / ws2_f32
    } else {
        ws2_f32 // propagate NaN / 0 so the kernel-side guard trips
    };
    let gs_region = arena.region("gemma4_nvfp4_w_global_scale", 4, 4)?;
    unsafe { gs_region.copy_from_host(&alpha_f32.to_le_bytes())? };
    let global_scale_ptr = gs_region.device_ptr();

    // (4) `.input_scale`  F32 scalar  → device scalar verbatim
    //
    // Unlike `weight_scale_2`, this is the activation-side scale
    // used by the input quantization kernel. The kernel multiplies
    // input rows by `1 / input_scale` before NVFP4 packing — same
    // convention modelopt uses — so we forward the encode form
    // verbatim and let the kernel side do the reciprocal. (Storing
    // the reciprocal here would diverge from the obvious
    // checkpoint-to-device mapping and force every kernel call site
    // to remember which scale was pre-inverted.)
    let ins_name = format!("{base}.input_scale");
    let (isi, ie) = pool.must_get(&ins_name)?;
    if ie.dtype != DType::F32 {
        return Err(corrupt(format!(
            "{ins_name}: expected F32, got {:?}", ie.dtype
        )));
    }
    if !(ie.shape.is_empty() || ie.shape == [1]) {
        return Err(corrupt(format!(
            "{ins_name}: expected scalar shape ([] or [1]), got {:?}", ie.shape
        )));
    }
    let ins_raw = pool.bytes_of(isi, ie);
    if ins_raw.len() != 4 {
        return Err(corrupt(format!(
            "{ins_name}: expected 4 bytes (F32 scalar), got {}", ins_raw.len()
        )));
    }
    let is_region = arena.region("gemma4_nvfp4_w_input_scale", 4, 4)?;
    unsafe { is_region.copy_from_host(ins_raw)? };
    let input_scale_ptr = is_region.device_ptr();

    Ok(Gemma4Nvfp4LinearLoaded {
        shape,
        packed_ptr,
        sfb_natural_ptr,
        sfb_cutlass_ptr: 0,
        global_scale_ptr,
        input_scale_ptr,
        packed_bytes,
        sfb_bytes: 0,
        bf16_ptr: 0,
    })
}

/// Upload all three NVFP4 MLP linears for one layer. Convenience
/// wrapper around `upload_gemma4_nvfp4_linear`; returns
/// (gate, up, down) in that order, matching
/// `Gemma4Nvfp4LayerLoaded::nvfp4_linears` and the forward path's
/// natural execution order.
pub fn upload_gemma4_nvfp4_layer_mlp(
    arena: &HbmArena<'_>,
    pool: &Gemma4Nvfp4ShardPool,
    arch: &Gemma4Arch,
    layer_idx: usize,
) -> Result<(Gemma4Nvfp4LinearLoaded, Gemma4Nvfp4LinearLoaded, Gemma4Nvfp4LinearLoaded)> {
    let prefix = &arch.weight_prefix;
    let mut out: [Option<Gemma4Nvfp4LinearLoaded>; 3] = [None, None, None];
    for (i, kind) in Gemma4Nvfp4MlpKind::ALL.iter().enumerate() {
        let base = format!("{prefix}.layers.{layer_idx}.{}", kind.name());
        out[i] = Some(upload_gemma4_nvfp4_linear(arena, pool, &base, kind.shape_for(arch))?);
    }
    Ok((out[0].unwrap(), out[1].unwrap(), out[2].unwrap()))
}

fn corrupt(detail: String) -> RvllmError {
    RvllmError::Loader {
        err: LoaderError::Corrupt { detail },
        ctx: LoaderCtx {
            path: PathBuf::from("(gemma4 nvfp4 upload)"),
            tensor: None,
        },
        bt: std::backtrace::Backtrace::capture(),
    }
}
