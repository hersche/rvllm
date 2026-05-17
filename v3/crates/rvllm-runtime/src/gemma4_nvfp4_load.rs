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
use rvllm_loader::gemma4_arch::{Gemma4Arch, Gemma4LayerType};
use rvllm_loader::gemma4_nvfp4_weights::{
    validate_gemma4_nvfp4_inventory, Gemma4Nvfp4LayerLoaded, Gemma4Nvfp4LinearLoaded,
    Gemma4Nvfp4LoadedModel, Gemma4Nvfp4MlpKind, Gemma4Nvfp4OutsideText,
};
use rvllm_loader::mistral35_weights::Nvfp4LinearShape;
use rvllm_loader::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use rvllm_loader::weights::F16Weight;
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

/// Upload one bf16/f16 tensor verbatim, with dtype + shape
/// validation. Used for retained-bf16 attention projections,
/// layernorms, q_norm/k_norm, and layer_scalar. Mirrors
/// `mistral35_load::upload_typed_tensor`; localized here so the
/// Gemma path stays self-contained until Phase 5 cleanup
/// promotes both to a shared helper.
fn upload_typed_tensor(
    arena: &HbmArena<'_>,
    pool: &Gemma4Nvfp4ShardPool,
    region_name: &'static str,
    tensor_name: &str,
    expected_dtype: DType,
    expected_shape: Option<&[usize]>,
) -> Result<F16Weight> {
    let (si, e) = pool.must_get(tensor_name)?;
    let bytes_per_elem: usize = match expected_dtype {
        DType::Bf16 | DType::F16 => 2,
        DType::F32 => 4,
        DType::Fp8E4M3 | DType::U8 => 1,
        other => {
            return Err(corrupt(format!(
                "upload_typed_tensor: dtype {:?} not supported for {tensor_name}",
                other
            )))
        }
    };
    if e.dtype != expected_dtype {
        return Err(corrupt(format!(
            "{tensor_name}: dtype={:?} but loader expected {:?}",
            e.dtype, expected_dtype
        )));
    }
    if let Some(want) = expected_shape {
        if e.shape != want {
            return Err(corrupt(format!(
                "{tensor_name}: shape={:?} but loader expected {:?}",
                e.shape, want
            )));
        }
    }
    let elem_count: usize = e.shape.iter().product();
    let expect_bytes = elem_count * bytes_per_elem;
    let raw = pool.bytes_of(si, e);
    if raw.len() != expect_bytes {
        return Err(corrupt(format!(
            "{tensor_name}: mmap len={} but shape={:?} × {} = {} bytes",
            raw.len(),
            e.shape,
            bytes_per_elem,
            expect_bytes
        )));
    }
    let region = arena.region(region_name, raw.len(), 16)?;
    unsafe { region.copy_from_host(raw)? };
    Ok(F16Weight {
        offset_bytes: region.device_ptr(),
        shape: e.shape.clone(),
    })
}

/// Upload every weight tensor for one Gemma 4 layer: the four
/// norms, the per-head q_norm/k_norm gammas, the layer_scalar
/// residual multiplier, the bf16 attention projections (v_proj
/// optional for `attention_k_eq_v` global layers), and the three
/// NVFP4 MLP linears. Returns a populated `Gemma4Nvfp4LayerLoaded`.
pub fn upload_gemma4_nvfp4_layer(
    arena: &HbmArena<'_>,
    pool: &Gemma4Nvfp4ShardPool,
    arch: &Gemma4Arch,
    layer_idx: usize,
) -> Result<Gemma4Nvfp4LayerLoaded> {
    let prefix = &arch.weight_prefix;
    let h = arch.hidden_size;
    let base = format!("{prefix}.layers.{layer_idx}");

    let input_layernorm = upload_typed_tensor(
        arena, pool, "gemma4n_input_ln",
        &format!("{base}.input_layernorm.weight"),
        DType::Bf16, Some(&[h])
    )?;
    let post_attention_layernorm = upload_typed_tensor(
        arena, pool, "gemma4n_post_attn_ln",
        &format!("{base}.post_attention_layernorm.weight"),
        DType::Bf16, Some(&[h])
    )?;
    let pre_feedforward_layernorm = upload_typed_tensor(
        arena, pool, "gemma4n_pre_ff_ln",
        &format!("{base}.pre_feedforward_layernorm.weight"),
        DType::Bf16, Some(&[h])
    )?;
    let post_feedforward_layernorm = upload_typed_tensor(
        arena, pool, "gemma4n_post_ff_ln",
        &format!("{base}.post_feedforward_layernorm.weight"),
        DType::Bf16, Some(&[h])
    )?;
    let layer_scalar = upload_typed_tensor(
        arena, pool, "gemma4n_layer_scalar",
        &format!("{base}.layer_scalar"),
        DType::Bf16, Some(&[1])
    )?;
    // q_norm/k_norm shape is [head_dim]; head_dim differs by layer
    // type (sliding=256, global=512 on 31B). The bring-up's actual
    // forward path is the canonical gate — shape None lets either
    // pass through this load step.
    let q_norm = upload_typed_tensor(
        arena, pool, "gemma4n_q_norm",
        &format!("{base}.self_attn.q_norm.weight"),
        DType::Bf16, None
    )?;
    let k_norm = upload_typed_tensor(
        arena, pool, "gemma4n_k_norm",
        &format!("{base}.self_attn.k_norm.weight"),
        DType::Bf16, None
    )?;

    // Attention projections (bf16). Shape unconstrained here; the
    // bring-up validates against the arch-derived per-layer
    // expectation (sliding/global heads differ on 31B).
    let q_proj = upload_typed_tensor(
        arena, pool, "gemma4n_q_proj",
        &format!("{base}.self_attn.q_proj.weight"),
        DType::Bf16, None
    )?;
    let k_proj = upload_typed_tensor(
        arena, pool, "gemma4n_k_proj",
        &format!("{base}.self_attn.k_proj.weight"),
        DType::Bf16, None
    )?;
    // v_proj is absent on global layers when `attention_k_eq_v=true`;
    // checked by presence in the pool, not via arch flags (the arch
    // currently doesn't surface per-layer k_eq_v).
    let v_key = format!("{base}.self_attn.v_proj.weight");
    let v_proj = if pool.tensors.contains_key(&v_key) {
        Some(upload_typed_tensor(
            arena, pool, "gemma4n_v_proj", &v_key, DType::Bf16, None
        )?)
    } else {
        // Global layer with k_eq_v aliasing — runtime forward will
        // reuse k_proj. Sanity-check: only global layers may have
        // v_proj absent.
        match arch.layer_types.get(layer_idx) {
            Some(Gemma4LayerType::GlobalAttention) => None,
            _ => return Err(corrupt(format!(
                "v_proj absent on layer {layer_idx} but layer_type is not GlobalAttention"
            ))),
        }
    };
    let o_proj = upload_typed_tensor(
        arena, pool, "gemma4n_o_proj",
        &format!("{base}.self_attn.o_proj.weight"),
        DType::Bf16, None
    )?;

    // MLP NVFP4 trio.
    let (gate_proj, up_proj, down_proj) =
        upload_gemma4_nvfp4_layer_mlp(arena, pool, arch, layer_idx)?;

    Ok(Gemma4Nvfp4LayerLoaded {
        input_layernorm,
        post_attention_layernorm,
        pre_feedforward_layernorm,
        post_feedforward_layernorm,
        layer_scalar,
        q_norm,
        k_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
        gate_proj,
        up_proj,
        down_proj,
    })
}

/// Upload the outside-the-stack text weights (embed_tokens +
/// final_norm). Lm-head is tied to embed_tokens so no separate
/// upload is needed.
pub fn upload_gemma4_nvfp4_outside_text(
    arena: &HbmArena<'_>,
    pool: &Gemma4Nvfp4ShardPool,
    arch: &Gemma4Arch,
) -> Result<Gemma4Nvfp4OutsideText> {
    let prefix = &arch.weight_prefix;
    let embed_tokens = upload_typed_tensor(
        arena, pool, "gemma4n_embed_tokens",
        &format!("{prefix}.embed_tokens.weight"),
        DType::Bf16, Some(&[arch.vocab_size, arch.hidden_size])
    )?;
    let final_norm = upload_typed_tensor(
        arena, pool, "gemma4n_final_norm",
        &format!("{prefix}.norm.weight"),
        DType::Bf16, Some(&[arch.hidden_size])
    )?;
    Ok(Gemma4Nvfp4OutsideText { embed_tokens, final_norm })
}

/// Top-level loader: reads the model directory, validates the
/// NVFP4 inventory, opens the shard pool, and uploads every
/// text-side weight (outside + all 60 layers). Returns a
/// fully-populated `Gemma4Nvfp4LoadedModel`.
///
/// Vision tower upload is deferred to Phase 3c — the text
/// forward path can be validated first and vision splice
/// integrated after.
pub fn load_gemma4_nvfp4_text(
    arena: &HbmArena<'_>,
    model_dir: &Path,
    arch: &Gemma4Arch,
) -> Result<Gemma4Nvfp4LoadedModel> {
    let pool = Gemma4Nvfp4ShardPool::open(model_dir)?;

    // Inventory pass — cheap; refuses checkpoints with quantized
    // attention or missing MLP tensors before we touch the GPU.
    // Pool stores (shard_idx, entry) tuples; the validator wants
    // a plain entry map — strip the shard index.
    let entry_map: BTreeMap<String, TensorEntry> = pool
        .tensors
        .iter()
        .map(|(k, (_si, e))| (k.clone(), e.clone()))
        .collect();
    let inv = validate_gemma4_nvfp4_inventory(arch, &arch.weight_prefix, &entry_map)?;
    eprintln!(
        "[gemma4-nvfp4-load] inventory: layers={} mlp_nvfp4={} attn_bf16={}",
        inv.num_layers,
        inv.counts.weight,
        inv.counts.attn_bf16,
    );

    let outside = upload_gemma4_nvfp4_outside_text(arena, &pool, arch)?;
    eprintln!(
        "[gemma4-nvfp4-load] outside uploaded (embed_tokens + final_norm)"
    );

    let mut layers = Vec::with_capacity(arch.num_hidden_layers);
    for li in 0..arch.num_hidden_layers {
        layers.push(upload_gemma4_nvfp4_layer(arena, &pool, arch, li)?);
        if li == 0 || li == arch.num_hidden_layers - 1 || (li + 1) % 10 == 0 {
            eprintln!(
                "[gemma4-nvfp4-load] layer {}/{} uploaded",
                li + 1, arch.num_hidden_layers,
            );
        }
    }

    Ok(Gemma4Nvfp4LoadedModel { outside, layers })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// GPU smoke: opens the on-disk NVFP4 checkpoint and uploads
    /// just layer 0 (~330 MB on device, small enough to coexist
    /// with a running rvllm-serve on GB10's unified memory). Proves
    /// the Phase 2 + 3a upload chain works end-to-end on hardware
    /// without disturbing production.
    ///
    /// Run:  GEMMA4_NVFP4_DIR=/home/r00t/Gemma-4-31B-IT-NVFP4 \
    ///       cargo test -p rvllm-runtime --features cuda \
    ///         gemma4_nvfp4_load::tests::ondisk_layer0_upload \
    ///         -- --ignored --nocapture
    #[test]
    #[ignore]
    fn ondisk_layer0_upload_smoke() {
        let dir = match std::env::var("GEMMA4_NVFP4_DIR") {
            Ok(v) => PathBuf::from(v),
            Err(_) => {
                eprintln!("GEMMA4_NVFP4_DIR unset — skipping GPU smoke");
                return;
            }
        };
        let arch = rvllm_loader::gemma4_arch::Gemma4Arch::from_dir(&dir)
            .expect("Gemma4Arch::from_dir");
        eprintln!(
            "[gpu-smoke] arch: layers={} hidden={} intermediate={} prefix={}",
            arch.num_hidden_layers, arch.hidden_size, arch.intermediate_size,
            arch.weight_prefix,
        );

        // 512 MiB arena — fits layer 0 with headroom; leaves rvllm-
        // serve's working set undisturbed on GB10 (unified memory,
        // ~70 GB free with prod active).
        let ctx = rvllm_mem::context::CudaContextHandle::init(0)
            .expect("CudaContextHandle::init");
        let arena = rvllm_mem::HbmArena::new(&ctx, 512 * 1024 * 1024)
            .expect("HbmArena::new");

        let pool = Gemma4Nvfp4ShardPool::open(&dir).expect("ShardPool::open");
        eprintln!("[gpu-smoke] pool open: {} tensors, {} shards",
                  pool.tensors.len(), pool.mmaps.len());

        let layer0 = upload_gemma4_nvfp4_layer(&arena, &pool, &arch, 0)
            .expect("upload_gemma4_nvfp4_layer(0)");
        eprintln!(
            "[gpu-smoke] layer 0 uploaded: \n  \
             input_ln.shape={:?}\n  \
             q_proj.shape={:?}\n  \
             k_proj.shape={:?}\n  \
             v_proj.shape={:?}\n  \
             gate_proj.shape=N={} K={}\n  \
             gate_proj.packed_ptr=0x{:x} sfb_natural_ptr=0x{:x}\n  \
             gate_proj.global_scale_ptr=0x{:x} input_scale_ptr=0x{:x}\n  \
             down_proj.packed_bytes={}",
            layer0.input_layernorm.shape,
            layer0.q_proj.shape,
            layer0.k_proj.shape,
            layer0.v_proj.as_ref().map(|w| &w.shape),
            layer0.gate_proj.shape.n, layer0.gate_proj.shape.k,
            layer0.gate_proj.packed_ptr, layer0.gate_proj.sfb_natural_ptr,
            layer0.gate_proj.global_scale_ptr, layer0.gate_proj.input_scale_ptr,
            layer0.down_proj.packed_bytes,
        );

        // Sanity: every device pointer must be non-zero (HbmArena
        // never returns zero on a successful allocation), and
        // packed/scale byte counts must match the on-disk values.
        for (name, lin) in [
            ("gate", &layer0.gate_proj),
            ("up",   &layer0.up_proj),
            ("down", &layer0.down_proj),
        ] {
            assert!(lin.packed_ptr != 0, "{name}: packed_ptr is null");
            assert!(lin.sfb_natural_ptr != 0, "{name}: sfb_natural_ptr is null");
            assert!(lin.global_scale_ptr != 0, "{name}: global_scale_ptr is null");
            assert!(lin.input_scale_ptr != 0, "{name}: input_scale_ptr is null");
            assert_eq!(lin.packed_bytes, lin.shape.packed_bytes(),
                       "{name}: packed_bytes != shape.packed_bytes()");
        }
        assert!(layer0.q_proj.offset_bytes != 0, "q_proj.offset_bytes is null");
        assert!(layer0.k_proj.offset_bytes != 0, "k_proj.offset_bytes is null");
        assert!(layer0.o_proj.offset_bytes != 0, "o_proj.offset_bytes is null");
        eprintln!("[gpu-smoke] all pointers non-null + byte counts match");
    }
}
