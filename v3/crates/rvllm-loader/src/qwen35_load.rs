//! Qwen 3.5 27B dense weight loader (Phase 1a — outside tensors).
//!
//! The on-disk format is identical to Qwen 3.6 (`model.language_model.*`
//! prefix, BF16 outside tensors, FP8 e4m3 + BF16 block-128
//! `weight_scale_inv` per-projection). The only divergence for the
//! dense Qwen 3.5 is the MLP block (`mlp.{gate,up,down}_proj.weight`
//! replacing the 256-expert MoE), which lands in Phase 1b.
//!
//! Phase 1a (this commit): `load_qwen35_outside` uploads
//!   * `model.language_model.embed_tokens.weight`         BF16 → F16
//!   * `model.language_model.norm.weight`                 BF16 → F16
//!     (+1.0 bias for GemmaRMSNorm kernel convention)
//!   * `lm_head.weight`                                   BF16 → F16
//!   * `lm_head.weight` again, CPU-quantized to FP8 per-tensor scale
//!     for the existing `fp8_gemv` lm-head path.
//!
//! Phase 1b: `load_qwen35_model` will add the 64 per-layer entries
//! (16 full-attn + 48 linear-attn + 64 dense MLPs) and optional
//! vision tower (Qwen3-VL ViT, shape-identical to Qwen 3.6's).

use std::collections::BTreeMap;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use rvllm_core::{DType, LoaderCtx, LoaderError, Result, RvllmError};
use rvllm_mem::HbmArena;

use crate::fp8_quant::{check_clamp_gate, quantize_per_tensor_ref, FP8_E4M3_MAX};
use crate::qwen35_weights::{
    Qwen35DenseMlpBlock, Qwen35FullAttnLayer, Qwen35Layer, Qwen35LayerAttn,
    Qwen35LinearAttnLayer, Qwen35LoadedModel, Qwen35LoadedOutside,
};
use crate::safetensors::{ShardHeader, ShardIndex, TensorEntry};
use crate::weights::{F16Weight, Fp8Weight};

/// Same `model.language_model.*` prefix as Qwen 3.6.
const QWEN35_PREFIX: &str = "model.language_model";

struct ShardMap {
    _mmap: Mmap,
    header: ShardHeader,
}

impl ShardMap {
    fn open(path: &Path) -> Result<Self> {
        let f = std::fs::File::open(path).map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|source| RvllmError::Io {
            err: rvllm_core::IoError::from(&source),
            path: path.to_path_buf(),
            source,
        })?;
        let header = ShardHeader::parse(path, &mmap)?;
        Ok(Self { _mmap: mmap, header })
    }
    fn bytes(&self) -> &[u8] { &self._mmap }
}

/// Per-load context: opened shards + tensor index. Mirrors the
/// private `LoadCtx` in `qwen36_load.rs`; kept duplicate here so
/// per-family loaders stay independent during the multi-week
/// bring-up. A future refactor can consolidate the two LoadCtx into
/// a shared helper without touching this file's call sites.
struct LoadCtx<'a> {
    shards: Vec<ShardMap>,
    tensors: BTreeMap<String, (usize, TensorEntry)>,
    model_dir: &'a Path,
    arena: &'a HbmArena<'a>,
}

impl<'a> LoadCtx<'a> {
    fn new(model_dir: &'a Path, arena: &'a HbmArena<'a>) -> Result<Self> {
        let idx = ShardIndex::resolve(model_dir)?;
        let mut shards = Vec::with_capacity(idx.shards.len());
        for p in &idx.shards {
            shards.push(ShardMap::open(p)?);
        }
        let mut tensors: BTreeMap<String, (usize, TensorEntry)> = BTreeMap::new();
        for (si, sm) in shards.iter().enumerate() {
            for (name, entry) in &sm.header.tensors {
                tensors.insert(name.clone(), (si, entry.clone()));
            }
        }
        Ok(Self { shards, tensors, model_dir, arena })
    }

    fn bytes_of(&self, si: usize, e: &TensorEntry) -> &[u8] {
        let s = self.shards[si].bytes();
        let start = e.file_offset as usize;
        &s[start..start + e.nbytes as usize]
    }

    fn must_get(&self, name: &str) -> Result<(usize, TensorEntry)> {
        self.tensors.get(name).cloned().ok_or_else(|| RvllmError::Loader {
            err: LoaderError::MissingTensor { name: name.to_string() },
            ctx: LoaderCtx { path: self.model_dir.to_path_buf(), tensor: Some(name.to_string()) },
            bt: std::backtrace::Backtrace::capture(),
        })
    }

    fn upload_f16(&self, region_name: &'static str, hf_name: &str) -> Result<(F16Weight, u64)> {
        self.upload_f16_with_bias(region_name, hf_name, 0.0)
    }

    fn upload_f16_with_bias(
        &self,
        region_name: &'static str,
        hf_name: &str,
        bias: f32,
    ) -> Result<(F16Weight, u64)> {
        let (si, e) = self.must_get(hf_name)?;
        let mut buf = tensor_to_f16_bytes(&e, self.bytes_of(si, &e), self.model_dir)?;
        if bias != 0.0 {
            let n = buf.len() / 2;
            for i in 0..n {
                let lo = buf[i * 2];
                let hi = buf[i * 2 + 1];
                let bits = u16::from_le_bytes([lo, hi]);
                let v = f16::from_bits(bits).to_f32() + bias;
                let new_bits = f16::from_f32(v).to_bits();
                let nb = new_bits.to_le_bytes();
                buf[i * 2] = nb[0];
                buf[i * 2 + 1] = nb[1];
            }
        }
        let region = self.arena.region(region_name, buf.len(), 16)?;
        unsafe { region.copy_from_host(&buf)? };
        Ok((F16Weight { offset_bytes: region.device_ptr(), shape: e.shape.clone() }, buf.len() as u64))
    }

    /// FP8 e4m3 + BF16 block-128 weight_scale_inv companion. Identical
    /// algorithm to qwen36_load::upload_fp8_blockwise.
    #[allow(dead_code)] // used in Phase 1b
    fn upload_fp8_blockwise(
        &self,
        region_name: &'static str,
        weight_name: &str,
    ) -> Result<Fp8Weight> {
        let (wsi, we) = self.must_get(weight_name)?;
        if we.dtype != DType::Fp8E4M3 {
            return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: we.name.clone(),
                    expected: DType::Fp8E4M3,
                    got: we.dtype,
                },
                ctx: LoaderCtx { path: self.model_dir.to_path_buf(), tensor: Some(we.name.clone()) },
                bt: std::backtrace::Backtrace::capture(),
            });
        }
        let raw = self.bytes_of(wsi, &we);
        let region = self.arena.region(region_name, raw.len(), 16)?;
        unsafe { region.copy_from_host(raw)? };
        let scale_name = format!("{weight_name}_scale_inv");
        let (ssi, se) = self.must_get(&scale_name)?;
        let scale_bytes = match se.dtype {
            DType::Bf16 => bf16_bytes_to_f32_bytes(self.bytes_of(ssi, &se)),
            DType::F32 => self.bytes_of(ssi, &se).to_vec(),
            _ => return Err(RvllmError::Loader {
                err: LoaderError::DtypeMismatch {
                    tensor: se.name.clone(),
                    expected: DType::F32,
                    got: se.dtype,
                },
                ctx: LoaderCtx { path: self.model_dir.to_path_buf(), tensor: Some(se.name.clone()) },
                bt: std::backtrace::Backtrace::capture(),
            }),
        };
        let bs_region = self.arena.region("qwen35_fp8_blockscale", scale_bytes.len(), 16)?;
        unsafe { bs_region.copy_from_host(&scale_bytes)? };
        let one = 1.0f32;
        let one_r = self.arena.region("qwen35_fp8_scale", 4, 4)?;
        unsafe { one_r.copy_from_host(&one.to_le_bytes())? };
        let weight_n = we.shape[0];
        let weight_k = if we.shape.len() >= 2 { we.shape[1] } else { 0 };
        let n_blocks = (weight_n + 127) / 128;
        let k_blocks = (weight_k + 127) / 128;
        Ok(Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: one_r.device_ptr(),
            shape: we.shape.clone(),
            scale: 1.0,
            clamp_ppm: 0.0,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: Some(bs_region.device_ptr()),
            blockscale_n_blocks: n_blocks as u32,
            blockscale_k_blocks: k_blocks as u32,
        })
    }

    fn upload_lm_head_fp8(
        &self,
        f16_weight: &F16Weight,
        f16_bytes_len: u64,
        tensor_name: &str,
    ) -> Result<(Fp8Weight, u64)> {
        let (si, e) = self.must_get(tensor_name)?;
        let f16_bytes = tensor_to_f16_bytes(&e, self.bytes_of(si, &e), self.model_dir)?;
        debug_assert_eq!(f16_bytes.len() as u64, f16_bytes_len);
        use rayon::prelude::*;
        let f32_vals: Vec<f32> = f16_bytes.par_chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();
        let q = quantize_per_tensor_ref(&f32_vals);
        check_clamp_gate(tensor_name, q.clamp_ppm, self.model_dir)?;
        let fp8: Vec<u8> = f32_vals.par_iter()
            .map(|v| fp8_e4m3_encode((*v / q.scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)))
            .collect();
        let region = self.arena.region("qwen35_lm_head_fp8", fp8.len(), 16)?;
        unsafe { region.copy_from_host(&fp8)? };
        let scale_region = self.arena.region("qwen35_lm_head_fp8_scale", 4, 4)?;
        unsafe { scale_region.copy_from_host(&q.scale.to_le_bytes())? };
        Ok((Fp8Weight {
            offset_bytes: region.device_ptr(),
            scale_ptr: scale_region.device_ptr(),
            shape: f16_weight.shape.clone(),
            scale: q.scale,
            clamp_ppm: q.clamp_ppm,
            dtype: DType::Fp8E4M3,
            channelscale_ptr: None,
            blockscale_ptr: None,
            blockscale_n_blocks: 0,
            blockscale_k_blocks: 0,
        }, fp8.len() as u64))
    }
}

/// Phase 1a entry point: outside-the-stack tensors only.
pub fn load_qwen35_outside(
    model_dir: &Path,
    arena: &HbmArena,
) -> Result<Qwen35LoadedOutside> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    load_outside_via_ctx(&ctx)
}

/// Phase 1b entry point: outside tensors + every per-layer block
/// (linear-attn or full-attn) + dense MLP block. Vision tower is
/// optional and not loaded here yet (Phase 3 covers the Qwen3-VL
/// ViT upload).
pub fn load_qwen35_model(
    model_dir: &Path,
    arena: &HbmArena,
    layer_types: &[crate::load::LayerAttnType],
) -> Result<Qwen35LoadedModel> {
    let ctx = LoadCtx::new(model_dir, arena)?;
    let outside = load_outside_via_ctx(&ctx)?;

    let mut layers: Vec<Qwen35Layer> = Vec::with_capacity(layer_types.len());
    let mut n_full = 0usize;
    let mut n_linear = 0usize;
    let load_started = std::time::Instant::now();

    for (l, ty) in layer_types.iter().enumerate() {
        let attn = match ty {
            crate::load::LayerAttnType::Full => {
                n_full += 1;
                Qwen35LayerAttn::Full(load_full_attn_layer(&ctx, l)?)
            }
            crate::load::LayerAttnType::Linear
            | crate::load::LayerAttnType::SlidingAttention => {
                n_linear += 1;
                Qwen35LayerAttn::Linear(load_linear_attn_layer(&ctx, l)?)
            }
        };
        let mlp = load_dense_mlp_block(&ctx, l)?;
        if l == 0 || l == layer_types.len() - 1 || (l + 1) % 16 == 0 {
            eprintln!(
                "[qwen35-loader] layer {l}/{total} loaded ({attn_kind}) \
                 [{elapsed:.1}s elapsed, arena.used={:.2} GiB]",
                arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
                total = layer_types.len(),
                attn_kind = match ty {
                    crate::load::LayerAttnType::Full => "full",
                    crate::load::LayerAttnType::Linear => "linear",
                    crate::load::LayerAttnType::SlidingAttention => "sliding",
                },
                elapsed = load_started.elapsed().as_secs_f64(),
            );
        }
        layers.push(Qwen35Layer { attn, mlp });
    }

    eprintln!(
        "[qwen35-loader] per-layer upload complete: \
         {n_full} full-attn + {n_linear} linear-attn layers, \
         dense MLP on every layer. \
         Total wall: {:.1}s, arena.used={:.2} GiB.",
        load_started.elapsed().as_secs_f64(),
        arena.used() as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    let vision = load_qwen35_vision(&ctx).ok();
    if vision.is_some() {
        eprintln!("[qwen35-loader] vision tower loaded \
                   (27 ViT blocks + PatchMerger)");
    } else {
        eprintln!("[qwen35-loader] vision tower SKIPPED \
                   (no model.visual.* tensors or load failed)");
    }
    Ok(Qwen35LoadedModel { outside, layers, vision })
}

/// Qwen 3.5 vision tower load — byte-identical layout to Qwen 3.6
/// (`model.visual.*` prefix, 27 ViT blocks, patch_embed, pos_embed,
/// PatchMerger). Reuses `Qwen36Vision` via the existing struct in
/// `qwen36_weights`. The only difference vs `qwen36_load`'s sibling
/// is the arena region-name prefix (`qwen35_vis_*`) so a future
/// Codex audit can identify which family allocated each region.
fn load_qwen35_vision(
    ctx: &LoadCtx,
) -> Result<crate::qwen36_weights::Qwen36Vision> {
    use crate::qwen36_weights::{
        Qwen36PatchMerger, Qwen36Vision, Qwen36VisionBlock,
        Qwen36VisionPatchEmbed,
    };
    let started = std::time::Instant::now();
    const QWEN35_VISION_PREFIX: &str = "model.visual";

    let pe_w_name = format!("{QWEN35_VISION_PREFIX}.patch_embed.proj.weight");
    let pe_b_name = format!("{QWEN35_VISION_PREFIX}.patch_embed.proj.bias");
    let _ = ctx.must_get(&pe_w_name)?;
    let _ = ctx.must_get(&pe_b_name)?;
    let (proj_weight, _) = ctx.upload_f16("qwen35_vis_pe_w", &pe_w_name)?;
    let (proj_bias, _) = ctx.upload_f16("qwen35_vis_pe_b", &pe_b_name)?;
    let patch_embed = Qwen36VisionPatchEmbed { proj_weight, proj_bias };

    let pos_embed_name = format!("{QWEN35_VISION_PREFIX}.pos_embed.weight");
    let (pos_embed, _) = ctx.upload_f16("qwen35_vis_pos_embed", &pos_embed_name)?;

    let mut blocks = Vec::with_capacity(27);
    for i in 0..27 {
        let p = |s: &str| format!("{QWEN35_VISION_PREFIX}.blocks.{i}.{s}");
        let (norm1_w, _) = ctx.upload_f16("qwen35_vis_n1w", &p("norm1.weight"))?;
        let (norm1_b, _) = ctx.upload_f16("qwen35_vis_n1b", &p("norm1.bias"))?;
        let (qkv_w, _) = ctx.upload_f16("qwen35_vis_qkv_w", &p("attn.qkv.weight"))?;
        let (qkv_b, _) = ctx.upload_f16("qwen35_vis_qkv_b", &p("attn.qkv.bias"))?;
        let (proj_w, _) = ctx.upload_f16("qwen35_vis_o_w", &p("attn.proj.weight"))?;
        let (proj_b, _) = ctx.upload_f16("qwen35_vis_o_b", &p("attn.proj.bias"))?;
        let (norm2_w, _) = ctx.upload_f16("qwen35_vis_n2w", &p("norm2.weight"))?;
        let (norm2_b, _) = ctx.upload_f16("qwen35_vis_n2b", &p("norm2.bias"))?;
        let (fc1_w, _) = ctx.upload_f16("qwen35_vis_fc1w", &p("mlp.linear_fc1.weight"))?;
        let (fc1_b, _) = ctx.upload_f16("qwen35_vis_fc1b", &p("mlp.linear_fc1.bias"))?;
        let (fc2_w, _) = ctx.upload_f16("qwen35_vis_fc2w", &p("mlp.linear_fc2.weight"))?;
        let (fc2_b, _) = ctx.upload_f16("qwen35_vis_fc2b", &p("mlp.linear_fc2.bias"))?;
        blocks.push(Qwen36VisionBlock {
            norm1_w, norm1_b, qkv_w, qkv_b, proj_w, proj_b,
            norm2_w, norm2_b, fc1_w, fc1_b, fc2_w, fc2_b,
        });
    }

    let mp = |s: &str| format!("{QWEN35_VISION_PREFIX}.merger.{s}");
    let (norm_w, _) = ctx.upload_f16("qwen35_vis_mg_nw", &mp("norm.weight"))?;
    let (norm_b, _) = ctx.upload_f16("qwen35_vis_mg_nb", &mp("norm.bias"))?;
    let (mfc1_w, _) = ctx.upload_f16("qwen35_vis_mg_fc1w", &mp("linear_fc1.weight"))?;
    let (mfc1_b, _) = ctx.upload_f16("qwen35_vis_mg_fc1b", &mp("linear_fc1.bias"))?;
    let (mfc2_w, _) = ctx.upload_f16("qwen35_vis_mg_fc2w", &mp("linear_fc2.weight"))?;
    let (mfc2_b, _) = ctx.upload_f16("qwen35_vis_mg_fc2b", &mp("linear_fc2.bias"))?;
    let merger = Qwen36PatchMerger {
        norm_w, norm_b,
        fc1_w: mfc1_w, fc1_b: mfc1_b,
        fc2_w: mfc2_w, fc2_b: mfc2_b,
    };

    eprintln!(
        "[qwen35-loader] vision: 27 blocks + patch_embed + merger uploaded \
         in {:.1}s",
        started.elapsed().as_secs_f64(),
    );
    Ok(Qwen36Vision {
        patch_embed,
        pos_embed,
        blocks,
        merger,
    })
}

fn load_full_attn_layer(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen35FullAttnLayer> {
    let ln = |s: &str| format!("{QWEN35_PREFIX}.layers.{layer_idx}.{s}");

    // All four are GemmaRMSNorm-style; gamma centred at 0, add +1.
    let (input_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen35_input_ln", &ln("input_layernorm.weight"), 1.0)?;
    let (post_attention_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen35_post_attn_ln", &ln("post_attention_layernorm.weight"), 1.0)?;
    let (q_norm, _) = ctx.upload_f16_with_bias(
        "qwen35_q_norm", &ln("self_attn.q_norm.weight"), 1.0)?;
    let (k_norm, _) = ctx.upload_f16_with_bias(
        "qwen35_k_norm", &ln("self_attn.k_norm.weight"), 1.0)?;

    let q_proj = ctx.upload_fp8_blockwise("qwen35_q_proj", &ln("self_attn.q_proj.weight"))?;
    let k_proj = ctx.upload_fp8_blockwise("qwen35_k_proj", &ln("self_attn.k_proj.weight"))?;
    let v_proj = ctx.upload_fp8_blockwise("qwen35_v_proj", &ln("self_attn.v_proj.weight"))?;
    let o_proj = ctx.upload_fp8_blockwise("qwen35_o_proj", &ln("self_attn.o_proj.weight"))?;

    if layer_idx == 3 {
        eprintln!(
            "[qwen35-loader] full-attn layer {layer_idx}: q={:?} k={:?} v={:?} o={:?}",
            q_proj.shape, k_proj.shape, v_proj.shape, o_proj.shape,
        );
    }

    Ok(Qwen35FullAttnLayer {
        input_layernorm,
        post_attention_layernorm,
        q_norm,
        k_norm,
        q_proj,
        k_proj,
        v_proj,
        o_proj,
    })
}

fn load_linear_attn_layer(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen35LinearAttnLayer> {
    let ln = |s: &str| format!("{QWEN35_PREFIX}.layers.{layer_idx}.{s}");

    let (input_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen35_input_ln", &ln("input_layernorm.weight"), 1.0)?;
    let (post_attention_layernorm, _) = ctx.upload_f16_with_bias(
        "qwen35_post_attn_ln", &ln("post_attention_layernorm.weight"), 1.0)?;
    let (a_log, _) = ctx.upload_f16("qwen35_a_log", &ln("linear_attn.A_log"))?;
    let (dt_bias, _) = ctx.upload_f16("qwen35_dt_bias", &ln("linear_attn.dt_bias"))?;
    let (conv1d, _) = ctx.upload_f16("qwen35_conv1d", &ln("linear_attn.conv1d.weight"))?;
    let (in_proj_a, _) = ctx.upload_f16("qwen35_in_proj_a", &ln("linear_attn.in_proj_a.weight"))?;
    let (in_proj_b, _) = ctx.upload_f16("qwen35_in_proj_b", &ln("linear_attn.in_proj_b.weight"))?;
    // RMSNormGated: gamma centred at 1 already, no bias.
    let (norm, _) = ctx.upload_f16("qwen35_la_norm", &ln("linear_attn.norm.weight"))?;

    let in_proj_qkv =
        ctx.upload_fp8_blockwise("qwen35_la_in_qkv", &ln("linear_attn.in_proj_qkv.weight"))?;
    let in_proj_z =
        ctx.upload_fp8_blockwise("qwen35_la_in_z", &ln("linear_attn.in_proj_z.weight"))?;
    let out_proj =
        ctx.upload_fp8_blockwise("qwen35_la_out", &ln("linear_attn.out_proj.weight"))?;

    if layer_idx == 0 {
        eprintln!(
            "[qwen35-loader] linear-attn layer {layer_idx}: \
             qkv={:?} z={:?} out={:?} a_log={:?} conv1d={:?}",
            in_proj_qkv.shape, in_proj_z.shape, out_proj.shape,
            a_log.shape, conv1d.shape,
        );
    }

    Ok(Qwen35LinearAttnLayer {
        input_layernorm,
        post_attention_layernorm,
        a_log,
        dt_bias,
        conv1d,
        in_proj_a,
        in_proj_b,
        in_proj_qkv,
        in_proj_z,
        norm,
        out_proj,
    })
}

fn load_dense_mlp_block(ctx: &LoadCtx, layer_idx: usize) -> Result<Qwen35DenseMlpBlock> {
    let ln = |s: &str| format!("{QWEN35_PREFIX}.layers.{layer_idx}.mlp.{s}");
    let gate_proj = ctx.upload_fp8_blockwise("qwen35_gate_proj", &ln("gate_proj.weight"))?;
    let up_proj = ctx.upload_fp8_blockwise("qwen35_up_proj", &ln("up_proj.weight"))?;
    let down_proj = ctx.upload_fp8_blockwise("qwen35_down_proj", &ln("down_proj.weight"))?;
    if layer_idx == 0 {
        eprintln!(
            "[qwen35-loader] dense MLP layer {layer_idx}: \
             gate={:?} up={:?} down={:?}",
            gate_proj.shape, up_proj.shape, down_proj.shape,
        );
    }
    Ok(Qwen35DenseMlpBlock { gate_proj, up_proj, down_proj })
}

fn load_outside_via_ctx(ctx: &LoadCtx) -> Result<Qwen35LoadedOutside> {
    let embed_name = format!("{QWEN35_PREFIX}.embed_tokens.weight");
    let norm_name = format!("{QWEN35_PREFIX}.norm.weight");
    let lm_head_name = "lm_head.weight";

    let (embed_tokens, embed_tokens_bytes) =
        ctx.upload_f16("qwen35_embedding", &embed_name)?;
    // GemmaRMSNorm convention: kernel expects gamma centred at 1, the
    // checkpoint stores it centred at 0 — add +1.
    let (final_norm, final_norm_bytes) =
        ctx.upload_f16_with_bias("qwen35_final_norm", &norm_name, 1.0)?;
    let (lm_head, lm_head_bytes) = ctx.upload_f16("qwen35_lm_head", lm_head_name)?;

    let (lm_head_fp8, lm_head_fp8_bytes) =
        ctx.upload_lm_head_fp8(&lm_head, lm_head_bytes, lm_head_name)?;

    eprintln!(
        "[qwen35-loader] outside tensors uploaded: \
         embed_tokens {:?} ({:.1} MiB), \
         final_norm {:?} ({:.2} KiB), \
         lm_head_f16 {:?} ({:.1} MiB), \
         lm_head_fp8 {:?} ({:.1} MiB, scale={:.6e}, clamp_ppm={:.3})",
        embed_tokens.shape,
        embed_tokens_bytes as f64 / (1024.0 * 1024.0),
        final_norm.shape,
        final_norm_bytes as f64 / 1024.0,
        lm_head.shape,
        lm_head_bytes as f64 / (1024.0 * 1024.0),
        lm_head_fp8.shape,
        lm_head_fp8_bytes as f64 / (1024.0 * 1024.0),
        lm_head_fp8.scale,
        lm_head_fp8.clamp_ppm,
    );

    Ok(Qwen35LoadedOutside {
        embed_tokens,
        final_norm,
        lm_head,
        lm_head_fp8,
        embed_tokens_bytes,
        final_norm_bytes,
        lm_head_bytes,
        lm_head_fp8_bytes,
    })
}

// ── Helpers (mirror qwen36_load private helpers) ────────────────

fn tensor_to_f16_bytes(e: &TensorEntry, raw: &[u8], model_dir: &Path) -> Result<Vec<u8>> {
    match e.dtype {
        DType::F16 => Ok(raw.to_vec()),
        DType::Bf16 => Ok(bf16_bytes_to_f16_bytes(raw)),
        DType::F32 => Ok(f32_bytes_to_f16_bytes(raw)),
        _ => Err(RvllmError::Loader {
            err: LoaderError::DtypeMismatch {
                tensor: e.name.clone(),
                expected: DType::F16,
                got: e.dtype,
            },
            ctx: LoaderCtx { path: model_dir.to_path_buf(), tensor: Some(e.name.clone()) },
            bt: std::backtrace::Backtrace::capture(),
        }),
    }
}

fn bf16_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&f16::from_f32(as_f32).to_le_bytes());
    }
    out
}

fn bf16_bytes_to_f32_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 2;
    let mut out = Vec::with_capacity(n * 4);
    for i in 0..n {
        let lo = raw[2 * i];
        let hi = raw[2 * i + 1];
        let as_f32 = f32::from_bits(u32::from_le_bytes([0, 0, lo, hi]));
        out.extend_from_slice(&as_f32.to_le_bytes());
    }
    out
}

fn f32_bytes_to_f16_bytes(raw: &[u8]) -> Vec<u8> {
    let n = raw.len() / 4;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = f32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn fp8_e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let s: u8 = if v.to_bits() >> 31 != 0 { 0x80 } else { 0 };
    let a = v.abs();
    if a == 0.0 { return 0; }
    if a > FP8_E4M3_MAX { return s | 0x7e; }
    let bits = a.to_bits();
    let exp32 = ((bits >> 23) & 0xff) as i32 - 127;
    let mant32 = bits & 0x7f_ffff;
    let mut exp8 = exp32 + 7;
    if exp8 <= 0 {
        let shift = 1 - exp8;
        let full = mant32 | (1 << 23);
        let rshift = (20 + shift) as u32;
        let mut m = full >> rshift;
        let round_bit = if rshift > 0 { (full >> (rshift - 1)) & 1 } else { 0 };
        let sticky = if rshift > 1 {
            (full & ((1 << (rshift - 1)) - 1) != 0) as u32
        } else { 0 };
        m += round_bit & (sticky | (m & 1));
        if m >= 8 { return s | 0x08; }
        return s | (m as u8 & 0x07);
    }
    let trunc = mant32 >> 20;
    let round_bit = (mant32 >> 19) & 1;
    let sticky = (mant32 & 0x7_ffff) != 0;
    let m = trunc + (round_bit & (sticky as u32 | (trunc & 1)));
    if m >= 8 {
        exp8 += 1;
        if exp8 > 15 { return s | 0x7e; }
        return s | ((exp8 as u8 & 0x0f) << 3);
    }
    if exp8 > 15 { return s | 0x7e; }
    s | ((exp8 as u8 & 0x0f) << 3) | (m as u8 & 0x07)
}
