#!/usr/bin/env python3
"""HF Gemma 4 E4B audio tower reference-state dump.

Runs the HF audio_tower on a deterministic test input (1-second
440Hz sine wave resampled to 16 kHz mono) and writes f16 buffers
at each stage so the rvllm Rust forward can be validated layer-by-
layer with row-cosine diffs (same methodology as the PLE / Gemma
vision audits).

Output dir (`--dump-dir`, default `/tmp/e4b_audio_dump/`):

  audio_input_mel.bin              [T_mel, 128]  f16
    (computed by HF Gemma4AudioProcessor on the test waveform)
  audio_after_subsample.bin        [N, 1024]  f16
    (output of audio_tower.subsample_conv_projection)

For each layer L in --dump-layers (default 0,1,11):
  audio_layer{L}_input.bin         [N, 1024]  f16  (block input)
  audio_layer{L}_after_ffn1.bin    [N, 1024]  f16
  audio_layer{L}_after_attn.bin    [N, 1024]  f16  (post norm_post_attn + residual)
  audio_layer{L}_after_lconv.bin   [N, 1024]  f16
  audio_layer{L}_output.bin        [N, 1024]  f16  (post norm_out)

  audio_after_output_proj.bin      [N, 1536]  f16
  audio_after_embed_audio.bin      [N, 2560]  f16  (final, splice-ready)

Each .bin is raw little-endian f16, row-major contiguous. Pair
with v3/tools/cmp_e4b_audio.py for the row-cosine diff.

Run:
  python3 gemma4_e4b_audio_hf_dump.py \\
    --model /home/r00t/gemma4-e4b \\
    --dump-dir /tmp/e4b_audio_dump
"""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/home/r00t/gemma4-e4b")
    p.add_argument("--dump-dir", default="/tmp/e4b_audio_dump")
    p.add_argument("--dump-layers", default="0,1,11")
    p.add_argument("--freq", type=float, default=440.0)
    p.add_argument("--seconds", type=float, default=1.0)
    return p.parse_args()


def write_f16(path: Path, t: torch.Tensor):
    a = t.detach().to(torch.float16).contiguous().cpu().numpy()
    a.tofile(str(path))
    print(f"  wrote {path.name}  shape={tuple(a.shape)}  bytes={a.nbytes}")


def main():
    args = parse_args()
    dump_dir = Path(args.dump_dir)
    dump_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(s) for s in args.dump_layers.split(",")]

    sys.path.insert(0, "/home/r00t/.unsloth/studio/.venv_t5")
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    audio_tower = model.model.audio_tower
    embed_audio = model.model.embed_audio

    sample_rate = 16000
    n_samples = int(args.seconds * sample_rate)
    t = torch.linspace(0, args.seconds, n_samples)
    waveform = 0.5 * torch.sin(2 * math.pi * args.freq * t).unsqueeze(0)
    audio_inputs = processor.feature_extractor(
        [waveform.squeeze(0).numpy()], sampling_rate=sample_rate, return_tensors="pt"
    )
    mel = audio_inputs["input_features"].to("cuda", torch.bfloat16)
    mask = audio_inputs["input_features_mask"].to("cuda")
    print(f"mel shape: {mel.shape}")

    write_f16(dump_dir / "audio_input_mel.bin", mel[0].T)  # [T_mel, 128]

    with torch.no_grad():
        # ---- subsample with per-stage hooks ----
        ss = audio_tower.subsample_conv_projection
        # Stage 0 conv input is mel.unsqueeze(1) -> [B, 1, T, 128]
        h0_in = mel.unsqueeze(1)
        h0_conv = ss.layer0.conv(h0_in.to(ss.layer0.conv.weight.dtype))
        # CHW shape [B, c0, h0, w0]; write transposed-to-HWC for diff convenience.
        write_f16(dump_dir / "audio_subsample_stage0_conv.bin",
                  h0_conv[0].permute(1, 2, 0))  # [h0, w0, c0]
        h0_act = ss.layer0.act(ss.layer0.norm(h0_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        write_f16(dump_dir / "audio_subsample_stage0_act.bin",
                  h0_act[0].permute(1, 2, 0))  # [h0, w0, c0]
        h1_conv = ss.layer1.conv(h0_act.to(ss.layer1.conv.weight.dtype))
        write_f16(dump_dir / "audio_subsample_stage1_conv.bin",
                  h1_conv[0].permute(1, 2, 0))  # [h1, w1, c1]
        h1_act = ss.layer1.act(ss.layer1.norm(h1_conv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        write_f16(dump_dir / "audio_subsample_stage1_act.bin",
                  h1_act[0].permute(1, 2, 0))  # [h1, w1, c1]
        # final reshape input to input_proj_linear: [B, h1, w1*c1=1024]
        h1_flat = h1_act.permute(0, 2, 3, 1).contiguous().reshape(h1_act.shape[0], h1_act.shape[2], -1)
        write_f16(dump_dir / "audio_subsample_pre_proj.bin", h1_flat[0])  # [h1, 1024]
        h = ss.input_proj_linear(h1_flat)
        write_f16(dump_dir / "audio_after_subsample.bin", h[0])

        # ---- relative pos embed (shared across layers) ----
        pos_embed = audio_tower.rel_pos_enc(h)
        write_f16(dump_dir / "audio_pos_embed.bin", pos_embed[0])  # [13, 1024]

        # ---- run layers, dumping selected ----
        for li, layer in enumerate(audio_tower.layers):
            if li in layers:
                write_f16(dump_dir / f"audio_layer{li}_input.bin", h[0])
            # FFN1
            h_after_ffn1 = layer.feed_forward1(h)
            if li in layers:
                write_f16(dump_dir / f"audio_layer{li}_after_ffn1.bin", h_after_ffn1[0])
            # Attention block (pre-attn norm + attn + post-attn norm + residual)
            residual = h_after_ffn1
            gc = min(layer.gradient_clipping,
                     torch.finfo(layer.norm_pre_attn.weight.dtype).max)
            tmp = torch.clamp(h_after_ffn1, -gc, gc)
            tmp = layer.norm_pre_attn(tmp)
            tmp, _ = layer.self_attn(tmp, pos_embed, attention_mask=None)
            tmp = torch.clamp(tmp, -gc, gc)
            tmp = layer.norm_post_attn(tmp)
            h_after_attn = tmp + residual
            if li in layers:
                write_f16(dump_dir / f"audio_layer{li}_after_attn.bin", h_after_attn[0])
            # LConv1D
            h_after_lconv = layer.lconv1d(h_after_attn)
            if li in layers:
                write_f16(dump_dir / f"audio_layer{li}_after_lconv.bin", h_after_lconv[0])
            # FFN2 + norm_out
            h_after_ffn2 = layer.feed_forward2(h_after_lconv)
            h_clamped = torch.clamp(h_after_ffn2, -gc, gc)
            h_out = layer.norm_out(h_clamped)
            if li in layers:
                write_f16(dump_dir / f"audio_layer{li}_output.bin", h_out[0])
            h = h_out

        # ---- output_proj + embed_audio projection ----
        h_proj = audio_tower.output_proj(h)
        write_f16(dump_dir / "audio_after_output_proj.bin", h_proj[0])
        h_emb = embed_audio.embedding_projection(h_proj)
        write_f16(dump_dir / "audio_after_embed_audio.bin", h_emb[0])

    print(f"Done. Dumped {len(list(dump_dir.iterdir()))} files to {dump_dir}")


if __name__ == "__main__":
    main()
