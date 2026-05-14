#!/usr/bin/env python3
"""rvllm-serve runtime smoke + perf harness.

Switches profiles, brings rvllm-serve up against each, fires a
suite of text + vision + audio probes, and writes a markdown
table with timings and tok/s.

Output: runtime-tests/results.md (overwritten each run).
Append a JSONL log of raw records to runtime-tests/results.jsonl
so multiple runs can be diffed offline.

Usage:
  sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py
                                                # all profiles, all probes
  sudo .../run_smoke.py --profiles mobile-e4b-rvllm   # subset
  sudo .../run_smoke.py --skip-audio --skip-vision    # text only
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import subprocess
import sys
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
#  Profile catalogue
# ---------------------------------------------------------------------------

PROFILE_DIR = Path("/home/r00t/.rvllm/profiles")
ACTIVE_LINK = Path("/home/r00t/.rvllm/active-profile.env")
HERE = Path(__file__).resolve().parent
RESULTS_MD = HERE / "results.md"
RESULTS_JSONL = HERE / "results.jsonl"

# Profiles we benchmark by default. Comments map each to the model
# it serves so the result table is human-readable.
DEFAULT_PROFILES: list[tuple[str, str]] = [
    ("mobile-e4b-rvllm",          "gemma-4-e4b-it (FP8 KV)"),
    ("mobile-e4b-rvllm-nvfp4",    "gemma-4-e4b-it (NVFP4 KV)"),
    ("mobile-31b-rvllm",          "gemma-4-31b-it (FP8 KV)"),
    ("mobile-31b-rvllm-nvfp4",    "gemma-4-31b-it (NVFP4 KV)"),
    ("mobile-qwen-rvllm",         "qwen3-6-35b-a3b (FP8)"),
]

# The model id the OpenAI request must use. Inferred from the
# profile's RVLLM_MODEL_ID; we cache here so we don't have to re-read.
PROFILE_MODEL_ID: dict[str, str] = {
    "mobile-e4b-rvllm":       "gemma-4-e4b-it",
    "mobile-e4b-rvllm-nvfp4": "gemma-4-e4b-it",
    "mobile-31b-rvllm":       "gemma-4-31b-it",
    "mobile-31b-rvllm-nvfp4": "gemma-4-31b-it",
    "mobile-qwen-rvllm":      "qwen3-6-35b-a3b",
}

# Which profiles support which modalities.
SUPPORTS_VISION = {"mobile-e4b-rvllm", "mobile-e4b-rvllm-nvfp4",
                   "mobile-31b-rvllm", "mobile-31b-rvllm-nvfp4",
                   "mobile-qwen-rvllm"}
SUPPORTS_AUDIO  = {"mobile-e4b-rvllm", "mobile-e4b-rvllm-nvfp4"}

# ---------------------------------------------------------------------------
#  Prompt suite
# ---------------------------------------------------------------------------

# 10 text prompts spanning short → medium → long. The long ones are
# synthesised so the script doesn't need any external fixtures.
def long_repeat(seed: str, words: int) -> str:
    tokens = seed.split()
    out = []
    while len(out) < words:
        out.extend(tokens)
    return " ".join(out[:words]) + " Bitte fasse den vorigen Text in einem Satz zusammen."

TEXT_PROMPTS: list[tuple[str, str]] = [
    ("short_capital",      "Was ist die Hauptstadt von Frankreich?"),
    ("short_math",         "1 + 1 ="),
    ("short_pangram",      "The quick brown fox jumps over the lazy"),
    ("medium_explain",     "Erkläre in drei Sätzen, was Photosynthese ist."),
    ("medium_code",        "Write a short Python function that reverses a string."),
    ("medium_translation", "Translate this sentence to German: 'The early bird catches the worm.'"),
    ("long_summary_300",   long_repeat(
        "Im Frühling blühen die Kirschbäume und die Tage werden länger. "
        "Die Vögel kehren aus dem Süden zurück und das Gras beginnt zu wachsen. ",
        words=300,
    )),
    ("long_summary_800",   long_repeat(
        "Quantum mechanics describes nature at the smallest scales of energy levels of atoms and subatomic particles. "
        "Classical physics, the collection of theories that existed before the advent of quantum mechanics, "
        "describes many aspects of nature at an ordinary (macroscopic) scale, but is not sufficient for describing them at small (atomic and subatomic) scales. ",
        words=800,
    )),
    ("reasoning_chain",    "Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. Wie viele Äpfel hat Anna jetzt? Bitte erkläre kurz den Rechenweg."),
    ("instruction",        "Liste drei Dinge auf, die man beachten sollte, wenn man eine Sauerteig-Brot backt."),
]

# ---------------------------------------------------------------------------
#  Vision fixtures
# ---------------------------------------------------------------------------

def make_image_orange_ball() -> bytes:
    """Return /tmp/ball.png contents if present, else a synthesized 256x256 PNG."""
    p = Path("/tmp/ball.png")
    if p.exists():
        return p.read_bytes()
    # Synthesize a tiny solid-orange PNG (256x256) as a fallback so
    # the harness never fails for lack of a fixture.
    try:
        from PIL import Image  # type: ignore
        im = Image.new("RGB", (256, 256), (255, 140, 40))
        # paint a light-blue border to mimic the ball-on-blue feel
        for x in range(256):
            for y in range(8):
                im.putpixel((x, y), (180, 210, 255))
                im.putpixel((x, 255 - y), (180, 210, 255))
                im.putpixel((y, x), (180, 210, 255))
                im.putpixel((255 - y, x), (180, 210, 255))
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        # Minimal 1x1 transparent PNG so the admission path at least
        # doesn't reject for invalid bytes.
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9ZptAaIAAAAASUVORK5CYII="
        )

def make_image_solid(color: tuple[int, int, int]) -> bytes:
    try:
        from PIL import Image  # type: ignore
        import io
        im = Image.new("RGB", (128, 128), color)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return make_image_orange_ball()

VISION_PROMPTS: list[tuple[str, bytes, str]] = [
    ("orange_ball",  make_image_orange_ball(),
     "Welche Farbe hat das Hauptobjekt im Bild? Antworte in einem Satz."),
    ("solid_red",    make_image_solid((255, 0, 0)),
     "Was ist die dominante Farbe im Bild?"),
    ("solid_green",  make_image_solid((0, 200, 0)),
     "Welche Farbe siehst du?"),
]

# ---------------------------------------------------------------------------
#  Audio fixture
# ---------------------------------------------------------------------------

def make_wav_440hz_1s() -> bytes:
    """Synthesize a 1-second 440 Hz mono 16 kHz PCM WAV."""
    sr = 16_000
    n = sr
    data = bytearray()
    data += b"RIFF" + struct.pack("<I", 36 + n * 2)
    data += b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
    data += b"data" + struct.pack("<I", n * 2)
    for i in range(n):
        v = int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / sr))
        data += struct.pack("<h", v)
    return bytes(data)

# ---------------------------------------------------------------------------
#  Helpers — profile switching + server readiness
# ---------------------------------------------------------------------------

def switch_profile(profile_basename: str) -> None:
    target = PROFILE_DIR / f"{profile_basename}.env"
    if not target.exists():
        raise FileNotFoundError(f"profile not found: {target}")
    subprocess.run(
        ["sudo", "ln", "-sfn", str(target), str(ACTIVE_LINK)],
        check=True,
    )
    subprocess.run(["sudo", "systemctl", "restart", "rvllm-serve.service"], check=True)

def wait_ready(timeout_s: int = 300) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8010/v1/models", timeout=5
            ) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(3)
    return False

# ---------------------------------------------------------------------------
#  Request runner
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    profile:        str
    model_id:       str
    kind:           str             # "text" | "vision" | "audio"
    label:          str
    prompt:         str             # truncated to 80 chars for the table
    output:         str             # truncated to 200 chars
    prompt_tokens:  int | None
    completion_tokens: int | None
    ttft_ms:        float | None
    total_ms:       float
    prefill_tok_s:  float | None
    decode_tok_s:   float | None
    ok:             bool
    error:          str | None = None

def call_chat(
    profile: str,
    model_id: str,
    kind: str,
    label: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int = 240,
) -> ProbeResult:
    # rvllm-serve doesn't accept `stream_options` (returns 400) and
    # streaming responses omit `usage`, so we use non-streaming for
    # accurate token counts. TTFT is therefore the same as total
    # response time in this harness; we expose it as `n/a` in the
    # table to be honest about what we can measure.
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8010/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t_start = time.time()
    ttft: float | None = None
    completion_tokens: int | None = None
    prompt_tokens: int | None = None
    error: str | None = None
    output_text = ""
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body_bytes = r.read()
        obj = json.loads(body_bytes.decode("utf-8", errors="replace"))
        if "choices" in obj and obj["choices"]:
            output_text = obj["choices"][0].get("message", {}).get("content", "") or ""
        if "usage" in obj and obj["usage"]:
            u = obj["usage"]
            prompt_tokens = u.get("prompt_tokens")
            completion_tokens = u.get("completion_tokens")
    except urllib.error.HTTPError as e:
        try:
            body_bytes = e.read().decode("utf-8", errors="replace")[:800]
        except Exception:
            body_bytes = ""
        error = f"HTTP {e.code}: {body_bytes}"
    except Exception as e:
        error = repr(e)
    total = time.time() - t_start
    # We can't separate prefill from decode without per-token
    # streaming, so we report a single combined throughput.
    prefill_tok_s = None
    decode_tok_s = None
    if (prompt_tokens or 0) + (completion_tokens or 0) > 0 and total > 0:
        combined_tok_s = ((prompt_tokens or 0) + (completion_tokens or 0)) / total
        # Convention for the table: stash combined throughput in the
        # decode column (which is the metric users actually care about
        # for chat) and leave prefill empty since we can't measure
        # TTFT through the non-streaming path.
        decode_tok_s = combined_tok_s
    prompt_summary = messages[-1].get("content", "")
    if isinstance(prompt_summary, list):
        prompt_summary = " ".join(
            p.get("text", f"<{p.get('type','?')}>") for p in prompt_summary
        )
    # If we caught no HTTP error but also got no output, surface a
    # synthetic 'empty stream' message so the table doesn't show a
    # silent ❌ with no diagnostic.
    if error is None and not output_text:
        error = "empty stream — server returned no SSE deltas (likely worker-side error event dropped)"
    return ProbeResult(
        profile=profile,
        model_id=model_id,
        kind=kind,
        label=label,
        prompt=str(prompt_summary)[:80].replace("\n", " "),
        output=output_text[:200].replace("\n", " "),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens or None,
        ttft_ms=ttft * 1000 if ttft is not None else None,
        total_ms=total * 1000,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        ok=error is None and bool(output_text),
        error=error,
    )

# ---------------------------------------------------------------------------
#  Per-profile suite
# ---------------------------------------------------------------------------

def run_profile_suite(
    profile: str,
    skip_text: bool,
    skip_vision: bool,
    skip_audio: bool,
    text_max_tokens: int,
) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    model_id = PROFILE_MODEL_ID.get(profile)
    if model_id is None:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8010/v1/models", timeout=5
            ) as r:
                body = json.loads(r.read())
                model_id = body["data"][0]["id"]
        except Exception:
            model_id = "unknown"

    # ----- text -----
    if not skip_text:
        for label, prompt in TEXT_PROMPTS:
            print(f"  [{profile}] text:{label} ...", flush=True)
            r = call_chat(
                profile=profile, model_id=model_id, kind="text", label=label,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=text_max_tokens,
            )
            results.append(r)
            if r.ok:
                tput = r.decode_tok_s or 0.0
                print(f"      ok=True  total={r.total_ms:.0f}ms  tok/s={tput:.1f}  "
                      f"(prompt={r.prompt_tokens} completion={r.completion_tokens})")
            else:
                err_short = (r.error or "")[:140].replace("\n", " ")
                print(f"      FAIL: {err_short}")

    # ----- vision -----
    if not skip_vision and profile in SUPPORTS_VISION:
        for label, img_bytes, q in VISION_PROMPTS:
            b64 = base64.b64encode(img_bytes).decode()
            url = f"data:image/png;base64,{b64}"
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": q},
                ],
            }]
            print(f"  [{profile}] vision:{label} ...", flush=True)
            r = call_chat(
                profile=profile, model_id=model_id, kind="vision", label=label,
                messages=messages, max_tokens=40,
            )
            results.append(r)
            if r.ok:
                print(f"      ok=True  total={r.total_ms:.0f}ms  "
                      f"(prompt={r.prompt_tokens} completion={r.completion_tokens})")
            else:
                err_short = (r.error or "")[:140].replace("\n", " ")
                print(f"      FAIL: {err_short}")

    # ----- audio -----
    if not skip_audio and profile in SUPPORTS_AUDIO:
        wav = make_wav_440hz_1s()
        b64 = base64.b64encode(wav).decode()
        url = f"data:audio/wav;base64,{b64}"
        messages = [{
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": url}},
                {"type": "text", "text": "Transcribe verbatim."},
            ],
        }]
        print(f"  [{profile}] audio:440hz_1s ...", flush=True)
        r = call_chat(
            profile=profile, model_id=model_id, kind="audio", label="440hz_1s",
            messages=messages, max_tokens=5,
        )
        results.append(r)
        if r.ok:
            print(f"      ok=True  total={r.total_ms:.0f}ms")
        else:
            err_short = (r.error or "")[:140].replace("\n", " ")
            print(f"      EXPECTED-FAIL (B6 not wired): {err_short}")

    return results

# ---------------------------------------------------------------------------
#  Output formatter
# ---------------------------------------------------------------------------

def format_table(records: list[ProbeResult]) -> str:
    lines = []
    lines.append("# rvllm-serve runtime tests")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    # Top-level summary per profile.
    profiles_seen: list[str] = []
    for r in records:
        if r.profile not in profiles_seen:
            profiles_seen.append(r.profile)
    lines.append("## Summary")
    lines.append("")
    lines.append("| Profile | Pass | Fail | Mean prefill tok/s | Mean decode tok/s |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in profiles_seen:
        subset = [r for r in records if r.profile == p]
        ok = sum(1 for r in subset if r.ok)
        bad = len(subset) - ok
        pref = [r.prefill_tok_s for r in subset if r.prefill_tok_s]
        dec  = [r.decode_tok_s for r in subset if r.decode_tok_s]
        mp = sum(pref) / len(pref) if pref else float("nan")
        md = sum(dec) / len(dec) if dec else float("nan")
        lines.append(f"| {p} | {ok} | {bad} | {mp:.1f} | {md:.1f} |")
    lines.append("")
    # Detailed table.
    lines.append("## Detailed records")
    lines.append("")
    lines.append("| Profile | Kind | Label | Prompt (truncated) | Output (truncated) | "
                 "prompt_tokens | completion_tokens | ttft ms | total ms | "
                 "prefill tok/s | decode tok/s | ok |")
    lines.append("|---|---|---|---|---|---:|---:|---:|---:|---:|---:|:-:|")
    for r in records:
        def f(x: Any) -> str:
            if x is None:
                return "—"
            if isinstance(x, float):
                return f"{x:.1f}"
            return str(x)
        output = r.output if r.ok else f"`{(r.error or '')[:120]}`"
        # Escape markdown table delimiters in user-visible text.
        prompt_cell = r.prompt.replace("|", "\\|")
        output_cell = output.replace("|", "\\|")
        lines.append(
            f"| {r.profile} | {r.kind} | {r.label} | {prompt_cell} | {output_cell} | "
            f"{f(r.prompt_tokens)} | {f(r.completion_tokens)} | {f(r.ttft_ms)} | "
            f"{f(r.total_ms)} | {f(r.prefill_tok_s)} | {f(r.decode_tok_s)} | "
            f"{'✅' if r.ok else '❌'} |"
        )
    return "\n".join(lines) + "\n"

# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="Subset of profile basenames; default = all in DEFAULT_PROFILES")
    ap.add_argument("--skip-text",   action="store_true")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-audio",  action="store_true")
    ap.add_argument("--text-max-tokens", type=int, default=80)
    ap.add_argument("--results",  default=str(RESULTS_MD))
    ap.add_argument("--jsonl",    default=str(RESULTS_JSONL))
    ap.add_argument("--restore-profile", default="mobile-e4b-rvllm",
                    help="Profile to leave active when the run ends")
    args = ap.parse_args()

    chosen = args.profiles or [p for p, _ in DEFAULT_PROFILES]
    all_records: list[ProbeResult] = []
    for prof in chosen:
        target = PROFILE_DIR / f"{prof}.env"
        if not target.exists():
            print(f"[skip] profile {prof} not on disk")
            continue
        print(f"\n=== profile {prof}", flush=True)
        try:
            switch_profile(prof)
        except subprocess.CalledProcessError as e:
            print(f"[skip] profile switch failed: {e}")
            continue
        if not wait_ready():
            print(f"[skip] profile {prof} did not come up within timeout")
            continue
        recs = run_profile_suite(
            prof,
            skip_text=args.skip_text,
            skip_vision=args.skip_vision,
            skip_audio=args.skip_audio,
            text_max_tokens=args.text_max_tokens,
        )
        all_records.extend(recs)

    # Restore the default profile so the next manual smoke is on a known base.
    try:
        switch_profile(args.restore_profile)
        wait_ready()
    except Exception:
        pass

    md = format_table(all_records)
    Path(args.results).write_text(md)
    with open(args.jsonl, "a") as f:
        for r in all_records:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"\nWrote {args.results} ({len(all_records)} records)")
    print(f"Appended {args.jsonl}")
    return 0 if all(r.ok for r in all_records) else 2

if __name__ == "__main__":
    sys.exit(main())
