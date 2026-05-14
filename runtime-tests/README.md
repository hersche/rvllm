# rvllm-serve runtime-tests

A self-contained smoke + perf harness that loops over the
profiles in `~/.rvllm/profiles/`, restarts `rvllm-serve.service`
against each one, fires a fixed suite of 10 text prompts (short
→ 800-word context), 3 image prompts, and 1 audio prompt
(E4B-only), and writes a markdown table with timings, prefill /
decode tok/s, and pass/fail per record.

## Run

```bash
sudo /home/r00t/workspace/upstream/rvllm-serve/runtime-tests/run_smoke.py
```

(Needs `sudo` because it restarts the systemd service and rewrites
the `active-profile.env` symlink between profiles.)

Subset to specific profiles:

```bash
sudo .../run_smoke.py --profiles mobile-e4b-rvllm mobile-e4b-rvllm-nvfp4
```

Skip a modality:

```bash
sudo .../run_smoke.py --skip-vision --skip-audio
```

Default: walks the 5 mobile profiles (E4B-FP8, E4B-NVFP4,
31B-FP8, 31B-NVFP4, Qwen-FP8); other Rusty-mode profiles can be
added via `--profiles`.

## Outputs

- `runtime-tests/results.md`     — overwritten every run, markdown summary + per-record table
- `runtime-tests/results.jsonl`  — appended each run, machine-readable

## What's measured

- prompt tokens (from response `usage.prompt_tokens`)
- completion tokens
- TTFT — time from request to first SSE token
- total wall time
- derived prefill tok/s = prompt_tokens / TTFT
- derived decode  tok/s = completion_tokens / (total - TTFT)

Decode tok/s captures the inverse of average inter-token time
during generation. Prefill tok/s is a single sample so it carries
warm-up noise; treat it as an order-of-magnitude check, not a
benchmark number.

## Audio

E4B audio admission is wired (B2-B4) but the encoder forward (B6)
is not yet integrated. The harness still fires one audio probe to
exercise the fail-fast path; expect `ok=False` with a B6 error
message on the E4B profiles until the encoder lands.

## Adding a new probe

Edit `TEXT_PROMPTS`, `VISION_PROMPTS`, or
`make_wav_440hz_1s()` at the top of `run_smoke.py` and re-run.
