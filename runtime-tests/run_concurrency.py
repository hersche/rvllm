#!/usr/bin/env python3
"""Profile-aware concurrent chat benchmark for rvllm-serve.

This complements run_smoke.py: it uses the same external profile switching
flow, but measures queue/decode batching behavior by sending multiple
OpenAI-compatible chat requests concurrently to the active service.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROFILE_DIR = Path("/home/r00t/.rvllm/profiles")
ACTIVE_LINK = Path("/home/r00t/.rvllm/active-profile.env")
HERE = Path(__file__).resolve().parent

PROFILE_MODEL_ID: dict[str, str] = {
    "mobile-qwen-rvllm-nvfp4-spec": "qwen3-6-35b-a3b",
    "mobile-qwen35-rvllm-nvfp4-spec": "qwen3-6-27b",
    "mobile-31b-nvfp4w-rvllm-spec": "gemma-4-31b-it-nvfp4",
}

PROMPTS: list[tuple[str, str, str]] = [
    ("capital", "Was ist die Hauptstadt von Frankreich? Antworte kurz.", "paris"),
    ("math", "1 + 1 = ? Antworte nur mit der Zahl.", "2"),
    (
        "translation",
        "Translate this sentence to German: 'The early bird catches the worm.'",
        "wurm",
    ),
    (
        "reasoning",
        "Anna hat 3 Äpfel. Sie gibt Tom 2 davon und kauft dann 5 weitere. "
        "Wie viele Äpfel hat Anna jetzt? Antworte nur mit der Zahl und dem Wort Äpfel.",
        "6",
    ),
]


@dataclass
class RequestResult:
    profile: str
    concurrency: int
    request_index: int
    label: str
    ok: bool
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    output: str
    error: str | None = None


@dataclass
class SummaryResult:
    profile: str
    model_id: str
    concurrency: int
    requested: int
    ok: int
    fail: int
    wall_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_tok_s: float
    completion_tok_s: float
    requests_s: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float


def switch_profile(profile_basename: str) -> None:
    target = PROFILE_DIR / f"{profile_basename}.env"
    if not target.exists():
        raise FileNotFoundError(f"profile not found: {target}")
    subprocess.run(["sudo", "ln", "-sfn", str(target), str(ACTIVE_LINK)], check=True)
    subprocess.run(["sudo", "systemctl", "restart", "rvllm-serve.service"], check=True)


def wait_ready(timeout_s: int = 300, expected_model_id: str | None = None) -> bool:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(
                "http://127.0.0.1:8010/v1/models", timeout=5
            ) as r:
                if r.status == 200:
                    if expected_model_id is None:
                        return True
                    body = json.loads(r.read().decode("utf-8", errors="replace"))
                    ids = [m.get("id") for m in body.get("data", [])]
                    if expected_model_id in ids:
                        return True
        except Exception:
            pass
        time.sleep(3)
    return False


def quality_ok(label: str, expected: str, output: str) -> tuple[bool, str | None]:
    text = output.lower()
    if expected.lower() not in text:
        return False, f"expected {expected!r}"
    if label == "translation" and "frühe" not in text and "fruehe" not in text:
        return False, "expected German idiom with fruehe/frühe"
    if label == "reasoning" and ("4 äpfel" in text or "4 apfel" in text or "10" in text):
        return False, "expected 3 - 2 + 5 = 6 apples"
    return True, None


def send_chat_sync(
    profile: str,
    model_id: str,
    concurrency: int,
    request_index: int,
    max_tokens: int,
    timeout_s: int,
) -> RequestResult:
    label, prompt, expected = PROMPTS[request_index % len(PROMPTS)]
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8010/v1/chat/completions",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    error: str | None = None
    output = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
        obj: dict[str, Any] = json.loads(body.decode("utf-8", errors="replace"))
        if obj.get("choices"):
            output = obj["choices"][0].get("message", {}).get("content", "") or ""
        usage = obj.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")[:400]
        except Exception:
            body = ""
        error = f"HTTP {e.code}: {body}"
    except Exception as e:
        error = repr(e)
    latency_ms = (time.perf_counter() - start) * 1000
    ok = False
    if error is None:
        if not output:
            error = "empty output"
        else:
            ok, error = quality_ok(label, expected, output)
    return RequestResult(
        profile=profile,
        concurrency=concurrency,
        request_index=request_index,
        label=label,
        ok=ok,
        latency_ms=latency_ms,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        output=output[:200].replace("\n", " "),
        error=error,
    )


def percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((len(values) - 1) * pct / 100)
    return values[idx]


async def run_level(
    profile: str,
    model_id: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
    timeout_s: int,
) -> tuple[SummaryResult, list[RequestResult]]:
    start = time.perf_counter()
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> RequestResult:
        async with sem:
            return await asyncio.to_thread(
                send_chat_sync,
                profile,
                model_id,
                concurrency,
                i,
                max_tokens,
                timeout_s,
            )

    rows = await asyncio.gather(*(one(i) for i in range(requests)))
    wall_ms = (time.perf_counter() - start) * 1000
    ok_rows = [r for r in rows if r.ok]
    latencies = [r.latency_ms for r in rows]
    total_prompt = sum(r.prompt_tokens or 0 for r in rows)
    total_completion = sum(r.completion_tokens or 0 for r in rows)
    wall_s = max(wall_ms / 1000, 1e-9)
    summary = SummaryResult(
        profile=profile,
        model_id=model_id,
        concurrency=concurrency,
        requested=requests,
        ok=len(ok_rows),
        fail=len(rows) - len(ok_rows),
        wall_ms=wall_ms,
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
        total_tok_s=(total_prompt + total_completion) / wall_s,
        completion_tok_s=total_completion / wall_s,
        requests_s=len(rows) / wall_s,
        avg_latency_ms=sum(latencies) / len(latencies),
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
    )
    return summary, rows


def format_markdown(summaries: list[SummaryResult]) -> str:
    lines = [
        "# rvllm-serve concurrent runtime benchmark",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "",
        "| Profile | Concurrency | Pass | Fail | Wall ms | total tok/s | completion tok/s | req/s | avg latency ms | p95 latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lines.append(
            f"| {s.profile} | {s.concurrency} | {s.ok} | {s.fail} | "
            f"{s.wall_ms:.1f} | {s.total_tok_s:.1f} | {s.completion_tok_s:.1f} | "
            f"{s.requests_s:.2f} | {s.avg_latency_ms:.1f} | {s.p95_latency_ms:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", nargs="+", required=True)
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--request-timeout", type=int, default=300)
    ap.add_argument("--restore-profile", default="mobile-qwen-rvllm-nvfp4-spec")
    ap.add_argument("--results", default=str(HERE / "concurrency-results.md"))
    ap.add_argument("--jsonl", default=str(HERE / "concurrency-results.jsonl"))
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    summaries: list[SummaryResult] = []
    request_rows: list[RequestResult] = []

    try:
        for profile in args.profiles:
            model_id = PROFILE_MODEL_ID.get(profile)
            print(f"\n=== profile {profile}", flush=True)
            switch_profile(profile)
            if not wait_ready(expected_model_id=model_id):
                print(f"[skip] profile {profile} did not become ready")
                continue
            if model_id is None:
                with urllib.request.urlopen("http://127.0.0.1:8010/v1/models", timeout=5) as r:
                    body = json.loads(r.read().decode("utf-8", errors="replace"))
                    model_id = body["data"][0]["id"]
            for level in levels:
                print(f"  concurrency={level} requests={args.requests}", flush=True)
                summary, rows = asyncio.run(
                    run_level(
                        profile,
                        model_id,
                        level,
                        args.requests,
                        args.max_tokens,
                        args.request_timeout,
                    )
                )
                summaries.append(summary)
                request_rows.extend(rows)
                print(
                    f"    ok={summary.ok}/{summary.requested} "
                    f"total_tok/s={summary.total_tok_s:.1f} "
                    f"completion_tok/s={summary.completion_tok_s:.1f} "
                    f"avg_latency={summary.avg_latency_ms:.1f}ms",
                    flush=True,
                )
    finally:
        try:
            switch_profile(args.restore_profile)
            wait_ready(expected_model_id=PROFILE_MODEL_ID.get(args.restore_profile))
        except Exception as e:
            print(f"[warn] restore profile {args.restore_profile} failed: {e}")

    Path(args.results).write_text(format_markdown(summaries))
    with open(args.jsonl, "a") as f:
        for s in summaries:
            f.write(json.dumps({"kind": "summary", **asdict(s)}) + "\n")
        for r in request_rows:
            f.write(json.dumps({"kind": "request", **asdict(r)}) + "\n")
    print(f"\nWrote {args.results} ({len(summaries)} summary rows)")
    print(f"Appended {args.jsonl}")
    return 0 if all(s.fail == 0 for s in summaries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
