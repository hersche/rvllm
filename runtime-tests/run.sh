#!/usr/bin/env bash
# runtime-tests: matrix runner across rvllm-serve profiles × modalities.
#
# Per cell: prompt, response, prompt_tokens, completion_tokens,
# duration_ms, prefill_tok/s, decode_tok/s.
#
# Profiles: 11 (see PROFILES array below). The script switches the
# active-profile symlink, restarts rvllm-serve, waits for readiness,
# fires 10 text + 3 vision + (3 audio on E4B only) requests, records
# all metrics, and writes a single timestamped markdown file.
#
# Per-cell fail-soft: 180s curl --max-time; on timeout the row records
# TIMEOUT and the matrix continues.

set -uo pipefail

ENDPOINT="http://127.0.0.1:8010"
PROFILE_DIR="/home/r00t/.rvllm/profiles"
ACTIVE="/home/r00t/.rvllm/active-profile.env"
IMAGE="/tmp/ball.png"
AUDIO_DIR="/home/r00t/audio_examples"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y-%m-%dT%H-%M-%S)"
OUT="${SCRIPT_DIR}/results-${TS}.md"
TIMEOUT=180

# === Profile rows ===
# Format: profile_filename | display_label | model_id | weight | kv | hadamard | audio_capable(yes|no)
PROFILES=(
  "mobile-31b-rvllm-fp8kv.env                | Gemma 4 31B  / FP8 / FP8 KV       | gemma-4-31b-it    | fp8   | fp8   | n/a | no"
  "mobile-31b-rvllm-nvfp4-hadamard-on.env    | Gemma 4 31B  / FP8 / NVFP4 KV/HAD | gemma-4-31b-it    | fp8   | nvfp4 | on  | no"
  "mobile-31b-rvllm-nvfp4-hadamard-off.env   | Gemma 4 31B  / FP8 / NVFP4 KV     | gemma-4-31b-it    | fp8   | nvfp4 | off | no"
  "mobile-e4b-rvllm.env                      | Gemma 4 E4B  / F16 / FP8 KV       | gemma-4-e4b-it    | f16   | fp8   | n/a | yes"
  "mobile-e4b-rvllm-nvfp4-hadamard-on.env    | Gemma 4 E4B  / F16 / NVFP4 KV/HAD | gemma-4-e4b-it    | f16   | nvfp4 | on  | yes"
  "mobile-e4b-rvllm-nvfp4-hadamard-off.env   | Gemma 4 E4B  / F16 / NVFP4 KV     | gemma-4-e4b-it    | f16   | nvfp4 | off | yes"
  "mobile-mistral35-rvllm.env                | Mistral 3.5  / NVFP4 / NVFP4 KV   | mistral-3.5-nvfp4 | nvfp4 | nvfp4 | n/a | no"
  "mobile-qwen35-rvllm.env                   | Qwen 3.5 27B / FP8 / F16 KV       | qwen3-5-27b-dense | fp8   | f16   | n/a | no"
  "mobile-qwen35-rvllm-nvfp4.env             | Qwen 3.5 27B / FP8 / NVFP4 KV     | qwen3-5-27b-dense | fp8   | nvfp4 | n/a | no"
  "mobile-qwen-rvllm.env                     | Qwen 3.6 35B / FP8 / F16 KV       | qwen3-6-35b-a3b   | fp8   | f16   | n/a | no"
  "mobile-qwen-rvllm-nvfp4.env               | Qwen 3.6 35B / FP8 / NVFP4 KV     | qwen3-6-35b-a3b   | fp8   | nvfp4 | n/a | no"
)

# === 10 text prompts (varied length, German + English + code + math) ===
TEXT_PROMPTS=(
  "Hi."
  "Wie heißt die Hauptstadt von Frankreich?"
  "1 + 1 = ?"
  "Erzähl mir einen kurzen Witz auf Deutsch."
  "Write a one-line Python function that returns the square of its argument."
  "Erkläre kurz und präzise, was der Unterschied zwischen einem Compiler und einem Interpreter ist."
  "List three benefits of using version control in software projects."
  "Schreibe ein 4-zeiliges Gedicht über den Frühling."
  "Wenn ein Zug um 14:35 in Zürich abfährt und nach 2 Stunden 27 Minuten in Bern ankommt, wann kommt er an?"
  "Gib mir eine kurze Zusammenfassung in drei Sätzen, wie eine Hash-Map funktioniert."
)

# === Per-prompt token caps (rough match to expected output length) ===
TEXT_CAPS=(20 40 20 80 60 200 120 80 60 120)

# === 3 vision prompts on /tmp/ball.png ===
VISION_PROMPTS=(
  "Was zeigt das Bild?"
  "Beschreibe das Bild bitte ausführlich."
  "List the dominant colors in this image."
)
VISION_CAPS=(60 120 40)

# === 3 audio files (E4B only) ===
AUDIO_FILES=(
  "${AUDIO_DIR}/das_ist_ein_test.mp3"
  "${AUDIO_DIR}/wie_ist_das_wetter_heute.ogg"
  "${AUDIO_DIR}/i_am_a_human.ogg"
)

# ---------------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# Write the file header once.
{
  printf '# rvllm-serve runtime-tests — %s\n\n' "$TS"
  printf 'Endpoint: \`%s\`\n\n' "$ENDPOINT"
  printf 'Cell metrics: prompt_tokens / completion_tokens / duration_ms / prefill_tok-s / decode_tok-s.\n'
  printf 'Prefill tok/s = prompt_tokens / (duration_ms/1000) (effective prompt-side throughput, includes vision/audio splice).\n'
  printf 'Decode tok/s = completion_tokens / (duration_ms/1000) (effective decode-side throughput).\n\n'
  printf '## Profiles\n\n'
  printf '| # | Profile file | Label | Weight | KV | Hadamard | Audio |\n'
  printf '|---|---|---|---|---|---|---|\n'
  i=1
  for row in "${PROFILES[@]}"; do
    IFS='|' read -r file label model weight kv had aud <<<"$row"
    printf '| %d | `%s` | %s | %s | %s | %s | %s |\n' "$i" \
      "$(echo "$file" | xargs)" "$(echo "$label" | xargs)" \
      "$(echo "$weight" | xargs)" "$(echo "$kv" | xargs)" \
      "$(echo "$had" | xargs)" "$(echo "$aud" | xargs)"
    i=$((i+1))
  done
  printf '\n## Results\n\n'
} > "$OUT"

# --- Helpers ---

emit_table_header() {
  local title="$1"
  {
    printf '\n### %s\n\n' "$title"
    printf '| profile | modality | prompt | response (head) | prompt_tok | completion_tok | duration_ms | prefill_t/s | decode_t/s |\n'
    printf '|---|---|---|---|---:|---:|---:|---:|---:|\n'
  } >> "$OUT"
}

# Truncate strings safely for the markdown column.
ms_str() { awk -v ms="$1" 'BEGIN{printf "%d", ms+0}'; }
trunc() { local s="$1" n="${2:-80}"; if [ "${#s}" -gt "$n" ]; then echo "${s:0:$n}…"; else echo "$s"; fi; }
md_escape() {
  printf '%s' "$1" | tr '\n' ' ' | sed -e 's/|/\\|/g' -e 's/^ *//; s/ *$//'
}

# Switch profile + restart + wait. Sets $PROFILE_LABEL etc as side effects.
activate_profile() {
  local file="$1"
  sudo ln -sfn "${PROFILE_DIR}/${file}" "${ACTIVE}"
  sudo systemctl restart rvllm-serve
  local waited=0
  until curl -fsS "${ENDPOINT}/v1/models" -o /dev/null 2>/dev/null; do
    sleep 5
    waited=$((waited+5))
    if [ "$waited" -ge 300 ]; then
      log "TIMEOUT waiting for $file"
      return 1
    fi
  done
  return 0
}

# Fire one text chat request. Echoes a JSONL-style record.
fire_text() {
  local model="$1" prompt="$2" max_new="$3"
  local payload
  payload=$(jq -nc --arg m "$model" --arg p "$prompt" --argjson n "$max_new" \
    '{model:$m, temperature:0, max_tokens:$n, messages:[{role:"user", content:$p}]}')
  local t0 t1 body http_ms
  t0=$(date +%s%3N)
  body=$(curl -sS --max-time "$TIMEOUT" -H 'Content-Type: application/json' \
         -X POST "${ENDPOINT}/v1/chat/completions" -d "$payload" 2>/dev/null) || body='{"error":"curl_failed"}'
  t1=$(date +%s%3N)
  http_ms=$((t1 - t0))
  printf '%s\t%s\n' "$http_ms" "$body"
}

# Fire one vision chat request (image is bundled inline).
fire_vision() {
  local model="$1" prompt="$2" max_new="$3" img_b64="$4"
  local payload
  payload=$(jq -nc --arg m "$model" --arg p "$prompt" --argjson n "$max_new" --arg img "$img_b64" \
    '{model:$m, temperature:0, max_tokens:$n,
      messages:[{role:"user", content:[
        {type:"image_url", image_url:{url:("data:image/png;base64," + $img)}},
        {type:"text", text:$p}
      ]}]}')
  local t0 t1 body http_ms
  t0=$(date +%s%3N)
  body=$(curl -sS --max-time "$TIMEOUT" -H 'Content-Type: application/json' \
         -X POST "${ENDPOINT}/v1/chat/completions" -d "$payload" 2>/dev/null) || body='{"error":"curl_failed"}'
  t1=$(date +%s%3N)
  http_ms=$((t1 - t0))
  printf '%s\t%s\n' "$http_ms" "$body"
}

# Fire one audio transcription. Returns http_ms\tbody.
fire_audio() {
  local model="$1" audio_file="$2"
  local t0 t1 body http_ms
  t0=$(date +%s%3N)
  body=$(curl -sS --max-time "$TIMEOUT" \
         -F "file=@${audio_file}" -F "model=${model}" \
         "${ENDPOINT}/v1/audio/transcriptions" 2>/dev/null) || body='{"error":"curl_failed"}'
  t1=$(date +%s%3N)
  http_ms=$((t1 - t0))
  printf '%s\t%s\n' "$http_ms" "$body"
}

# Parse a chat-completion response. Echoes:
#   content\tprompt_tok\tcompletion_tok
parse_chat() {
  local body="$1"
  local content pt ct
  content=$(echo "$body" | jq -r '.choices[0].message.content // .error.message // "PARSE_FAIL"' 2>/dev/null)
  pt=$(echo "$body" | jq -r '.usage.prompt_tokens // 0' 2>/dev/null)
  ct=$(echo "$body" | jq -r '.usage.completion_tokens // 0' 2>/dev/null)
  printf '%s\t%s\t%s\n' "$content" "$pt" "$ct"
}

# Parse audio transcription response. Echoes content (no usage in spec).
parse_audio() {
  local body="$1"
  echo "$body" | jq -r '.text // .error.message // "PARSE_FAIL"' 2>/dev/null
}

# Compute prefill/decode tok/s safely (guard div by 0).
tok_per_sec() {
  awk -v toks="$1" -v ms="$2" 'BEGIN{
    if (ms+0 == 0) { print "n/a"; exit }
    printf "%.1f", (toks+0) * 1000.0 / (ms+0)
  }'
}

write_row_text() {
  # args: label, modality, prompt, content, pt, ct, ms
  local label="$1" mod="$2" prompt="$3" content="$4" pt="$5" ct="$6" ms="$7"
  local pre dec
  pre=$(tok_per_sec "$pt" "$ms")
  dec=$(tok_per_sec "$ct" "$ms")
  printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(md_escape "$label")" "$mod" \
    "$(md_escape "$(trunc "$prompt" 70)")" \
    "$(md_escape "$(trunc "$content" 100)")" \
    "$pt" "$ct" "$ms" "$pre" "$dec" >> "$OUT"
}

# ----------------------- main loop -----------------------

# Pre-encode the image once.
IMG_B64=$(base64 -w0 "$IMAGE")

profile_no=0
for row in "${PROFILES[@]}"; do
  profile_no=$((profile_no+1))
  IFS='|' read -r file label model weight kv had audio_capable <<<"$row"
  file=$(echo "$file" | xargs); label=$(echo "$label" | xargs)
  model=$(echo "$model" | xargs); audio_capable=$(echo "$audio_capable" | xargs)

  log "=== [$profile_no/${#PROFILES[@]}] $label  ($file) ==="
  if ! activate_profile "$file"; then
    {
      printf '\n### %s\n\nPROFILE FAILED TO ACTIVATE.\n' "$label"
    } >> "$OUT"
    continue
  fi

  emit_table_header "$label"

  # Warm-up the engine + KV (one tiny prompt). Result discarded.
  fire_text "$model" "OK." 4 >/dev/null

  # Text x 10
  for i in "${!TEXT_PROMPTS[@]}"; do
    p="${TEXT_PROMPTS[$i]}"
    cap="${TEXT_CAPS[$i]}"
    log "  text[$((i+1))/10] $(trunc "$p" 50)"
    IFS=$'\t' read -r ms body <<<"$(fire_text "$model" "$p" "$cap")"
    IFS=$'\t' read -r content pt ct <<<"$(parse_chat "$body")"
    write_row_text "$label" "text" "$p" "$content" "$pt" "$ct" "$ms"
  done

  # Vision x 3
  for i in "${!VISION_PROMPTS[@]}"; do
    p="${VISION_PROMPTS[$i]}"
    cap="${VISION_CAPS[$i]}"
    log "  vision[$((i+1))/3] $(trunc "$p" 50)"
    IFS=$'\t' read -r ms body <<<"$(fire_vision "$model" "$p" "$cap" "$IMG_B64")"
    IFS=$'\t' read -r content pt ct <<<"$(parse_chat "$body")"
    write_row_text "$label" "vision" "[ball.png] $p" "$content" "$pt" "$ct" "$ms"
  done

  # Audio x 3 (E4B only)
  if [ "$audio_capable" = "yes" ]; then
    for a in "${AUDIO_FILES[@]}"; do
      base=$(basename "$a")
      log "  audio: $base"
      IFS=$'\t' read -r ms body <<<"$(fire_audio "$model" "$a")"
      content=$(parse_audio "$body")
      # The transcriptions endpoint returns no usage block; report
      # "-" for tok counters and only duration_ms.
      printf '| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n' \
        "$(md_escape "$label")" "audio" \
        "$(md_escape "$base")" "$(md_escape "$(trunc "$content" 100)")" \
        "-" "-" "$ms" "-" "-" >> "$OUT"
    done
  fi
done

log "Matrix complete. Results: $OUT"
echo "$OUT"
