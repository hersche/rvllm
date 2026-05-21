#!/usr/bin/env bash
# Spec-decode A/B + byte-equivalence harness for rvllm-serve.
#
# Drives qwen35 / qwen36 / mistral35 / gemma4-nvfp4 spec decoders
# through a fixed set of zeroclaw-shape prompts, captures wall
# time + md5 with spec ON vs OFF, and reports the perf delta +
# byte-equivalence verdict per workload.
#
# Why this exists:
#   * task #36 / #35 / #34 all need a repeatable A/B run to track
#     whether kernel-level changes (e.g. NVFP4 batched prefill
#     tuning, spec session refactors, drafter rewrites) move
#     accept-rate × wall in the right direction.
#   * The model bring-up path is expensive (60-90s for big
#     families) so flipping spec OFF/ON via systemd restart per
#     prompt is the wrong shape — this harness flips spec via
#     the ENV var that the active profile reads + a single
#     restart at the start of each ON/OFF arm.
#
# Usage:
#   v3/tools/spec_decode_bench.sh [--model qwen3-6-27b] [--prompts file] [--runs 3]
#
# Output (stdout): a table with per-workload wall + md5 + delta.
# Exit code 0 iff every workload is byte-equivalent across
# spec=0 and spec=1 (regression gate). Non-zero if any md5 diff.

set -u
SERVICE=rvllm-serve
PROFILE_PATH=/home/r00t/workspace/upstream/rvllm-serve/profiles/gb10/qwen36-26b-nvfp4
MODEL=qwen3-6-27b
SPEC_ENV=RVLLM_QWEN35_SPEC_DECODE
RUNS=3
MAX_TOKENS=80
PROMPTS_FILE=""
RESULTS=/tmp/spec_decode_bench_$$.csv

while (( "$#" )); do
  case "$1" in
    --model)        MODEL=$2; shift 2;;
    --profile)      PROFILE_PATH=$2; shift 2;;
    --service)      SERVICE=$2; shift 2;;
    --spec-env)     SPEC_ENV=$2; shift 2;;
    --runs)         RUNS=$2; shift 2;;
    --max-tokens)   MAX_TOKENS=$2; shift 2;;
    --prompts)      PROMPTS_FILE=$2; shift 2;;
    --results)      RESULTS=$2; shift 2;;
    *)              echo "unknown arg: $1" >&2; exit 1;;
  esac
done

# Default prompt set — zeroclaw-shape (system+repeated tools+user
# question), simple Q&A, echo task. Designed to span low/medium/
# high n-gram match rates so accept-rate variation is visible.
declare -A PROMPTS
if [[ -n "$PROMPTS_FILE" ]]; then
  # Load `name\tprompt` lines from a TSV file.
  while IFS=$'\t' read -r name prompt; do
    [[ -z "$name" || -z "$prompt" || "$name" == \#* ]] && continue
    PROMPTS[$name]=$prompt
  done < "$PROMPTS_FILE"
else
  PROMPTS[smalltalk]="Was ist 2+2? Erkläre kurz."
  PROMPTS[zeroclaw]="System: Du bist Rusty. Tool: brain(action) für DB. Tool: ha(state) für Smart-Home. Tool: web(url) für Internet. Tool: brain(action) für DB. Tool: ha(state) für Smart-Home. Tool: web(url) für Internet. Tool: brain(action) für DB. Tool: ha(state) für Smart-Home. Tool: web(url) für Internet. Tool: brain(action) für DB. Tool: ha(state) für Smart-Home. Tool: web(url) für Internet. Tool: brain(action) für DB. Tool: ha(state) für Smart-Home. Tool: web(url) für Internet. User: Erkläre mir kurz wie ich brain.action benutzen würde. Welche Aktionen gibt es?"
  PROMPTS[echo]="Lies und beschreibe kurz: Linux ist ein freies, quelloffenes Betriebssystem, das ursprünglich von Linus Torvalds entwickelt wurde. Es basiert auf dem Unix-Prinzip und wird heute von Millionen Menschen weltweit verwendet."
fi

wait_ready() {
  local i
  for i in $(seq 1 60); do
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 \
           http://127.0.0.1:8010/v1/models 2>/dev/null || echo 000)
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 5
  done
  echo "service didn't come up" >&2
  return 1
}

set_spec_in_profile() {
  local val=$1
  sudo sed -i "s/^export ${SPEC_ENV}=.*/export ${SPEC_ENV}=${val}/" "$PROFILE_PATH"
}

# Pure-bash JSON escape (handles \, ", and \n only — enough for
# the prompts above; non-ASCII pass through as-is which jq's
# downstream parser handles correctly).
json_escape() {
  local s=$1
  s=${s//\\/\\\\}
  s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}
  printf '%s' "$s"
}

bench_one() {
  local name=$1 prompt=$2
  local pj
  pj=$(json_escape "$prompt")
  local times=()
  local md5_first=""
  local i
  for i in $(seq 1 "$RUNS"); do
    local body t resp
    body="{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$pj\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0.0}"
    resp=$(curl -s --max-time 300 \
           -w '\n__TIME__%{time_total}\n' \
           http://127.0.0.1:8010/v1/chat/completions \
           -d "$body")
    t=$(printf '%s' "$resp" | sed -n 's/^__TIME__//p')
    local content
    content=$(printf '%s' "$resp" | sed '$d;$d' | jq -r '.choices[0].message.content // ""' 2>/dev/null || echo "")
    local md5
    md5=$(printf '%s' "$content" | md5sum | awk '{print $1}')
    if [[ -z "$md5_first" ]]; then md5_first=$md5; fi
    times+=("$t")
  done
  # Average (numerically, not avoiding bash-isms here).
  local sum=0 t
  for t in "${times[@]}"; do
    sum=$(/home/r00t/.vllm-exp/bin/python3 -c "print($sum + $t)")
  done
  local avg
  avg=$(/home/r00t/.vllm-exp/bin/python3 -c "print($sum / $RUNS)")
  printf '%s\t%s\t%s\n' "$name" "$avg" "$md5_first"
}

run_arm() {
  local label=$1 spec=$2
  echo "── ${label} (${SPEC_ENV}=${spec}) ──" >&2
  set_spec_in_profile "$spec"
  sudo systemctl restart "$SERVICE"
  wait_ready || exit 2
  local name
  for name in "${!PROMPTS[@]}"; do
    local row
    row=$(bench_one "$name" "${PROMPTS[$name]}")
    printf 'arm=%s\t%s\n' "$label" "$row" | tee -a "$RESULTS"
  done
}

> "$RESULTS"
echo "results CSV: $RESULTS" >&2
run_arm spec_off 0
run_arm spec_on  1

# Restore profile to spec=0 (production default).
set_spec_in_profile 0
sudo systemctl restart "$SERVICE"
wait_ready || true

# Report.
echo
echo "================ Spec A/B summary ================"
printf '%-20s | %-12s | %-12s | %-7s | %-7s | %-12s\n' \
  "workload" "eager wall" "spec wall" "Δ%" "md5≡" "md5_eager"
printf -- '%.0s-' {1..90}; echo
declare -A EAGER SPEC MD5E MD5S
while IFS=$'\t' read -r arm name wall md5; do
  arm=${arm#arm=}
  case "$arm" in
    spec_off) EAGER[$name]=$wall; MD5E[$name]=$md5;;
    spec_on)  SPEC[$name]=$wall; MD5S[$name]=$md5;;
  esac
done < "$RESULTS"

regress=0
for name in "${!EAGER[@]}"; do
  ew=${EAGER[$name]:-0}
  sw=${SPEC[$name]:-0}
  pct=$(/home/r00t/.vllm-exp/bin/python3 -c "
ew, sw = $ew, $sw
if ew > 0:
    print(f'{((sw - ew) / ew) * 100:+.1f}')
else:
    print('NA')
")
  eqv="✓"
  if [[ "${MD5E[$name]}" != "${MD5S[$name]}" ]]; then
    eqv="✗"
    regress=1
  fi
  printf '%-20s | %12.3f | %12.3f | %7s | %-7s | %-12.12s\n' \
    "$name" "$ew" "$sw" "$pct" "$eqv" "${MD5E[$name]}"
done

echo
if (( regress )); then
  echo "REGRESSION: at least one workload diverged. (see $RESULTS)"
  exit 1
fi
echo "All workloads byte-equivalent across spec OFF/ON. (see $RESULTS)"
