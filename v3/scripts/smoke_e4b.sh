#!/usr/bin/env bash
# E4B-it first-light smoke (A5 anchor).
#
# Flips ~/.rvllm/active-profile.env to the E4B profile, restarts
# rvllm-serve, runs a text-only smoke plus (optional) vision smoke,
# and restores the previously-active profile + service when done.
#
# Run **only** when you've freed the GPU for E4B. The script never
# permanently displaces the active profile — even on Ctrl-C the trap
# restores it.
#
# Usage:
#   ./scripts/smoke_e4b.sh             # text-only smoke
#   ./scripts/smoke_e4b.sh --vision    # also probe with /tmp/ball.png
#
set -euo pipefail

PORT=8010
PROFILE_DIR=/home/r00t/.rvllm/profiles
ACTIVE=/home/r00t/.rvllm/active-profile.env
E4B_PROFILE="${PROFILE_DIR}/mobile-e4b-rvllm.env"
WANT_VISION=0
[[ "${1:-}" == "--vision" ]] && WANT_VISION=1

if [[ ! -f "$E4B_PROFILE" ]]; then
    echo "ERROR: $E4B_PROFILE missing; deploy v3/MOBILE_E4B_PROFILE_TEMPLATE.env first."
    exit 2
fi

# Snapshot the current active profile so we can restore it.
SNAP="$(mktemp -t rvllm-active.XXXX.env)"
cp "$ACTIVE" "$SNAP"
echo "[smoke] snapshotted current active-profile → $SNAP"

restore() {
    echo "[smoke] restoring previous active-profile"
    sudo cp "$SNAP" "$ACTIVE"
    sudo systemctl restart rvllm-serve.service || true
    rm -f "$SNAP"
}
trap restore EXIT

# Activate E4B and restart the service.
sudo cp "$E4B_PROFILE" "$ACTIVE"

# If user wants to test vision, flip MAX_IMAGES=8 on for this run.
if [[ "$WANT_VISION" == "1" ]]; then
    echo "[smoke] enabling vision admission (MAX_IMAGES=8) for this run"
    sudo sed -i 's/^RVLLM_VISION_MAX_IMAGES=0$/RVLLM_VISION_MAX_IMAGES=8/' "$ACTIVE"
fi

sudo systemctl restart rvllm-serve.service
echo "[smoke] waiting for :$PORT to accept connections..."
for i in $(seq 1 90); do
    if curl -s --max-time 1 "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
        echo "[smoke] :$PORT up after ${i}s"
        break
    fi
    sleep 1
done
curl -s "http://127.0.0.1:${PORT}/v1/models" | head -c 200; echo

# Text smoke.
echo "[smoke] text completion (German)"
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --max-time 60 \
    -d '{"model":"gemma-4-e4b-it","messages":[{"role":"user","content":"Wer bist du? Antworte in einem Satz."}],"max_tokens":80,"temperature":0.2}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'

# Vision smoke.
if [[ "$WANT_VISION" == "1" ]]; then
    if [[ ! -f /tmp/ball.png ]]; then
        echo "[smoke] /tmp/ball.png missing — skipping vision smoke."
    else
        echo "[smoke] vision completion (ball.png)"
        B64=$(base64 -w0 /tmp/ball.png)
        curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            --max-time 120 \
            -d "{\"model\":\"gemma-4-e4b-it\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${B64}\"}},{\"type\":\"text\",\"text\":\"Was zeigt das Bild?\"}]}],\"max_tokens\":80,\"temperature\":0.2}" \
            | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"])'
    fi
fi

echo "[smoke] done"
