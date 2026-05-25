#!/bin/bash
# install-and-restart.sh — atomic build → md5-verify → restart → ready-wait
# for rvllm-serve. Eliminates the binary-staleness class of error that
# wasted multiple sessions:
#
#   * `cargo build` puts the fresh binary in `v3/target/release/`
#   * The systemd unit reads from `/home/r00t/.rvllm/bin/rvllm-server`
#     (which IS a symlink to the v3/target/release one on the dev
#     setup, but ANY install that breaks the symlink invariant — like
#     a `cp` that overwrites the symlink with a file — leaves the
#     symlink-vs-file ambiguity that this script tries to detect)
#   * PTX is loaded fresh per restart from `kernels/sm_121/`, so
#     kernel-only changes (no Rust dispatch) DO take effect without
#     rebuild — but the manifest sha must match the .so / .ptx files
#     on disk
#
# Usage:
#   scripts/install-and-restart.sh             # release build + restart + ready-wait
#   scripts/install-and-restart.sh --skip-build  # just restart + ready-wait
#   scripts/install-and-restart.sh --no-zeroclaw # don't restart zeroclaw too
#
# Exits non-zero if any step fails. Logs every step + timing.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
V3_BIN="$REPO_ROOT/v3/target/release/rvllm-server"
INSTALL_BIN="/home/r00t/.rvllm/bin/rvllm-server"

SKIP_BUILD=0
NO_ZEROCLAW=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --no-zeroclaw) NO_ZEROCLAW=1 ;;
        -h|--help)
            sed -n '2,21p' "$0"
            exit 0
            ;;
        *) echo "unknown arg: $arg (see --help)" >&2; exit 2 ;;
    esac
done

log() {
    printf '[%(%H:%M:%S)T] %s\n' -1 "$*"
}

run_step() {
    local label="$1"; shift
    local t0
    t0=$(date +%s.%N)
    log "→ $label"
    if "$@"; then
        local t1
        t1=$(date +%s.%N)
        log "  ✓ $label ($(awk "BEGIN { printf \"%.2fs\", $t1 - $t0 }"))"
    else
        local rc=$?
        log "  ✗ $label FAILED (rc=$rc)"
        exit $rc
    fi
}

build_release() {
    cd "$REPO_ROOT/v3"
    cargo build --release --bin rvllm-server --features cuda,gb10 2>&1 \
        | tail -3
}

stop_services() {
    sudo systemctl stop rvllm-serve
    if [ "$NO_ZEROCLAW" = 0 ]; then
        sudo systemctl stop zeroclaw 2>/dev/null || true
    fi
}

verify_md5_match() {
    if [ ! -f "$V3_BIN" ]; then
        log "  ✗ v3 binary missing at $V3_BIN — build must run first"
        return 1
    fi
    if [ ! -e "$INSTALL_BIN" ]; then
        log "  ! install path missing — creating symlink $INSTALL_BIN → $V3_BIN"
        sudo mkdir -p "$(dirname "$INSTALL_BIN")"
        sudo ln -sfn "$V3_BIN" "$INSTALL_BIN"
    fi
    if [ -L "$INSTALL_BIN" ]; then
        # Symlink — they MUST resolve to the same inode
        local install_target v3_real
        install_target=$(readlink -f "$INSTALL_BIN")
        v3_real=$(readlink -f "$V3_BIN")
        if [ "$install_target" != "$v3_real" ]; then
            log "  ✗ symlink drift: $INSTALL_BIN → $install_target (expected $v3_real)"
            log "    fixing symlink..."
            sudo ln -sfn "$V3_BIN" "$INSTALL_BIN"
        fi
    fi
    local md5_install md5_v3
    md5_install=$(md5sum "$INSTALL_BIN" | awk '{print $1}')
    md5_v3=$(md5sum "$V3_BIN" | awk '{print $1}')
    if [ "$md5_install" != "$md5_v3" ]; then
        log "  ✗ md5 mismatch: install=$md5_install v3=$md5_v3"
        log "    if install is a regular file (not symlink), it was clobbered"
        log "    by a manual cp. Recovering by re-symlinking:"
        sudo rm -f "$INSTALL_BIN"
        sudo ln -sfn "$V3_BIN" "$INSTALL_BIN"
        md5_install=$(md5sum "$INSTALL_BIN" | awk '{print $1}')
        if [ "$md5_install" != "$md5_v3" ]; then
            log "  ✗ recovery failed — md5 still mismatch"
            return 1
        fi
    fi
    log "  ✓ md5 match: $md5_v3"
}

start_rvllm() {
    sudo systemctl start rvllm-serve
}

wait_for_ready() {
    local t0
    t0=$(date +%s)
    while ! curl -sf http://127.0.0.1:8010/v1/models >/dev/null 2>&1; do
        sleep 3
        local elapsed=$(( $(date +%s) - t0 ))
        if [ "$elapsed" -gt 180 ]; then
            log "  ✗ /v1/models not ready after 180s — check journalctl -u rvllm-serve"
            return 1
        fi
    done
    sleep 2  # extra grace for the cuda_worker setup to finalize
}

start_zeroclaw() {
    if [ "$NO_ZEROCLAW" = 0 ]; then
        sudo systemctl start zeroclaw 2>/dev/null || log "  ! zeroclaw start skipped (not installed?)"
    fi
}

smoke_test() {
    # 5-token greedy "hi" to detect immediate breakage. Doesn't validate
    # output content (model may produce different first-5 tokens after
    # changes) — just that the request completes without error.
    local model
    model=$(curl -sf http://127.0.0.1:8010/v1/models \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" \
        2>/dev/null || echo "unknown")
    local resp
    resp=$(curl -s -H 'Content-Type: application/json' \
        http://127.0.0.1:8010/v1/chat/completions \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5,\"temperature\":0.0}" \
        --max-time 60)
    if echo "$resp" | grep -q '"choices"'; then
        local content
        content=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "<parse-fail>")
        log "  ✓ smoke ($model): \"$content\""
    else
        log "  ✗ smoke failed: $resp"
        return 1
    fi
}

# ---- main ----
log "=== install-and-restart (skip_build=$SKIP_BUILD no_zeroclaw=$NO_ZEROCLAW) ==="

if [ "$SKIP_BUILD" = 0 ]; then
    run_step "cargo build --release rvllm-server" build_release
fi
run_step "systemctl stop rvllm-serve [+ zeroclaw]" stop_services
run_step "md5 verify install symlink ↔ v3/target" verify_md5_match
run_step "systemctl start rvllm-serve" start_rvllm
run_step "wait for /v1/models ready" wait_for_ready
run_step "systemctl start zeroclaw" start_zeroclaw
run_step "smoke /v1/chat/completions" smoke_test

log "=== done ==="
