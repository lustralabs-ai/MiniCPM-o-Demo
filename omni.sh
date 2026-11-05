#!/usr/bin/env bash
# Start / stop the MiniCPM-o demo (llama.cpp-omni backend) without docker.
#
#   ./omni.sh start            start backend -> worker -> gateway, wait for health, register worker
#   ./omni.sh stop             stop gateway -> worker -> backend, killing whole process groups
#   ./omni.sh restart
#   ./omni.sh status
#   ./omni.sh logs [backend|worker|gateway]   (tail -f)
#
#   llama-omni-server :22501 -> worker :22401 -> gateway :8016
#
# The demo is cpp-only for now: fit-capture relies on llama.cpp-omni features (session.reset,
# session.say). The vllm-omni and torch stacks this script used to run were dropped; see git
# history. "cpp" is still accepted after the command (./omni.sh start cpp).
#
# Paths and options can be overridden with the environment variables below.

set -uo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ENV="${PY_ENV:-$HOME/py}"
PYTHON="${PYTHON:-$PY_ENV/bin/python}"
STATE_DIR="${OMNI_STATE_DIR:-/tmp/minicpm-omni}"

HF_MODEL="${HF_MODEL:-$HOME/models/MiniCPM-o-4.5}"   # token2wav assets only
LLAMA_OMNI_DIR="${LLAMA_OMNI_DIR:-$HOME/lustra/llama.cpp-omni}"
GGUF_MODEL="${GGUF_MODEL:-$HOME/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf}"
CPP_CTX="${CPP_CTX:-8192}"
DUPLEX_MAX_DURATION="${DUPLEX_MAX_DURATION:-0}"   # seconds, 0 = unlimited
BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-300}"

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- configuration ------------------------------------------------------------

STACK=cpp   # names the run dir and the gateway worker id
BACKEND_PORT=22501; WORKER_PORT=22401; GW_PORT=8016
GW_INTERNAL_PORT=$((GW_PORT + 1))
COMPONENTS=(backend worker gateway)
RUN_DIR="$STATE_DIR/$STACK"

component_port() {
    case "$1" in
        backend) echo "$BACKEND_PORT" ;;
        worker)  echo "$WORKER_PORT" ;;
        gateway) echo "$GW_INTERNAL_PORT" ;;
    esac
}

# Prints the command line for a component, one argument per line.
component_cmd() {
    case "$1" in
        backend)
            # llama-omni-server always binds 0.0.0.0; --host is ignored.
            printf '%s\n' "$LLAMA_OMNI_DIR/build/bin/llama-omni-server" \
                -m "$GGUF_MODEL" -ngl 99 --port "$BACKEND_PORT" -c "$CPP_CTX" ;;
        worker)
            printf '%s\n' "$PYTHON" worker.py --host 127.0.0.1 --port "$WORKER_PORT" --gpu-id 0 \
                --backend-server-url "http://127.0.0.1:$BACKEND_PORT" ;;
        gateway)
            printf '%s\n' "$PYTHON" gateway.py --http --host 127.0.0.1 \
                --port "$GW_PORT" --internal-port "$GW_INTERNAL_PORT" \
                --duplex-max-duration "$DUPLEX_MAX_DURATION" ;;
    esac
}

component_cwd() {
    case "$1" in
        backend) echo "$LLAMA_OMNI_DIR" ;;   # token2wav looks for tools/omni/assets relative to cwd
        *)       echo "$DEMO_DIR" ;;
    esac
}

component_env() {
    case "$1" in
        # On this APU the audio encoder's default GPU lookup misses the iGPU and
        # falls back to CPU, which leaves duplex mode stuck listening.
        # HIP graphs get recaptured on every turn, which stalls the second
        # token2wav window of each reply for 2-3 s; running without them is faster.
        # rocBLAS's hipBLASLt path makes the fp16 token2wav GEMMs ~17% faster on gfx1151.
        # OMNI_T2W_* let token2wav build a speaker prompt from the session's ref audio.
        backend) echo "MTMD_BACKEND_DEVICE=ROCm0 GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-1}" \
                      "ROCBLAS_USE_HIPBLASLT=${ROCBLAS_USE_HIPBLASLT:-1}" \
                      "OMNI_T2W_ASSET_DIR=$HF_MODEL/assets/token2wav OMNI_T2W_PYTHON=$PYTHON" ;;
    esac
}

component_timeout() {
    [ "$1" = backend ] && echo "$BACKEND_TIMEOUT" || echo 60
}

# --- process helpers ----------------------------------------------------------

pidfile() { echo "$RUN_DIR/$1.pid"; }
logfile() { echo "$RUN_DIR/$1.log"; }

component_pid() {
    local f; f="$(pidfile "$1")"
    [ -f "$f" ] || return 1
    local pid; pid="$(cat "$f")"
    kill -0 "$pid" 2>/dev/null && echo "$pid"
}

port_owner() {
    ss -ltnpH "sport = :$1" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2
}

healthy() {
    curl -sf -m 3 "http://127.0.0.1:$1/health" >/dev/null 2>&1
}

start_component() {
    local comp="$1"
    local port; port="$(component_port "$comp")"

    if component_pid "$comp" >/dev/null; then
        log "$comp already running (pid $(component_pid "$comp"))"
        return 0
    fi
    local owner; owner="$(port_owner "$port")"
    if [ -n "$owner" ]; then
        die "$comp: port $port is already used by pid $owner ($(ps -o comm= -p "$owner"))"
    fi

    local -a cmd env_args=()
    mapfile -t cmd < <(component_cmd "$comp")
    local env_str; env_str="$(component_env "$comp")"
    [ -n "$env_str" ] && env_args=($env_str)
    [ -x "${cmd[0]}" ] || die "$comp: ${cmd[0]} not found or not executable"

    local lf; lf="$(logfile "$comp")"
    : > "$lf"
    log "starting $comp on :$port (log: $lf)"
    # setsid gives each component its own process group so stop can take down
    # child processes too (worker subprocesses).
    (
        cd "$(component_cwd "$comp")" || exit 1
        exec env PATH="$PY_ENV/bin:$PATH" "${env_args[@]}" setsid "${cmd[@]}" >>"$lf" 2>&1 </dev/null
    ) &
    echo $! > "$(pidfile "$comp")"

    local timeout; timeout="$(component_timeout "$comp")"
    local t0=$SECONDS
    until healthy "$port"; do
        if ! component_pid "$comp" >/dev/null; then
            echo "---- last lines of $lf ----" >&2
            tail -n 30 "$lf" >&2
            die "$comp exited during startup"
        fi
        if (( SECONDS - t0 > timeout )); then
            tail -n 30 "$lf" >&2
            die "$comp not healthy after ${timeout}s (still running; see $lf)"
        fi
        sleep 2
    done
    log "$comp healthy after $((SECONDS - t0))s"
}

stop_component() {
    local comp="$1"
    local pid
    if ! pid="$(component_pid "$comp")"; then
        rm -f "$(pidfile "$comp")"
        # e.g. started from a shell with a different /tmp or OMNI_STATE_DIR
        local port; port="$(component_port "$comp")"
        if [ -n "$port" ] && ss -ltnH "sport = :$port" 2>/dev/null | grep -q .; then
            log "warning: $comp port $port is in use but has no pid file in $RUN_DIR; not stopping it"
        fi
        return 0
    fi
    log "stopping $comp (pid $pid)"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "$comp did not exit after 30s, sending SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    fi
    # stragglers left in the group
    sleep 1
    pkill -KILL -g "$pid" 2>/dev/null
    rm -f "$(pidfile "$comp")"
}

register_worker() {
    local url="http://127.0.0.1:$GW_INTERNAL_PORT/internal/workers/$STACK-worker"
    local body="{\"endpoint\":\"127.0.0.1:$WORKER_PORT\",\"gpu_group\":\"gpu-0\"}"
    for _ in $(seq 15); do
        if curl -sf -m 5 -X PUT -H 'content-type: application/json' --data "$body" "$url" >/dev/null; then
            log "registered $STACK-worker with gateway"
            return 0
        fi
        sleep 2
    done
    die "could not register worker at $url"
}

# --- commands -----------------------------------------------------------------

cmd_start() {
    mkdir -p "$RUN_DIR"
    # check every port before launching anything, so a conflict doesn't leave a half-started stack
    local comp port owner
    for comp in "${COMPONENTS[@]}" gateway-public; do
        [ "$comp" = gateway-public ] && port="$GW_PORT" || port="$(component_port "$comp")"
        [ "$comp" != gateway-public ] && component_pid "$comp" >/dev/null && continue
        owner="$(port_owner "$port")"
        [ -n "$owner" ] && die "port $port is already used by pid $owner ($(ps -o args= -p "$owner" | cut -c1-80))"
    done
    for comp in "${COMPONENTS[@]}"; do
        start_component "$comp"
    done
    register_worker
    log "demo is up: http://127.0.0.1:$GW_PORT"
}

cmd_stop() {
    local i
    for (( i=${#COMPONENTS[@]}-1; i>=0; i-- )); do
        stop_component "${COMPONENTS[i]}"
    done
}

cmd_status() {
    echo "llama.cpp-omni demo  (http://127.0.0.1:$GW_PORT)"
    for comp in "${COMPONENTS[@]}"; do
        local port; port="$(component_port "$comp")"
        local pid state
        if pid="$(component_pid "$comp")"; then
            healthy "$port" && state="healthy" || state="running, not healthy"
        else
            pid="-"
            local owner; owner="$(port_owner "$port")"
            [ -n "$owner" ] && state="stopped (port $port used by pid $owner, not started by omni.sh)" || state="stopped"
        fi
        printf '  %-8s :%-6s pid %-8s %s\n' "$comp" "$port" "$pid" "$state"
    done
}

cmd_logs() {
    local comp="${1:-}"
    if [ -n "$comp" ]; then
        tail -n 50 -F "$(logfile "$comp")"
    else
        local files=()
        for c in "${COMPONENTS[@]}"; do files+=("$(logfile "$c")"); done
        tail -n 20 -F "${files[@]}"
    fi
}

usage() { sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

cmd="${1:-}"
shift || true
# Accept the old stack argument for the one stack that is left.
case "${1:-}" in
    cpp|all) shift ;;
    vllm|torch) die "the $1 stack was removed; the demo is cpp-only (see the header of $0)" ;;
esac

case "$cmd" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    restart) cmd_stop; cmd_start ;;
    status)  cmd_status ;;
    logs)    cmd_logs "${1:-}" ;;
    *)       usage ;;
esac
