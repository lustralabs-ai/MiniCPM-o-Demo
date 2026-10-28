#!/usr/bin/env bash
# Start / stop the MiniCPM-o demo stacks without docker.
#
#   ./omni.sh start  <stack>      start backend -> worker -> gateway, wait for health, register worker
#   ./omni.sh stop   <stack|all>  stop a stack (gateway -> worker -> backend), killing whole process groups
#   ./omni.sh restart <stack>
#   ./omni.sh status [stack|all]
#   ./omni.sh logs   <stack> [backend|bridge|worker|gateway]   (tail -f)
#
# Stacks (each has its own ports, so they can run side by side if GPU memory allows):
#   vllm   vllm-omni :8099 -> bridge :22500 -> worker :22400 -> gateway :8006
#   cpp    llama-omni-server :22501       -> worker :22401 -> gateway :8016
#   torch  py_backend.server :22502       -> worker :22402 -> gateway :8026
#
# Paths and options can be overridden with the environment variables below.

set -uo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_ENV="${PY_ENV:-$HOME/py}"
PYTHON="${PYTHON:-$PY_ENV/bin/python}"
VLLM_BIN="${VLLM_BIN:-$PY_ENV/bin/vllm}"
STATE_DIR="${OMNI_STATE_DIR:-/tmp/minicpm-omni}"

HF_MODEL="${HF_MODEL:-$HOME/models/MiniCPM-o-4.5}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-$HOME/lustra/vllm-omni}"
VLLM_DEPLOY="${VLLM_DEPLOY:-vllm_omni/deploy/minicpmo_4_5_apu.yaml}"
LLAMA_OMNI_DIR="${LLAMA_OMNI_DIR:-$HOME/lustra/llama.cpp-omni}"
GGUF_MODEL="${GGUF_MODEL:-$HOME/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q8_0.gguf}"
CPP_CTX="${CPP_CTX:-8192}"
DUPLEX_MAX_DURATION="${DUPLEX_MAX_DURATION:-0}"   # seconds, 0 = unlimited
BACKEND_TIMEOUT="${BACKEND_TIMEOUT:-900}"         # vllm-omni takes a few minutes to load

STACKS=(vllm cpp torch)

die() { echo "error: $*" >&2; exit 1; }
log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- per-stack configuration -------------------------------------------------

stack_config() {
    case "$1" in
        vllm)  BACKEND_PORT=8099;  BRIDGE_PORT=22500; WORKER_PORT=22400; GW_PORT=8006; COMPONENTS=(backend bridge worker gateway) ;;
        cpp)   BACKEND_PORT=22501; BRIDGE_PORT="";    WORKER_PORT=22401; GW_PORT=8016; COMPONENTS=(backend worker gateway) ;;
        torch) BACKEND_PORT=22502; BRIDGE_PORT="";    WORKER_PORT=22402; GW_PORT=8026; COMPONENTS=(backend worker gateway) ;;
        *) die "unknown stack '$1' (expected: ${STACKS[*]})" ;;
    esac
    GW_INTERNAL_PORT=$((GW_PORT + 1))
    # the worker talks to the bridge when there is one, otherwise to the backend
    WORKER_UPSTREAM_PORT="${BRIDGE_PORT:-$BACKEND_PORT}"
    RUN_DIR="$STATE_DIR/$1"
}

component_port() {
    case "$1" in
        backend) echo "$BACKEND_PORT" ;;
        bridge)  echo "$BRIDGE_PORT" ;;
        worker)  echo "$WORKER_PORT" ;;
        gateway) echo "$GW_INTERNAL_PORT" ;;
    esac
}

# Prints the command line for a component, one argument per line.
component_cmd() {
    local stack="$1" comp="$2"
    case "$stack/$comp" in
        vllm/backend)
            printf '%s\n' "$VLLM_BIN" serve "$HF_MODEL" --omni \
                --deploy-config "$VLLM_DEPLOY" --trust-remote-code \
                --served-model-name openbmb/MiniCPM-o-4_5 \
                --chat-template vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja \
                --chat-template-content-format openai \
                --host 127.0.0.1 --port "$BACKEND_PORT" ;;
        vllm/bridge)
            printf '%s\n' "$PYTHON" -m py_backend.vllm_omni_bridge \
                --host 127.0.0.1 --port "$BRIDGE_PORT" --upstream "http://127.0.0.1:$BACKEND_PORT" ;;
        cpp/backend)
            # llama-omni-server always binds 0.0.0.0; --host is ignored.
            printf '%s\n' "$LLAMA_OMNI_DIR/build/bin/llama-omni-server" \
                -m "$GGUF_MODEL" -ngl 99 --port "$BACKEND_PORT" -c "$CPP_CTX" ;;
        torch/backend)
            printf '%s\n' "$PYTHON" -m py_backend.server \
                --host 127.0.0.1 --port "$BACKEND_PORT" --gpu-id 0 --model-path "$HF_MODEL" ;;
        */worker)
            printf '%s\n' "$PYTHON" worker.py --host 127.0.0.1 --port "$WORKER_PORT" --gpu-id 0 \
                --backend-server-url "http://127.0.0.1:$WORKER_UPSTREAM_PORT" ;;
        */gateway)
            printf '%s\n' "$PYTHON" gateway.py --http --host 127.0.0.1 \
                --port "$GW_PORT" --internal-port "$GW_INTERNAL_PORT" \
                --duplex-max-duration "$DUPLEX_MAX_DURATION" ;;
    esac
}

component_cwd() {
    case "$1/$2" in
        vllm/backend) echo "$VLLM_OMNI_DIR" ;;
        cpp/backend)  echo "$LLAMA_OMNI_DIR" ;;   # token2wav looks for tools/omni/assets relative to cwd
        *)            echo "$DEMO_DIR" ;;
    esac
}

component_env() {
    case "$1/$2" in
        # On this APU the audio encoder's default GPU lookup misses the iGPU and
        # falls back to CPU, which leaves duplex mode stuck listening.
        cpp/backend) echo "MTMD_BACKEND_DEVICE=ROCm0" ;;
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
    local stack="$1" comp="$2"
    local port; port="$(component_port "$comp")"

    if component_pid "$comp" >/dev/null; then
        log "$stack/$comp already running (pid $(component_pid "$comp"))"
        return 0
    fi
    local owner; owner="$(port_owner "$port")"
    if [ -n "$owner" ]; then
        die "$stack/$comp: port $port is already used by pid $owner ($(ps -o comm= -p "$owner"))"
    fi

    local -a cmd env_args=()
    mapfile -t cmd < <(component_cmd "$stack" "$comp")
    local env_str; env_str="$(component_env "$stack" "$comp")"
    [ -n "$env_str" ] && env_args=($env_str)
    [ -x "${cmd[0]}" ] || die "$stack/$comp: ${cmd[0]} not found or not executable"

    local lf; lf="$(logfile "$comp")"
    : > "$lf"
    log "starting $stack/$comp on :$port (log: $lf)"
    # setsid gives each component its own process group so stop can take down
    # child processes too (vllm-omni stage engines, worker subprocesses).
    (
        cd "$(component_cwd "$stack" "$comp")" || exit 1
        exec env PATH="$PY_ENV/bin:$PATH" "${env_args[@]}" setsid "${cmd[@]}" >>"$lf" 2>&1 </dev/null
    ) &
    echo $! > "$(pidfile "$comp")"

    local timeout; timeout="$(component_timeout "$comp")"
    local t0=$SECONDS
    until healthy "$port"; do
        if ! component_pid "$comp" >/dev/null; then
            echo "---- last lines of $lf ----" >&2
            tail -n 30 "$lf" >&2
            die "$stack/$comp exited during startup"
        fi
        if (( SECONDS - t0 > timeout )); then
            tail -n 30 "$lf" >&2
            die "$stack/$comp not healthy after ${timeout}s (still running; see $lf)"
        fi
        sleep 2
    done
    log "$stack/$comp healthy after $((SECONDS - t0))s"
}

stop_component() {
    local stack="$1" comp="$2"
    local pid; pid="$(component_pid "$comp")" || { rm -f "$(pidfile "$comp")"; return 0; }
    log "stopping $stack/$comp (pid $pid)"
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in $(seq 30); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        log "$stack/$comp did not exit after 30s, sending SIGKILL"
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    fi
    # stragglers left in the group (e.g. vllm stage engines)
    sleep 1
    pkill -KILL -g "$pid" 2>/dev/null
    rm -f "$(pidfile "$comp")"
}

register_worker() {
    local stack="$1"
    local url="http://127.0.0.1:$GW_INTERNAL_PORT/internal/workers/$stack-worker"
    local body="{\"endpoint\":\"127.0.0.1:$WORKER_PORT\",\"gpu_group\":\"gpu-0\"}"
    for _ in $(seq 15); do
        if curl -sf -m 5 -X PUT -H 'content-type: application/json' --data "$body" "$url" >/dev/null; then
            log "registered $stack-worker with gateway"
            return 0
        fi
        sleep 2
    done
    die "could not register worker at $url"
}

# --- commands -----------------------------------------------------------------

cmd_start() {
    local stack="$1"
    stack_config "$stack"
    mkdir -p "$RUN_DIR"
    # check every port before launching anything, so a conflict doesn't leave
    # a half-started stack (or a multi-minute vllm load) behind
    local comp port owner
    for comp in "${COMPONENTS[@]}" gateway-public; do
        [ "$comp" = gateway-public ] && port="$GW_PORT" || port="$(component_port "$comp")"
        [ "$comp" != gateway-public ] && component_pid "$comp" >/dev/null && continue
        owner="$(port_owner "$port")"
        [ -n "$owner" ] && die "$stack: port $port is already used by pid $owner ($(ps -o args= -p "$owner" | cut -c1-80))"
    done
    for comp in "${COMPONENTS[@]}"; do
        start_component "$stack" "$comp"
    done
    register_worker "$stack"
    log "$stack is up: http://127.0.0.1:$GW_PORT"
}

cmd_stop() {
    local stack="$1"
    stack_config "$stack"
    local i
    for (( i=${#COMPONENTS[@]}-1; i>=0; i-- )); do
        stop_component "$stack" "${COMPONENTS[i]}"
    done
}

cmd_status() {
    local stack="$1"
    stack_config "$stack"
    echo "$stack  (http://127.0.0.1:$GW_PORT)"
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
    local stack="$1" comp="${2:-}"
    stack_config "$stack"
    if [ -n "$comp" ]; then
        tail -n 50 -F "$(logfile "$comp")"
    else
        local files=()
        for c in "${COMPONENTS[@]}"; do files+=("$(logfile "$c")"); done
        tail -n 20 -F "${files[@]}"
    fi
}

expand_stacks() {
    if [ "${1:-all}" = all ]; then echo "${STACKS[@]}"; else echo "$1"; fi
}

usage() { sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
    start)   [ -n "${2:-}" ] || usage; for s in $(expand_stacks "$2"); do cmd_start "$s"; done ;;
    stop)    for s in $(expand_stacks "${2:-all}"); do cmd_stop "$s"; done ;;
    restart) [ -n "${2:-}" ] || usage; for s in $(expand_stacks "$2"); do cmd_stop "$s"; cmd_start "$s"; done ;;
    status)  for s in $(expand_stacks "${2:-all}"); do cmd_status "$s"; done ;;
    logs)    [ -n "${2:-}" ] || usage; cmd_logs "$2" "${3:-}" ;;
    *)       usage ;;
esac
