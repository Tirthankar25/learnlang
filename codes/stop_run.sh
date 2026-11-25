#!/usr/bin/env bash
# Stops everything run_all.sh starts and checks that the GPU is free again:
#   - the LeLa app (uvicorn app:app on port 7800)
#   - vLLM (start_vllm.sh -> python3 -m vllm.entrypoints.openai.api_server on port 8001)
#     and its worker processes (VLLM::EngineCore, multiprocessing spawn / resource_tracker)
#   - any run_all.sh still waiting in the foreground
# Sends SIGTERM first, waits, then SIGKILL for whatever is left.
# Finally shows GPU memory and says whether the GPU is clear.
#
# Usage: ./stop_run.sh            stop LeLa processes (yours only)
#        ./stop_run.sh --all-gpu  also kill any other process of yours still holding the GPU
# LELA_GPU picks the GPU to check (default 0, same as run_all.sh).
set -uo pipefail

GPU="${LELA_GPU:-0}"
ALL_GPU=0
[ "${1:-}" = "--all-gpu" ] && ALL_GPU=1
ME="$(id -u)"
SELF=$$

# ---------- find processes ----------
declare -A PIDS=()
add() { for p in "$@"; do
          [[ "$p" =~ ^[0-9]+$ ]] || continue
          [ "$p" = "$SELF" ] && continue
          [ "$p" = "$PPID" ] && continue
          kill -0 "$p" 2>/dev/null || continue
          PIDS[$p]=1
        done; }

# Listeners on the two ports.
port_pids() {
    local port=$1
    if command -v ss >/dev/null; then
        ss -ltnpH "sport = :$port" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2
    fi
    if command -v lsof >/dev/null; then
        lsof -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null
    fi
    if command -v fuser >/dev/null; then
        fuser -n tcp "$port" 2>/dev/null | tr -s ' ' '\n'
    fi
}
add $(port_pids 7800) $(port_pids 8001)

# By command line (only this user's processes).
for pat in 'run_all\.sh' 'start_vllm\.sh' 'uvicorn app:app' \
           'vllm\.entrypoints' 'vllm serve' 'VLLM::' \
           'multiprocessing\.spawn' 'multiprocessing\.resource_tracker'; do
    for p in $(pgrep -u "$ME" -f "$pat" 2>/dev/null); do
        # multiprocessing helpers are only ours if they belong to vLLM / the app.
        if [[ "$pat" == multiprocessing* ]]; then
            cwd=$(readlink "/proc/$p/cwd" 2>/dev/null || true)
            parent=$(ps -o args= -p "$(ps -o ppid= -p "$p" | tr -d ' ')" 2>/dev/null || true)
            [[ "$cwd" == "$(cd "$(dirname "$0")" && pwd)"* || "$parent" == *vllm* || "$parent" == *uvicorn* || "$parent" == *VLLM* ]] || continue
        fi
        add "$p"
    done
done

# Children and process-group members of everything found (vLLM runs in its own setsid group).
for _ in 1 2 3; do
    for p in "${!PIDS[@]}"; do
        add $(pgrep -P "$p" 2>/dev/null)
        pgid=$(ps -o pgid= -p "$p" 2>/dev/null | tr -d ' ')
        [ -n "$pgid" ] && [ "$pgid" != "$(ps -o pgid= -p $SELF | tr -d ' ')" ] && add $(pgrep -g "$pgid" 2>/dev/null)
    done
done

# Processes holding the GPU (nvidia-smi first; fuser on /dev/nvidia* if it lists nothing).
gpu_pids() {
    local list
    list=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null | tr -d ' ')
    if [ -z "$list" ] && command -v fuser >/dev/null; then
        list=$(fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\n')
        if [ -z "$list" ] && sudo -n true 2>/dev/null; then
            list=$(sudo fuser /dev/nvidia* 2>/dev/null | tr -s ' ' '\n')
        fi
    fi
    echo "$list" | grep -E '^[0-9]+$' | sort -u
}
for p in $(gpu_pids); do
    owner=$(ps -o uid= -p "$p" 2>/dev/null | tr -d ' ')
    args=$(ps -o args= -p "$p" 2>/dev/null)
    [ "$owner" = "$ME" ] || continue
    if [ $ALL_GPU = 1 ] || [[ "$args" == *vllm* || "$args" == *VLLM* || "$args" == *uvicorn* || "$args" == *multiprocessing* ]]; then
        add "$p"
    fi
done

# ---------- stop them ----------
if [ ${#PIDS[@]} -eq 0 ]; then
    echo "--> No LeLa / vLLM processes found."
else
    echo "--> Stopping ${#PIDS[@]} process(es):"
    for p in "${!PIDS[@]}"; do ps -o pid=,args= -p "$p" 2>/dev/null | cut -c1-120 | sed 's/^/    /'; done
    kill -TERM "${!PIDS[@]}" 2>/dev/null

    for _ in $(seq 1 20); do                     # up to 10 s for a clean exit
        left=()
        for p in "${!PIDS[@]}"; do kill -0 "$p" 2>/dev/null && left+=("$p"); done
        [ ${#left[@]} -eq 0 ] && break
        sleep 0.5
    done
    if [ ${#left[@]} -gt 0 ]; then
        echo "--> ${#left[@]} still running after SIGTERM, sending SIGKILL: ${left[*]}"
        kill -KILL "${left[@]}" 2>/dev/null
        sleep 1
    fi
    still=()
    for p in "${!PIDS[@]}"; do kill -0 "$p" 2>/dev/null && still+=("$p"); done
    [ ${#still[@]} -gt 0 ] && echo "!! Could not stop: ${still[*]} (try: sudo kill -9 ${still[*]})"
fi

# ---------- check the GPU ----------
if ! nvidia-smi >/dev/null 2>&1; then
    echo "!! nvidia-smi fails, so GPU memory can't be checked."; exit 1
fi
# The driver frees memory a moment after the process exits; give it a few seconds.
for _ in $(seq 1 10); do
    USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
    [ "${USED_MB:-0}" -le 1000 ] && break
    sleep 1
done

echo
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv -i "$GPU"
APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader -i "$GPU" 2>/dev/null)
[ -n "$APPS" ] && { echo "Compute processes on GPU $GPU:"; echo "$APPS" | sed 's/^/    /'; }

echo
if [ "${USED_MB:-0}" -le 1000 ]; then
    echo "OK: GPU $GPU is clear (${USED_MB} MiB in use, desktop/driver only). Safe to run ./run_all.sh."
    exit 0
fi

echo "!! GPU $GPU still has ${USED_MB} MiB in use."
LEFT=$(gpu_pids)
if [ -n "$LEFT" ]; then
    echo "Processes still holding /dev/nvidia*:"
    for p in $LEFT; do ps -o pid=,user=,args= -p "$p" 2>/dev/null | cut -c1-120 | sed 's/^/    /'; done
    echo "If they are yours and safe to stop: ./stop_run.sh --all-gpu  (or sudo kill -9 <pid> for other users)."
else
    echo "No process holding the GPU could be found. Try: sudo fuser -v /dev/nvidia*"
    echo "(without sudo, other users' or root's processes are invisible). If nothing shows,"
    echo "the memory is stuck in the driver: sudo nvidia-smi --gpu-reset -i $GPU (no display on it) or reboot."
fi
exit 1
