#!/usr/bin/env bash
# Starts vLLM, waits until it is healthy, then runs the LeLa server.
# Ctrl-C (or uvicorn exiting) stops vLLM too.
set -euo pipefail
cd "$(dirname "$0")"

if [ -d "lela-env" ]; then
    # shellcheck disable=SC1091
    source lela-env/bin/activate
fi

# Everything runs on one GPU (LELA_GPU, default 0). vLLM takes 55% of it (~22 GB of 40 GB:
# Llama 3.1 8B weights ~16 GB + KV cache); the rest (~18 GB) is for ASR + TTS, which need ~5-7 GB.
GPU="${LELA_GPU:-0}"
export LELA_VLLM_MEM="${LELA_VLLM_MEM:-0.55}"

VLLM_PID=""
cleanup() {
    if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "--> Stopping vLLM ($VLLM_PID)..."
        kill -- "-$VLLM_PID" 2>/dev/null || kill "$VLLM_PID"   # whole process group: vLLM spawns workers
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
# Registered BEFORE anything blocks; in the original it came after uvicorn and never ran.
trap cleanup EXIT INT TERM

echo "--> 1. Starting vLLM engine on GPU $GPU (127.0.0.1:8001), log: vllm.log"
CUDA_VISIBLE_DEVICES="$GPU" setsid ./start_vllm.sh > vllm.log 2>&1 &
VLLM_PID=$!

echo "Waiting for vLLM to become healthy..."
for _ in $(seq 1 200); do                       # ~10 minutes max
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM exited during startup. Last lines of vllm.log:"; tail -n 30 vllm.log; exit 1
    fi
    if curl -sf http://127.0.0.1:8001/health > /dev/null; then break; fi
    sleep 3
done
curl -sf http://127.0.0.1:8001/health > /dev/null || { echo "vLLM not healthy after timeout"; exit 1; }
echo "--> vLLM is ready!"

[ -f prompts/kannada_teacher.wav ] || { echo "Missing prompts/kannada_teacher.wav"; exit 1; }

echo "--> 2. Starting LeLa (ASR + TTS on GPU $GPU) on port 7800 (open via https or an SSH tunnel to localhost; see review)"
CUDA_VISIBLE_DEVICES="$GPU" uvicorn app:app --host "${LELA_HOST:-127.0.0.1}" --port 7800
