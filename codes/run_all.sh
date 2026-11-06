#!/usr/bin/env bash
# Starts vLLM, waits until it is healthy, then runs the LeLa server.
# Ctrl-C (or uvicorn exiting) stops vLLM too.
set -euo pipefail
cd "$(dirname "$0")"

VENV="${LELA_VENV:-$HOME/.virtualenvs/lelaenv}"
if [ -f "$VENV/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
elif [ -d "lela-env" ]; then
    # shellcheck disable=SC1091
    source lela-env/bin/activate
else
    echo "No venv found at $VENV (set LELA_VENV to point at yours)"; exit 1
fi

# Everything runs on one ~11 GB GPU (RTX 2080 Ti, 11264 MiB; LELA_GPU picks it, default 0). Budget:
#   vLLM 72% (~7.9 GB): Llama 3.1 8B AWQ 4-bit weights ~5.4 GiB + warm-up + KV cache.
#     At 60% vLLM had no room left for the KV cache ("No available memory for the cache blocks").
#   The rest (~3 GB) for the app: Whisper small (~0.6 GB), MMS-TTS (~0.2 GB) and CUDA contexts.
#   IndicConformer (~2.5 GB on GPU) does not fit too, so it runs on CPU by default here
#   (slower, roughly 1-3 s per utterance). Try LELA_CONFORMER_DEVICE=auto to put it on the GPU
#   with a smaller vLLM share (LELA_VLLM_MEM=0.66 LELA_VLLM_MAX_LEN=2048) if latency matters.
#   IndicF5 (LELA_TTS=indicf5) needs ~2 GB more and will not fit on this card.
# app.py must ask vLLM for the same model name it serves, so both read LELA_LLM_MODEL.
GPU="${LELA_GPU:-0}"
export LELA_VLLM_MEM="${LELA_VLLM_MEM:-0.72}"
export LELA_LLM_MODEL="${LELA_LLM_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"
export LELA_TTS="${LELA_TTS:-mms}"
export LELA_CONFORMER_DEVICE="${LELA_CONFORMER_DEVICE:-cpu}"

# Fail early with a clear message instead of a vLLM traceback.
if ! nvidia-smi > /dev/null 2>&1; then
    echo "nvidia-smi fails. Usually a driver/library version mismatch after a driver update:"
    echo "reboot (or reload the nvidia kernel modules), then check that nvidia-smi works."
    nvidia-smi || true
    exit 1
fi
USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
if [ "${USED_MB:-0}" -gt 1000 ]; then
    echo "!! GPU $GPU already has ${USED_MB} MiB in use (an old vLLM still running?). LeLa needs nearly all of it."
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i "$GPU" || true
    echo "Stop those processes first, or continue at your own risk in 5 s (Ctrl-C to abort)."
    sleep 5
fi

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

echo "--> 1. Starting vLLM ($LELA_LLM_MODEL) on GPU $GPU (127.0.0.1:8001), log: vllm.log"
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

if [ "$LELA_TTS" = "indicf5" ] && [ ! -f prompts/kannada_teacher.wav ]; then
    echo "Missing prompts/kannada_teacher.wav (needed for LELA_TTS=indicf5)"; exit 1
fi

echo "--> 2. Starting LeLa (ASR + TTS on GPU $GPU) on port 7800 (open via https or an SSH tunnel to localhost; see review)"
CUDA_VISIBLE_DEVICES="$GPU" uvicorn app:app --host "${LELA_HOST:-127.0.0.1}" --port 7800
