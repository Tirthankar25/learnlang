#!/usr/bin/env bash
# Serves the tutor LLM on 127.0.0.1:8001 (OpenAI-compatible API).
# Auth: no token in this file. Run `hf auth login` (or `huggingface-cli login`) once,
# or export HF_TOKEN in your shell / an untracked .env before starting.
#
# Sized for a single ~11 GB GPU (RTX 2080 Ti) shared with ASR + TTS (see run_all.sh):
#   - Llama 3.1 8B Instruct, 4-bit AWQ: ~5.7 GB of weights instead of ~16 GB in bf16.
#   - AWQ kernels need float16 (vLLM rejects bfloat16 for AWQ; Turing has no bf16 anyway).
#     On Turing (compute 7.5) vLLM uses the plain AWQ kernel, not the faster Marlin one (needs 8.0).
#   - 3072-token context and 4 concurrent sequences keep the KV cache and warm-up small.
#     A LeLa turn is ~1-2k tokens (lesson state + last 12 lines + 400-token reply).
# On a bigger GPU you can go back to the full model:
#   LELA_LLM_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct LELA_VLLM_DTYPE=bfloat16 LELA_VLLM_MEM=0.55
set -euo pipefail

MODEL="${LELA_LLM_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4}"

# 127.0.0.1: only app.py needs this port; 0.0.0.0 exposed an unauthenticated LLM to the network.
# --enforce-eager skips CUDA graphs, which saves ~0.5-1 GB; keep it on a 12 GB card.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype "${LELA_VLLM_DTYPE:-float16}" \
    --host 127.0.0.1 \
    --port 8001 \
    --gpu-memory-utilization "${LELA_VLLM_MEM:-0.72}" \
    --max-model-len "${LELA_VLLM_MAX_LEN:-3072}" \
    --max-num-seqs "${LELA_VLLM_MAX_SEQS:-4}" \
    --enforce-eager
