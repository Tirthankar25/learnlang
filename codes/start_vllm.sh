#!/usr/bin/env bash

# If your Hugging Face access token isn't saved globally, uncomment:

# Launches Llama-3.1-8B on port 8001 with bounded memory
# python3 -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --port 8001 \
#     --host 0.0.0.0 \
#     --gpu-memory-utilization 0.50 \
#     --max-model-len 4096 \
#     --trust-remote-code \
#     --enforce-eager

python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --port 8001 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.55 \
    --max-model-len 4096 \
    --trust-remote-code \
    --enforce-eager

# vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --port 8001 \
#     --gpu-memory-utilization 0.55 \
#     --max-model-len 4096 \
#     --dtype bfloat16
