#!/usr/bin/env bash
# Starts the Ollama server with real request-level parallelism enabled.
#
# By default `ollama serve` auto-detects OLLAMA_NUM_PARALLEL and, on this
# machine's 16GB GPU with qwen2.5:14b-instruct-q4_K_M loaded (~10-12GB),
# auto-detection lands on effectively serial request handling -- confirmed
# empirically (4 concurrent requests took ~23s, matching 4x a single
# request's latency, not the ~10s true parallelism would produce).
# OLLAMA_NUM_PARALLEL=4 was benchmarked as the highest value that still fits
# this GPU at CANDIDATE_RANKING_OLLAMA_NUM_CTX=8192 (~14.7GB used, ~1.7GB
# headroom, 100% GPU, no CPU fallback). Going higher risks OOM or falling
# back to (much slower) CPU/shared-memory execution.
#
# Usage: ./scripts/start_ollama.sh
# Restart after changing this file's OLLAMA_NUM_PARALLEL value by re-running
# it -- it will not stop an already-running `ollama serve` for you.

set -euo pipefail

if curl -s -o /dev/null -m 1 http://localhost:11434/api/version; then
    echo "ollama serve is already running on :11434 -- stop it first (kill the" >&2
    echo "existing process) if you need to change OLLAMA_NUM_PARALLEL." >&2
    exit 1
fi

export OLLAMA_NUM_PARALLEL=4

nohup ollama serve > "${OLLAMA_LOG_PATH:-/tmp/ollama_serve.log}" 2>&1 &
disown

for _ in $(seq 1 20); do
    if curl -s -o /dev/null -m 1 http://localhost:11434/api/version; then
        echo "ollama serve up (pid $!) with OLLAMA_NUM_PARALLEL=$OLLAMA_NUM_PARALLEL"
        exit 0
    fi
    sleep 1
done

echo "ollama serve did not come up within 20s -- check ${OLLAMA_LOG_PATH:-/tmp/ollama_serve.log}" >&2
exit 1
