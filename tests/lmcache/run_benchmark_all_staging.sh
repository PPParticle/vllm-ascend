#!/bin/bash
# Run all working set benchmarks with staging config.
# Each test point gets a fresh server and clean cache.
#
# Usage:
#   source /usr/local/Ascend/cann-8.5.1/set_env.sh
#   bash tests/lmcache/run_benchmark_all_staging.sh
#
#   START_FROM=ctx16k_ws8 bash tests/lmcache/run_benchmark_all_staging.sh

set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required environment
: "${VLLM_PATH:?Set VLLM_PATH to vllm source directory}"
: "${VLLM_ASCEND_PATH:?Set VLLM_ASCEND_PATH to vllm-ascend source directory}"
: "${PYTHON:?Set PYTHON to python interpreter path}"
: "${MODEL_PATH:?Set MODEL_PATH to model directory}"
: "${SSD_PATH:?Set SSD_PATH to SSD cache directory}"
: "${RESULTS_DIR:?Set RESULTS_DIR to results output directory}"

# CANN environment
source /usr/local/Ascend/cann-8.5.1/set_env.sh 2>/dev/null || true

PORT=8100
LOG="$RESULTS_DIR/benchmark_server.log"

# Phase 2 config: with staging isolation
CPU_SIZE=8
STAGING_SIZE=2
SSD_SIZE=30

TEST_POINTS=(
    ctx8k_ws8
    ctx8k_ws16
    ctx8k_ws32
    ctx16k_ws8
    ctx16k_ws16
    ctx16k_ws32
    ctx32k_ws8
    ctx32k_ws16
)

mkdir -p "$RESULTS_DIR" "$SSD_PATH"

kill_server() {
    echo "  [setup] Killing existing server..."
    lsof -ti :$PORT 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    sleep 3
    ps aux | grep "VLLM::EngineCore" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
    echo "  [setup] Server stopped."
}

clear_cache() {
    echo "  [setup] Clearing SSD cache..."
    rm -rf "$SSD_PATH"/* 2>/dev/null || true
}

start_server() {
    HOT_SIZE=$((CPU_SIZE - STAGING_SIZE))
    echo "  [setup] Starting vLLM server (CPU=${CPU_SIZE}GB [hot=${HOT_SIZE}GB + staging=${STAGING_SIZE}GB], SSD=${SSD_SIZE}GB)..."
    PYTHONPATH=$VLLM_PATH:$VLLM_ASCEND_PATH:$PYTHONPATH \
    VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    LMCACHE_LOCAL_CPU=True \
    LMCACHE_MAX_LOCAL_CPU_SIZE=$CPU_SIZE \
    LMCACHE_STAGING_CPU_SIZE=$STAGING_SIZE \
    LMCACHE_LOCAL_DISK=file://$SSD_PATH/ \
    LMCACHE_MAX_LOCAL_DISK_SIZE=$SSD_SIZE \
    LMCACHE_CHUNK_SIZE=128 \
    LMCACHE_SAVE_UNFULL_CHUNK=True \
    nohup $PYTHON -u -m vllm.entrypoints.openai.api_server \
        --port $PORT \
        --model $MODEL_PATH \
        --trust-remote-code \
        --block-size 128 \
        --dtype float16 \
        --max-model-len 32768 \
        --max-num-batched-tokens 32768 \
        --max-num-seqs 8 \
        --gpu-memory-utilization 0.90 \
        --enforce-eager \
        --no-enable-prefix-caching \
        --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}' \
        > "$LOG" 2>&1 &

    SERVER_PID=$!
    echo "  [setup] Server PID: $SERVER_PID, waiting for ready..."

    for i in $(seq 1 60); do
        if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
            RESP=$(curl -sf http://localhost:$PORT/v1/chat/completions \
                -H "Content-Type: application/json" \
                -d "{\"model\":\"$MODEL_PATH\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":2,\"temperature\":0}" \
                --max-time 30 2>/dev/null || echo "FAIL")
            if echo "$RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d['choices'][0]['message']['content']" 2>/dev/null; then
                echo "  [setup] Server fully ready after ~${i}x5s"
                return 0
            fi
        fi
        sleep 5
    done
    echo "  [setup] ERROR: Server failed to start!"
    tail -20 "$LOG"
    return 1
}

START_FROM="${START_FROM:-}"
skipping=0
if [ -n "$START_FROM" ]; then
    skipping=1
fi

echo "========================================"
echo "LMCache Working Set Benchmark - Staging"
echo "Config: CPU=${CPU_SIZE}GB (hot=$((CPU_SIZE - STAGING_SIZE))GB + staging=${STAGING_SIZE}GB), SSD=${SSD_SIZE}GB"
echo "Test points: ${#TEST_POINTS[@]}"
echo "========================================"
echo ""

for tp in "${TEST_POINTS[@]}"; do
    if [ $skipping -eq 1 ]; then
        if [ "$tp" = "$START_FROM" ]; then
            skipping=0
        else
            echo "SKIP: $tp (before START_FROM)"
            continue
        fi
    fi

    echo ""
    echo "############################################"
    echo "  Test: $tp"
    echo "############################################"

    kill_server
    clear_cache
    start_server || { echo "FAILED to start server for $tp, skipping"; continue; }

    echo "  Running benchmark for $tp..."
    $PYTHON "$SCRIPT_DIR/benchmark_working_set.py" --no-manage-server --only "$tp" \
        2>&1 | tee "$RESULTS_DIR/${tp}_staging_output.log"

    echo "  Done: $tp"
done

kill_server
echo ""
echo "========================================"
echo "All test points completed!"
echo "Results in: $RESULTS_DIR/"
echo "========================================"
