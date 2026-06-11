#!/bin/bash
# LMCache Working Set Benchmark Runner
# Sources CANN environment and runs the benchmark script.
#
# Usage:
#   source /usr/local/Ascend/cann-8.5.1/set_env.sh
#   bash tests/lmcache/run_benchmark_ws.sh
#
#   # Or run specific test points:
#   bash tests/lmcache/run_benchmark_ws.sh --only ctx8k_ws8 ctx16k_ws16
#
#   # Connect to already-running server:
#   bash tests/lmcache/run_benchmark_ws.sh --no-manage-server

set -euo pipefail

# Resolve script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Required environment (set externally or defaults)
: "${VLLM_PATH:?Set VLLM_PATH to vllm source directory}"
: "${VLLM_ASCEND_PATH:?Set VLLM_ASCEND_PATH to vllm-ascend source directory}"
: "${PYTHON:?Set PYTHON to python interpreter path (e.g. /path/to/venv/bin/python3)}"
: "${MODEL_PATH:?Set MODEL_PATH to model directory}"
: "${SSD_PATH:?Set SSD_PATH to SSD cache directory}"
: "${RESULTS_DIR:?Set RESULTS_DIR to results output directory}"

# CANN environment (required for NPU)
source /usr/local/Ascend/cann-8.5.1/set_env.sh 2>/dev/null || true

mkdir -p "$RESULTS_DIR" "$SSD_PATH"

echo "========================================"
echo "LMCache Working Set Benchmark"
echo "========================================"
echo "Python:    $($PYTHON --version)"
echo "Model:     $MODEL_PATH"
echo "SSD:       $SSD_PATH"
echo "Results:   $RESULTS_DIR"
echo "========================================"
echo ""

# Pass config to benchmark script via environment
export VLLM_PATH VLLM_ASCEND_PATH PYTHON MODEL_PATH SSD_PATH RESULTS_DIR

exec "$PYTHON" "$SCRIPT_DIR/benchmark_working_set.py" "$@"
