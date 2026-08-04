#!/bin/bash
# APC-on server for the T/E/T LMCache disk-read benchmark.
# CONFIG in {baseline, nostg, stg}.
# LMCACHE_MAX_LOCAL_CPU_SIZE is total CPU memory; staging is subtracted from it.

if [ ! -f /home/fcqiao/vllm-ascend-v018-env/bin/activate ]; then
  echo "vLLM environment not found" >&2
  exit 1
fi
# shellcheck source=/dev/null
source /home/fcqiao/vllm-ascend-v018-env/bin/activate || exit 1
for script in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  /usr/local/Ascend/nnal/atb/set_env.sh \
  /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh
do
  # shellcheck source=/dev/null
  [ -f "$script" ] && source "$script" || true
done

# Vendor environment scripts are not guaranteed to be nounset-safe.
set -u

ASCEND_ARCH=$(uname -i)
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/local/Ascend/ascend-toolkit/latest/${ASCEND_ARCH}-linux/devlib"
export PYTHONPATH="/home/fcqiao/vllm-v018:/home/fcqiao/vllm-ascend-v0.18.0:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONHASHSEED=0

CONFIG=${CONFIG:-stg}
CPU=${CPU:-6}
STAGING=${STAGING:-3}
DISK=${DISK:-18}
KVBLOCKS=${KVBLOCKS-256}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
MODEL_PATH=${MODEL_PATH:-/data/models/Qwen2.5-14B-Instruct}
SSD_PATH=${SSD_PATH:-/data/lmcache-disk/}
# DEBUG is required for per-chunk disk bandwidth and allocator retry analysis.
export LMCACHE_LOG_LEVEL=${LMCACHE_LOG_LEVEL:-DEBUG}

case "$CONFIG" in
  baseline|nostg|stg) ;;
  *)
    echo "CONFIG must be one of baseline, nostg, stg; got: $CONFIG" >&2
    exit 2
    ;;
esac

COMMON=(
  --port 8100
  --model "$MODEL_PATH"
  --served-model-name qwen2.5-14b
  --trust-remote-code
  --block-size 128
  --dtype float16
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization 0.90
  --enforce-eager
  --enable-prefix-caching
)
if [ -n "$KVBLOCKS" ]; then
  COMMON+=(--num-gpu-blocks-override "$KVBLOCKS")
fi

echo "[serve] config=$CONFIG cpu_total=${CPU}GiB staging=${STAGING}GiB disk=${DISK}GiB"
echo "[serve] apc_blocks=$KVBLOCKS max_num_seqs=$MAX_NUM_SEQS max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
python3 -c 'import lmcache, vllm; print("[serve] lmcache=", getattr(lmcache, "__file__", "?"), "version=", getattr(lmcache, "__version__", "?")); print("[serve] vllm=", getattr(vllm, "__file__", "?"), "version=", getattr(vllm, "__version__", "?"))' || true

if [ "$CONFIG" = "baseline" ]; then
  unset LMCACHE_LOCAL_CPU LMCACHE_MAX_LOCAL_CPU_SIZE LMCACHE_STAGING_CPU_SIZE
  unset LMCACHE_LOCAL_DISK LMCACHE_MAX_LOCAL_DISK_SIZE LMCACHE_EXTRA_CONFIG
  export VLLM_PLUGINS=ascend,ascend_model_loader,ascend_service_profiling
  exec python3 -m vllm.entrypoints.openai.api_server "${COMMON[@]}"
fi

export VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="$CPU"
export LMCACHE_STAGING_CPU_SIZE="$STAGING"
export LMCACHE_LOCAL_DISK="file://${SSD_PATH%/}/"
export LMCACHE_MAX_LOCAL_DISK_SIZE="$DISK"
export LMCACHE_CHUNK_SIZE=128
export LMCACHE_SAVE_UNFULL_CHUNK=True
# O_DIRECT breaks writes on the /data loop device (0-byte files); disabled.
if [ -z "${LMCACHE_EXTRA_CONFIG:-}" ]; then
  export LMCACHE_EXTRA_CONFIG='{"use_odirect": false}'
fi

exec python3 -m vllm.entrypoints.openai.api_server "${COMMON[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
