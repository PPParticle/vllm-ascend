#!/bin/bash
# SCBench T/E/T disk-retrieve bench (16k context). Same methodology as gsm8k/gpqa.
# ctx_len=16384 -> each prompt ~129 chunks (3.1 GiB). T=1, E=3 (>= APC+hot 320 blocks).
# T+E = 4 prompts = ~12.4 GiB < disk 18 GiB (budget 14.4) -> E does NOT evict T from disk.
# MAX_MODEL_LEN=20480 (fits 16k prompt + query + 256 output). REPEATS=3 for more samples.
set -uo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) && cd "$SCRIPT_DIR"
PY=/home/fcqiao/vllm-ascend-v018-env/bin/python
export HF_HOME=/data/hf-cache HF_HUB_CACHE=/data/hf-cache/hub HF_HUB_OFFLINE=1
export SSD_PATH=/data/lmcache-disk/
MODEL_PATH=/data/models/Qwen2.5-14B-Instruct
RESULT_DIR=$SCRIPT_DIR/perf_results
mkdir -p "$RESULT_DIR"
SUMMARY=$SCRIPT_DIR/scbench_diskread.txt
: > "$SUMMARY"
REPEATS=${REPEATS:-2}

ACTIVE_SPID=""; ACTIVE_PGID=""
stop_server() { [ -n "$ACTIVE_SPID" ] && kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true; ACTIVE_SPID=""; }
trap stop_server EXIT

run_one() {
  local config=$1 cpu=$2 stg=$3 rep=$4
  local label="scbench_pressure_${config}_r$(printf '%02d' "$rep")"
  local slog="$SCRIPT_DIR/srv_${label}.log"; local result="$RESULT_DIR/diskread_scbench_${label}.json"
  echo "############## $label CPU=${cpu}G STG=${stg}G ctx=16k ##############"
  find /data/lmcache-disk -mindepth 1 -delete 2>/dev/null
  setsid env CONFIG=$config CPU=$cpu STAGING=$stg DISK=18 KVBLOCKS=256 \
    MAX_NUM_SEQS=8 MAX_MODEL_LEN=20480 MAX_NUM_BATCHED_TOKENS=20480 \
    MODEL_PATH=$MODEL_PATH SSD_PATH=$SSD_PATH bash serve_readhot.sh > "$slog" 2>&1 &
  ACTIVE_SPID=$!; ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_SPID" | tr -d ' ')
  for _ in $(seq 1 240); do
    curl -fsS http://localhost:8100/health >/dev/null 2>&1 && break
    kill -0 "$ACTIVE_SPID" 2>/dev/null || { echo "[$label] DIED"; tail -20 "$slog"; return 1; }
    sleep 3
  done
  echo "[$label] ready; $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$slog" | head -1)"
  local disk_args=(); [ "$config" != "baseline" ] && disk_args+=(--expect-disk)
  SERVER_LOG="$slog" SSD_PATH=$SSD_PATH PYTHONUNBUFFERED=1 "$PY" bench_readhot.py \
    --dataset scbench --tokenizer "$MODEL_PATH" \
    --warmup 2 --rounds 1 --concurrency 8 --warmup-conc 1 --warmup-mt 1 \
    --max-tokens 256 --ctx-len 16384 --cache-dir /data/hf-cache/hub \
    --mode pressure --target-chunks 0 --evict-count 4 --evict-blocks 384 \
    --evict-concurrency 8 --evict-mt 1 --chunk-size 128 \
    --apc-blocks 256 --hot-blocks 256 --chunk-mib 24 --disk-gib 18 --disk-headroom 0.9 \
    --drain-stable-polls 8 \
    --cpu-gib "$cpu" --staging-gib "$stg" \
    --config "$config" --tag "$label" --out "$RESULT_DIR" "${disk_args[@]}" 2>&1 | tee /tmp/${label}.txt
  [ -f "$result" ] && python3 -c "
import json;d=json.load(open('$result'))
print('  -> TTFT=%.3fs cond=%s disk_read=%.0fMiB acc=%.0f%% staging_fail=%s'%(
  d['ttft_sec']['mean'],d['conditioning_valid'],d['log_analysis']['disk_read']['mib'],
  d['accuracy']*100,d['allocator_analysis']['staging_alloc_failures']))" | tee -a "$SUMMARY"
  stop_server; sleep 5
}

for rep in $(seq 1 "$REPEATS"); do
  run_one baseline 0 0 "$rep"
  run_one nostg   6 0 "$rep"
  run_one stg     6 3 "$rep"
done
echo "============== SCBENCH 16k SUMMARY =============="; cat "$SUMMARY"; echo "DONE"
