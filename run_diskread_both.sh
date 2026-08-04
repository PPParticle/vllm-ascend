#!/bin/bash
# Reproducible APC-miss + LocalCPU-hot-miss + LMCache-disk-hit benchmark.
# Default mode is "pressure": T -> drain -> E -> immediately measure T once.
# Set MODE=quiescent to drain E before measuring the plain disk-only path.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
cd "$SCRIPT_DIR" || exit 1
PY=${PY:-/home/fcqiao/vllm-ascend-v018-env/bin/python}
MODEL_PATH=${MODEL_PATH:-/data/models/Qwen2.5-14B-Instruct}
RESULT_DIR=${RESULT_DIR:-$SCRIPT_DIR/perf_results}
mkdir -p "$RESULT_DIR"
export HF_HOME=/data/hf-cache
export HF_HUB_CACHE=/data/hf-cache/hub
export HF_HUB_OFFLINE=1
export SSD_PATH=/data/lmcache-disk/

MODE=${MODE:-pressure}
REPEATS=${REPEATS:-1}
APC_BLOCKS=${APC_BLOCKS:-256}
HOT_BLOCKS_MAX=${HOT_BLOCKS_MAX:-256}
TARGET_CHUNKS=${TARGET_CHUNKS:-96}
EVICT_BLOCKS=${EVICT_BLOCKS:-384}
EVICT_COUNT=${EVICT_COUNT:-256}
CONCURRENCY=${CONCURRENCY:-8}
DISK_GIB=${DISK_GIB:-18}
MAX_TOKENS=${MAX_TOKENS:-256}
TTFT_SLO_SECONDS=${TTFT_SLO_SECONDS:-0}
LATENCY_SLO_SECONDS=${LATENCY_SLO_SECONDS:-0}

SUMMARY="$SCRIPT_DIR/diskread_both.txt"
SUMMARY_JSONL="$SCRIPT_DIR/diskread_both.jsonl"
printf "dataset\tconfig\tmode\trepeat\tttft_mean\tttft_p50\tttft_p95\tttft_p99\tlatency_mean\tlatency_p50\tlatency_p95\tlatency_p99\tqps\tserved_prompt_tps\toutput_tps\ttotal_served_tps\tapc_miss_tokens\texternal_planned_miss_tokens\tlmcache_retrieve_shortfall\talloc_fail\talloc_retry\tno_evict\tnot_busy_looping\tsleep_events\tsleep_sec\tevict_fail_metric\tdisk_prefetch_chunks\tretrieved_tokens_mean\tdisk_read_mib\tdisk_read_ops\trequest_failures\tapc_miss_valid\tpressure_valid\n" > "$SUMMARY"
: > "$SUMMARY_JSONL"

ACTIVE_SPID=""
ACTIVE_PGID=""

stop_server() {
  if [ -z "$ACTIVE_SPID" ]; then
    return
  fi
  if [ -n "$ACTIVE_PGID" ]; then
    kill -TERM -- "-$ACTIVE_PGID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      pgrep -g "$ACTIVE_PGID" >/dev/null 2>&1 || break
      sleep 1
    done
    if pgrep -g "$ACTIVE_PGID" >/dev/null 2>&1; then
      kill -KILL -- "-$ACTIVE_PGID" 2>/dev/null || true
    fi
  elif kill -0 "$ACTIVE_SPID" 2>/dev/null; then
    kill -TERM "$ACTIVE_SPID" 2>/dev/null || true
  fi
  wait "$ACTIVE_SPID" 2>/dev/null || true
  ACTIVE_SPID=""
  ACTIVE_PGID=""
}

trap stop_server EXIT
trap 'stop_server; exit 130' INT
trap 'stop_server; exit 143' TERM

reset_disk() {
  mkdir -p "$SSD_PATH"
  local resolved
  resolved=$(readlink -f -- "$SSD_PATH")
  if [ "$resolved" != "/data/lmcache-disk" ]; then
    echo "Refusing to clear unexpected SSD_PATH: $resolved" >&2
    return 1
  fi
  find "$resolved" -mindepth 1 -delete
}

append_summary() {
  local json_path=$1
  "$PY" - "$json_path" >> "$SUMMARY" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    row = json.load(handle)

tag_parts = row["tag"].rsplit("_r", 1)
repeat = tag_parts[1] if len(tag_parts) == 2 else "?"
config = row.get("config", tag_parts[0].rsplit("_", 1)[-1])
ttft = row["ttft_sec"]
latency = row["latency_sec"]
throughput = row["throughput"]
miss = row["cache_miss_analysis"]
events = row["log_analysis"]["events"]
disk_read = row["log_analysis"]["disk_read"]

def number(value, digits=6):
    return "" if value is None else f"{value:.{digits}f}"

values = [
    row["dataset"],
    config,
    row["mode"],
    repeat,
    number(ttft["mean"]),
    number(ttft["p50"]),
    number(ttft["p95"]),
    number(ttft["p99"]),
    number(latency["mean"]),
    number(latency["p50"]),
    number(latency["p95"]),
    number(latency["p99"]),
    number(throughput["successful_qps"]),
    number(throughput["prompt_tokens_per_sec"], 2),
    number(throughput["output_tokens_per_sec"], 2),
    number(throughput["total_tokens_per_sec"], 2),
    number(miss["apc"]["misses"], 0),
    number(miss["external"]["misses"], 0),
    number(miss["lmcache_retrieve_tokens"]["retrieve_shortfall"], 0),
    str(events["alloc_fail"]),
    str(events["alloc_retry"]),
    str(events["no_evict_candidate"]),
    str(events["not_busy_looping"]),
    str(events["allocator_sleep"]),
    number(row["log_analysis"]["allocator_sleep_seconds"], 3),
    number(row["allocator_analysis"]["local_cpu_evict_failed_delta"], 0),
    str(row["log_analysis"]["disk_prefetch_chunks"]),
    number(row["log_analysis"]["retrieved_length_tokens"]["mean"], 1),
    number(disk_read["mib"], 2),
    str(disk_read["operations"]),
    str(row["failures"]),
    str(row["apc_miss_valid"]),
    str(row["pressure_valid"]),
]
print("\t".join(values))
PY
}

run_one() {
  local dataset=$1
  local config=$2
  local cpu_gib=$3
  local staging_gib=$4
  local repeat=$5
  local label
  label="${dataset}_${MODE}_${config}_r$(printf '%02d' "$repeat")"
  local server_log="$SCRIPT_DIR/srv_${label}.log"
  local bench_log="/tmp/${label}.txt"
  local result_json="$RESULT_DIR/diskread_${dataset}_${label}.json"

  echo "############## $label CPU=${cpu_gib}G STG=${staging_gib}G ##############"
  if ! "$PY" -c 'import socket; s=socket.socket(); s.bind(("0.0.0.0", 8100)); s.close()'; then
    echo "[$label] port 8100 is already occupied; refusing to use an old server" >&2
    return 1
  fi
  reset_disk || return 1

  setsid env \
    CONFIG="$config" \
    CPU="$cpu_gib" \
    STAGING="$staging_gib" \
    DISK="$DISK_GIB" \
    KVBLOCKS="$APC_BLOCKS" \
    MAX_NUM_SEQS="$CONCURRENCY" \
    MAX_NUM_BATCHED_TOKENS=16384 \
    MODEL_PATH="$MODEL_PATH" \
    SSD_PATH="$SSD_PATH" \
    bash "$SCRIPT_DIR/serve_readhot.sh" > "$server_log" 2>&1 &
  ACTIVE_SPID=$!
  ACTIVE_PGID=$(ps -o pgid= -p "$ACTIVE_SPID" | tr -d ' ')

  local ready=0
  for _ in $(seq 1 200); do
    if curl -fsS http://localhost:8100/health >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "$ACTIVE_SPID" 2>/dev/null; then
      echo "[$label] server died during startup" >&2
      tail -30 "$server_log"
      stop_server
      return 1
    fi
    sleep 3
  done
  if [ "$ready" -ne 1 ]; then
    echo "[$label] server health timeout" >&2
    tail -30 "$server_log"
    stop_server
    return 1
  fi

  echo "[$label] ready; $(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$server_log" | head -1)"
  if [ "$config" = "stg" ]; then
    if ! grep -q "Staging allocator created:" "$server_log" || \
       ! grep -q "hot_cache reduced from" "$server_log"; then
      echo "[$label] staging patch was not activated; refusing invalid run" >&2
      tail -50 "$server_log"
      stop_server
      return 1
    fi
  fi
  if [ "$config" != "baseline" ] && ! grep -qiE "O_DIRECT.*True|use_odirect.*true" "$server_log"; then
    echo "[$label] WARNING: could not confirm O_DIRECT=True from startup log" >&2
  fi

  local disk_args=()
  if [ "$config" != "baseline" ]; then
    disk_args+=(--expect-disk)
  fi

  SERVER_LOG="$server_log" SSD_PATH="$SSD_PATH" PYTHONUNBUFFERED=1 \
    "$PY" "$SCRIPT_DIR/bench_readhot.py" \
      --dataset "$dataset" \
      --tokenizer "$MODEL_PATH" \
      --warmup 64 \
      --rounds 1 \
      --concurrency "$CONCURRENCY" \
      --warmup-conc 1 \
      --warmup-mt 1 \
      --max-tokens "$MAX_TOKENS" \
      --mode "$MODE" \
      --target-chunks "$TARGET_CHUNKS" \
      --evict-count "$EVICT_COUNT" \
      --evict-blocks "$EVICT_BLOCKS" \
      --evict-concurrency "$CONCURRENCY" \
      --evict-mt 1 \
      --apc-blocks "$APC_BLOCKS" \
      --hot-blocks "$HOT_BLOCKS_MAX" \
      --chunk-size 128 \
      --chunk-mib 24 \
      --disk-gib "$DISK_GIB" \
      --cpu-gib "$cpu_gib" \
      --staging-gib "$staging_gib" \
      --ttft-slo-seconds "$TTFT_SLO_SECONDS" \
      --latency-slo-seconds "$LATENCY_SLO_SECONDS" \
      --summary-jsonl "$SUMMARY_JSONL" \
      --out "$RESULT_DIR" \
      --config "$config" \
      --tag "$label" \
      "${disk_args[@]}" 2>&1 | tee "$bench_log"
  local bench_status=${PIPESTATUS[0]}

  if [ "$bench_status" -eq 0 ] && [ -f "$result_json" ]; then
    if ! append_summary "$result_json"; then
      echo "[$label] failed to append summary" >&2
      bench_status=1
    fi
  else
    echo "[$label] benchmark failed with status $bench_status" >&2
  fi

  stop_server
  sleep 3
  return "$bench_status"
}

run_dataset() {
  local dataset=$1
  for repeat in $(seq 1 "$REPEATS"); do
    run_one "$dataset" baseline 0 0 "$repeat" || return 1

    # Alternate nostg/stg order across repetitions to reduce order bias.
    if [ $((repeat % 2)) -eq 1 ]; then
      run_one "$dataset" nostg 6 0 "$repeat" || return 1
      run_one "$dataset" stg   6 3 "$repeat" || return 1
    else
      run_one "$dataset" stg   6 3 "$repeat" || return 1
      run_one "$dataset" nostg 6 0 "$repeat" || return 1
    fi
  done
}

case "$MODE" in
  pressure|quiescent) ;;
  *)
    echo "MODE must be pressure or quiescent; got: $MODE" >&2
    exit 2
    ;;
esac

run_dataset gsm8k || exit 1
run_dataset gpqa || exit 1

echo
echo "============== SUMMARY =============="
cat "$SUMMARY"
echo "Raw JSONL: $SUMMARY_JSONL"
echo "DONE"
