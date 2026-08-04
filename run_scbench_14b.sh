#!/bin/bash
# SCBench (SharedContextBench) on Qwen2.5-14B — cold vs warm TTFT, per test point fresh server.
# KV/token = 192KB (48L x 8 KV heads x 128 dim x fp16 x 2).
#
# SIZING LESSON (verified by repro): store-time puts go to BOTH cpu and disk; if
# cpu+disk cannot hold the working set with headroom, disk self-evicts early data
# and warm-miss re-stores cascade-evict the next warm targets -> 0% hits, worse
# than recompute. Fix: CPU layer >= working set (lookups then hit CPU, disk churn
# is irrelevant).
#
#   ctx8k_ws8   12.9G -> CPU=14 SSD=6
#   ctx8k_ws16  25.8G -> CPU=27 SSD=6
#   ctx16k_ws8  25.8G -> CPU=27 SSD=6
#   ctx8k_ws32  51.6G -> CPU=40 SSD=17 (~78% in CPU)
#   ctx16k_ws16 51.6G -> CPU=40 SSD=17
#   ctx32k_ws8  51.6G -> CPU=40 SSD=17
# Two phases: nostg (no staging), stg (LMCACHE_STAGING_CPU_SIZE=2).
cd /home/fcqiao/v018-tests
source /home/fcqiao/vllm-ascend-v018-env/bin/activate
for s in /usr/local/Ascend/ascend-toolkit/set_env.sh /usr/local/Ascend/nnal/atb/set_env.sh /usr/local/Ascend/nnal/atb/latest/atb/set_env.sh; do [ -f "$s" ] && source "$s" || true; done
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/$(uname -i)-linux/devlib
export VLLM_PATH=/home/fcqiao/vllm-v018
export VLLM_ASCEND_PATH=/home/fcqiao/vllm-ascend-v0.18.0
export PYTHON=/home/fcqiao/vllm-ascend-v018-env/bin/python
export MODEL_PATH=/data/models/Qwen2.5-14B-Instruct
export SSD_PATH=/data/lmcache-disk/
export HF_HOME=/data/hf-cache HF_HUB_CACHE=/data/hf-cache/hub HF_HUB_OFFLINE=1
export PYTHONHASHSEED=0

# test_point cpu_gb ssd_gb
POINTS=(
  "ctx8k_ws8 14 6"
  "ctx8k_ws16 27 6"
  "ctx16k_ws8 27 6"
  "ctx8k_ws32 40 17"
  "ctx16k_ws16 40 17"
  "ctx32k_ws8 40 17"
)

for PHASE in nostg stg; do
  if [ "$PHASE" = "stg" ]; then export LMCACHE_STAGING_CPU_SIZE=2; else unset LMCACHE_STAGING_CPU_SIZE; fi
  export RESULTS_DIR=/home/fcqiao/v018-tests/perf_results/scbench_${PHASE}
  mkdir -p "$RESULTS_DIR"
  find "$SSD_PATH" -mindepth 1 -delete 2>/dev/null   # fresh disk per phase (py skips clear on first point)
  echo "================ PHASE: $PHASE ================"
  for spec in "${POINTS[@]}"; do
    set -- $spec
    TP=$1; CPUSZ=$2; SSDSZ=$3
    echo "########## [$PHASE] $TP (CPU=${CPUSZ}G SSD=${SSDSZ}G) ##########"
    CPU_CACHE_GB=$CPUSZ SSD_CACHE_GB=$SSDSZ SERVER_LOG=/home/fcqiao/v018-tests/srv_scbench_${PHASE}_${TP}.log \
      $PYTHON scbench_14b.py --only "$TP" 2>&1 | tee /home/fcqiao/v018-tests/scbench_${PHASE}_${TP}.log
  done
done
echo "ALL SCBENCH DONE"
