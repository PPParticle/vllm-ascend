#!/usr/bin/env python3
"""
LMCache Working Set Benchmark with SCBench (SharedContextBench)
===============================================================

Tests LMCache performance with different working set sizes and context lengths.
Uses SCBench dataset from Microsoft (microsoft/SCBench).

Test Matrix (8 points):
  ctx=8k:  WS=8, 16, 32
  ctx=16k: WS=8, 16, 32
  ctx=32k: WS=8, 16      (WS32+ctx32k excluded: needs 56GB > 31GB available)

For each test point:
  1. Start fresh vLLM server (no prefix cache, LMCache enabled)
  2. Cold pass:  send N unique contexts + query_1  -> fills cache, measure TTFT
  3. Warm pass:  send N same contexts + query_2    -> cache hit,  measure TTFT
  4. Record and compare cold vs warm TTFT

Usage:
  # Set environment variables:
  export VLLM_PATH=<vllm_source_path>
  export VLLM_ASCEND_PATH=<vllm_ascend_source_path>
  export PYTHON=<venv_python_path>
  export MODEL_PATH=<model_path>
  export SSD_PATH=<ssd_cache_path>
  export RESULTS_DIR=<results_output_path>

  source /usr/local/Ascend/cann-8.5.1/set_env.sh
  $PYTHON tests/lmcache/benchmark_working_set.py

  # Skip server management (connect to running server):
  $PYTHON tests/lmcache/benchmark_working_set.py --no-manage-server

  # Only run specific test points:
  $PYTHON tests/lmcache/benchmark_working_set.py --only ctx8k_ws8 ctx16k_ws16
"""

import argparse
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time

import requests

# ======================== Configuration ========================

MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
MODEL_NAME = MODEL_PATH  # Must match server's served_model_name
SERVER_PORT = 8100
SERVER_URL = f"http://localhost:{SERVER_PORT}"

CPU_CACHE_GB = 8
SSD_CACHE_GB = 20
SSD_PATH = os.environ.get("SSD_PATH", "/tmp/lmcache_ssd")
SERVER_LOG = os.environ.get("SERVER_LOG", "/tmp/benchmark_server.log")

MAX_OUTPUT_TOKENS = 10
MAX_NUM_SEQS = 8

# KV bytes per token: 28 layers × 2(K+V) × 4 KV heads × 128 dim × 2 bytes(fp16)
BYTES_PER_TOKEN = 28 * 2 * 4 * 128 * 2  # 57,344 bytes ≈ 56 KB

CONTEXT_LENGTHS = [8192, 16384, 32768]
WORKING_SET_SIZES = [8, 16, 32]

# Full test matrix (exclude WS32+ctx32k)
TEST_MATRIX = [
    (ctx, ws)
    for ctx in CONTEXT_LENGTHS
    for ws in WORKING_SET_SIZES
    if not (ctx == 32768 and ws == 32)
]

SCBENCH_SUBSET = "scbench_qa_eng"  # 69 samples, 5+ multi-turns, context > 70k tokens
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/tmp/lmcache_results")


# ======================== Server Management ========================


def get_server_env(baseline=False):
    """Build environment variables for vLLM server."""
    env = os.environ.copy()
    base_env = {
        "PYTHONPATH": f"{os.environ.get('VLLM_PATH', '')}:{os.environ.get('VLLM_ASCEND_PATH', '')}:{env.get('PYTHONPATH', '')}",
        "VLLM_PLUGINS": "ascend,ascend_model_loader,ascend_service_profiling",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    }
    if not baseline:
        base_env.update({
            "VLLM_PLUGINS": "ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling",
            "LMCACHE_LOCAL_CPU": "True",
            "LMCACHE_MAX_LOCAL_CPU_SIZE": str(CPU_CACHE_GB),
            "LMCACHE_LOCAL_DISK": f"file://{SSD_PATH}",
            "LMCACHE_MAX_LOCAL_DISK_SIZE": str(SSD_CACHE_GB),
            "LMCACHE_CHUNK_SIZE": "128",
            "LMCACHE_SAVE_UNFULL_CHUNK": "True",
        })
    env.update(base_env)
    return env


def build_server_cmd(max_model_len, baseline=False):
    """Build vLLM server command."""
    cmd = [
        os.environ.get("PYTHON", sys.executable),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--port",
        str(SERVER_PORT),
        "--model",
        MODEL_PATH,
        "--trust-remote-code",
        "--block-size",
        "128",
        "--dtype",
        "float16",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_model_len),
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--gpu-memory-utilization",
        "0.90",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    if not baseline:
        cmd.extend([
            "--kv-transfer-config",
            json.dumps(
                {"kv_connector": "LMCacheAscendConnector", "kv_role": "kv_both"}
            ),
        ])
    return cmd


def start_server(max_model_len=32768, baseline=False):
    """Start vLLM server as subprocess."""
    env = get_server_env(baseline=baseline)
    cmd = build_server_cmd(max_model_len, baseline=baseline)

    log_fh = open(SERVER_LOG, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT, preexec_fn=os.setsid
    )
    return proc


def wait_for_server(timeout=300):
    """Wait for server health endpoint to return 200."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                # Extra sleep: vLLM returns health=200 before model is fully loaded
                time.sleep(3)
                return True
        except requests.ConnectionError:
            pass
        time.sleep(5)
    return False


def stop_server(proc):
    """Stop vLLM server process group."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def clear_ssd_cache():
    """Clear SSD cache directory."""
    if os.path.exists(SSD_PATH):
        for entry in os.listdir(SSD_PATH):
            p = os.path.join(SSD_PATH, entry)
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)


def kill_orphan_servers():
    """Kill any lingering vLLM server on our port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{SERVER_PORT}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            pid = pid.strip()
            if pid.isdigit():
                os.kill(int(pid), signal.SIGKILL)
                print(f"  Killed orphan PID {pid}")
    except Exception:
        pass


# ======================== Data Loading ========================


def load_scbench_data(subset, context_len, working_set_size, tokenizer, offset=0):
    """Load SCBench data and truncate contexts to target length.

    Args:
        offset: skip first N valid samples to avoid cache overlap across test points.

    Returns list of dicts with:
      - context: truncated context text
      - context_tokens: actual token count
      - cold_query: first multi-turn query
      - warm_query: second multi-turn query
    """
    from datasets import load_dataset

    ds = load_dataset(
        "microsoft/SCBench", subset, split="test", cache_dir="/tmp/hf_cache/hub"
    )

    # Reserve tokens for: chat template (~30) + query (~200) + output (MAX_OUTPUT_TOKENS)
    max_context_tokens = context_len - MAX_OUTPUT_TOKENS - 250

    valid = []
    for item in ds:
        ctx = item["context"]
        tokens = tokenizer.encode(ctx, add_special_tokens=False)

        if len(tokens) >= max_context_tokens and len(item["multi_turns"]) >= 2:
            truncated = tokenizer.decode(tokens[:max_context_tokens])
            valid.append(
                {
                    "context": truncated,
                    "context_tokens": max_context_tokens,
                    "cold_query": item["multi_turns"][0]["input"],
                    "warm_query": item["multi_turns"][1]["input"],
                }
            )

    if len(valid) < offset + working_set_size:
        print(
            f"  WARNING: Only {len(valid)} valid samples, need {offset + working_set_size} (offset={offset})"
        )
        working_set_size = max(0, min(working_set_size, len(valid) - offset))

    return valid[offset:offset + working_set_size], working_set_size


# ======================== TTFT Measurement ========================


def measure_ttft(messages, max_tokens=MAX_OUTPUT_TOKENS, timeout=120):
    """Measure TTFT via streaming chat completion.

    Returns (ttft_seconds, full_response_text) or (None, None) on error.
    """
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.time()
    ttft = None
    full_text = ""

    try:
        resp = requests.post(url, json=payload, stream=True, timeout=timeout)
        resp.raise_for_status()

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            payload_str = line[6:]
            if payload_str.strip() == "[DONE]":
                break
            try:
                data = json.loads(payload_str)
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    if ttft is None:
                        ttft = time.time() - start
                    full_text += content
            except json.JSONDecodeError:
                pass

    except Exception as e:
        print(f"    [ERROR] Request failed: {e}")
        return None, None

    if ttft is None:
        ttft = time.time() - start

    return ttft, full_text


def run_pass(samples, pass_type):
    """Run cold or warm pass. Returns list of (ttft_ms, response)."""
    results = []
    label = "COLD" if pass_type == "cold" else "WARM"
    query_key = "cold_query" if pass_type == "cold" else "warm_query"

    for i, sample in enumerate(samples):
        messages = [
            {"role": "system", "content": sample["context"]},
            {"role": "user", "content": sample[query_key]},
        ]

        ttft_s, response = measure_ttft(messages)
        ttft_ms = ttft_s * 1000 if ttft_s is not None else None
        results.append({"ttft_ms": ttft_ms, "response": response or ""})

        if ttft_ms is not None:
            print(f"    [{label}] {i + 1:>3}/{len(samples)}: TTFT={ttft_ms:.1f}ms")
        else:
            print(f"    [{label}] {i + 1:>3}/{len(samples)}: FAILED")

    return results


def compute_stats(ttfts):
    """Compute summary statistics from a list of TTFT values (ms)."""
    if not ttfts:
        return {}
    return {
        "avg_ms": round(statistics.mean(ttfts), 1),
        "median_ms": round(statistics.median(ttfts), 1),
        "p90_ms": round(sorted(ttfts)[int(len(ttfts) * 0.9)], 1) if len(ttfts) >= 2 else round(ttfts[0], 1),
        "min_ms": round(min(ttfts), 1),
        "max_ms": round(max(ttfts), 1),
        "count": len(ttfts),
    }


# ======================== Test Execution ========================


def run_test_point(ctx_len, ws_size, tokenizer, sample_offset=0, server_proc=None,
                   manage_server=True, baseline_mode=False):
    """Run a single (context_len, working_set_size) test point.

    If manage_server=True, starts/stops server automatically.
    sample_offset: skip first N samples to avoid cache overlap.
    baseline_mode: if True, skip warm pass (no LMCache).
    Returns (result_dict, server_proc) or (None, server_proc) on failure.
    """
    test_key = f"ctx{ctx_len // 1024}k_ws{ws_size}"
    kv_total_gb = ws_size * ctx_len * BYTES_PER_TOKEN / (1024**3)
    mode_tag = " [BASELINE]" if baseline_mode else ""

    print(f"\n{'=' * 65}")
    print(f"  {test_key}  |  KV total: {kv_total_gb:.1f} GB{mode_tag}")
    print(f"{'=' * 65}")

    # --- Server lifecycle ---
    if manage_server:
        if server_proc is not None:
            print("  Stopping previous server...")
            stop_server(server_proc)
            clear_ssd_cache()
            kill_orphan_servers()
            time.sleep(5)

        print("  Starting vLLM server...")
        server_proc = start_server(max_model_len=max(ctx_len, 8192), baseline=baseline_mode)

        print("  Waiting for server ready...", end="", flush=True)
        if not wait_for_server(timeout=300):
            print(" FAILED (timeout)")
            stop_server(server_proc)
            return None, None
        print(" OK")

    # --- Load data ---
    print(f"  Loading SCBench data (subset={SCBENCH_SUBSET}, offset={sample_offset})...")
    samples, actual_ws = load_scbench_data(
        SCBENCH_SUBSET, ctx_len, ws_size, tokenizer, offset=sample_offset
    )
    print(f"  Prepared {actual_ws} samples, ~{ctx_len} tokens each")

    # --- Cold pass ---
    print(f"\n  Phase 1: Cold pass ({actual_ws} requests)...")
    cold_results = run_pass(samples, "cold")

    # --- Warm pass (skip in baseline mode) ---
    warm_results = []
    if not baseline_mode:
        print(f"\n  Phase 2: Warm pass ({actual_ws} requests)...")
        warm_results = run_pass(samples, "warm")
    else:
        print(f"\n  Phase 2: SKIPPED (baseline mode)")

    # --- Compute metrics ---
    cold_ttfts = [r["ttft_ms"] for r in cold_results if r["ttft_ms"] is not None]
    warm_ttfts = [r["ttft_ms"] for r in warm_results if r["ttft_ms"] is not None]

    result = {
        "test_key": test_key,
        "context_len": ctx_len,
        "working_set_size": actual_ws,
        "kv_total_gb": round(kv_total_gb, 1),
        "bytes_per_token": BYTES_PER_TOKEN,
        "cold": compute_stats(cold_ttfts),
    }
    if warm_ttfts:
        result["warm"] = compute_stats(warm_ttfts)

    # Speedup
    if result["cold"] and result.get("warm"):
        result["speedup"] = round(
            result["cold"]["avg_ms"] / result["warm"]["avg_ms"], 2
        )
    else:
        result["speedup"] = 0.0

    # Per-test summary
    print(f"\n  {'─' * 50}")
    if result["cold"]:
        print(
            f"  Cold TTFT: avg={result['cold']['avg_ms']:.1f}ms  "
            f"p50={result['cold']['median_ms']:.1f}ms  "
            f"p90={result['cold']['p90_ms']:.1f}ms"
        )
    if result.get("warm"):
        print(
            f"  Warm TTFT: avg={result['warm']['avg_ms']:.1f}ms  "
            f"p50={result['warm']['median_ms']:.1f}ms  "
            f"p90={result['warm']['p90_ms']:.1f}ms"
        )
        print(f"  Speedup:   {result['speedup']:.2f}x")
    if baseline_mode:
        print(f"  (baseline, no warm pass)")
    else:
        print("  Insufficient data for summary")

    return result, server_proc


# ======================== Main ========================


def main():
    parser = argparse.ArgumentParser(description="LMCache Working Set Benchmark")
    parser.add_argument(
        "--no-manage-server",
        action="store_true",
        help="Connect to already-running server instead of managing lifecycle",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Only run specific test points (e.g. --only ctx8k_ws8 ctx16k_ws16)",
    )
    parser.add_argument(
        "--cpu-cache-gb",
        type=int,
        default=CPU_CACHE_GB,
        help=f"CPU cache size in GB (default: {CPU_CACHE_GB})",
    )
    parser.add_argument(
        "--ssd-cache-gb",
        type=int,
        default=SSD_CACHE_GB,
        help=f"SSD cache size in GB (default: {SSD_CACHE_GB})",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Baseline mode: no LMCache, single cold pass only",
    )
    args = parser.parse_args()

    manage_server = not args.no_manage_server
    baseline_mode = args.baseline

    # Override cache sizes from args
    cpu_cache_gb = args.cpu_cache_gb
    ssd_cache_gb = args.ssd_cache_gb

    # Filter test matrix if --only specified
    test_points = TEST_MATRIX
    if args.only:
        requested = set(args.only)
        test_points = [
            (ctx, ws)
            for ctx, ws in TEST_MATRIX
            if f"ctx{ctx // 1024}k_ws{ws}" in requested
        ]
        if not test_points:
            print(f"ERROR: No matching test points for: {args.only}")
            print(f"Available: {[f'ctx{c//1024}k_ws{w}' for c, w in TEST_MATRIX]}")
            sys.exit(1)

    # Header
    mode_label = "BASELINE (No LMCache)" if baseline_mode else "LMCache Enabled"
    print("=" * 65)
    print(f"  Working Set Benchmark with SCBench — {mode_label}")
    print("=" * 65)
    print(f"  Model:       {MODEL_PATH}")
    if not baseline_mode:
        print(f"  CPU Cache:   {cpu_cache_gb} GB")
        print(f"  SSD Cache:   {ssd_cache_gb} GB")
    else:
        print(f"  Cache:       DISABLED (baseline)")
    print(f"  Server:      {'managed' if manage_server else 'external'}")
    print(f"  Test points: {len(test_points)}")
    for ctx, ws in test_points:
        kv = ws * ctx * BYTES_PER_TOKEN / (1024**3)
        cpu_cap = cpu_cache_gb / (ctx * BYTES_PER_TOKEN / (1024**3))
        print(
            f"    ctx={ctx // 1024}k  WS={ws:>2}  KV={kv:>5.1f}GB  "
            f"(CPU fits ~{cpu_cap:.0f} contexts)"
        )
    print()

    # Check server connectivity
    if not manage_server:
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code != 200:
                print(f"ERROR: Server at {SERVER_URL} returned {resp.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"ERROR: Cannot connect to server at {SERVER_URL}: {e}")
            sys.exit(1)
        print(f"Server at {SERVER_URL} is ready.")

    # Load tokenizer
    print("Loading tokenizer...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Tokenizer loaded.\n")

    # Run tests
    all_results = {}
    server_proc = None
    # Track cumulative sample offset per context length to avoid cache overlap
    ctx_offsets = {}  # {ctx_len: total_samples_used_so_far}

    for idx, (ctx_len, ws_size) in enumerate(test_points):
        test_key = f"ctx{ctx_len // 1024}k_ws{ws_size}"
        sample_offset = ctx_offsets.get(ctx_len, 0)
        print(f"\n{'#' * 65}")
        print(f"  Test {idx + 1}/{len(test_points)}: {test_key}  (sample_offset={sample_offset})")
        print(f"{'#' * 65}")

        result, server_proc = run_test_point(
            ctx_len,
            ws_size,
            tokenizer,
            sample_offset=sample_offset,
            server_proc=server_proc,
            manage_server=manage_server,
            baseline_mode=baseline_mode,
        )

        if result:
            all_results[test_key] = result
            # Advance offset for this context length
            ctx_offsets[ctx_len] = sample_offset + result["working_set_size"]
        else:
            print(f"  SKIPPED: {test_key} (server error)")

    # Final cleanup
    if manage_server and server_proc:
        print("\nCleaning up server...")
        stop_server(server_proc)
        clear_ssd_cache()
        kill_orphan_servers()

    # ======================== Final Report ========================
    print(f"\n{'=' * 80}")
    print("  FINAL RESULTS")
    print(f"{'=' * 80}")

    if not all_results:
        print("  No results collected.")
        sys.exit(1)

    # Summary table
    if baseline_mode:
        header = (
            f"  {'Test':<15} {'KV(GB)':>7} "
            f"{'Cold Avg':>10} {'Cold P90':>10}"
        )
        print(header)
        print(f"  {'─' * 50}")
        for key, r in all_results.items():
            c = r.get("cold", {})
            if c:
                print(
                    f"  {key:<15} {r['kv_total_gb']:>6.1f}G "
                    f"{c['avg_ms']:>9.1f}ms {c['p90_ms']:>9.1f}ms"
                )
            else:
                print(f"  {key:<15} {r['kv_total_gb']:>6.1f}G  (insufficient data)")
    else:
        header = (
            f"  {'Test':<15} {'KV(GB)':>7} "
            f"{'Cold Avg':>10} {'Cold P90':>10} "
            f"{'Warm Avg':>10} {'Warm P90':>10} "
            f"{'Speedup':>8}"
        )
        print(header)
        print(f"  {'─' * 75}")
        for key, r in all_results.items():
            c = r.get("cold", {})
            w = r.get("warm")
            if c and w:
                print(
                    f"  {key:<15} {r['kv_total_gb']:>6.1f}G "
                    f"{c['avg_ms']:>9.1f}ms {c['p90_ms']:>9.1f}ms "
                    f"{w['avg_ms']:>9.1f}ms {w['p90_ms']:>9.1f}ms "
                    f"{r['speedup']:>7.2f}x"
                )
            elif c:
                print(
                    f"  {key:<15} {r['kv_total_gb']:>6.1f}G "
                    f"{c['avg_ms']:>9.1f}ms {c['p90_ms']:>9.1f}ms  (no warm data)"
                )
            else:
                print(f"  {key:<15} {r['kv_total_gb']:>6.1f}G  (insufficient data)")

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = "baseline" if baseline_mode else "working_set"
    results_file = os.path.join(RESULTS_DIR, f"{prefix}_{ts}.json")

    output = {
        "metadata": {
            "model": MODEL_PATH,
            "mode": "baseline" if baseline_mode else "lmcache",
            "cpu_cache_gb": cpu_cache_gb if not baseline_mode else 0,
            "ssd_cache_gb": ssd_cache_gb if not baseline_mode else 0,
            "scbench_subset": SCBENCH_SUBSET,
            "bytes_per_token": BYTES_PER_TOKEN,
            "timestamp": ts,
        },
        "results": all_results,
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {results_file}")
    print()


if __name__ == "__main__":
    main()
