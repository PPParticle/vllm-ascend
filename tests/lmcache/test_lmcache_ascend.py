#!/usr/bin/env python3
"""Test LMCache Ascend integration with vllm openai API.

Sends two requests with the same long prefix, verifies:
1. Service starts without errors
2. LMCacheAscendConnector is loaded
3. Second request shows cache hit (lower TTFT)

Usage:
    # Set environment variables first:
    export MODEL_PATH=<your_model_path>
    export VLLM_PATH=<vllm_source_path>
    export VLLM_ASCEND_PATH=<vllm_ascend_source_path>

    python3 tests/lmcache/test_lmcache_ascend.py
"""
import json
import os
import subprocess
import sys
import time

import requests

# Configuration (override via environment variables)
MODEL = os.environ.get("MODEL_PATH", os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct"))
PORT = int(os.environ.get("SERVER_PORT", "8100"))
BASE_URL = f"http://localhost:{PORT}"

# A long shared prefix to ensure cache hit
SHARED_PREFIX = (
    "The history of artificial intelligence began in antiquity, with myths, "
    "stories and rumors of artificial beings endowed with intelligence or "
    "consciousness by master craftsmen. The seeds of modern AI were planted by "
    "classical philosophers who attempted to describe the process of human "
    "thinking as the mechanical manipulation of symbols. This work culminated "
    "in the invention of the programmable digital computer in the 1940s, a "
    "machine based on the abstract essence of mathematical reasoning. This "
    "device and the ideas behind it inspired a handful of scientists to begin "
    "seriously discussing the possibility of building an electronic brain. "
    "The field of AI research was founded at a workshop held on the campus of "
    "Dartmouth College during the summer of 1956. The attendees became the "
    "leaders of AI research for decades. Many of them predicted that machines "
    "as intelligent as humans would exist within a generation, and they were "
    "given millions of dollars to make this vision come true. "
) * 3  # Repeat to make it longer for cache hit


def start_server():
    """Start vllm API server with LMCacheAscendConnector."""
    vllm_path = os.environ.get("VLLM_PATH", "")
    vllm_ascend_path = os.environ.get("VLLM_ASCEND_PATH", "")
    pythonpath = os.pathsep.join(filter(None, [vllm_path, vllm_ascend_path,
                                                os.environ.get("PYTHONPATH", "")]))

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--port", str(PORT),
        "--model", MODEL,
        "--trust-remote-code",
        "--block-size", "128",
        "--dtype", "float16",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "1",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--kv-transfer-config",
        json.dumps({
            "kv_connector": "LMCacheAscendConnector",
            "kv_role": "kv_both",
        }),
    ]
    env = {
        "PYTHONPATH": pythonpath,
        "VLLM_PLUGINS": "ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling",
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "LMCACHE_LOCAL_CPU": "True",
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "2",
        "LMCACHE_CHUNK_SIZE": "128",
        "LMCACHE_SAVE_UNFULL_CHUNK": "True",
    }
    full_env = {**os.environ, **env}

    print(f"Starting vllm server on port {PORT}")
    print(f"  Model: {MODEL}")
    proc = subprocess.Popen(
        cmd, env=full_env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return proc


def wait_for_server(timeout=300):
    """Wait until the server is ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"Server ready after {time.time()-start:.1f}s")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def send_request(prompt, label):
    """Send a chat completion request and return TTFT."""
    t0 = time.time()
    try:
        r = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 32,
                "temperature": 0,
            },
            timeout=120,
        )
        ttft = time.time() - t0
        if r.status_code != 200:
            print(f"[{label}] Error {r.status_code}: {r.text[:200]}")
            return None
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        print(f"[{label}] TTFT: {ttft:.2f}s | Response: {content[:80]}")
        return ttft
    except Exception as e:
        print(f"[{label}] Exception: {e}")
        return None


def main():
    proc = start_server()

    # Stream server output
    import threading

    def print_output():
        for line in proc.stdout:
            try:
                msg = line.decode("utf-8", errors="replace").rstrip()
                if any(kw in msg for kw in [
                    "LMCache", "lmcache", "cache", "connector", "error",
                    "Error", "ERROR", "Uvicorn", "Application startup",
                    "ready", "NPU", "npu", "ASCEND",
                ]):
                    print(f"  [server] {msg}")
            except Exception:
                pass

    t = threading.Thread(target=print_output, daemon=True)
    t.start()

    print("Waiting for server...")
    if not wait_for_server():
        print("Server failed to start!")
        proc.terminate()
        proc.wait(timeout=10)
        sys.exit(1)

    # Test 1: First request (cold)
    prompt1 = SHARED_PREFIX + "What is the capital of France?"
    ttft1 = send_request(prompt1, "REQ1-cold")

    # Test 2: Second request (same prefix, should hit cache)
    prompt2 = SHARED_PREFIX + "What is the capital of Germany?"
    ttft2 = send_request(prompt2, "REQ2-cache")

    # Test 3: Third request (same prefix again)
    prompt3 = SHARED_PREFIX + "What is the capital of Japan?"
    ttft3 = send_request(prompt3, "REQ3-cache")

    print("\n--- Results ---")
    if ttft1 and ttft2:
        ratio = ttft2 / ttft1
        print(f"REQ1 (cold):  {ttft1:.2f}s")
        print(f"REQ2 (cache): {ttft2:.2f}s (ratio: {ratio:.2f}x)")
        if ttft3:
            print(f"REQ3 (cache): {ttft3:.2f}s")
        if ratio < 0.8:
            print("PASS: Cache hit likely (TTFT reduced)")
        else:
            print("WARN: No significant TTFT reduction (cache may not be working)")
    else:
        print("FAIL: Could not complete requests")

    proc.terminate()
    proc.wait(timeout=10)
    print("Server stopped.")


if __name__ == "__main__":
    main()
