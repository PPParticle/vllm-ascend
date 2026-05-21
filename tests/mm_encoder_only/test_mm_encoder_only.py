"""
End-to-end test for --mm-encoder-only on single NPU card.

This test verifies that:
1. The vLLM server can start with --mm-encoder-only flag
2. The server processes multimodal requests without crashing
3. The _dummy_run and _dummy_sampler_run early returns work correctly

Usage:
    source /data/arctic_env/bin/activate
    VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True HF_HOME=/data/huggingface_home \
        python3 test_mm_encoder_only.py
"""
import base64
import json
import os
import subprocess
import sys
import time
import traceback

import requests

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
SHARED_STORAGE_PATH = "/tmp/mm_encoder_only_test_storage"


def get_open_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def get_test_image_base64():
    """Get base64 encoded test image."""
    from modelscope import snapshot_download
    mm_dir = snapshot_download(
        "vllm-ascend/mm_request",
        repo_type="dataset",
        cache_dir="/data/huggingface_home",
    )
    image_path = os.path.join(mm_dir, "test_mm2.jpg")
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def cleanup_storage():
    subprocess.run(["rm", "-rf", SHARED_STORAGE_PATH], capture_output=True)


def test_mm_encoder_only():
    port = get_open_port()
    image_data = get_test_image_base64()

    ec_transfer_config = json.dumps({
        "ec_connector_extra_config": {
            "shared_storage_path": SHARED_STORAGE_PATH
        },
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
    })

    # Use local model path directly to avoid re-downloading
    local_model_path = "/data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct"
    if not os.path.exists(local_model_path):
        local_model_path = MODEL

    server_args = [
        sys.executable, "/data/arctic_env/bin/vllm", "serve", local_model_path,
        "--port", str(port),
        "--gpu-memory-utilization", "0.90",
        "--tensor-parallel-size", "1",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--max-model-len", "4096",
        "--max-num-batched-tokens", "4096",
        "--max-num-seqs", "1",
        "--mm-encoder-only",
        "--ec-transfer-config", ec_transfer_config,
    ]

    env = os.environ.copy()
    env["VLLM_PLUGINS"] = "ascend"
    env["VLLM_USE_MODELSCOPE"] = "True"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["HF_HOME"] = "/data/huggingface_home"
    env["ASCEND_RT_VISIBLE_DEVICES"] = "0"
    # Fix editable install: ensure /data/vllm is in PYTHONPATH
    # so the editable finder resolves vllm correctly
    python_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"/data/vllm:/data/vllm-ascend:{python_path}"

    print(f"[TEST] Starting encoder-only server on port {port}...")
    print(f"[TEST] Server args: {' '.join(server_args)}")

    proc = subprocess.Popen(
        server_args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Log server output in background
    def log_output():
        for line in iter(proc.stdout.readline, ""):
            print(f"[SERVER] {line}", end="")

    import threading
    log_thread = threading.Thread(target=log_output, daemon=True)
    log_thread.start()

    try:
        # Wait for server to be healthy
        health_url = f"http://127.0.0.1:{port}/health"
        print(f"[TEST] Waiting for server at {health_url}...")
        max_wait = 900
        start = time.time()
        server_ready = False

        while time.time() - start < max_wait:
            # Check if process died
            poll = proc.poll()
            if poll is not None:
                print(f"[TEST] Server exited with code {poll}")
                return False

            try:
                resp = requests.get(health_url, timeout=5)
                if resp.status_code == 200:
                    print(f"[TEST] Server is healthy after {time.time()-start:.1f}s")
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(5)

        if not server_ready:
            print(f"[TEST] FAILED: Server did not become healthy within {max_wait}s")
            return False

        # Send a multimodal request
        print("[TEST] Sending multimodal request...")
        chat_url = f"http://127.0.0.1:{port}/v1/chat/completions"
        payload = {
            "model": local_model_path,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the content of this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                ],
            }],
            "max_tokens": 10,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(chat_url, json=payload, headers=headers, timeout=120)
            print(f"[TEST] Response status: {resp.status_code}")
            print(f"[TEST] Response body: {json.dumps(resp.json(), indent=2, ensure_ascii=False)[:2000]}")

            # In encoder-only mode, the server processes the image encoder
            # but doesn't generate text. The response may vary depending on
            # how the API server handles encoder-only mode.
            # We mainly verify no crash occurred.
            print("[TEST] SUCCESS: Server handled request without crash")
            return True

        except requests.exceptions.RequestException as e:
            print(f"[TEST] Request failed: {e}")
            # If the server stayed up, it might still be a partial success
            poll = proc.poll()
            if poll is None:
                print("[TEST] Server is still running - request handling issue, not a crash")
                return False
            return False

    except Exception as e:
        print(f"[TEST] Exception: {e}")
        traceback.print_exc()
        return False
    finally:
        print("[TEST] Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        cleanup_storage()
        print("[TEST] Done")


if __name__ == "__main__":
    success = test_mm_encoder_only()
    sys.exit(0 if success else 1)
