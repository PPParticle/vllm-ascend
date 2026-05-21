"""
离线 API 方式使用 mm_encoder_only：单次启动，多次 encode。

使用 LLM 类直接在进程内调用，无需启动 HTTP 服务器。

Usage:
    source /data/arctic_env/bin/activate && \
    PYTHONPATH=/data/vllm:/data/vllm-ascend \
    VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True \
    VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=/data/huggingface_home \
    python3 test_mm_encoder_only_offline.py
"""
import base64
import json
import os
import sys

# 确保 spawn 子进程能找到 CANN 的 acl 模块
_cann_paths = [
    "/usr/local/Ascend/cann-8.5.1/python/site-packages",
    "/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe",
]
existing = set(os.environ.get("PYTHONPATH", "").split(":"))
for p in _cann_paths:
    if p not in existing:
        os.environ["PYTHONPATH"] = p + ":" + os.environ.get("PYTHONPATH", "")

from vllm import LLM, SamplingParams
from vllm.config.ec_transfer import ECTransferConfig

MODEL = "/data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct"
SHARED_STORAGE = "/tmp/mm_encoder_only_offline_storage"


def get_test_image_b64():
    from modelscope import snapshot_download
    mm_dir = snapshot_download(
        "vllm-ascend/mm_request",
        repo_type="dataset",
        cache_dir="/data/huggingface_home",
    )
    with open(os.path.join(mm_dir, "test_mm2.jpg"), "rb") as f:
        return base64.b64encode(f.read()).decode()


def main():
    # 清理旧存储
    os.system(f"rm -rf {SHARED_STORAGE}")

    image_b64 = get_test_image_b64()
    print("[INFO] 测试图片加载完成")

    # ============ Step 1: 启动 Encoder（单次） ============
    print("[INFO] 正在初始化 LLM (mm_encoder_only + ec_producer)...")
    encoder = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        # 关键参数
        mm_encoder_only=True,
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_producer",
            ec_connector_extra_config={
                "shared_storage_path": SHARED_STORAGE,
            },
        ),
    )
    print("[INFO] Encoder 初始化完成，可以开始多次 encode")

    # ============ Step 2: 多次 encode ============
    sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

    # 构造多轮不同的请求
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "Describe image 1"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "Describe image 2"},
                ],
            }
        ],
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "Compare these two images"},
                ],
            }
        ],
    ]

    for i, messages in enumerate(conversations):
        print(f"\n[INFO] === 第 {i+1} 次 encode ===")
        outputs = encoder.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for output in outputs:
            print(f"  request_id: {output.request_id}")
            for comp in output.outputs:
                print(f"  text: {comp.text!r}")
            print(f"  finished: {output.finished}")

    # ============ Step 3: 检查 EC 存储中的缓存 ============
    print(f"\n[INFO] === EC 共享存储内容 ({SHARED_STORAGE}) ===")
    if os.path.exists(SHARED_STORAGE):
        for root, dirs, files in os.walk(SHARED_STORAGE):
            for f in files:
                path = os.path.join(root, f)
                size = os.path.getsize(path)
                print(f"  {path} ({size} bytes)")
    else:
        print("  (存储目录不存在)")

    # ============ Step 4: 也可以批量 encode ============
    print(f"\n[INFO] === 批量 encode (3 张图一起) ===")
    batch_messages = [conv for conv in conversations]
    outputs = encoder.chat(
        messages=batch_messages,
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    print(f"  批量处理了 {len(outputs)} 个请求")

    print("\n[INFO] 全部完成")


if __name__ == "__main__":
    main()
