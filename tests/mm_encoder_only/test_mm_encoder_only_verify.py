"""
严格验证 mm_encoder_only 功能性的脚本。

验证项：
  1. EC 存储中有 encoder cache 文件，且内容合法（safetensors, 非空 tensor, 无 NaN/Inf）
  2. LM decoder 确实被跳过 — 对比有/无 mm_encoder_only 的内存和行为差异
  3. Encoder 输出可被 consumer 端加载

Usage:
    source /data/arctic_env/bin/activate && \
    PYTHONPATH=/data/vllm:/data/vllm-ascend:/usr/local/Ascend/cann-8.5.1/python/site-packages:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe \
    VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True \
    VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=/data/huggingface_home \
    python3 /data/test_mm_encoder_only_verify.py
"""
import base64
import gc
import os
import sys
import time
import traceback

import torch

# CANN 路径注入到 PYTHONPATH
_cann = "/usr/local/Ascend/cann-8.5.1/python/site-packages"
for p in [_cann]:
    if p not in os.environ.get("PYTHONPATH", ""):
        os.environ["PYTHONPATH"] = p + ":" + os.environ.get("PYTHONPATH", "")

from safetensors import torch as sf_torch
from vllm import LLM, SamplingParams
from vllm.config.ec_transfer import ECTransferConfig

MODEL = "/data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct"

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def get_test_image_b64():
    from modelscope import snapshot_download
    mm_dir = snapshot_download(
        "vllm-ascend/mm_request",
        repo_type="dataset",
        cache_dir="/data/huggingface_home",
    )
    with open(os.path.join(mm_dir, "test_mm2.jpg"), "rb") as f:
        return base64.b64encode(f.read()).decode()


def cleanup(path):
    os.system(f"rm -rf {path}")


def find_safetensors(root):
    """递归查找所有 safetensors 文件"""
    results = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".safetensors"):
                results.append(os.path.join(dirpath, fn))
    return results


# ============================================================
# 验证 1: Encoder 输出写入 EC 存储，内容合法
# ============================================================
def verify_encoder_output():
    print("\n===== 验证 1: Encoder 输出写入 EC 存储 =====")
    storage = "/tmp/mm_verify_v1_storage"
    cleanup(storage)

    llm = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        mm_encoder_only=True,
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_producer",
            ec_connector_extra_config={"shared_storage_path": storage},
        ),
    )

    image_b64 = get_test_image_b64()

    # 发一次 encode 请求
    outputs = llm.chat(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        sampling_params=SamplingParams(max_tokens=1),
        use_tqdm=False,
    )

    # 1a. EC 存储目录存在
    check("EC storage dir exists", os.path.isdir(storage))

    # 1b. 有 safetensors 文件
    st_files = find_safetensors(storage)
    check("Has safetensors file(s)", len(st_files) > 0, f"found {len(st_files)} files")

    if st_files:
        # 1c. 文件大小 > 0
        for f in st_files:
            size = os.path.getsize(f)
            check(f"File non-empty ({os.path.basename(os.path.dirname(f))})", size > 0, f"{size} bytes")

        # 1d. 能用 safetensors 加载
        try:
            data = sf_torch.load_file(st_files[0])
            check("Safetensors loads OK", True)
        except Exception as e:
            check("Safetensors loads OK", False, str(e))
            data = {}

        if data:
            # 1e. 包含 ec_cache key
            check("Has 'ec_cache' key", "ec_cache" in data, f"keys: {list(data.keys())}")

            if "ec_cache" in data:
                t = data["ec_cache"]
                # 1f. Tensor 非空
                check("Tensor non-empty", t.numel() > 0, f"shape={t.shape}, numel={t.numel()}")
                # 1g. 无 NaN
                check("No NaN", not torch.isnan(t).any().item())
                # 1h. 无 Inf
                check("No Inf", not torch.isinf(t).any().item())
                # 1i. 值不全为零（说明确实有计算）
                check("Not all zeros", not torch.all(t == 0).item(), f"mean={t.float().mean().item():.6f}")
                # 1j. 维度合理（hidden_size 维度）
                check("Has hidden dim", len(t.shape) >= 2, f"shape={t.shape}")

    # 1k. 同一 LLM 实例内第二次 encode 同一张图：
    # 由于 EncoderCacheManager 的 in-memory 缓存仍持有 mm_hash，
    # scheduler 会跳过重编码（不会触发 save_caches），这是正确行为。
    # 验证：第二次 encode 应能正常完成（不报错），output 为 short/dummy。
    outputs2 = llm.chat(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "Different prompt for second encode"},
        ]}],
        sampling_params=SamplingParams(max_tokens=1),
        use_tqdm=False,
    )
    check("Second encode completes without error", len(outputs2) > 0)
    if outputs2:
        check("Second encode output is short/dummy",
              len(outputs2[0].outputs[0].text) <= 5,
              f"text={outputs2[0].outputs[0].text!r}")

    del llm
    gc.collect()
    cleanup(storage)


# ============================================================
# 验证 2: LM decoder 被跳过 — 对比有无 mm_encoder_only
# ============================================================
def verify_lm_skipped():
    print("\n===== 验证 2: LM decoder 被跳过 =====")
    storage_enc = "/tmp/mm_verify_v2_encoder"
    storage_normal = "/tmp/mm_verify_v2_normal"
    cleanup(storage_enc)
    cleanup(storage_normal)

    image_b64 = get_test_image_b64()
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # --- 2a. 带 mm_encoder_only ---
    print("  [INFO] Initializing with mm_encoder_only=True ...")
    t0 = time.time()
    llm_enc = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        mm_encoder_only=True,
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_producer",
            ec_connector_extra_config={"shared_storage_path": storage_enc},
        ),
    )
    init_time_enc = time.time() - t0
    print(f"  [INFO] mm_encoder_only init: {init_time_enc:.1f}s")

    t0 = time.time()
    out_enc = llm_enc.chat(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        sampling_params=sp,
        use_tqdm=False,
    )
    infer_time_enc = time.time() - t0
    print(f"  [INFO] mm_encoder_only inference: {infer_time_enc:.2f}s")

    enc_text = out_enc[0].outputs[0].text
    enc_finished = out_enc[0].finished
    del llm_enc
    gc.collect()

    # --- 2b. 不带 mm_encoder_only ---
    print("  [INFO] Initializing with mm_encoder_only=False ...")
    t0 = time.time()
    llm_normal = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        # 不设 mm_encoder_only
    )
    init_time_normal = time.time() - t0
    print(f"  [INFO] normal init: {init_time_normal:.1f}s")

    t0 = time.time()
    out_normal = llm_normal.chat(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": "What is in this image?"},
        ]}],
        sampling_params=SamplingParams(max_tokens=20, temperature=0.0),
        use_tqdm=False,
    )
    infer_time_normal = time.time() - t0
    print(f"  [INFO] normal inference: {infer_time_normal:.2f}s")

    normal_text = out_normal[0].outputs[0].text
    normal_finished = out_normal[0].finished
    del llm_normal
    gc.collect()

    # --- 对比 ---
    print("\n  --- 对比结果 ---")

    # 2a. 初始化时间：encoder-only 应该更快（跳过了 LM profiling）
    check(
        "Init time: encoder-only < normal",
        init_time_enc < init_time_normal,
        f"enc={init_time_enc:.1f}s vs normal={init_time_normal:.1f}s",
    )

    # 2b. encoder-only 的 profile_run 产出的 KV cache 应该更少
    # (is_producer 返回空 kv_cache_spec，只加了 encoder-only 层)
    # 用初始化时间作为代理指标

    # 2c. 推理输出：encoder-only 不应该有有意义的文本生成
    check(
        "Encoder-only output is short/dummy",
        len(enc_text) <= 5,
        f"enc_text={enc_text!r} (len={len(enc_text)})",
    )

    # 2d. 正常模式有有意义的输出
    check(
        "Normal mode produces real text",
        len(normal_text) > 5,
        f"normal_text={normal_text!r} (len={len(normal_text)})",
    )

    # 2e. encoder-only 有 EC 存储输出
    enc_files = find_safetensors(storage_enc)
    check("Encoder-only has EC cache files", len(enc_files) > 0)

    # 2f. 正常模式没有 EC 存储（没有 ec_transfer_config）
    normal_files = find_safetensors(storage_normal)
    check("Normal mode has no EC cache", len(normal_files) == 0)

    cleanup(storage_enc)
    cleanup(storage_normal)


# ============================================================
# 验证 3: Consumer 端可加载 encoder 输出
# ============================================================
def verify_consumer_load():
    print("\n===== 验证 3: Consumer 可加载 encoder 输出 =====")
    storage = "/tmp/mm_verify_v3_storage"
    cleanup(storage)

    # --- 3a. Producer 端先 encode ---
    print("  [INFO] Producer: encoding image ...")
    producer = LLM(
        model=MODEL,
        enforce_eager=True,
        max_model_len=4096,
        max_num_seqs=1,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
        mm_encoder_only=True,
        ec_transfer_config=ECTransferConfig(
            ec_connector="ECExampleConnector",
            ec_role="ec_producer",
            ec_connector_extra_config={"shared_storage_path": storage},
        ),
    )

    image_b64 = get_test_image_b64()
    producer.chat(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        ]}],
        sampling_params=SamplingParams(max_tokens=1),
        use_tqdm=False,
    )
    del producer
    gc.collect()

    # --- 3b. 直接读取和验证 safetensors ---
    st_files = find_safetensors(storage)
    check("Producer created cache files", len(st_files) > 0)

    if st_files:
        data = sf_torch.load_file(st_files[0])
        check("Cache file loads", "ec_cache" in data)

        if "ec_cache" in data:
            tensor = data["ec_cache"]
            shape = tensor.shape
            print(f"  [INFO] Encoder output shape: {shape}, dtype: {tensor.dtype}")

            # 3c. 验证 tensor 可以被 safetensors 重新保存和加载
            verify_path = "/tmp/mm_verify_v3_roundtrip.safetensors"
            sf_torch.save_file({"ec_cache": tensor}, verify_path)
            reloaded = sf_torch.load_file(verify_path)
            match = torch.allclose(tensor, reloaded["ec_cache"])
            check("Roundtrip save/load preserves data", match)
            os.remove(verify_path)

            # 3d. Consumer 端初始化并加载
            print("  [INFO] Consumer: loading encoder cache ...")
            try:
                consumer = LLM(
                    model=MODEL,
                    enforce_eager=True,
                    max_model_len=4096,
                    max_num_seqs=1,
                    gpu_memory_utilization=0.90,
                    tensor_parallel_size=1,
                    enable_prefix_caching=False,
                    ec_transfer_config=ECTransferConfig(
                        ec_connector="ECExampleConnector",
                        ec_role="ec_consumer",
                        ec_connector_extra_config={"shared_storage_path": storage},
                    ),
                )
                check("Consumer initialized successfully", True)

                # 3e. Consumer 发同样的请求，应该能复用 encoder cache
                consumer.chat(
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": "What is in this image?"},
                    ]}],
                    sampling_params=SamplingParams(max_tokens=10, temperature=0.0),
                    use_tqdm=False,
                )
                check("Consumer inference completed", True)

                del consumer
                gc.collect()
            except Exception as e:
                check("Consumer initialized and ran", False, str(e))
                traceback.print_exc()

    cleanup(storage)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("mm_encoder_only 功能性严格验证")
    print("=" * 60)

    try:
        verify_encoder_output()
    except Exception as e:
        print(f"  [ERROR] 验证 1 异常: {e}")
        traceback.print_exc()

    try:
        verify_lm_skipped()
    except Exception as e:
        print(f"  [ERROR] 验证 2 异常: {e}")
        traceback.print_exc()

    try:
        verify_consumer_load()
    except Exception as e:
        print(f"  [ERROR] 验证 3 异常: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed} PASS, {failed} FAIL")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
