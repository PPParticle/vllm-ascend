"""
等价于上游 vLLM 的 test_encoder_instance_zero_kv_cache 测试。

上游源文件: vllm/tests/v1/engine/test_engine_core.py
上游测试函数: test_encoder_instance_zero_kv_cache

验证内容:
  1. ec_producer: KV cache blocks == 1, groups/tensors 为空, EC connector 是 producer, chunked prefill 禁用
  2. ec_consumer: KV cache blocks > 1, groups 非空, EC connector 是 consumer

Usage:
    source /data/arctic_env/bin/activate && \
    PYTHONPATH=/data/vllm:/data/vllm-ascend:/usr/local/Ascend/cann-8.5.1/python/site-packages:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe \
    VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True \
    VLLM_WORKER_MULTIPROC_METHOD=spawn HF_HOME=/data/huggingface_home \
    python3 /data/test_mm_encoder_only_upstream_equivalent.py
"""
import os
import sys
import traceback

# 上游测试使用 llava-hf/llava-1.5-7b-hf，我们使用本地已有的 Qwen2.5-VL-3B-Instruct
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


# ============================================================
# 测试: ec_producer 角色的 KV cache 配置
# 等价于上游 test_encoder_instance_zero_kv_cache("ec_producer", 0.01, False, False)
# ============================================================
def test_ec_producer():
    print("\n===== 测试: ec_producer KV cache 配置 =====")
    print("  (等价于上游 test_encoder_instance_zero_kv_cache ec_producer)")

    from vllm.config import (
        CacheConfig,
        ECTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.utils.torch_utils import set_default_torch_num_threads
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.abstract import Executor

    model_config = ModelConfig(
        model=MODEL,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        disable_hybrid_kv_cache_manager=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.01,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECExampleConnector",
        ec_role="ec_producer",
        ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test_producer"},
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        ec_transfer_config=ec_transfer_config,
    )

    executor_class = Executor.get_class(vllm_config)
    print(f"  executor_class: {executor_class}")

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

    # 上游 Check: encoder_cache_manager should exist
    check(
        "encoder_cache_manager exists",
        engine_core.scheduler.encoder_cache_manager is not None,
    )

    # 上游 Check 1: num_blocks should be 1 (only null_block)
    kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
    print(f"  kv_cache_config: num_blocks={kv_cache_config.num_blocks}, "
          f"groups={len(kv_cache_config.kv_cache_groups)}, "
          f"tensors={len(kv_cache_config.kv_cache_tensors)}")
    check(
        "ec_producer: num_blocks == 1",
        kv_cache_config.num_blocks == 1,
        f"got {kv_cache_config.num_blocks}",
    )

    # 上游 Check 2: kv_cache_groups should be empty
    check(
        "ec_producer: kv_cache_groups is empty",
        len(kv_cache_config.kv_cache_groups) == 0,
        f"got {len(kv_cache_config.kv_cache_groups)}",
    )

    # 上游 Check 3: kv_cache_tensors should be empty
    check(
        "ec_producer: kv_cache_tensors is empty",
        len(kv_cache_config.kv_cache_tensors) == 0,
        f"got {len(kv_cache_config.kv_cache_tensors)}",
    )

    # 上游 Check 4: EC connector is initialized and is producer
    check(
        "ec_producer: EC connector exists",
        engine_core.scheduler.ec_connector is not None,
    )
    if engine_core.scheduler.ec_connector is not None:
        check(
            "ec_producer: EC connector is producer",
            engine_core.scheduler.ec_connector.is_producer,
        )

    # 上游 Check 5: chunked prefill is disabled
    check(
        "ec_producer: chunked prefill disabled",
        not vllm_config.scheduler_config.enable_chunked_prefill,
        f"enable_chunked_prefill={vllm_config.scheduler_config.enable_chunked_prefill}",
    )

    del engine_core

    # 上游用 @create_new_process_for_each_test() 每个测试在独立子进程中运行。
    # 我们在同一进程中，需要重置全局 _EC_CONNECTOR_AGENT 避免污染 consumer 测试。
    import vllm.distributed.ec_transfer.ec_transfer_state as ec_state
    ec_state._EC_CONNECTOR_AGENT = None


# ============================================================
# 测试: ec_consumer 角色的 KV cache 配置
# 等价于上游 test_encoder_instance_zero_kv_cache("ec_consumer", 0.7, False, False)
# ============================================================
def test_ec_consumer():
    print("\n===== 测试: ec_consumer KV cache 配置 =====")
    print("  (等价于上游 test_encoder_instance_zero_kv_cache ec_consumer)")

    from vllm.config import (
        CacheConfig,
        ECTransferConfig,
        ModelConfig,
        SchedulerConfig,
        VllmConfig,
    )
    from vllm.utils.torch_utils import set_default_torch_num_threads
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.executor.abstract import Executor

    model_config = ModelConfig(
        model=MODEL,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=512,
        max_model_len=512,
        disable_hybrid_kv_cache_manager=True,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=16,
        gpu_memory_utilization=0.7,
        cache_dtype="auto",
        enable_prefix_caching=False,
    )
    ec_transfer_config = ECTransferConfig(
        ec_connector="ECExampleConnector",
        ec_role="ec_consumer",
        ec_connector_extra_config={"shared_storage_path": "/tmp/ec_test_consumer"},
    )

    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        ec_transfer_config=ec_transfer_config,
    )

    executor_class = Executor.get_class(vllm_config)
    print(f"  executor_class: {executor_class}")

    with set_default_torch_num_threads(1):
        engine_core = EngineCore(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=True,
        )

    # 上游 Check: encoder_cache_manager should exist
    check(
        "encoder_cache_manager exists",
        engine_core.scheduler.encoder_cache_manager is not None,
    )

    # 上游 Check 1: num_blocks should be > 1
    kv_cache_config = engine_core.scheduler.kv_cache_manager.kv_cache_config
    print(f"  kv_cache_config: num_blocks={kv_cache_config.num_blocks}, "
          f"groups={len(kv_cache_config.kv_cache_groups)}, "
          f"tensors={len(kv_cache_config.kv_cache_tensors)}")
    check(
        "ec_consumer: num_blocks > 1",
        kv_cache_config.num_blocks > 1,
        f"got {kv_cache_config.num_blocks}",
    )

    # 上游 Check 2: kv_cache_groups should NOT be empty
    check(
        "ec_consumer: kv_cache_groups not empty",
        len(kv_cache_config.kv_cache_groups) > 0,
        f"got {len(kv_cache_config.kv_cache_groups)}",
    )

    # 上游 Check 3: EC connector is consumer
    check(
        "ec_consumer: EC connector exists",
        engine_core.scheduler.ec_connector is not None,
    )
    if engine_core.scheduler.ec_connector is not None:
        check(
            "ec_consumer: EC connector is consumer (not producer)",
            not engine_core.scheduler.ec_connector.is_producer,
        )

    del engine_core


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("mm_encoder_only 上游等价测试")
    print("等价于: test_encoder_instance_zero_kv_cache")
    print("=" * 60)

    try:
        test_ec_producer()
    except Exception as e:
        print(f"  [ERROR] ec_producer 测试异常: {e}")
        traceback.print_exc()

    try:
        test_ec_consumer()
    except Exception as e:
        print(f"  [ERROR] ec_consumer 测试异常: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"结果: {passed} PASS, {failed} FAIL")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
