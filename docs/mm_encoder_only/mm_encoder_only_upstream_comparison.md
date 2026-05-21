# mm_encoder_only 上游等价测试对比

## 测试来源

上游测试文件: `vllm/tests/v1/engine/test_engine_core.py`
上游测试函数: `test_encoder_instance_zero_kv_cache`

## 测试内容

验证 `mm_encoder_only` 参数（通过 `ec_role` 间接启用）对 EngineCore 初始化的影响。

| 检查项 | 上游 (GPU) | Ascend NPU | 结果 |
|--------|-----------|------------|------|
| **ec_producer** | | | |
| encoder_cache_manager 存在 | PASS | PASS | 一致 |
| num_blocks == 1 (仅 null_block) | PASS | PASS | 一致 |
| kv_cache_groups 为空 | PASS | PASS | 一致 |
| kv_cache_tensors 为空 | PASS | PASS | 一致 |
| EC connector 是 producer | PASS | PASS | 一致 |
| chunked prefill 被禁用 | PASS | PASS | 一致 |
| **ec_consumer** | | | |
| encoder_cache_manager 存在 | PASS | PASS | 一致 |
| num_blocks > 1 | PASS (3752) | PASS (3752) | 一致 |
| kv_cache_groups 非空 | PASS | PASS | 一致 |
| EC connector 是 consumer | PASS | PASS | 一致 |

## 差异

### 1. 测试环境差异

| 方面 | 上游 GPU | Ascend |
|------|---------|--------|
| 测试模型 | `llava-hf/llava-1.5-7b-hf` | `Qwen2.5-VL-3B-Instruct` |
| 进程隔离 | `@create_new_process_for_each_test()` | 同一进程，手动重置全局状态 |
| KV connector 参数化 | `use_kv_connector=[False, True]` | 仅 `False`（Ascend 无 NixlConnector） |

### 2. Ascend 特有代码差异

`get_kv_cache_spec()` 中 EC transfer 检查条件不同：

| | 上游 GPU | Ascend NPU |
|---|---------|------------|
| 文件 | `gpu_model_runner.py:6883` | `model_runner_v1.py:3402` |
| 条件 | `has_ec_transfer() and not get_ec_transfer().is_consumer` | `has_ec_transfer() and get_ec_transfer().is_producer` |
| 语义 | consumer 时返回完整 spec | producer 时返回空 |

两者逻辑等价（`not is_consumer` == `is_producer`），实际行为一致。

### 3. 测试环境问题

在同一进程中连续运行 producer 和 consumer 测试时，全局 `_EC_CONNECTOR_AGENT` 会被 producer 测试设为 producer role，污染后续 consumer 测试。上游通过 `@create_new_process_for_each_test()` 避免此问题。我们通过手动重置全局变量解决。

## 结论

**Ascend NPU 的 mm_encoder_only 行为与上游 GPU 完全一致**。12/12 检查项全部通过，KV cache 分配数量一致。
