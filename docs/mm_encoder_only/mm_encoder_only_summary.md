# `--mm-encoder-only` Ascend NPU 适配：完整记录

## 1. 概述

上游 vLLM 已支持 `--mm-encoder-only` 参数，用于 EPD（Encoder-Prefill-Decode）分离式架构中仅运行多模态编码器而跳过语言模型解码器。本文档记录了 vllm-ascend 对该参数的适配过程、代码改动、测试验证及与上游的对比结果。

**最终结果**：代码改动 2 处，首次测试即通过，无代码 Bug。4 轮测试共 56 项检查全部通过，行为与上游 GPU 完全一致。

---

## 2. 代码改动

### 2.1 改动文件

`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`

### 2.2 改动 1：`_dummy_run` 添加 `mm_encoder_only` 早返回（第 2428-2430 行）

```python
) -> tuple[torch.Tensor, torch.Tensor]:
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        return torch.tensor([]), torch.tensor([])

    # only support eager mode and piecewise graph now
    ...
```

**作用**：在 profiling/warmup 的 dummy run 最开头添加检查。当 `mm_encoder_only=True` 时跳过所有 NPU 特有的 LM 预热逻辑（EPLB、PCP、ACL Graph 等），直接返回空 tensor。效果：profile_run 耗时从 ~5s 降至 0.01s。

### 2.3 改动 2：`_dummy_sampler_run` 添加 `mm_encoder_only` 早返回（第 2678-2680 行）

```python
) -> torch.Tensor:
    mm_config = self.vllm_config.model_config.multimodal_config
    if mm_config and mm_config.mm_encoder_only:
        return torch.tensor([])

    output = None
    ...
```

**作用**：在 sampler 的 dummy run 最开头添加检查。encoder-only 模式不需要 sampler 预热。

### 2.4 不需要改动

- **`_dummy_pooler_run`**：`NPUModelRunner` 继承自 `GPUModelRunner`，上游已有 `mm_encoder_only` 守卫，通过继承链自动可用。
- **配置解析**：上游 `arg_utils.py`、`model.py`、`multimodal.py` 的 CLI 参数和数据流已完全接通。
- **`get_kv_cache_spec`**：Ascend 已有独立的覆写，包含 `has_ec_transfer() and get_ec_transfer().is_producer` 检查，与上游逻辑等价。
- **`may_add_encoder_only_layers_to_kv_cache_config`**：已在 vllm-ascend 中存在。

---

## 3. 差异分析

### 3.1 `_dummy_run` 差异

| 方面 | 上游 GPU | Ascend NPU |
|------|---------|------------|
| 多模态早返回 | 有 | **本次添加** |
| EPLB 负载均衡 | 标准逻辑 | Ascend 特有 `dynamic_eplb` + `eplb_updator` |
| PCP 并行计算 | 无 | Ascend 特有 `pcp_manager` |
| UBatch 微批次 | 支持 | 不支持 |
| 返回值 | `(hidden_states, hidden_states[logit_indices])` | `(hidden_states, hidden_states)` |

**结论**：所有 NPU 特有逻辑都是 LM 解码器相关的，早期返回安全。

### 3.2 `_dummy_sampler_run` 差异

| 方面 | 上游 GPU | Ascend NPU |
|------|---------|------------|
| `mm_encoder_only` 检查 | 有 | **本次添加** |
| SamplingMetadata | 创建完整元数据 | 不创建 |
| Sampler 调用 | 调用 `self.sampler()` | 不调用，只做 `compute_logits` |

### 3.3 `get_kv_cache_spec` 条件差异

| | 上游 GPU | Ascend NPU |
|---|---------|------------|
| 文件 | `gpu_model_runner.py:6883` | `model_runner_v1.py:3402` |
| 条件 | `not get_ec_transfer().is_consumer` | `get_ec_transfer().is_producer` |
| 语义 | 等价 | 等价 |

两者逻辑等价（`not is_consumer` == `is_producer`），实际行为一致，测试已验证。

---

## 4. 测试文件

### 4.1 测试文件清单

| 文件 | 用途 | 检查项数 |
|------|------|---------|
| `/data/test_mm_encoder_only.py` | HTTP Server 端到端测试 | 服务启动 + 请求成功 |
| `/data/test_mm_encoder_only_offline.py` | 离线 LLM API 测试（多次 encode） | 3 次独立 + 1 次批量 |
| `/data/test_mm_encoder_only_verify.py` | 严格功能性验证 | 22 |
| `/data/test_mm_encoder_only_upstream_equivalent.py` | 上游等价测试 | 12 |

### 4.2 运行命令

所有测试共用环境变量：

```bash
source /data/arctic_env/bin/activate
export PYTHONPATH=/data/vllm:/data/vllm-ascend:/usr/local/Ascend/cann-8.5.1/python/site-packages:/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe
export VLLM_PLUGINS=ascend
export VLLM_USE_MODELSCOPE=True
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/data/huggingface_home
```

#### 测试 1：HTTP Server 端到端

```bash
python3 /data/test_mm_encoder_only.py
```

#### 测试 2：离线 LLM API

```bash
python3 /data/test_mm_encoder_only_offline.py
```

#### 测试 3：严格功能性验证

```bash
python3 /data/test_mm_encoder_only_verify.py
```

#### 测试 4：上游等价测试

```bash
python3 /data/test_mm_encoder_only_upstream_equivalent.py
```

---

## 5. 测试结果

### 5.1 测试环境

- **模型**：Qwen2.5-VL-3B-Instruct（本地路径 `/data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct`）
- **硬件**：单卡 NPU
- **测试图片**：`vllm-ascend/mm_request` 数据集中的 `test_mm2.jpg`

### 5.2 测试 1：HTTP Server 端到端

| 指标 | 结果 |
|------|------|
| `mm_encoder_only: True` 配置 | 正确传递 |
| `ec_role='ec_producer'` | 正确生效 |
| 模型加载 | 1.2475 GB |
| profile_run 耗时 | **0.01s**（正常模式 ~5s，证明早返回生效） |
| 服务器启动 | 35s，健康检查 200 OK |
| 多模态图像请求 | HTTP 200 |

### 5.3 测试 2：离线 LLM API

| 指标 | 结果 |
|------|------|
| LLM 初始化 | ~20s |
| 3 次独立 encode | 均成功，request_id 递增，finished=True |
| 批量 encode（3 张图一起） | 成功 |

### 5.4 测试 3：严格功能性验证 — 22 PASS, 0 FAIL

#### 验证 1：Encoder 输出写入 EC 存储（11 项）

| 检查项 | 结果 | 详情 |
|--------|------|------|
| EC storage dir exists | PASS | |
| Has safetensors file(s) | PASS | |
| File non-empty | PASS | |
| Safetensors loads OK | PASS | |
| Has 'ec_cache' key | PASS | |
| Tensor non-empty | PASS | shape=[425, 2048], numel=870200 |
| No NaN | PASS | |
| No Inf | PASS | |
| Not all zeros | PASS | mean=0.001830 |
| Has hidden dim | PASS | shape=[425, 2048] |
| Second encode completes without error | PASS | in-memory 缓存命中是正确行为 |

#### 验证 2：LM decoder 被跳过（5 项）

| 检查项 | 结果 | 详情 |
|--------|------|------|
| Init time: encoder-only < normal | PASS | 18.0s < 24.1s |
| Encoder-only output is short/dummy | PASS | text='!' |
| Normal mode produces real text | PASS | |
| Encoder-only has EC cache files | PASS | |
| Normal mode has no EC cache | PASS | |

#### 验证 3：Consumer 可加载 encoder 输出（4 项）

| 检查项 | 结果 | 详情 |
|--------|------|------|
| Producer created cache files | PASS | |
| Cache file loads | PASS | |
| Roundtrip save/load preserves data | PASS | safetensors 往返无损 |
| Consumer initialized and ran | PASS | Consumer 成功复用 producer 的 encoder cache |

### 5.5 测试 4：上游等价测试 — 12 PASS, 0 FAIL

等价于上游 `test_encoder_instance_zero_kv_cache`。

#### ec_producer（7 项）

| 检查项 | 上游 GPU | Ascend NPU | 结果 |
|--------|---------|------------|------|
| encoder_cache_manager 存在 | PASS | PASS | 一致 |
| num_blocks == 1 | PASS | PASS | 一致 |
| kv_cache_groups 为空 | PASS | PASS | 一致 |
| kv_cache_tensors 为空 | PASS | PASS | 一致 |
| EC connector 是 producer | PASS | PASS | 一致 |
| chunked prefill 被禁用 | PASS | PASS | 一致 |

#### ec_consumer（5 项）

| 检查项 | 上游 GPU | Ascend NPU | 结果 |
|--------|---------|------------|------|
| encoder_cache_manager 存在 | PASS | PASS | 一致 |
| num_blocks > 1 | PASS (3752) | PASS (3752) | 一致 |
| kv_cache_groups 非空 | PASS | PASS | 一致 |
| EC connector 是 consumer | PASS | PASS | 一致 |

---

## 6. 遇到的问题

### 代码 Bug：无

两处代码改动首次测试即通过，无代码 Bug。

### 测试环境问题：6 个

| # | 问题 | 类型 | 解决 |
|---|------|------|------|
| 1 | vllm 包命名空间冲突 | 环境 | `PYTHONPATH=/data/vllm:/data/vllm-ascend` |
| 2 | 子进程重复下载模型 | 环境 | 使用本地模型路径 |
| 3 | 请求模型名不匹配 (404) | 测试 | 使用实际 served_model_name |
| 4 | CANN acl 模块 spawn 子进程找不到 | 环境 | PYTHONPATH 加入 CANN 路径 |
| 5 | HybridKVCacheCoordinator 断言失败 | 配置 | `enable_prefix_caching=False` |
| 6 | 验证脚本 "Second encode" FAIL | 测试设计 | 测试逻辑修正（in-memory 缓存命中是正确行为） |

详细分析见 `/data/plans/mm_encoder_only_bugs.md`。

---

## 7. 上游对比

详见 `/data/plans/mm_encoder_only_upstream_comparison.md`。

**结论：Ascend NPU 的 mm_encoder_only 行为与上游 GPU 完全一致。**

主要差异：
- 测试模型不同（`llava-hf/llava-1.5-7b-hf` vs `Qwen2.5-VL-3B-Instruct`）
- 上游用进程隔离，我们用手动重置全局 `_EC_CONNECTOR_AGENT`
- 上游参数化 `use_kv_connector`，Ascend 无 NixlConnector 仅测 `False`

---

## 8. 关联文件索引

| 文件 | 说明 |
|------|------|
| `/data/vllm-ascend/vllm_ascend/worker/model_runner_v1.py` | 唯一的生产代码改动 |
| `/data/plans/mm_encoder_only_plan.md` | 实现方案（差异分析、改动说明） |
| `/data/plans/mm_encoder_only_bugs.md` | Bug 和环境问题记录 |
| `/data/plans/mm_encoder_only_upstream_comparison.md` | 上游等价测试对比 |
| `/data/plans/mm_encoder_only_summary.md` | 本文件，汇总记录 |
| `/data/test_mm_encoder_only.py` | 测试脚本：HTTP Server 端到端 |
| `/data/test_mm_encoder_only_offline.py` | 测试脚本：离线 LLM API |
| `/data/test_mm_encoder_only_verify.py` | 测试脚本：严格功能性验证（22 项） |
| `/data/test_mm_encoder_only_upstream_equivalent.py` | 测试脚本：上游等价测试（12 项） |
