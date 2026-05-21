# `--mm-encoder-only` 参数支持：完整实现记录

## 1. 背景

上游 vLLM 已支持 `--mm-encoder-only` 参数，允许模型仅运行多模态编码器（MM Encoder）而跳过语言模型（LM）解码器。该功能分布在三层：

1. **配置/参数解析** — `arg_utils.py`、`model.py`、`multimodal.py`：CLI 参数和数据流已完全接通
2. **GPU Model Runner 守卫** — `gpu_model_runner.py`：在 `_dummy_run`、`_dummy_sampler_run`、`_dummy_pooler_run` 中添加提前返回
3. **Encoder-only KV Cache** — `may_add_encoder_only_layers_to_kv_cache_config`：已存在于 vllm-ascend

## 2. 差异分析

### 2.1 `_dummy_run` 差异

| 方面 | 上游 GPU | Ascend NPU |
|------|---------|------------|
| 多模态早返回 | **有** (`mm_encoder_only` 检查) | **缺失** |
| EPLB 负载均衡 | 标准逻辑 | Ascend 特有 `dynamic_eplb` + `eplb_updator` |
| PCP 并行计算 | 无 | Ascend 特有 `pcp_manager` 初始化 |
| UBatch 微批次 | 支持 | 不支持 (`ubatch_slices=None`) |
| Mixed batch | 支持 | 不支持 (`NotImplementedError`) |
| 返回值 | `(hidden_states, hidden_states[logit_indices])` | `(hidden_states, hidden_states)` |
| 序列长度 | 直接用 `max_query_len` | 图捕获时用固定常量 `SEQ_LEN_WITH_MAX_PA_WORKSPACE=6144` |

**结论**：可以安全添加相同的 `mm_encoder_only` 早返回逻辑。

### 2.2 `_dummy_sampler_run` 差异

| 方面 | 上游 GPU | Ascend NPU |
|------|---------|------------|
| `mm_encoder_only` 检查 | **有** | **缺失** |
| 隐藏状态处理 | 用 `torch.rand_like` 替换 | 直接使用原始值 |
| SamplingMetadata | 创建完整元数据 | **不创建** |
| Sampler 调用 | 调用 `self.sampler()` | **不调用**，只做 `compute_logits` |

**结论**：同样可以添加 `mm_encoder_only` 早返回。

### 2.3 `_dummy_pooler_run` 分析

- `NPUModelRunner` 继承自 `GPUModelRunner`（`class NPUModelRunner(GPUModelRunner)`）
- 上游 `GPUModelRunner` 已定义 `_dummy_pooler_run`，且包含 `mm_encoder_only` 守卫
- **通过继承链自动可用，不需要额外改动**

### 2.4 NPU 特有初始化在 encoder-only 模式下的必要性

| NPU 特有步骤 | encoder-only 下是否需要 | 原因 |
|------------|----------------------|------|
| `eplb_updator.forward_before()` | 不需要 | EPLB 是 MoE 专家负载均衡，encoder-only 不运行 LM |
| `pcp_manager.init_batch_info()` | 不需要 | PCP 是上下文并行，encoder-only 不运行 LM |
| `AscendAttentionState` 设置 | 不需要 | LM 解码器注意力状态 |
| `update_cos_sin()` | 不需要 | 旋转位置编码，用于 LM |
| `lmhead_tp_enable` dummy logits | 不需要 | LM 头部 TP 优化 |
| ACL Graph 捕获 | 不需要 | encoder-only 不需要图捕获 |

**结论**：所有 NPU 特有逻辑都是 LM 解码器相关的，早期返回安全。

## 3. 实现改动

### 改动文件

`vllm-ascend/vllm_ascend/worker/model_runner_v1.py`

### 改动 1：`_dummy_run` 添加早返回（第 2428-2430 行）

在方法最开头（所有 Ascend 逻辑之前）添加：

```python
mm_config = self.vllm_config.model_config.multimodal_config
if mm_config and mm_config.mm_encoder_only:
    return torch.tensor([]), torch.tensor([])
```

### 改动 2：`_dummy_sampler_run` 添加早返回（第 2678-2680 行）

在方法最开头添加：

```python
mm_config = self.vllm_config.model_config.multimodal_config
if mm_config and mm_config.mm_encoder_only:
    return torch.tensor([])
```

### 不需要改动

- `_dummy_pooler_run`：通过继承已自动支持
- 配置解析：已在上游完全接通

## 4. 端到端测试

### 测试环境

- **模型**：Qwen2.5-VL-3B-Instruct（从 ModelScope 下载，存于 `/data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct`）
- **硬件**：单卡 NPU
- **测试脚本**：`/data/test_mm_encoder_only.py`

### 测试配置

```bash
vllm serve /data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct \
    --port <port> \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size 1 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 1 \
    --mm-encoder-only \
    --ec-transfer-config '{"ec_connector_extra_config":{"shared_storage_path":"/tmp/mm_encoder_only_test_storage"},"ec_connector":"ECExampleConnector","ec_role":"ec_producer"}'
```

### 验证结果

| 指标 | 结果 |
|------|------|
| `mm_encoder_only: True` 配置 | 正确传递 |
| `ec_role='ec_producer'` | 正确生效 |
| 模型加载 | 1.2475 GB，成功 |
| `profile_run` 耗时 | **0.01s**（对比正常数十秒，证明早返回生效） |
| 服务器启动 | 35s，健康检查 200 OK |
| 多模态图像请求 | HTTP 200，正常返回 chat completion |

### 关键日志

```
Loading model weights took 1.2475 GB
init engine (profile, create kv cache, warmup model) took 0.01 s
Application startup complete.
GET /health HTTP/1.1" 200 OK
POST /v1/chat/completions HTTP/1.1" 200 OK
```

### 如何复现验证

```bash
# 1. 激活环境
source /data/arctic_env/bin/activate

# 2. 运行测试脚本
VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HF_HOME=/data/huggingface_home \
    python3 /data/test_mm_encoder_only.py

# 3. 或手动启动服务器验证
PYTHONPATH=/data/vllm:/data/vllm-ascend \
VLLM_PLUGINS=ascend VLLM_USE_MODELSCOPE=True \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
HF_HOME=/data/huggingface_home \
vllm serve /data/huggingface_home/Qwen/Qwen2___5-VL-3B-Instruct \
    --port 8000 --gpu-memory-utilization 0.90 --tensor-parallel-size 1 \
    --enforce-eager --max-model-len 4096 --max-num-seqs 1 \
    --mm-encoder-only \
    --ec-transfer-config '{"ec_connector_extra_config":{"shared_storage_path":"/tmp/ec_storage"},"ec_connector":"ECExampleConnector","ec_role":"ec_producer"}'
```

## 5. 代码无 Bug

实现首次测试即通过，无代码 Bug。测试过程中遇到的环境问题（vllm 命名空间冲突、子进程重复下载模型、请求模型名不匹配）已记录在 `/data/plans/mm_encoder_only_bugs.md`。
