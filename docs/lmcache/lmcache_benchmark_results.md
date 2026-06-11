# LMCache vLLM-Ascend 测试记录

---

## 测试环境

### 硬件

| 项目 | 规格 |
|------|------|
| NPU | Ascend 910B2C, 60GB HBM |
| CPU | x86_64 |
| SSD | 本地磁盘，用于 KV cache 磁盘缓存 |

### 软件

| 组件 | 版本 |
|------|------|
| vLLM | main (~v0.19-v0.20), commit `6f786f2c5` |
| vLLM-Ascend | 0.1.dev2943+g39b8abb7e |
| LMCache | v0.4.4 + staging 补丁 | fork: `https://github.com/PPParticle/LMCache.git`, 分支 `feat/staging_cpu_size` |
| LMCache-Ascend | v0.1.dev113, commit `e7fa9fe` |
| PyTorch | 2.9.0+cpu |
| torch_npu | 2.9.0rc1 |
| CANN | 8.5.1 |
| Python | 3.11.6 |

### 模型与数据集

| 项目 | 说明 |
|------|------|
| 模型 | Qwen2.5-7B-Instruct |
| 数据集 | SCBench (microsoft/SCBench), `scbench_qa_eng` 子集（69 samples, 5+ multi-turns, context > 70k tokens） |
| KV 大小 | 56 KB/token (28 layers × 2(K+V) × 4 KV heads × 128 dim × fp16) |

### 测试矩阵

```
ctx=8k:   ws=8, 16, 32    (KV: 3.5GB, 7.0GB, 14.0GB)
ctx=16k:  ws=8, 16, 32    (KV: 7.0GB, 14.0GB, 28.0GB)
ctx=32k:  ws=8, 16        (KV: 14.0GB, 28.0GB, ws32 需 56GB 超出可用容量)
```

共 8 个测试点。每个测试点流程：
1. 启动全新 vLLM 服务（禁用 prefix cache，启用 LMCache）
2. Cold pass：发送 N 个不同上下文 + query_1 → 填充缓存，测量 TTFT
3. Warm pass：发送 N 个相同上下文 + query_2 → 缓存命中，测量 TTFT

---

## 实验参数设置

### 服务启动参数

```
python3 -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model <model_path> \
    --trust-remote-code \
    --block-size 128 \
    --dtype float16 \
    --max-model-len <ctx_len>      # 8192 / 16384 / 32768
    --max-num-batched-tokens <ctx_len>
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

### 环境变量

| 变量 | Phase 1 值 | Phase 2 值 | 说明 |
|------|-----------|-----------|------|
| `VLLM_PLUGINS` | `ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling` | 同左 | 全部列出 |
| `LMCACHE_LOCAL_CPU` | True | True | 启用 CPU 缓存 |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | 8 | 8 | CPU 缓存总量 (GB) |
| `LMCACHE_STAGING_CPU_SIZE` | — (未设置) | 2 | SSD staging 暂存区 (GB) |
| `LMCACHE_LOCAL_DISK` | `file://<ssd_path>/` | 同左 | SSD 路径 |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | 20 | 30 | SSD 容量 (GB) |
| `LMCACHE_CHUNK_SIZE` | 128 | 128 | 分块大小 |
| `LMCACHE_SAVE_UNFULL_CHUNK` | True | True | 保存不完整 chunk |

### Baseline 对比

Baseline 使用相同服务参数，但不设置 LMCache 环境变量，`VLLM_PLUGINS` 不含 `ascend_kv_connector`，不传 `--kv-transfer-config`。

---

## 测试脚本

### 目录结构

```
tests/lmcache/
├── test_lmcache_ascend.py        # 功能验证：发送两个相同前缀请求，确认缓存命中
├── benchmark_working_set.py      # 性能测试：SCBench 工作集基准测试（8 测试点）
├── run_benchmark_ws.sh           # 运行脚本：Working Set Benchmark 启动器
├── run_benchmark_all.sh          # 运行脚本：全量测试（Phase 1 配置）
└── run_benchmark_all_staging.sh  # 运行脚本：全量测试（Phase 2 staging 配置）
```

### 功能验证：`test_lmcache_ascend.py`

验证 LMCache 在 Ascend NPU 上的基本 store/retrieve 功能。

```bash
# 前置：source CANN 环境，启动 vLLM 服务（参考 lmcache_deployment.md）
python3 tests/lmcache/test_lmcache_ascend.py
```

流程：
1. 向运行中的服务发送两个具有相同长前缀的请求
2. 验证第二个请求的 TTFT 低于第一个（缓存命中）

### 性能测试：`benchmark_working_set.py`

SCBench 工作集基准测试，自动管理服务启停，测量 8 个测试点的 cold/warm TTFT。

```bash
# 全量测试（自动启停服务）
source /usr/local/Ascend/cann-8.5.1/set_env.sh
bash tests/lmcache/run_benchmark_ws.sh

# 只跑指定测试点
bash tests/lmcache/run_benchmark_ws.sh --only ctx8k_ws8 ctx16k_ws16

# 连接已运行的服务（不自动启停）
bash tests/lmcache/run_benchmark_ws.sh --no-manage-server
```

输出：JSON 结果文件保存到 `<results_dir>/working_set_<timestamp>.json`

---

## 第一部分：功能性验证

### 验证目标

确认 LMCache 在 Ascend NPU 上的 store/retrieve 功能正确工作，包括 CPU 缓存和 SSD 磁盘缓存两条路径。由于 `/proc/PID/io` 在 loop-mounted 磁盘上不准确，采用源码插桩方式验证数据实际来源。

### 插桩方法

在 LMCache 的 storage backend 源文件中添加 `print` 追踪语句：

| 插桩文件 | 插桩位置 | 追踪内容 |
|---------|---------|---------|
| `storage_manager.py` | `batched_contains` | 每个 backend 的命中 chunk 数 |
| `storage_manager.py` | `get_block_mapping` | 实际命中的 backend 名称 |
| `local_cpu_backend.py` | `get_blocking` | CPU 内存读取确认 |
| `local_disk_backend.py` | `get_blocking` | SSD 磁盘读取确认及文件路径 |

**注意**: 某些系统 Python 实际加载 `lib64/python3.11/site-packages/` 下的模块，插桩时需修改 `lib64/` 路径下的文件。

### CPU 缓存命中验证

发送请求后，LMCache 通过 `LocalCPUBackend` 直接命中：

```
[TRACE_CONTAINS] backend=LocalCPUBackend, hit_chunks=2/2
[BACKEND_HIT] ✅ backend=LocalCPUBackend, hit_chunks=2/2
[CPU_READ] ✅ Data from CPU memory, key=CacheEngineKey(...)
```

**结论**: CPU 缓存 store → retrieve 链路正常。

### SSD 磁盘缓存命中验证

设置 `LMCACHE_MAX_LOCAL_CPU_SIZE=1`（1GB CPU），发送大量请求填满 CPU 缓存，使最早的请求被 LRU 淘汰到磁盘。然后重复发送最早请求，触发 SSD 读取：

```
[TRACE_CONTAINS] backend=LocalCPUBackend, hit_chunks=0/2     ← CPU 已被淘汰
[TRACE_CONTAINS] backend=LocalDiskBackend, hit_chunks=2/2    ← SSD 命中
[BACKEND_HIT] ✅ backend=LocalDiskBackend, hit_chunks=2/2
[DISK_READ] ✅ Data from SSD disk, path=.../local_disk/...-half.pt
```

**结论**: SSD 磁盘缓存 store → LRU 淘汰 → SSD retrieve 链路正常。

### LMCache 存储架构理解

通过插桩验证，确认了 LMCache 的存储层级行为：

1. **`batched_put(location=None)`** 同时存入 CPU 和 Disk 两个 backend
2. **`batched_contains`** 按创建顺序遍历：CPU 优先 → Disk 其次
3. **`LocalDiskBackend` 硬依赖 `LocalCPUBackend`**（`assert local_cpu_backend is not None`），不可禁用，CPU 缓存始终被创建作为 Disk 后端的 I/O 缓冲区
4. **LRU 淘汰机制**: CPU 缓存满后，旧数据被淘汰但保留在磁盘上（如果 Disk backend 可用）
5. **磁盘缓存不跨重启持久化**: 内存索引随进程丢失，`.pt` 文件虽在但无法命中，需重新 warm-up

### 命中率验证

服务端日志确认 LMCache 命中率：

| 请求 Tokens | LMCache Hit Tokens | 命中率 |
|-------------|-------------------|--------|
| 593 | 593 | 100% |
| 1129 | 1129 | 100% |
| 2195 | 2195 | 100% |
| 3322 | 3322 | 100% |

### SSD 额外开销测量

| 来源 | TTFT |
|------|------|
| Cold（无缓存） | 49.4ms |
| CPU cache hit | 37.8ms |
| SSD cache hit | 38.3ms |
| **SSD vs CPU 开销** | **+2.2ms** |

**结论**: SSD 磁盘读取额外开销仅 +2.2ms，对 7B+ 模型可忽略。

### 配置踩坑记录

#### `VLLM_PLUGINS` 不全导致 connector 注册失败

- **现象**: `ValueError: Unsupported connector type: LMCacheAscendConnector`
- **原因**: `VLLM_PLUGINS=ascend` 只加载 `model_runner_plugins` 组的 `ascend` 插件，`general_plugins` 组的 `ascend_kv_connector` 被过滤掉
- **解决**: `VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling`
- **根因**: vLLM 插件系统按组分类注册，vllm-ascend 注册了多组插件：

  | 插件名 | 所属组 | 作用 |
  |--------|--------|------|
  | `ascend` | `model_runner_plugins` | 模型加载、推理执行 |
  | `ascend_kv_connector` | `general_plugins` | KV connector 注册 |
  | `ascend_model_loader` | `general_plugins` | 模型加载器 |
  | `ascend_service_profiling` | `general_plugins` | 性能分析 |

#### `LMCACHE_CHUNK_SIZE` + `LMCACHE_SAVE_UNFULL_CHUNK` 导致缓存失效

- **现象**: LMCache store 存入 0 tokens，retrieve 始终 miss，hit tokens=0
- **原因**: 默认 `chunk_size=256`、`save_unfull_chunk=False`。当请求 prompt tokens 不足 256 时，最后一个不满的 chunk 被丢弃。若 prompt 总量不足 256 tokens，所有 chunk 都被丢弃
- **解决**:
  ```bash
  LMCACHE_CHUNK_SIZE=128        # 降低 chunk 粒度，与 --block-size 匹配
  LMCACHE_SAVE_UNFULL_CHUNK=True # 保存不满的 chunk
  ```
- **注意**: `chunk_size` 应与 vllm 的 `--block-size` 匹配或为其约数

---

## 第二部分：性能测试

测试分两个阶段，逐步发现和解决性能问题。

### Baseline（原始 vLLM-Ascend，无 LMCache）

每个请求重新计算全部 prefix，TTFT 完全取决于 prefill 计算。

| 场景 | KV 总量 | Baseline TTFT (ms) |
|------|---------|--------------------|
| ctx8k_ws8 | 3.5GB | 538 |
| ctx8k_ws16 | 7.0GB | 537 |
| ctx8k_ws32 | 14.0GB | 537 |
| ctx16k_ws8 | 7.0GB | 1,234 |
| ctx16k_ws16 | 14.0GB | 1,233 |
| ctx16k_ws32 | 28.0GB | 1,233 |
| ctx32k_ws8 | 14.0GB | 3,196 |
| ctx32k_ws16 | 28.0GB | 3,198 |

### Phase 1: Working Set Benchmark（CPU 8GB + SSD 20GB，无 staging 隔离）

#### 三层性能区间

| KV 总量 | 缓存状态 | Speedup | 说明 |
|---------|---------|---------|------|
| ≤ 8 GB | CPU 命中 | 6.6-9.4x | 全部在 hot_cache |
| 8-20 GB | CPU+SSD | 2.3-5.2x | 部分 SSD 命中 |
| ≥ 28 GB | 溢出 | **0.87-0.90x** ❌ | SSD 读取需要 CPU staging，但 CPU 已满 |

#### 详细数据

| Test | KV Total | Cold (ms) | Warm (ms) | Speedup | 状态 |
|------|----------|-----------|-----------|---------|------|
| ctx8k_ws8 | 3.5GB | ~500 | ~53 | **9.38x** | ✅ 全在 hot_cache |
| ctx16k_ws8 | 7.0GB | ~1,162 | ~124 | **9.38x** | ✅ 全在 hot_cache |
| ctx32k_ws8 | 14.0GB | ~3,066 | ~585 | **5.24x** | ✅ CPU+SSD |
| ctx8k_ws16 | 7.0GB | ~497 | ~178 | **2.79x** | ✅ CPU+SSD |
| ctx16k_ws16 | 14.0GB | ~1,133 | ~485 | **2.34x** | ✅ CPU+SSD |
| ctx32k_ws16 | 28.0GB | ~3,048 | ~3,393 | **0.90x** ❌ | 溢出，反而变慢 |
| ctx8k_ws32 | 14.0GB | ~472 | ~536 | **0.88x** ❌ | 溢出，反而变慢 |
| ctx16k_ws32 | 28.0GB | ~1,094 | ~1,253 | **0.87x** ❌ | 溢出，反而变慢 |

#### 问题分析

**根因**: `LocalDiskBackend` 从 SSD 加载数据时，需要 `local_cpu_backend.allocate()` 分配 CPU 暂存空间。CPU 内存池（8GB）已被 hot_cache 占满 → 分配失败 → 缓存命中率为 0% → 只有存储开销，没有命中收益。

#### LMCache cold 开销

| 上下文长度 | Cold 额外开销 | 百分比 |
|-----------|-------------|--------|
| ctx8k | +40~90ms | 8-18% |
| ctx16k | +100~180ms | 9-15% |
| ctx32k | +350ms | 11% |

### Phase 2: SSD Staging 隔离（CPU 8GB = 6GB hot + 2GB staging + SSD 30GB）

#### 改动说明

将 CPU 内存分成两部分：hot_cache 存储区 + SSD staging 暂存区。

**修改文件** (LMCache 上游, 3 个):
1. `lmcache/v1/config.py`: 添加 `staging_cpu_size` 配置项
2. `lmcache/v1/storage_backend/local_cpu_backend.py`: 新增 `allocate_staging()` / `free_staging()` 方法
3. `lmcache/v1/storage_backend/local_disk_backend.py`: 改用 staging allocator

#### 最终结果

| 场景 | KV 总量 | Cold (ms) | Warm (ms) | Speedup | 命中路径 |
|------|---------|-----------|-----------|---------|---------|
| ctx8k_ws8 | 3.5GB | 538 | 74 | **7.23x** | hot_cache |
| ctx8k_ws16 | 7.0GB | 537 | 95 | **5.68x** | hot+SSD |
| ctx8k_ws32 | 14.0GB | 537 | 118 | **4.53x** | hot+SSD |
| ctx16k_ws8 | 7.0GB | 1,234 | 185 | **6.66x** | hot+SSD |
| ctx16k_ws16 | 14.0GB | 1,233 | 167 | **7.39x** | hot+SSD |
| ctx16k_ws32 | 28.0GB | 1,233 | 414 | **2.98x** | SSD staging |
| ctx32k_ws8 | 14.0GB | 3,196 | 707 | **4.52x** | hot+SSD |
| ctx32k_ws16 | 28.0GB | 3,198 | 726 | **4.40x** | SSD staging |

#### Staging 隔离前后对比

| 场景 | 无 staging | 有 staging | 变化 |
|------|-----------|-----------|------|
| 全在 hot_cache (ws8) | 9.38x | 6.66-7.23x | hot_cache 缩小 2GB，略降 |
| SSD 可容纳 (ws16) | ~2.3-5.2x | 5.68-7.39x | SSD staging 更可靠 |
| **之前溢出变慢 (ws32)** | **0.87-0.90x** ❌ | **2.98-4.53x** ✅ | 从负优化变大幅加速! |

#### 配置建议

| KV 总量 | 推荐配置 |
|---------|---------|
| < 6GB | `staging_cpu_size=0`，全部用于 hot_cache |
| 6-30GB | `staging_cpu_size=2`，hot_cache + SSD staging |
| > 30GB | 增大 SSD 容量，`staging_cpu_size=4` |

### 关键结论

1. **SSD staging 隔离有效**: SSD 读取不再与 hot_cache 争用 CPU 内存
2. **SSD 容量必须足够**: 30GB SSD 可以容纳 28GB KV，配合 staging 实现可靠加速
3. **hot_cache 缩小有代价**: KV < 6GB 时，staging 占用的 2GB 导致 ws8 降幅约 2-3x（仍远优于 baseline）
4. **全部场景实现加速**: 2.98x-7.39x，无负优化
