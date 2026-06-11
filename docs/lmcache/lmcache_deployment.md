# LMCache vLLM-Ascend 部署指南

## 1. 环境准备

```bash
# 1. 安装系统依赖（编译 triton 扩展需要 Python 开发头文件）
sudo dnf install python3-devel   # 或 apt install python3-devel

# 2. Source CANN 环境（设置 LD_LIBRARY_PATH、ATB 库路径等，必须执行）
source /usr/local/Ascend/cann-8.5.1/set_env.sh

# 3. 确保模型文件可访问（共享目录需检查父目录执行权限）
# 如遇 Permission denied，将模型复制到本地路径
```

**注意事项**:
- CANN `set_env.sh` 必须在启动服务前 source，否则 `OSError: Could not load library`
- 某些系统 Python 实际加载 `lib64/python3.11/site-packages/`，修改 `lib/` 不生效
- spawn 子进程不继承 `sys.path`，CANN 路径需加入 `PYTHONPATH`

---

## 2. 版本兼容性

以下为经过验证的组件版本组合：

| 组件 | 版本 | 来源 | 说明 |
|------|------|------|------|
| **vLLM** | main (~v0.19-v0.20), commit `6f786f2c5` | https://github.com/vllm-project/vllm | 推理引擎，上游未修改 |
| **vLLM-Ascend** | 0.1.dev2943+g39b8abb7e | https://github.com/vllm-project/vllm-ascend | Ascend NPU 适配层，新增 1 个 connector 文件 |
| **LMCache** | v0.4.4 + staging 补丁 | https://github.com/PPParticle/LMCache (`feat/staging_cpu_size` 分支) | 缓存引擎核心，fork 自上游 v0.4.4，新增 SSD staging 隔离 |
| **LMCache-Ascend** | 0.1.dev113 | https://github.com/LMCache/LMCache-Ascend | LMCache 的 NPU 适配插件，monkey-patch 方式接入 |
| **CANN** | 8.5.1 | 系统安装 | Ascend 计算框架 |
| **PyTorch** | 2.9.0+cpu | pip | CPU 版本即可，NPU 由 torch_npu 提供 |
| **torch_npu** | 2.9.0rc1 | pip | Ascend NPU 的 PyTorch 后端 |
| **Python** | 3.11.x | — | 3.10-3.12 均可 |

**版本选择说明**:
- vLLM 和 vLLM-Ascend 版本需匹配，具体兼容关系参考 [vLLM-Ascend README](https://github.com/vllm-project/vllm-ascend)
- LMCache 需使用 fork 仓库的 `feat/staging_cpu_size` 分支（基于上游 v0.4.4 + SSD staging 隔离补丁），不支持直接 `pip install lmcache`
- LMCache-Ascend 版本需与 LMCache v0.4.x 兼容

---

## 3. 源码安装

```bash
# 1. Clone vLLM（建议 checkout 到指定 commit）
git clone https://github.com/vllm-project/vllm.git
cd vllm && git checkout 6f786f2c5 && pip install -e . && cd ..

# 2. Clone vLLM-Ascend
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -e . && cd ..

# 3. Clone 并安装 LMCache（含 SSD staging 隔离补丁）
#    注意：必须使用 fork 仓库，上游 pip 版本不含 staging 隔离功能
git clone -b feat/staging_cpu_size https://github.com/PPParticle/LMCache.git
cd LMCache && NO_CUDA_EXT=1 pip install -e . --no-build-isolation && cd ..

# 4. Clone 并安装 LMCache-Ascend（需要 --recurse-submodules 拉取 kvcache-ops）
git clone --recurse-submodules https://github.com/LMCache/LMCache-Ascend.git
cd LMCache-Ascend && pip install -v --no-build-isolation -e .
```

---

## 4. 验证 Import 链路

```python
import lmcache              # v0.4.4 ✓
import lmcache_ascend       # monkey-patch 生效 ✓
import lmcache.v1.gpu_connector  # NPU connector ✓
```

---

## 5. 启动服务

```bash
PYTHONPATH=<vllm_path>:<vllm-ascend_path>:$PYTHONPATH \
VLLM_PLUGINS=ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=8 \
LMCACHE_STAGING_CPU_SIZE=2 \
LMCACHE_LOCAL_DISK=file://<disk_cache_path>/ \
LMCACHE_MAX_LOCAL_DISK_SIZE=30 \
LMCACHE_CHUNK_SIZE=128 \
LMCACHE_SAVE_UNFULL_CHUNK=True \
python3 -m vllm.entrypoints.openai.api_server \
    --port 8100 \
    --model <model_path> \
    --trust-remote-code \
    --block-size 128 \
    --dtype float16 \
    --max-model-len 32768 \
    --max-num-batched-tokens 32768 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

---

## 6. 参数说明

### vLLM 参数

| 参数 | 值 | 说明 |
|------|---|------|
| `--block-size` | 128 | vLLM KV cache block 大小，必须与 `LMCACHE_CHUNK_SIZE` 匹配 |
| `--max-model-len` | 32768 | 最大上下文长度 |
| `--gpu-memory-utilization` | 0.90 | NPU 显存利用率 |
| `--enforce-eager` | — | 禁用 CUDA graph（Ascend NPU 兼容） |
| `--kv-transfer-config` | `{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}` | 指定使用 LMCache connector |

### 环境变量

| 变量 | 值 | 说明 |
|------|---|------|
| `VLLM_PLUGINS` | `ascend,ascend_kv_connector,ascend_model_loader,ascend_service_profiling` | **必须全部列出**，只写 `ascend` 会丢失 kv_connector 插件 |
| `VLLM_WORKER_MULTIPROC_METHOD` | `spawn` | 子进程启动方式 |
| `PYTHONPATH` | `<vllm_path>:<vllm-ascend_path>` | 源码路径 |

### LMCache 缓存参数

| 变量 | 值 | 说明 |
|------|---|------|
| `LMCACHE_LOCAL_CPU` | True | 启用 CPU 内存缓存 |
| `LMCACHE_MAX_LOCAL_CPU_SIZE` | 8 | CPU 缓存总量 (GB) |
| `LMCACHE_STAGING_CPU_SIZE` | 2 | SSD staging 暂存区大小 (GB)，从 CPU 总量中隔离。实际 hot_cache = 8 - 2 = 6GB |
| `LMCACHE_LOCAL_DISK` | `file://<path>/` | SSD 磁盘缓存路径 |
| `LMCACHE_MAX_LOCAL_DISK_SIZE` | 30 | SSD 磁盘缓存容量 (GB) |
| `LMCACHE_CHUNK_SIZE` | 128 | KV cache 分块大小，必须与 `--block-size` 匹配 |
| `LMCACHE_SAVE_UNFULL_CHUNK` | True | 保存不完整 chunk（否则小请求无法缓存） |

### 缓存配置建议

| KV 总量 | 推荐配置 |
|---------|---------|
| < 6GB | `staging_cpu_size=0`，全部 CPU 用于 hot_cache |
| 6-30GB | `staging_cpu_size=2`，hot_cache + SSD staging |
| > 30GB | 增大 SSD 容量，`staging_cpu_size=4` |
