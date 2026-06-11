# LMCache vLLM-Ascend 架构说明

## 1. LMCache 应用背景

LMCache 是一个 KV Cache 复用库，通过缓存已计算的 KV Cache 避免重复 prefill，从而降低 TTFT。在 LLM 服务中，多个请求经常共享相同的前缀（如 system prompt、长上下文文档），LMCache 将这些共享前缀的 KV Cache 缓存到 CPU 内存或 SSD 磁盘，后续请求命中缓存后跳过对应部分的 prefill 计算。

LMCache-Ascend 是 LMCache 的 Ascend NPU 适配插件。LMCache 原生只支持 CUDA GPU，LMCache-Ascend 在 import 时自动将 CUDA 调用替换为 NPU 调用，用户无需手动适配。

---

## 2. 四个部件的关系

```
┌─ vLLM (上游，未修改) ──────────────────────────────┐
│  提供 KVConnectorBase_V1 抽象接口                     │
│  和 LMCacheConnectorV1 委托层                        │
└────────────────────────────────────────────────────┘
         ↕
┌─ vLLM-Ascend (本项目，新增 1 个文件) ───────────────┐
│  lmcache_ascend_connector.py                         │
│    import lmcache_ascend → 触发 NPU 适配              │
│    re-export LMCacheConnectorV1                      │
└────────────────────────────────────────────────────┘
         ↕
┌─ LMCache (fork 安装，含 staging 隔离补丁) ──────────┐
│  核心引擎：LMCacheEngine, StorageBackend              │
│  新增 staging_cpu_size 配置及 allocate_staging()     │
│  默认使用 CUDA，被 LMCache-Ascend 替换                │
└────────────────────────────────────────────────────┘
         ↕
┌─ LMCache-Ascend (pip 安装，未修改) ─────────────────┐
│  自动将 CUDA 调用替换为 NPU 调用                      │
└────────────────────────────────────────────────────┘
```

**关键点**: 用户只需要在 vLLM-Ascend 中新增 1 个 5 行文件。vLLM 和 LMCache-Ascend 通过 pip 安装无需修改。LMCache 需从 fork 仓库安装（包含 SSD staging 隔离补丁）。

### 数据流（单机 KV 复用场景）

```
请求到达 → vllm scheduler 调用 get_num_new_matched_tokens()
                    ↓
         lmcache lookup_client.lookup(token_ids)
                    ↓
         返回命中 token 数 → scheduler 分配 block
                    ↓
         worker forward: start_load_kv() → lmcache_engine.retrieve()
                    ↓
         NPU connector 将 KV 从 lmcache buffer 搬到 vllm paged KV cache
                    ↓
         只计算未命中的 token → save_kv_layer() → lmcache_engine.store()
                    ↓
         NPU connector 将新 KV 存入 lmcache 供后续请求复用
```

---

## 3. 代码改动

### 本项目（vLLM-Ascend）的改动

新增 1 个文件，修改 0 个文件。

#### `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py`

```python
# SPDX-License-Identifier: Apache-2.0
import lmcache_ascend  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1

__all__ = ["LMCacheConnectorV1"]
```

**作用**: 当 vLLM 通过 `--kv-transfer-config` 指定 `LMCacheAscendConnector` 时，加载此文件。`import lmcache_ascend` 自动触发 NPU 适配，然后重导出 `LMCacheConnectorV1`。

### LMCache 上游的改动（staging 隔离）

LMCache 上游的 3 个文件被修改，用于支持 SSD staging 内存隔离：

| 文件 | 修改内容 |
|-----|---------|
| `lmcache/v1/config.py` | 添加 `staging_cpu_size` 配置项 |
| `lmcache/v1/storage_backend/local_cpu_backend.py` | 新增 `allocate_staging()` / `free_staging()` 方法 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | SSD 读取改用 staging allocator |

**解决的问题**: SSD 读取时需要 CPU 暂存空间，但 CPU 池被 hot_cache 占满时分配失败 → 缓存溢出反而变慢。将 CPU 内存分为 hot_cache + staging 两部分后解决。

---

## 4. 版本信息

| 组件 | 版本 | 来源 |
|------|------|------|
| **vLLM** | main (~v0.19-v0.20) | git clone, commit `6f786f2c5` |
| **vLLM-Ascend** | 0.1.dev2943+g39b8abb7e | git clone, commit `39b8abb7` |
| **LMCache** | v0.4.4 + staging 补丁 | fork: `https://github.com/PPParticle/LMCache.git`, 分支 `feat/staging_cpu_size` |
| **LMCache-Ascend** | v0.1.dev113 | git clone (editable), commit `e7fa9fe` |
| **PyTorch** | 2.9.0+cpu | pip |
| **torch_npu** | 2.9.0rc1 | pip |
| **CANN** | 8.5.1 | 系统安装 |
| **NPU** | Ascend 910B2C | 60GB HBM |
| **Python** | 3.11.6 | — |
