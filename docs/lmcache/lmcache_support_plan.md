# LMCache vLLM-Ascend 变更记录

## 1. 原始状态

原始 vLLM-Ascend 不支持 LMCache。vLLM 上游已有 KV Cache 复用的抽象接口（`KVConnectorBase_V1` + `LMCacheConnectorV1`），但仅有 CUDA GPU 的实现。Ascend NPU 上无法使用 LMCache 的 KV Cache 缓存能力。

## 2. 变更内容

### 2.1 vLLM-Ascend 新增连接器注册入口

新增 `lmcache_ascend_connector.py`（5 行），作用是：
- 将 `LMCacheAscendConnector` 注册为 vLLM 可识别的 connector 名称
- `import lmcache_ascend` 触发 LMCache-Ascend 的 NPU monkey-patch，将所有 CUDA 调用替换为 NPU 实现
- 重导出 `LMCacheConnectorV1`，使其在 Ascend 上可用

vLLM 和 vLLM-Ascend 的主体代码**零改动**。

### 2.2 LMCache 上游新增 SSD Staging 隔离

LMCache 的 `LocalDiskBackend` 从 SSD 加载数据时需要 CPU 暂存空间，但 CPU 内存池被 hot_cache 占满时会导致分配失败，缓存溢出场景反而变慢。新增 staging 隔离机制，将 CPU 内存分为 hot_cache + staging 两部分，解决此问题。

修改 3 个文件：`config.py`、`local_cpu_backend.py`、`local_disk_backend.py`。

代码位于 fork 仓库：`https://github.com/PPParticle/LMCache.git`，分支 `feat/staging_cpu_size`（基于上游 v0.4.4）。

### 2.3 性能结果

Qwen2.5-7B-Instruct, Ascend 910B2C, SCBench 数据集，全场景 2.98x-7.39x TTFT 加速。

详见 `docs/lmcache/lmcache_benchmark_results.md`。

---

## 3. 文件变更清单

### vLLM-Ascend 仓库

#### 新增文件

| 文件路径 | 说明 |
|---------|------|
| `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py` | 连接器注册入口（5 行） |
| `docs/lmcache/lmcache_architecture.md` | 架构说明：LMCache 应用背景、四部件关系、层次结构、代码改动明细 |
| `docs/lmcache/lmcache_deployment.md` | 部署指南：环境准备、源码安装、启动服务、参数说明 |
| `docs/lmcache/lmcache_benchmark_results.md` | 测试记录：测试环境、参数设置、测试脚本、功能验证、性能数据 |
| `docs/lmcache/lmcache_support_plan.md` | 本文件，变更记录 |
| `tests/lmcache/README.md` | 测试脚本执行流程说明 |
| `tests/lmcache/test_lmcache_ascend.py` | 功能验证脚本 |
| `tests/lmcache/benchmark_working_set.py` | SCBench 工作集基准测试脚本 |
| `tests/lmcache/run_benchmark_ws.sh` | 单点基准测试运行脚本 |
| `tests/lmcache/run_benchmark_all.sh` | Phase 1 全量测试运行脚本 |
| `tests/lmcache/run_benchmark_all_staging.sh` | Phase 2 staging 全量测试运行脚本 |

#### 修改文件

无。vLLM-Ascend 主体代码无任何修改。

### LMCache 上游（staging 隔离，fork 仓库）

Fork: `https://github.com/PPParticle/LMCache.git`，分支 `feat/staging_cpu_size`

| 文件路径 | 修改内容 |
|---------|---------|
| `lmcache/v1/config.py` | 添加 `staging_cpu_size` 配置项 |
| `lmcache/v1/storage_backend/local_cpu_backend.py` | 新增 `allocate_staging()` / `free_staging()` 方法 |
| `lmcache/v1/storage_backend/local_disk_backend.py` | `load_bytes_from_disk()` 和 `batched_get_non_blocking()` 改用 staging allocator |

### LMCache-Ascend（NPU 适配插件，独立仓库）

LMCache-Ascend 通过 monkey-patch 接入，不修改 LMCache 源码。详见 `docs/lmcache/lmcache_architecture.md` 第 3.2 节。
