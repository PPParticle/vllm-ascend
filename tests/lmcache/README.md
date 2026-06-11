# LMCache vLLM-Ascend 测试脚本

## 前置条件

### 1. 完成部署

按照 `docs/lmcache/lmcache_deployment.md` 完成 vLLM、vLLM-Ascend、LMCache、LMCache-Ascend 的安装。

### 2. 设置环境变量

所有脚本依赖以下环境变量（在脚本外设置或通过 `.env` 文件加载）：

```bash
# 必须设置
export WORKSPACE=<your_workspace>          # 工作目录，如 /home/user
export VLLM_PATH=$WORKSPACE/vllm           # vllm 源码路径
export VLLM_ASCEND_PATH=$WORKSPACE/vllm-ascend  # vllm-ascend 源码路径
export PYTHON=$WORKSPACE/venv/bin/python3  # Python 解释器路径
export MODEL_PATH=<your_model_path>        # 模型路径，如 $WORKSPACE/models/Qwen2.5-7B-Instruct
export SSD_PATH=<your_ssd_path>            # SSD 缓存路径
export RESULTS_DIR=$WORKSPACE/results      # 结果输出目录

# CANN 环境（必须 source）
source /usr/local/Ascend/cann-8.5.1/set_env.sh
```

### 3. 安装依赖

```bash
pip install requests
```

---

## 脚本说明

### `test_lmcache_ascend.py` — 功能验证

快速验证 LMCache 在 Ascend NPU 上的 store/retrieve 功能。

**流程**：
1. 自动启动 vLLM 服务（带 LMCache connector）
2. 发送 3 个具有相同长前缀的请求
3. 对比 cold/warm TTFT，确认缓存命中

**运行**：

```bash
# 设置环境变量（修改脚本顶部的 MODEL 为你的模型路径）
python3 tests/lmcache/test_lmcache_ascend.py
```

**预期输出**：
```
[REQ1-cold]  TTFT: 1.20s | Response: ...
[REQ2-cache] TTFT: 0.35s | Response: ...
[REQ3-cache] TTFT: 0.36s | Response: ...

--- Results ---
REQ1 (cold):  1.20s
REQ2 (cache): 0.35s (ratio: 0.29x)
PASS: Cache hit likely (TTFT reduced)
```

---

### `benchmark_working_set.py` — SCBench 工作集基准测试

测量 8 个测试点（不同上下文长度 × 不同工作集大小）的 cold/warm TTFT。自动管理服务启停。

**运行**：

```bash
# 全量测试（每个测试点自动重启服务、清空缓存）
bash tests/lmcache/run_benchmark_ws.sh

# 只跑指定测试点
python3 tests/lmcache/benchmark_working_set.py --only ctx8k_ws8 ctx16k_ws16

# 连接已运行的服务（不自动启停）
python3 tests/lmcache/benchmark_working_set.py --no-manage-server
```

**输出**：JSON 结果文件到 `$RESULTS_DIR/working_set_<timestamp>.json`

---

### `run_benchmark_all.sh` — 全量测试（Phase 1：无 staging）

依次运行 8 个测试点，每个测试点重启服务、清空缓存，使用 CPU 8GB + SSD 20GB 配置。

```bash
# 运行全量测试
bash tests/lmcache/run_benchmark_all.sh

# 从指定测试点恢复（跳过已完成的）
START_FROM=ctx16k_ws8 bash tests/lmcache/run_benchmark_all.sh
```

---

### `run_benchmark_all_staging.sh` — 全量测试（Phase 2：staging 隔离）

与 `run_benchmark_all.sh` 相同流程，但使用 CPU 8GB（6GB hot + 2GB staging）+ SSD 30GB 配置。

```bash
bash tests/lmcache/run_benchmark_all_staging.sh

# 从指定测试点恢复
START_FROM=ctx16k_ws8 bash tests/lmcache/run_benchmark_all_staging.sh
```

**两个脚本的区别**：

| 参数 | `run_benchmark_all.sh` | `run_benchmark_all_staging.sh` |
|------|----------------------|------------------------------|
| CPU 缓存总量 | 8GB | 8GB |
| Staging 大小 | 不设置 | 2GB |
| 实际 hot_cache | 8GB | 6GB |
| SSD 容量 | 20GB | 30GB |

---

## 测试矩阵

```
ctx=8k:   ws=8, 16, 32    (KV: 3.5GB, 7.0GB, 14.0GB)
ctx=16k:  ws=8, 16, 32    (KV: 7.0GB, 14.0GB, 28.0GB)
ctx=32k:  ws=8, 16        (KV: 14.0GB, 28.0GB, ws32 需 56GB 超出可用容量)
```

共 8 个测试点，每个测试点耗时约 3-5 分钟（含服务启动）。
