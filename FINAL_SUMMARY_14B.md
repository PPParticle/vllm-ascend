# Qwen2.5-14B × vLLM-Ascend v0.18.0 × LMCache — 结项说明书

> 测试日期: 2026-08-03 ~ 2026-08-04
> 硬件: Ascend 910B 单卡 (64GB HBM), 主机内存 64GB, /data 59GB (可用 18GB, loop device)
> 模型: Qwen2.5-14B-Instruct (48 layers, 8 KV heads, head_dim 128, fp16, KV=192KiB/token=24MiB/chunk)
> 软件: vLLM v0.18.0 + vLLM-Ascend v0.18.0 + LMCache fork (feat/staging_cpu_size) + LMCache-Ascend

---

## 一、关键 Bug 修复

### Bug 1: ctypes UnboundLocalError → lmcache 引擎初始化失败，静默降级为纯重算
- `LMCache/lmcache/v1/memory_management.py::_allocate_cpu_memory()` 中本地补丁在分支内 `import ctypes`，
  使 `ctypes` 成为整个函数的局部变量；正常 pinned 路径执行到 `ctypes.c_uint8 * size` 时该 import 尚未发生 →
  `UnboundLocalError` → 引擎 init 失败 → "degraded mode (recompute)"。
- **修复**: 删除分支内多余的局部 `import ctypes`（模块顶部已有）。此前所有 nostg/stg 数据实际都是纯重算。

### Bug 2: PYTHONHASHSEED 未设置 → builtin hash 跨进程不一致
- lmcache 从 vLLM 导入确定性哈希失败（v0.18 无该接口）→ 回退 Python `hash()`；未设 seed 时每进程不同 →
  store/lookup key 不匹配。**修复**: serve 脚本 `export PYTHONHASHSEED=0`。

---

## 二、实验方法：T/E/T 确定性条件化（disk-retrieve 微基准）

直接测"APC miss + hot miss + disk hit"这条路径，三段式：
1. **T1**: warm 小批 T(target) → drain → T 的 KV 落 disk
2. **E**: warm 不相交的 E(eviction)，大小 > APC + 最大 hot → 把 T 从 APC/hot 挤出 → 此时 T 只在 disk
3. **T2**: 精确测一次 T → 保证 disk 命中

保证手段：
- `cache_marker`（sha256）让 T/E 在首个 cache block 分叉，不共享 block hash
- `/metrics` 前后差值校验 `conditioning_valid`（APC 命中=0、disk 读操作数达标、无失败）
- 日志窗口分析：staging_alloc_fail、disk_read 带宽、allocator sleep
- **O_DIRECT 已关闭**（/data 是 loop device，O_DIRECT 写产生 0 字节文件）
- `pressure` 模式：E 后不等其 disk 写完立刻测 T → E 的 pending 写压 staging 分配器（staging 该发力的场景）

缓存配额（KV=24MiB/chunk, block=chunk=128 token）：
- APC = 256 blocks = 16384 token = 3 GiB
- hot = 6 GiB（nostg: CPU=6,STG=0）；stg: CPU=6,STG=3 → hot=3 + staging=3（同等 6G 总 CPU 预算）
- disk = 18 GiB（/data 全部空闲）

---

## 三、结果（pressure 模式，全部 O_DIRECT off）

### GSM8K（8-shot 前缀 ~2k token，T=7 prompt/96 chunk）
| 配置 | TTFT | QPS | acc | disk_read | cond | staging_fail |
|---|---|---|---|---|---|---|
| baseline（重算） | 1.16s | 0.77 | 71.4% | 0 | ✓ | — |
| nostg（同步 disk 读） | **8.05s** | 0.44 | 71.4% | 2206MiB | ✓ | 0 |
| **stg**（staging 重叠） | **0.89s** | 0.81 | 71.4% | 2206MiB | ✓ | 0 |

**stg 比 nostg 快 9×**，且 cond 全部 True（干净 disk 命中），精度无损。

### GPQA（前缀 ~650 token，T=16 prompt）
| 配置 | TTFT | acc | disk_read | cond |
|---|---|---|---|---|
| baseline | 0.53s | 33.3% | 0 | ✓ |
| nostg | 0.68s | 33.3% | 2032MiB | ✓ |
| stg | 0.64s | 33.3% | 2032MiB | ✓ |

GPQA 的 nostg disk 读本就快（0.68s 读 2GiB ≈ 盘速），staging 余地小（+6%）。

### SCBench（16k 共享上下文，T=2 prompt/258 chunk，2 并发 disk 读）
| 配置 | TTFT(r1/r2) | cond | disk_read | staging_fail |
|---|---|---|---|---|
| baseline（重算 2×16k） | 8.09 / 8.08s | ✓ | 0 | — |
| nostg（2 并发同步 disk 读） | 15.57 / 13.54s | ✓ | 6175MiB | 0 |
| stg（staging 重叠） | 2.21 / 1.28s | **✗** | 3072MiB | 28 / 20 |

stg 名义快 7-12×，但 **cond=False**：staging 池(3GiB=128 chunk) < 单个 16k prompt(129 chunk) → staging_alloc_fail → 第 2 个 prompt 的 disk 读被丢（disk_read 仅 3072MiB=1 prompt）→ 结果不完整、不可信。

---

## 四、分析与结论

### 1. staging 的收益是**并发驱动**的
| 场景 | T 并发数 | nostg TTFT | stg TTFT | staging 加速 |
|---|---|---|---|---|
| GSM8K | 7 | 8.05s | 0.89s | **9×** |
| SCbench T=2 | 2 | 15.0s | 1.7s | ~9×（但 cond 失败）|
| GPQA | 16（但读快）| 0.68s | 0.64s | +6% |

机制：多个请求并发从 disk 取回 KV 时，nostg **同步阻塞**（互相排队，8-15s，比重算还慢）；
staging 用独立 staging 池**重叠** disk→CPU→NPU 传输 → 接近盘速（0.89s 读 2.2GiB）。
**并发 disk 读越多/越大，nostg 越阻塞，staging 收益越大。**

### 2. nostg 的 disk 读在并发下**比重算还慢**
GSM8K nostg 8.05s vs baseline 重算 1.16s（7×更慢）；SCbench nostg 15s vs 重算 8s。
→ **没有 staging 时，lmcache 的 disk 读在并发下是净负收益**；staging 把它从"比重算慢"变成"比重算快"。

### 3. staging 池尺寸的硬约束（bug）
staging 池必须 **≥ 单次最大 retrieve**。SCbench 16k prompt=129 chunk，staging=3GiB=128 chunk，
差 1 chunk → staging_alloc_fail（20-28 次）→ 部分 disk 读被丢 → cond=False、结果不可信。
**修复方向**：staging 池 ≥ 最大单 prompt KV，或分配失败时回退重算而非丢弃。

### 4. 精度
GSM8K/GPQA 在 nostg/stg 下精度与 baseline 一致（71.4%/33.3%）→ disk retrieve 路径正确（staging 池未溢出时）。
SCbench 自由文本答案 + 16k 长上下文，精度提取不可靠（本轮 acc≈0，非重点）。

---

## 五、复现

```bash
# GSM8K + GPQA（pressure T/E/T）
bash run_diskread_both.sh
# SCbench 16k（pressure T/E/T）
bash run_scbench_diskread.sh
# SCbench cold/warm 拟合（CPU 常驻，对照）
bash run_scbench_14b.sh
```
原始结果: `perf_results/diskread_*.json`（含 TTFT/QPS/disk_read/staging_fail/conditioning_valid 等全字段）。
