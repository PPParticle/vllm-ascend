# LMCache Working Set Benchmark 完整结果

## 测试环境
- **模型**: Qwen2.5-7B-Instruct (28 layers, 4 KV heads, fp16)
- **硬件**: Ascend 910B2C (60GB HBM)
- **数据集**: SCBench (microsoft/SCBench) - 共享 system prompt + 不同 query
- **配置**: CPU 缓存 8GB, SSD 缓存 20GB, block_size=128
- **禁用**: vLLM prefix caching (--no-enable-prefix-caching)
- **KV 大小**: 56 KB/token, 每请求 context ~8k/16k/32k tokens

## 完整对比表

| Test | KV(GB) | Baseline | LMCache Cold | LMCache Warm | vs Baseline | 缓存状态 |
|------|--------|----------|-------------|-------------|-------------|---------|
| ctx8k_ws8   |  3.5 | 496.8ms | 428.3ms |  72.4ms | **6.86x** ✅ | CPU 命中 |
| ctx8k_ws16  |  7.0 | 490.5ms | 540.0ms |  74.1ms | **6.62x** ✅ | CPU 命中 |
| ctx8k_ws32  | 14.0 | 491.4ms | 583.4ms | 168.7ms | **2.91x** ⚠️ | CPU+SSD |
| ctx16k_ws8  |  7.0 | 1161.8ms | 1335.8ms | 123.9ms | **9.38x** ✅ | CPU 命中 |
| ctx16k_ws16 | 14.0 | 1163.0ms | 1346.0ms | 307.5ms | **3.78x** ⚠️ | CPU+SSD |
| ctx16k_ws32 | 28.0 | 1163.4ms | 1266.2ms | 1329.9ms | **0.87x** ❌ | 缓存溢出 |
| ctx32k_ws8  | 14.0 | 3065.5ms | 3411.2ms | 585.3ms | **5.24x** ⚠️ | CPU+SSD |
| ctx32k_ws16 | 28.0 | 3071.3ms | 3425.7ms | 3430.9ms | **0.90x** ❌ | 缓存溢出 |

> **vs Baseline** = Baseline TTFT / LMCache Warm TTFT（越高越好）
>
> ctx16k_ws32 和 ctx32k_ws16 为独立重启服务器重测结果，排除了缓存残留干扰

## 关键发现

### 1. 三层性能区间

| 区间 | KV 总量 | 缓存行为 | 加速比 | 绝对节省 |
|------|---------|---------|--------|---------|
| **CPU 命中** | ≤ 8 GB | 全部在 CPU 内存 | **6.6-9.4x** | 424-1038ms |
| **SSD 辅助** | 8-20 GB | CPU 溢出到 SSD | **2.9-5.2x** | 323-2480ms |
| **缓存溢出** | ≥ 28 GB | 容量不足，无法缓存 | **0.87-0.90x** | -167 ~ -360ms (变慢!) |

### 2. LMCache 开销

LMCache cold TTFT 普遍高于 baseline（缓存存储开销）：
- ctx8k: +40~90ms (8-18%)
- ctx16k: +100~180ms (9-15%)
- ctx32k: +350ms (11%)

当缓存有效命中时，这个开销远小于节省的时间。但当缓存溢出时，只有开销没有收益。

### 3. 最佳场景

**ctx16k_ws8 = 9.38x 加速**
- Baseline: 1162ms → LMCache Warm: 124ms
- 绝对节省: **1038ms** (近 1 秒)
- 原因: 7GB KV 完全在 CPU 缓存内，加载极快

**ctx32k_ws8 = 5.24x 加速**
- Baseline: 3066ms → LMCache Warm: 585ms
- 绝对节省: **2481ms** (2.5 秒!)
- 虽然比例不如 16k，但绝对时间节省最大

### 4. 溢出场景分析

KV = 28GB 刚好等于 CPU(8GB) + SSD(20GB) 总量，但实际无法有效缓存：
- 顺序存储 32 个 context 的 KV 时，前面的数据被驱逐为后面的腾出空间
- Warm pass 请求最早的 context 时，数据已被驱逐，必须重新计算
- LMCache 的查找+加载开销使 TTFT 反而比 baseline 更差

### 5. 结论

LMCache 在 working set 不超过缓存总容量约 **70%** 时效果显著：
- 推荐最大 working set: CPU 缓存可容纳的 context 数量 × 2（允许 SSD 辅助）
- 超过这个阈值，性能反而下降
- 对于生产环境，建议根据实际的 context 长度和并发量选择合适的缓存大小
