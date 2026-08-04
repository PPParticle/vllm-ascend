# vLLM-Ascend v0.18.0 × LMCache disk-retrieve 实验（Qwen2.5-14B）

T/E/T 确定性条件化微基准：精确测量 lmcache **disk-hit** 路径下 baseline / nostg / staging 三者的
TTFT、QPS、精度，并校验每次测量确实是 disk 命中（`conditioning_valid`）。

详细分析与原始数据见 `FINAL_SUMMARY_14B.md`。

## 结构

| 文件 | 作用 |
|---|---|
| `serve_readhot.sh` | vllm server（APC on，按 CONFIG 切 baseline/nostg/stg；CPU=hot+staging，staging 从 hot 扣）|
| `bench_readhot.py` | T/E/T bench 主体：T1 warm→drain→E evict→T2 measure；抓 metrics 校验 conditioning；支持 gsm8k/gpqa/scbench |
| `bench_unique.py` | 依赖：`request()`(TTFT/latency/ok)、prompt 构造、答案提取 |
| `run_diskread_both.sh` | GSM8K + GPQA runner（pressure 模式，baseline/nostg/stg）|
| `run_scbench_diskread.sh` | SCbench 16k runner（T=2 并发 disk 读）|
| `scbench_14b.py` / `run_scbench_14b.sh` | SCbench cold/warm 拟合（CPU 常驻对照）|
| `perf_results/diskread_*.json` | 原始结果（TTFT/QPS/disk_read/staging_fail/conditioning_valid 等全字段）|
| `FINAL_SUMMARY_14B.md` | 结项说明书（bug 修复、方法、结果、分析）|

## 运行

```bash
bash run_diskread_both.sh       # GSM8K + GPQA
bash run_scbench_diskread.sh    # SCbench 16k
```

## 关键配置
- APC = 256 blocks (16384 tok, 3 GiB)；hot = 6 GiB（nostg）/ hot 3 + staging 3（stg）；disk = 18 GiB
- O_DIRECT **关闭**（/data loop device 不支持）；PYTHONHASHSEED=0
- warm mt=1（填缓存），measure mt=256（含精度）
