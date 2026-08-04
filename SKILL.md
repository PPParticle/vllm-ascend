# SKILL.md — AI 驱动的 vLLM-Ascend 参数适配优化

> 用 AI 工具（Claude Code）完成 vLLM-Ascend 两个参数适配的全流程方法、提示词设计与迭代经验。
> - `feat/mm-encoder-only`：`--mm-encoder-only` 适配（Qwen2.5-VL，跳过 LM dummy run）
> - `feat/lmcache`（本仓库）：LMCache CPU+SSD 两级缓存适配 + 14B disk-retrieve 性能基准（Qwen2.5-14B）

---

## 一、方法：探索-实现-验证-迭代循环

每个适配任务都跑同一个循环，AI 做执行、人做裁判：

1. **架构定位**：AI 读 upstream GPU 实现 + Ascend 差异，定位"改哪个文件、哪段、什么不用改"。
2. **最小补丁**：AI 生成贴合 upstream 模式的最小改动。
3. **部署跑通**：AI 起 server、读日志，解插件注册/CANN env/离线缓存权限等部署坑。
4. **功能验证**：AI 写验证脚本（HTTP + offline + 严格断言）跑 PASS/FAIL。
5. **性能基准**：AI 设计并跑基准，**强制产出带"合格章"的原始数据**。
6. **根因诊断**：异常时 AI 读源码 + 插桩日志 + metrics 定位根因。
7. **迭代修复**：人纠偏，AI 改代码/基准，回到 3。

**分工**：AI 负责"广度搜索 + 深度诊断 + 机械实现"；人负责"目标 + 约束 + 纠偏 + 验收"。

## 二、提示词设计（4 种模式）

| 模式 | 用法 | 何时用 |
|---|---|---|
| **A 高层目标** | "测 SCbench，看 staging 能否超 nostg" | 探索期，让 AI 自主找路径 |
| **B 逐参数规格** | "max_num_seqs=8; APC=hot=…; disk 用最大; 要测精度; 保留 alloc_fail" | 收敛期，锁死可比基线 |
| **C 纠偏否决** | "这结果不能用，store cache 被打满了" / "关掉 O_DIRECT" | 最高频，人判合理性+指方向 |
| **D 强制可验证** | "把脚本/原始数据完整输出; mean 是什么" | 防 AI 报喜不报忧 |

**关键经验**：性能对比类任务一旦进入数据收集，**必须切到模式 B 逐项锁规格**——否则 AI 会为"数字好看"私自调参（本项目发生过：偷偷把 warm mt 256→8、hot 调大，污染基线）。

## 三、LMCache 适配（`feat/lmcache`，当前项目）

### 设计与脚本（可复现）

四层解耦：vLLM connector 接口 → vLLM-Ascend **1 个 5 行触发文件** → LMCache fork（CPU+SSD 后端）→ LMCache-Ascend（import 时 monkey-patch NPU 调用）。

**T/E/T 确定性条件化基准**（`bench_readhot.py`）：精确测"APC miss + hot miss + disk hit"——
1. **T1** warm 小批 target → drain 到 disk 文件数稳定；
2. **E** warm 不相交 eviction（> APC + 最大 hot）把 target 挤出 APC/hot；
3. **T2** 精确测一次 target，并用 `/metrics` 校验 `conditioning_valid`（APC 命中=0、disk 读操作数达标、无失败）。

`cache_marker`（sha256）让 T/E 在首个 block 分叉，不共享 hash。`pressure` 模式：E 后不等其 disk 写完即测 T，压 staging 分配器。

**配置**（脚本里锁死）：APC = 256 blocks = 32k token = 6 GiB；hot：nostg 6 GiB / stg 3 GiB + staging 3 GiB（staging 从 hot 扣，总 CPU 同为 6 GiB）；disk = 18 GiB（/data 全空闲）；APC on；O_DIRECT **关**（loop device 写 0 字节）；`PYTHONHASHSEED=0`；max_num_seqs=conc=8；warm mt=1（填缓存），measure mt=256（含精度）。

**运行**：`bash run_diskread_both.sh`（GSM8K/GPQA）、`bash run_scbench_diskread.sh`（SCbench 16k）。
**原始数据**：`perf_results/diskread_*.json`（TTFT/QPS/disk_read/staging_fail/conditioning_valid）。

**依赖与版本（精确 pin）**

| 组件 | 仓库 | 分支 | commit |
|---|---|---|---|
| vllm-ascend | `gitlink.org.cn/vllm-ascend/vllm-ascend-v0.18.0`（fork）| master | `72dc689`（release base）；mm-encoder 改动在 `feat/mm-encoder-only`；lmcache connector 已在树中 |
| LMCache | `github.com/PPParticle/LMCache` | feat/staging_cpu_size | **`d1be10a3`**（SSD staging 隔离 + ctypes 修复 + staging fallback + 并行 disk 读）|
| LMCache-Ascend | `github.com/PPParticle/LMCache-Ascend` | main | **`7ab03a6`**（单卡 retrieve to_gpu 修复 + event-based sync）|
| vllm | `github.com/vllm-project/vllm` | v0.18.0 | 基线；`transformers_utils/config.py` 的 speculator 测试补丁**不纳入**提交 |

连接：vllm-ascend 的 `lmcache_ascend_connector.py` `import lmcache_ascend` → LMCache-Ascend 在 import 时 monkey-patch LMCache 的 CUDA 调用为 NPU 调用。三者版本对齐在当前 main（vcs-generated，无固定版本号）。

### 结果

| 数据集 | baseline | nostg | stg | 说明 |
|---|---|---|---|---|
| GSM8K | 1.16s | 8.05s | **0.89s** | staging 9×；cond 全 True |
| GPQA | 0.53s | 0.68s | 0.64s | nostg disk 读本就快，staging +6% |
| SCbench 16k | 8.1s | 14.5s | 1.7s | staging ~9×，但 stg cond=False（池<单prompt） |

**机制**：staging 用独立池**重叠**并发 disk→CPU→NPU 传输，解 nostg 同步阻塞；并发 disk 读越多，nostg 越慢（可比重算还慢），staging 收益越大。

## 四、mm-encoder-only 适配（`feat/mm-encoder-only`）

`NPUModelRunner._dummy_run` / `_dummy_sampler_run` 各加 1 处 `mm_encoder_only` 早返回（profile_run 5s→0.01s）。AI 确认 NPU 专有逻辑（EPLB/PCP/UBatch）均为 LM decoder 相关 → 早返回安全。4 轮验证（HTTP/offline/22 项功能/12 项 upstream 等价）全 PASS，行为与 upstream GPU 一致。

## 五、真实踩坑（简练）

- **缓存静默降级**：性能"无差异"先查 metrics+unhealthy 日志——`ctypes` UnboundLocalError 让 lmcache init 必崩、静默退化为纯重算，此前数据全假。
- **跨进程哈希不一致**：builtin `hash()` 无 seed → store/lookup key 不匹配；`PYTHONHASHSEED=0` 修复。
- **基准设计即 bug**：每轮全新 prefix→命中率结构性 0；store 同时写两层+disk 无余量→级联自驱逐（thrashing）。靠对照实验（CPU 装得下 vs 装不下）定位。
- **测量须校验目标路径**：T2 用 `conditioning_valid` 挡假数据——"stg 0.25s 超快"曾是失败请求瞬间返回的假象。
- **O_DIRECT 在 loop device 静默失败**：/data 是 loop，O_DIRECT 写出 0 字节文件；看 du/st_size 比看日志快。
- **staging 池尺寸硬约束**：池必须 ≥ 单次最大 retrieve；SCbench 16k prompt=129 chunk，池=128 chunk，差 1→alloc_fail→cond 失效。
- **改参数前锁规格**：禁 AI 为"加速"私自调 warm mt / hot。

## 六、给后续 AI 适配任务的要点

1. 先验证缓存/功能真生效（metrics+日志），再谈性能。
2. 性能数据必须带"合格章"（conditioning_valid/alloc_fail/失败数）。
3. 异常先怀疑实验设计，用对照实验证伪。
4. 既报喜也报忧：加速倍数 + 失效条件一起写。
5. 探索用高层目标，收敛用逐参数规格，全程用纠偏+强制可验证。
