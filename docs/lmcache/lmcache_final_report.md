# vLLM Ascend 赛道结项书

## 一、团队信息

| 项目 | 内容 |
|------|------|
| 项目名称 | vLLM Ascend LMCache 后端及 encoder-only 适配 |
| 团队名称 | Lone-wolf |
| 队长姓名 | fcqiao |
| 队长联系方式 | 邮箱：qiao_sh_pudong@163.com |

## 二、环境依赖

| 组件 | 申请书版本 | 实际使用版本 | 说明 |
|------|-----------|-------------|------|
| vLLM | v0.18.0 | v0.19.1 | 与 vLLM-Ascend main 分支保持一致 |
| vLLM-Ascend | v0.18.0rc1 | 最新 main (commit 7b9438d5) | 与 GitHub 仓库 Main 分支保持一致 |
| Python | ≥ 3.10 | 3.11.6 | |
| CANN | 8.5.1 | 8.5.1 | 一致 |
| PyTorch | 2.9.0 | 2.9.0+cpu | CPU 版本，NPU 由 torch_npu 提供 |
| torch_npu | — | 2.9.0rc1 | Ascend NPU 的 PyTorch 后端 |
| LMCache | 0.3.12 | v0.4.4 + staging 补丁 | 升级到 v0.4.4，新增 SSD staging 隔离补丁 |
| LMCache-Ascend | — | v0.1.dev113 | NPU 适配插件，monkey-patch 方式接入 |

## 三、参数理解与适配结果

### 1. LMCache KV Cache 复用（`--kv-transfer-config`）

申请书计划通过 `--kv-offloading-backend lmcache` 参数适配。实施过程中发现 vLLM 上游已演进为 `--kv-transfer-config` 参数体系，LMCache 以 KV connector 方式接入。实际采用 `LMCacheAscendConnector` 方案。

**适配结果：**

| 组件 | 申请书预期 | 实际实现 |
|------|-----------|---------|
| 配置桥接 | Patch `_post_init_kv_transfer_config()` | 无需 patch，通过 `--kv-transfer-config` 直接指定 connector |
| Connector 实现 | 重写 lmcache_ascend_connector.py | 仅需 5 行新文件（import + re-export），NPU 适配由 LMCache-Ascend 自动完成 |
| 异步传输 | 集成 CpuNpuOffloadingHandler | 由 LMCache-Ascend monkey-patch 自动处理 |
| LMCache 库兼容性 | 需验证 CUDA 依赖 | LMCache-Ascend 在 import 时自动替换 CUDA 调用 |
| 存储后端 | 仅 CPU 内存 | 扩展支持 CPU 内存 + SSD 磁盘（新增 staging 隔离） |

**实际方案的优势：**

申请书方案需要对 vLLM-Ascend 进行大量改动（patch platform.py、重写 connector、修改 utils.py 等）。实际发现 vLLM 上游已将 LMCache 集成完善，且 LMCache-Ascend 通过 monkey-patch 完成全部 NPU 适配，vLLM-Ascend 只需新增 1 个 5 行文件即可。

**关键创新：SSD Staging 内存隔离**

在 benchmark 测试中发现，当 KV 总量接近 CPU+SSD 缓存总容量时，SSD 缓存命中率降为 0%。根因是 SSD 读取需要 CPU 暂存空间，但 CPU 内存已被 hot_cache 占满。设计了 staging 隔离方案，将 CPU 内存分为 hot_cache + staging 两部分，解决了此问题。

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `vllm_ascend/.../lmcache_ascend_connector.py` | 新增 | 5 行 connector 注册文件 |
| LMCache `lmcache/v1/config.py` | 修改 | 添加 `staging_cpu_size` 配置 |
| LMCache `lmcache/v1/storage_backend/local_cpu_backend.py` | 修改 | 新增 staging allocator 及 `allocate_staging()` / `free_staging()` |
| LMCache `lmcache/v1/storage_backend/local_disk_backend.py` | 修改 | SSD 读取改用 staging allocator |

### 2. `--mm-encoder-only`

**适配结果：**

| 组件 | 申请书预期 | 实际实现 |
|------|-----------|---------|
| 参数验证 | platform.py 添加验证逻辑 | 无需额外验证，上游已完整支持 |
| Encoder 执行路径 | 验证 AscendMMEncoderAttention | 已有实现，无需修改 |
| ACL Graph 兼容 | 验证空输出影响 | 早返回跳过 ACL Graph，无兼容问题 |
| `_dummy_run` 跳过 | 未在申请书中列出 | 发现需添加早返回 guard，跳过 LM profiling |
| `_dummy_sampler_run` 跳过 | 未在申请书中列出 | 同上，跳过 sampler 预热 |

**核心改动**（2 处，首次测试即通过）：

在 `NPUModelRunner` 的 `_dummy_run()` 和 `_dummy_sampler_run()` 中添加 `mm_encoder_only` 早返回 guard，跳过不必要的 LM 预热，profile_run 耗时从 ~5s 降至 0.01s。

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `vllm_ascend/worker/model_runner_v1.py` | 修改 | 2 处早返回 guard |

## 四、技术方案（实施结果）

### 1. LMCache 适配方案

**方案调整说明：**

申请书中计划通过 patch `_post_init_kv_transfer_config()` 和重写 connector 实现。实施中发现 vLLM 上游已通过 `--kv-transfer-config` 参数支持 LMCache，且 LMCache-Ascend 通过 monkey-patch 完成全部 NPU 适配，因此大幅简化了方案。

**实际实施步骤：**

1. **安装验证**：安装 LMCache v0.4.4 + LMCache-Ascend，验证 import 链路正常
2. **Connector 注册**：新增 `lmcache_ascend_connector.py`（5 行），注册 `LMCacheAscendConnector`
3. **功能性验证**：通过源码插桩验证 CPU 缓存和 SSD 缓存命中路径
4. **性能测试**：SCBench 工作集 benchmark 发现 SSD 缓存溢出问题
5. **Staging 隔离设计**：设计并实现 CPU 内存 hot_cache + staging 分池方案
6. **全量验证**：8 个测试点全部实现 2.98x-7.39x 加速

### 2. mm-encoder-only 适配方案

**方案调整说明：**

申请书中计划修改 platform.py 添加验证。实施中发现上游已完整支持参数传递和 Encoder 执行路径，仅需在 `model_runner_v1.py` 中添加早返回 guard。

**实际实施步骤：**

1. **上游对比**：分析 GPU model runner 的 `mm_encoder_only` 处理逻辑
2. **差异定位**：确认 `_dummy_run()` 和 `_dummy_sampler_run()` 需要添加 guard
3. **代码实现**：添加 2 处早返回，对齐上游 GPU 行为
4. **测试验证**：4 轮测试 56 项检查全部通过，与上游 GPU 完全一致

### 3. AI 工具使用

| 工具 | 用途 | 具体辅助方式 |
|------|------|-------------|
| Claude Code | Agent 框架，开发全流程 | 架构分析、环境调试、代码生成、benchmark 设计与执行、性能瓶颈诊断 |
| GLM-5 | 核心推理与生成 | 代码审查、方案讨论、错误日志分析 |

## 五、开发时间规划（实际执行）

| 阶段 | 计划时间 | 实际时间 | 状态 |
|------|---------|---------|------|
| 环境搭建 | 4/16-4/18 | 4/16-4/18 | ✅ |
| 上游代码熟悉 | 4/21-4/25 | 4/21-4/25 | ✅ |
| mm-encoder-only 适配 | 4/28-5/22 | 5/19-5/21 | ✅ 首次测试即通过 |
| LMCache 环境搭建 | 5/26-5/28 | 5/21-5/26 | ✅ 版本升级到 v0.4.4 |
| LMCache 功能验证 | 5/29-6/9 | 5/26-6/9 | ✅ CPU + SSD 路径验证 |
| LMCache 性能测试 | 6/10-6/16 | 6/9-6/10 | ✅ SCBench 全量测试 |
| Staging 隔离方案 | 未计划 | 6/10 | ✅ 新增，解决溢出问题 |
| 测试完善 | 6/17-6/18 | 已完成 | ✅ |
| Bug 修复 | 6/19-6/26 | 已完成 | ✅ 主要是环境配置问题 |
| 文档收尾 | 7/1-7/2 | 6/11 | ✅ |

## 六、预期成果对比

### 1. 功能成果

**LMCache：**

| 成果 | 申请书预期 | 实际达成 |
|------|-----------|---------|
| CLI 参数可用 | `--kv-offloading-backend lmcache` 可用 | `--kv-transfer-config` 指定 LMCacheAscendConnector 可用 |
| KV Cache 卸载 | KV Cache 从 NPU 卸载到 CPU | ✅ 支持 CPU 内存 + SSD 磁盘两级缓存 |
| 异步传输 | NPU Stream 异步传输 | ✅ 由 LMCache-Ascend monkey-patch 处理 |
| 精度一致 | 卸载前后精度一致 | ✅ 命中缓存后推理结果正确 |
| 测试覆盖 | 3 个测试用例 | ✅ SCBench 8 个测试点，功能验证 + 性能测试 |

**mm-encoder-only：**

| 成果 | 申请书预期 | 实际达成 |
|------|-----------|---------|
| CLI 参数可用 | `--mm-encoder-only` 可用 | ✅ |
| 模型支持 | Qwen2.5-VL、LLaVA | ✅ Qwen2.5-VL-3B-Instruct 验证通过 |
| 精度一致 | Ascend 输出与 GPU 一致 | ✅ 12 项上游等价测试全部通过 |
| 测试覆盖 | 2 个端到端测试 | ✅ 4 轮测试 56 项检查 |

### 2. 性能指标

**LMCache：**

| 指标 | 申请书目标 | 实际达成 |
|------|-----------|---------|
| NPU 显存节省 | KV Cache 成功卸载 | ✅ 支持 CPU + SSD 两级缓存，最大 36GB 缓存容量 |
| 传输延迟隐藏 | 异步传输延迟重叠 ≥ 50% | ✅ SSD 额外开销仅 +2.2ms |
| 端到端性能 | 无明显性能瓶颈 | ✅ 全场景 2.98x-7.39x TTFT 加速 |

**具体性能数据（Qwen2.5-7B-Instruct, Ascend 910B2C, SCBench）：**

| 测试场景 | KV 总量 | Cold TTFT | Warm TTFT | 加速比 |
|---------|---------|-----------|-----------|--------|
| ctx8k_ws8 | 3.5GB | 538ms | 74ms | **7.23x** |
| ctx8k_ws16 | 7.0GB | 537ms | 95ms | **5.68x** |
| ctx8k_ws32 | 14.0GB | 537ms | 118ms | **4.53x** |
| ctx16k_ws8 | 7.0GB | 1,234ms | 185ms | **6.66x** |
| ctx16k_ws16 | 14.0GB | 1,233ms | 167ms | **7.39x** |
| ctx16k_ws32 | 28.0GB | 1,233ms | 414ms | **2.98x** |
| ctx32k_ws8 | 14.0GB | 3,196ms | 707ms | **4.52x** |
| ctx32k_ws16 | 28.0GB | 3,198ms | 726ms | **4.40x** |

**mm-encoder-only：**

| 指标 | 申请书目标 | 实际达成 |
|------|-----------|---------|
| Encoder 前向延迟 | 与 GPU 相当 | ✅ encoder-only 模式 profile_run 从 ~5s 降至 0.01s |
| 特征输出精度 | cos similarity > 0.999 | ✅ 22 项功能验证 + 12 项等价测试全部通过 |

### 3. 超出申请书的成果

1. **SSD Staging 内存隔离机制**：申请书未计划。在性能测试中发现 SSD 缓存溢出问题后，设计并实现了 staging 隔离方案，将之前负优化的场景（0.87-0.90x）转为 2.98-4.53x 加速。
2. **LMCache fork 仓库**：`https://github.com/PPParticle/LMCache`，分支 `feat/staging_cpu_size`，包含 staging 隔离补丁。
3. **完整文档体系**：架构说明、部署指南、测试记录、变更记录 4 篇文档。
4. **Benchmark 脚本**：SCBench 工作集测试脚本，支持全量/单点/staging 三种模式。

## 七、团队介绍

| 姓名 | 单位/学校 | 角色 | 主要分工 | 相关经验/技能 |
|------|----------|------|---------|-------------|
| 乔凡超 | 华东师范大学 | 队长 | 总体适配方案设计、节奏把控 | Agent 使用技巧、AI infra 开发、微架构优化 |

## 八、交付清单

| 交付物 | 位置 |
|--------|------|
| vLLM-Ascend 源码改动 | GitHub: `PPParticle/vllm-ascend`, 分支 `feat/lmcache-support`、`feat/mm-encoder-only` |
| LMCache staging 隔离补丁 | GitHub: `PPParticle/LMCache`, 分支 `feat/staging_cpu_size` |
| 竞赛提交 PR | GitLink: `fcqiao/ccf-vllm-ascend`, 分支 `Lone-wolf` |
| SKILL.md（AI 工具使用记录） | GitLink PR 中，含总览 + lmcache + mm_encoder_only 三份 |
