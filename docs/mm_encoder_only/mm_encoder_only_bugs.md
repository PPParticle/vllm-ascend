# mm-encoder-only 实现过程中的 Bug 记录

## 代码改动

### 改动 1：`_dummy_run` 添加 `mm_encoder_only` 早返回
- **文件**：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2428-2430`
- **内容**：在方法最开头添加 `mm_encoder_only` 检查，直接返回空 tensor
- **结果**：无 bug，首次测试即通过

### 改动 2：`_dummy_sampler_run` 添加 `mm_encoder_only` 早返回
- **文件**：`vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2678-2680`
- **内容**：在方法最开头添加 `mm_encoder_only` 检查，直接返回空 tensor
- **结果**：无 bug，首次测试即通过

## 测试环境问题（非代码 bug）

### 问题 1：vllm 包命名空间冲突
- **现象**：`ImportError: cannot import name 'SamplingParams' from 'vllm'`
- **原因**：`/data` 在 `sys.path` 中导致 Python 将 `vllm` 识别为 namespace package 而非 regular package，editable install 的 finder 无法生效
- **解决**：设置 `PYTHONPATH=/data/vllm:/data/vllm-ascend`

### 问题 2：子进程重复下载模型
- **现象**：EngineCore 子进程重新下载模型文件到 `/root/.cache/modelscope/`
- **原因**：子进程未复用 HF_HOME 缓存
- **解决**：使用本地模型路径直接传给 `vllm serve`，跳过 ModelScope 下载

### 问题 3：请求模型名不匹配
- **现象**：API 返回 404 "The model does not exist"
- **原因**：请求 payload 中 model 字段使用了远程名称 `Qwen/Qwen2.5-VL-3B-Instruct`，而 served_model_name 是本地路径
- **解决**：使用实际 served_model_name

### 问题 4：CANN acl 模块在 spawn 子进程中找不到
- **现象**：`ModuleNotFoundError: No module named 'acl'`
- **原因**：CANN 的 Python 路径在主进程 sys.path 中但不在 PYTHONPATH 环境变量中，spawn 子进程不继承 sys.path
- **解决**：在 PYTHONPATH 中加入 `/usr/local/Ascend/cann-8.5.1/python/site-packages`

### 问题 5：HybridKVCacheCoordinator 断言失败
- **现象**：`AssertionError: HybridKVCacheCoordinator requires at least two attention groups`
- **原因**：encoder-only 模式下只有 encoder-only attention 层形成 1 个 KV cache group，但 `enable_prefix_caching=True`（LLM 类默认）触发 HybridKVCacheCoordinator 路径，要求 >=2 个 group
- **解决**：设置 `enable_prefix_caching=False`

### 问题 6：验证脚本 "Second encode also produces cache" FAIL（测试设计问题，非代码 bug）
- **现象**：验证脚本报 `[FAIL] Second encode also produces cache — found 0 files`
- **根因**：测试在同一个 LLM 实例的两次 encode 之间调用了 `cleanup(storage)` 清除 EC 存储磁盘文件。但 `EncoderCacheManager` 的 in-memory 缓存仍持有 mm_hash 条目（移至 `freeable` 但未被驱逐），model runner 的 `encoder_cache[mm_hash]` 也仍持有张量数据。因此：
  1. 第二次 encode 同一张图时，`check_and_update_cache()` 返回 `True`（scheduler 认为 encoder output 已缓存）
  2. `_execute_mm_encoder` 不被调用 → `save_caches` 不被调用 → 磁盘无新文件
- **结论**：这是正确行为。EC connector 的缓存命中机制正常工作，不应在 in-memory 缓存仍存在时期望磁盘重新写入
- **修复**：将测试改为验证第二次 encode 能正常完成（不报错）且输出仍为 short/dummy，而非期望磁盘产生新文件

## 端到端测试结果

### HTTP Server 模式（/data/test_mm_encoder_only.py）
- **profile_run 耗时**：0.01s（对比正常数十秒，证明早返回生效）
- **服务器启动**：35s
- **多模态请求**：HTTP 200

### 离线 LLM API 模式（/data/test_mm_encoder_only_offline.py）
- **LLM 初始化**：~20s
- **3 次独立 encode**：均成功，request_id 递增，finished=True
- **批量 encode（3 张图一起）**：成功
