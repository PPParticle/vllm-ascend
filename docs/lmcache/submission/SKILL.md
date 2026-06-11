# CCF vLLM Ascend Adaptation

## Overview

This submission adapts two features for vLLM-Ascend on Ascend NPU:

### 1. LMCache Local Disk Support

Integrate LMCache KV cache reuse with vLLM-Ascend, enabling SSD-backed local disk caching to reduce TTFT for repeated prefix scenarios.

- **vLLM-Ascend change**: 1 new connector file (`lmcache_ascend_connector.py`)
- **LMCache change**: SSD staging memory isolation (3 files modified in fork)
- **Result**: 2.98x-7.39x TTFT acceleration across all SCBench scenarios

See `lmcache/SKILL.md` for details.

### 2. mm-encoder-only Support

Adapt `--mm-encoder-only` for vLLM Ascend, skipping unnecessary LM dummy execution when only the multimodal encoder is required.

- **vLLM-Ascend change**: 1 file modified (`model_runner_v1.py`)
- **Result**: Aligned with upstream GPU runner behavior

See `mm_encoder_only/SKILL.md` for details.

## AI-Assisted Workflow

AI tools (Claude) were used throughout the adaptation process:

1. **Architecture analysis**: Analyzed the four-layer decoupled relationship between vLLM, vLLM-Ascend, LMCache, and LMCache-Ascend. Identified minimal code changes needed.

2. **Environment debugging**: Diagnosed deployment issues including plugin registration failures, cache configuration pitfalls, and Python module loading paths.

3. **Functional verification**: Verified CPU cache hit and SSD cache hit paths through source code instrumentation.

4. **Performance bottleneck diagnosis**: Discovered SSD cache failure at large working sets, traced root cause to CPU memory contention between hot_cache and SSD staging.

5. **Solution design and implementation**: Designed the staging memory isolation mechanism and implemented the code changes.

6. **Benchmark execution**: Ran full SCBench working set benchmarks (8 test points) and analyzed performance data.

7. **Code generation**: Generated minimal, targeted patches matching upstream behavior patterns.
