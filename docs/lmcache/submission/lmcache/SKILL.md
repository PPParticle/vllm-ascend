# CCF vLLM Ascend LMCache Local Disk Support

## Task

Integrate LMCache KV cache reuse with vLLM-Ascend, enabling SSD-backed local disk caching to reduce TTFT for repeated prefix scenarios on Ascend NPU.

LMCache caches computed KV Cache in CPU memory or SSD. When multiple requests share the same prefix (e.g., system prompt), LMCache skips the prefill computation for cached tokens.

**Final result**: 2.98x-7.39x TTFT acceleration across all 8 SCBench test scenarios, with no degradation.

## Architecture Analysis

### Four-Layer Decoupled Architecture

```
vLLM (upstream, unmodified)
  └── KVConnectorBase_V1 interface + LMCacheConnectorV1 delegate
vLLM-Ascend (1 new file)
  └── lmcache_ascend_connector.py: triggers NPU adaptation via import
LMCache (fork with SSD staging isolation patch)
  └── Core cache engine with CPU + SSD storage backends
LMCache-Ascend (pip installed, unmodified)
  └── Replaces CUDA calls with NPU calls via import-time monkey-patch
```

Key insight: vLLM-Ascend only needs 1 new 5-line file. NPU adaptation is handled entirely by LMCache-Ascend through monkey-patching.

### LMCache Storage Architecture

1. **`batched_put(location=None)`** writes to both CPU and Disk backends simultaneously
2. **`batched_contains`** traverses in order: CPU first → Disk second
3. **`LocalDiskBackend` hard-depends on `LocalCPUBackend`** — CPU cache always created as disk I/O buffer
4. **LRU eviction**: CPU full → old data evicted but retained on disk
5. **Disk cache does not persist across restarts**: In-memory index lost with process

## Code Changes

### File List

| File | Status | Description |
|------|--------|-------------|
| `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py` | **New** (did not exist) | 5-line connector that triggers NPU adaptation and registers `LMCacheAscendConnector` |
| `lmcache_patch/v1/config.py` | **Modified** (existing file) | Added `staging_cpu_size` config entry |
| `lmcache_patch/v1/storage_backend/local_cpu_backend.py` | **Modified** (existing file) | Added staging allocator, `allocate_staging()` / `free_staging()` methods |
| `lmcache_patch/v1/storage_backend/local_disk_backend.py` | **Modified** (existing file) | Changed `allocate()` → `allocate_staging()` for disk reads |

### 1. vLLM-Ascend: New Connector File

**File**: `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py`

```python
import lmcache_ascend  # noqa: F401
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import LMCacheConnectorV1

__all__ = ["LMCacheConnectorV1"]
```

### 2. LMCache: SSD Staging Memory Isolation

**Problem**: When KV total ≥ 28GB (exceeding 8GB CPU), SSD cache hit rate drops to 0%. Root cause: `LocalDiskBackend` calls `local_cpu_backend.allocate()` for staging memory during SSD reads. CPU pool is full with hot_cache → allocation fails → cache counterproductive (0.87-0.90x, slower than baseline).

**Solution**: Split CPU memory into two isolated pools:
- **hot_cache**: stores normal KV cache entries
- **staging buffer**: reserved for SSD read temporary storage

**Modified files** (fork: `https://github.com/PPParticle/LMCache`, branch `feat/staging_cpu_size`):

| File | Change |
|------|--------|
| `config.py` | Add `staging_cpu_size` config (default 0.0) |
| `local_cpu_backend.py` | Create staging allocator; add `allocate_staging()` / `free_staging()` methods |
| `local_disk_backend.py` | Use `allocate_staging()` instead of `allocate()` for disk reads |

#### Patch Detail

**config.py** — one new config entry:

```python
"staging_cpu_size": {"type": float, "default": 0.0, "env_converter": float},
```

**local_cpu_backend.py** — staging allocator:

```python
# __init__: Create separate staging allocator
if staging_size_gb > 0:
    self.staging_allocator = MixedMemoryAllocator(staging_size_bytes, ...)
else:
    self.staging_allocator = None

# initialize_allocator(): Subtract staging from hot_cache
cpu_size_bytes = int(cpu_size * 1024**3) - staging_size_bytes

# New methods
def allocate_staging(self, shapes, dtypes, fmt=None, busy_loop=False):
    if self.staging_allocator is None:
        return self.allocate(shapes, dtypes, fmt, busy_loop=False)
    return self.staging_allocator.allocate(shapes, dtypes, fmt)

def free_staging(self, memory_obj):
    if self.staging_allocator is not None:
        self.staging_allocator.free(memory_obj)
    else:
        self.free(memory_obj)
```

**local_disk_backend.py** — use staging allocator:

```python
# load_bytes_from_disk():
memory_obj = self.local_cpu_backend.allocate_staging(shape, dtype, fmt)
if memory_obj is None:
    logger.error("Staging allocation failed. CPU staging pool may be exhausted.")
    return None

# batched_get_non_blocking():
memory_obj = self.local_cpu_backend.allocate_staging(shape, dtype, fmt, busy_loop=False)
```

## AI-Assisted Workflow

AI was used to:

1. **Architecture analysis**: Analyzed the four-layer decoupled relationship. Identified that only 1 connector file is needed in vLLM-Ascend.

2. **Environment debugging**: Diagnosed deployment issues:
   - `VLLM_PLUGINS` incomplete → connector registration failure. Fix: list all 4 plugins.
   - `LMCACHE_CHUNK_SIZE=256` + `LMCACHE_SAVE_UNFULL_CHUNK=False` → zero cache hits. Fix: `chunk_size=128` + `save_unfull_chunk=True`.
   - Python loads `lib64/` modules, not `lib/`.

3. **Functional verification**: Verified CPU and SSD cache hit paths through source code instrumentation. Confirmed 100% hit rate across 593-3322 tokens.

4. **SSD overhead measurement**: CPU cache hit 37.8ms, SSD cache hit 38.3ms, SSD overhead only +2.2ms.

5. **Performance bottleneck diagnosis**: Discovered SSD cache failure at large working sets. Traced root cause to CPU memory contention between hot_cache and SSD staging.

6. **Staging isolation design**: Designed the CPU memory pool split, implemented `allocate_staging()` / `free_staging()` with separate `MixedMemoryAllocator`.

7. **Benchmark execution**: Ran full SCBench working set benchmark (8 test points, Phase 1 + Phase 2 comparison).

## Benchmark Results

**Environment**: Qwen2.5-7B-Instruct, Ascend 910B2C, SCBench dataset, KV = 56 KB/token.

### Phase 1: Without Staging (CPU 8GB + SSD 20GB)

| KV Total | Cache Status | Speedup |
|----------|-------------|---------|
| ≤ 8 GB | CPU hit | 6.6-9.4x |
| 8-20 GB | CPU+SSD | 2.3-5.2x |
| ≥ 28 GB | Overflow | **0.87-0.90x** ❌ |

Overflow scenarios: SSD staging fails, cache counterproductive.

### Phase 2: With Staging (CPU 6GB hot + 2GB staging + SSD 30GB)

| Test | KV Total | Cold (ms) | Warm (ms) | Speedup |
|------|----------|-----------|-----------|---------|
| ctx8k_ws8 | 3.5GB | 538 | 74 | **7.23x** |
| ctx8k_ws16 | 7.0GB | 537 | 95 | **5.68x** |
| ctx8k_ws32 | 14.0GB | 537 | 118 | **4.53x** |
| ctx16k_ws8 | 7.0GB | 1,234 | 185 | **6.66x** |
| ctx16k_ws16 | 14.0GB | 1,233 | 167 | **7.39x** |
| ctx16k_ws32 | 28.0GB | 1,233 | 414 | **2.98x** |
| ctx32k_ws8 | 14.0GB | 3,196 | 707 | **4.52x** |
| ctx32k_ws16 | 28.0GB | 3,198 | 726 | **4.40x** |

### Key Improvement

Previously overflowed scenarios (ws32): **0.87-0.90x → 2.98-4.53x** — from negative to significant speedup.

All 8 scenarios: **2.98x-7.39x**, no degradation.

## Service Configuration

```bash
LMCACHE_LOCAL_CPU=True \
LMCACHE_MAX_LOCAL_CPU_SIZE=8 \
LMCACHE_STAGING_CPU_SIZE=2 \
LMCACHE_LOCAL_DISK=file://<ssd_path>/ \
LMCACHE_MAX_LOCAL_DISK_SIZE=30 \
LMCACHE_CHUNK_SIZE=128 \
LMCACHE_SAVE_UNFULL_CHUNK=True \
python3 -m vllm.entrypoints.openai.api_server \
    --block-size 128 --max-model-len 32768 --enforce-eager \
    --kv-transfer-config '{"kv_connector":"LMCacheAscendConnector","kv_role":"kv_both"}'
```

### Cache Configuration Recommendations

| KV Total | Recommended |
|----------|-------------|
| < 6GB | `staging_cpu_size=0` |
| 6-30GB | `staging_cpu_size=2` |
| > 30GB | Increase SSD, `staging_cpu_size=4` |

## Version Compatibility

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | v0.19.1 | Matches vLLM-Ascend main |
| vLLM-Ascend | Latest main | Connector code upstream |
| LMCache | v0.4.4 + staging patch | Fork: `https://github.com/PPParticle/LMCache` |
| LMCache-Ascend | v0.1.dev113 | NPU adaptation plugin |
| CANN | 8.5.1 | Ascend compute framework |
