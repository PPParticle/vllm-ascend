# CCF vLLM Ascend mm-encoder-only Adaptation

## Task

Adapt the vLLM startup argument `--mm-encoder-only` for vLLM Ascend.

In upstream vLLM EPD (Encoder-Prefill-Decode) architecture, `--mm-encoder-only` enables running only the multimodal encoder while skipping the language model decoder. This is used when encoder and decoder run on separate instances.

## Architecture Analysis

### Ascend vs GPU Differences

The Ascend `NPUModelRunner` has additional NPU-specific logic in `_dummy_run()` that does not exist in the GPU runner:

| Aspect | Upstream GPU | Ascend NPU |
|--------|-------------|------------|
| EPLB load balancing | Standard | Ascend-specific `dynamic_eplb` + `eplb_updator` |
| PCP parallel compute | None | Ascend-specific `pcp_manager` |
| UBatch micro-batching | Supported | Not supported |

All NPU-specific logic is LM decoder related, so early return is safe.

### What Did NOT Need Changes

- **`_dummy_pooler_run`**: Inherited from GPU runner, upstream guard already works.
- **Configuration parsing**: Upstream CLI parameter flow fully connected.
- **`get_kv_cache_spec`**: Ascend already has equivalent override (`is_producer` ≡ `not is_consumer`).

## Code Changes

### File List

| File | Status | Description |
|------|--------|-------------|
| `vllm_ascend/worker/model_runner_v1.py` | **Modified** (existing file) | Added 2 early return guards for `mm_encoder_only` |

### Change 1: `_dummy_run` early return

### Change 1: `_dummy_run` early return

```python
mm_config = self.vllm_config.model_config.multimodal_config
if mm_config and mm_config.mm_encoder_only:
    return torch.tensor([]), torch.tensor([])
```

Effect: profile_run time reduced from ~5s to 0.01s.

### Change 2: `_dummy_sampler_run` early return

```python
mm_config = self.vllm_config.model_config.multimodal_config
if mm_config and mm_config.mm_encoder_only:
    return torch.tensor([])
```

## AI-Assisted Workflow

AI was used to:

1. Analyze the competition requirement and identify target parameter and its role in EPD architecture.
2. Read upstream GPU model runner to understand complete expected behavior for encoder-only mode.
3. Compare GPU vs Ascend runner to locate missing adaptation points and confirm safety of early returns.
4. Design 4 rounds of tests covering HTTP server, offline API, functional verification, and upstream equivalence.
5. Generate minimal code patches matching upstream runner behavior patterns.

## Validation

**Environment**: Qwen2.5-VL-3B-Instruct, Ascend 910B2C, single card.

### Test 1: HTTP Server End-to-End

- `mm_encoder_only: True` correctly passed, `ec_role='ec_producer'` activated
- Model load: 1.2475 GB, profile_run: **0.01s** (normal ~5s, early return confirmed)
- Server startup: 35s, health check 200 OK, image request HTTP 200

### Test 2: Offline LLM API

- 3 independent encodes: all successful, finished=True
- Batch encode (3 images): successful

### Test 3: Strict Functional Verification — 22 PASS, 0 FAIL

**Encoder output written to EC storage (11 items)**: All passed. Encoder output shape=[425, 2048], no NaN/Inf, not all zeros.

**LM decoder skipped (5 items)**: Init time encoder-only (18.0s) < normal (24.1s). Encoder-only produces dummy output, normal produces real text. EC cache files only present in encoder-only mode.

**Consumer loads encoder output (4 items)**: Safetensors roundtrip lossless, consumer successfully reused producer's encoder cache.

### Test 4: Upstream Equivalence — 12 PASS, 0 FAIL

Equivalent to upstream `test_encoder_instance_zero_kv_cache`.

| Check | Upstream GPU | Ascend NPU | Match |
|-------|-------------|------------|-------|
| encoder_cache_manager exists | PASS | PASS | Yes |
| num_blocks correct (producer=1, consumer>1) | PASS | PASS | Yes |
| kv_cache_groups status correct | PASS | PASS | Yes |
| EC connector role correct | PASS | PASS | Yes |
| Chunked prefill disabled (producer) | PASS | PASS | Yes |

**Conclusion: Ascend NPU mm_encoder_only behavior is fully consistent with upstream GPU.**

## Notes

The `worker.py` change (returning `CompilationTimes`) was superseded by upstream vLLM-Ascend main branch. Final submission only contains `model_runner_v1.py` adaptation.
