#!/usr/bin/env python3
"""APC-on LMCache disk-only benchmark with explicit T/E/T conditioning.

T (target) is written to LMCache disk first.  A disjoint E (eviction) set then
fills and exceeds both APC and LocalCPU hot-cache capacity.  The final T access
is measured exactly once, so it should be APC-miss + hot-miss + disk-hit.

Modes:
  pressure   Measure T immediately after E. Pending E disk writes can keep hot
             objects non-evictable and expose the staging allocator benefit.
  quiescent  Wait for E disk writes to drain. This validates a plain disk-only
             path, but no-staging may succeed by evicting ordinary hot objects.
"""

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from bench_unique import (
    _norm,
    build_gpqa_unique,
    build_gsm8k_unique,
    extract_gpqa_answer,
    extract_gsm8k_answer,
    request as do_request,
)


METRIC_BASES = (
    "vllm:prefix_cache_hits",
    "vllm:prefix_cache_queries",
    "vllm:external_prefix_cache_hits",
    "vllm:external_prefix_cache_queries",
    "vllm:prompt_tokens",
    "vllm:prompt_tokens_by_source",
    "vllm:generation_tokens",
    "vllm:request_success",
    "lmcache:num_requested_tokens",
    "lmcache:num_hit_tokens",
    "lmcache:num_lookup_tokens",
    "lmcache:num_lookup_hits",
    "lmcache:num_vllm_hit_tokens",
    "lmcache:local_cpu_evict_count",
    "lmcache:local_cpu_evict_keys_count",
    "lmcache:local_cpu_evict_failed_count",
    "lmcache:active_memory_objs_count",
    "lmcache:pinned_memory_objs_count",
)

LOG_PATTERNS = {
    "alloc_fail": re.compile(
        r"Memory allocation failed during async disk load|"
        r"Staging allocation failed for disk load",
        re.IGNORECASE,
    ),
    "alloc_retry": re.compile(
        r"Unable to allocate memory object after \d+ attempts?",
        re.IGNORECASE,
    ),
    "staging_alloc_fail": re.compile(
        r"Memory allocation failed during async disk load|"
        r"Staging allocation failed for disk load",
        re.IGNORECASE,
    ),
    "no_evict_candidate": re.compile(
        r"No eviction candidates found in local cpu backend",
        re.IGNORECASE,
    ),
    "allocator_sleep": re.compile(
        r"Waiting for [0-9.]+ seconds before retrying",
        re.IGNORECASE,
    ),
    "not_busy_looping": re.compile(
        r"Not busy looping because we are not immediately able to evict",
        re.IGNORECASE,
    ),
    "disk_space_fail": re.compile(
        r"No eviction candidates found\. Disk space under pressure",
        re.IGNORECASE,
    ),
    "partial_retrieve": re.compile(
        r"partial(?:ly)? retrieve|Returning partial results|"
        r"retrieve.*(?:failed|failure)",
        re.IGNORECASE,
    ),
    "disk_read": re.compile(r"Disk read size:", re.IGNORECASE),
    "disk_write": re.compile(r"Disk write size:", re.IGNORECASE),
    "disk_prefetch": re.compile(r"Prefetching \d+ keys from disk", re.IGNORECASE),
    "retrieve_response": re.compile(
        r"Responding to scheduler.*with retrieved length \d+", re.IGNORECASE
    ),
}

ALLOCATOR_SLEEP_RE = re.compile(
    r"Waiting for ([0-9.]+) seconds before retrying", re.IGNORECASE
)
DISK_READ_RE = re.compile(
    r"Disk read size: (\d+) bytes,\s*Bandwidth: ([0-9.]+) MB/s",
    re.IGNORECASE,
)
DISK_WRITE_RE = re.compile(
    r"Disk write size: (\d+) bytes,\s*Bandwidth: ([0-9.]+) MB/s",
    re.IGNORECASE,
)
DISK_PREFETCH_RE = re.compile(r"Prefetching (\d+) keys from disk", re.IGNORECASE)
RETRIEVED_LENGTH_RE = re.compile(
    r"Responding to scheduler.*?with retrieved length (\d+)", re.IGNORECASE
)


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


def distribution(values):
    """Return a consistent numeric distribution summary."""
    if not values:
        return {
            "count": 0,
            "mean": None,
            "stddev": None,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def run_batch(prompts, endpoint, model, max_tokens, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(
            executor.map(
                lambda prompt: do_request(endpoint, model, prompt, max_tokens),
                prompts,
            )
        )


def require_success(results, phase):
    failed = [result for result in results if not result.ok]
    if failed:
        examples = "; ".join(result.error for result in failed[:3])
        raise RuntimeError(
            f"{phase}: {len(failed)}/{len(results)} requests failed: {examples}"
        )


def disk_state(root):
    count = 0
    total_bytes = 0
    newest_mtime_ns = 0
    path = Path(root)
    if not path.exists():
        return {"files": 0, "bytes": 0, "newest_mtime_ns": 0}
    try:
        for entry in path.rglob("*"):
            if not entry.is_file():
                continue
            stat = entry.stat()
            count += 1
            total_bytes += stat.st_size
            newest_mtime_ns = max(newest_mtime_ns, stat.st_mtime_ns)
    except OSError:
        pass
    return {
        "files": count,
        "bytes": total_bytes,
        "newest_mtime_ns": newest_mtime_ns,
    }


def wait_disk_stable(root, label, timeout, poll, stable_polls, expect_files):
    if not expect_files:
        return 0.0, disk_state(root), disk_state(root)["files"]

    started = time.perf_counter()
    last = None
    stable = 0
    peak = 0
    while time.perf_counter() - started < timeout:
        time.sleep(poll)
        state = disk_state(root)
        peak = max(peak, state["files"])
        signature = (state["files"], state["bytes"], state["newest_mtime_ns"])
        stable = stable + 1 if signature == last else 0
        last = signature
        elapsed = time.perf_counter() - started
        if int(elapsed) % 10 < poll:
            print(
                f"    {label} drain t={elapsed:.0f}s files={state['files']} "
                f"bytes={state['bytes']} stable={stable}/{stable_polls}",
                flush=True,
            )
        if state["files"] > 0 and stable >= stable_polls:
            return elapsed, state, peak

    state = disk_state(root)
    raise TimeoutError(
        f"{label}: LMCache disk did not stabilize in {timeout}s "
        f"(files={state['files']}, bytes={state['bytes']})"
    )


def scrape_metrics(endpoint):
    values = {}
    try:
        response = requests.get(f"{endpoint.rstrip('/')}/metrics", timeout=10)
        response.raise_for_status()
        for line in response.text.splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                continue
            sample_token = fields[0]
            sample_name = sample_token.split("{", 1)[0]
            try:
                value = float(fields[-1])
            except ValueError:
                continue
            for base in METRIC_BASES:
                if sample_name == base or sample_name.startswith(f"{base}_"):
                    values[base] = values.get(base, 0.0) + value
                    if base == "vllm:prompt_tokens_by_source":
                        source_match = re.search(r'source="([^"]+)"', sample_token)
                        if source_match:
                            source_key = f"{base}:{source_match.group(1)}"
                            values[source_key] = values.get(source_key, 0.0) + value
                    break
    except requests.RequestException as exc:
        values["_error"] = f"{type(exc).__name__}: {exc}"
    return values


def metric_delta(before, after):
    delta = {}
    keys = set(METRIC_BASES)
    keys.update(
        key for key in set(before) | set(after)
        if key.startswith("vllm:prompt_tokens_by_source:")
    )
    for key in keys:
        if key in before and key in after:
            delta[key] = after[key] - before[key]
    return delta


def read_log_from(path, offset):
    if not path or not os.path.exists(path):
        return "", offset
    with open(path, "rb") as handle:
        handle.seek(offset)
        data = handle.read()
        return data.decode("utf-8", errors="replace"), handle.tell()


def count_log_events(text):
    return {name: len(pattern.findall(text)) for name, pattern in LOG_PATTERNS.items()}


def io_log_summary(text, pattern):
    matches = pattern.findall(text)
    sizes = [int(size) for size, _ in matches]
    bandwidths = [float(bandwidth) for _, bandwidth in matches]
    return {
        "operations": len(matches),
        "bytes": sum(sizes),
        "mib": sum(sizes) / (1024 ** 2),
        "bandwidth_mbps": distribution(bandwidths),
    }


def analyze_log(text):
    retrieved_lengths = [int(value) for value in RETRIEVED_LENGTH_RE.findall(text)]
    return {
        "events": count_log_events(text),
        "allocator_sleep_seconds": sum(
            float(value) for value in ALLOCATOR_SLEEP_RE.findall(text)
        ),
        "disk_read": io_log_summary(text, DISK_READ_RE),
        "disk_write": io_log_summary(text, DISK_WRITE_RE),
        "disk_prefetch_chunks": sum(
            int(value) for value in DISK_PREFETCH_RE.findall(text)
        ),
        "retrieved_length_tokens": distribution(retrieved_lengths),
    }


def count_alloc_fail(path):
    text, _ = read_log_from(path, 0)
    return analyze_log(text)["events"]["alloc_fail"] if text else -1


def cache_marker(dataset, group, index):
    digest = hashlib.sha256(
        f"lmcache-diskread-v2:{dataset}:{group}:{index}".encode("utf-8")
    ).hexdigest()
    # The digest is the first content in the prompt, so T and E diverge inside
    # their first cache block and cannot share a block hash.
    return f"{digest}\n"


def prompt_size(tokenizer, prompt, chunk_size):
    tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    return {
        "tokens": tokens,
        "full_blocks": tokens // chunk_size,
        "lmcache_chunks": math.ceil(tokens / chunk_size),
    }


def make_records(
    dataset,
    rows,
    builder,
    indices,
    group,
    suffix,
    tokenizer,
    chunk_size,
    max_prompts,
    max_chunks=0,
    min_full_blocks=0,
):
    records = []
    total_chunks = 0
    total_full_blocks = 0
    for index in indices:
        prompt = (
            cache_marker(dataset, group, index)
            + builder(rows[index], index)
            + suffix
        )
        size = prompt_size(tokenizer, prompt, chunk_size)
        if max_chunks and records and total_chunks + size["lmcache_chunks"] > max_chunks:
            break
        records.append(
            {
                "index": index,
                "prompt": prompt,
                **size,
            }
        )
        total_chunks += size["lmcache_chunks"]
        total_full_blocks += size["full_blocks"]
        if len(records) >= max_prompts:
            break
        if min_full_blocks and total_full_blocks >= min_full_blocks:
            break

    if min_full_blocks and total_full_blocks < min_full_blocks:
        raise RuntimeError(
            f"{group}: only built {total_full_blocks} full blocks; "
            f"need at least {min_full_blocks}. Increase --evict-count."
        )
    return records


def records_summary(records):
    return {
        "prompts": len(records),
        "tokens": sum(record["tokens"] for record in records),
        "full_blocks": sum(record["full_blocks"] for record in records),
        "lmcache_chunks": sum(record["lmcache_chunks"] for record in records),
        "min_tokens": min(record["tokens"] for record in records),
        "max_tokens": max(record["tokens"] for record in records),
    }


def find_metric(mapping, name):
    value = mapping.get(name)
    return value if isinstance(value, (int, float)) else None


def cache_miss_analysis(deltas):
    def query_hit_miss(query_name, hit_name):
        queries = find_metric(deltas, query_name)
        hits = find_metric(deltas, hit_name)
        misses = None if queries is None or hits is None else max(0.0, queries - hits)
        hit_rate = None if queries in (None, 0) or hits is None else hits / queries
        return {
            "unit": "tokens",
            "queries": queries,
            "hits": hits,
            "misses": misses,
            "hit_rate": hit_rate,
        }

    requested_tokens = find_metric(deltas, "lmcache:num_requested_tokens")
    hit_tokens = find_metric(deltas, "lmcache:num_hit_tokens")
    lookup_tokens = find_metric(deltas, "lmcache:num_lookup_tokens")
    lookup_hits = find_metric(deltas, "lmcache:num_lookup_hits")
    return {
        "apc": query_hit_miss(
            "vllm:prefix_cache_queries", "vllm:prefix_cache_hits"
        ),
        "external": query_hit_miss(
            "vllm:external_prefix_cache_queries",
            "vllm:external_prefix_cache_hits",
        ),
        "lmcache_retrieve_tokens": {
            "requested": requested_tokens,
            "hits": hit_tokens,
            "retrieve_shortfall": (
                None
                if requested_tokens is None or hit_tokens is None
                else max(0.0, requested_tokens - hit_tokens)
            ),
            "hit_rate": (
                None
                if requested_tokens in (None, 0) or hit_tokens is None
                else hit_tokens / requested_tokens
            ),
        },
        "lmcache_lookup": {
            "queries": lookup_tokens,
            "hits": lookup_hits,
            "lookup_misses": (
                None
                if lookup_tokens is None or lookup_hits is None
                else max(0.0, lookup_tokens - lookup_hits)
            ),
            "hit_rate": (
                None
                if lookup_tokens in (None, 0) or lookup_hits is None
                else lookup_hits / lookup_tokens
            ),
        },
    }


def build_scbench_prompt(row, ctx_len, tokenizer):
    """Truncate the SCBench shared context to ctx_len tokens, then append turn-1 query."""
    context = row["context"]
    if ctx_len and ctx_len > 0:
        tokens = tokenizer.encode(context, add_special_tokens=False)
        if len(tokens) > ctx_len:
            context = tokenizer.decode(tokens[:ctx_len])
    query = row["multi_turns"][0]["input"]
    return f"{context}\n\nQuestion: {query}\nAnswer:"


def scbench_answer_correct(text, gold):
    """Loose containment match for SCBench free-text answers (correctness sanity check)."""
    if not gold or not text:
        return False
    norm = lambda s: "".join(ch for ch in str(s).lower() if ch.isalnum())
    needle = norm(gold)
    return bool(needle) and needle in norm(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="qwen2.5-14b")
    parser.add_argument(
        "--tokenizer",
        default="/data/models/Qwen2.5-14B-Instruct",
        help="Local tokenizer path used for actual block accounting.",
    )
    parser.add_argument("--dataset", choices=["gsm8k", "gpqa", "scbench"], required=True)
    parser.add_argument("--ctx-len", type=int, default=0,
                        help="SCBench context truncation in tokens (e.g. 32000)")
    parser.add_argument("--cache-dir", default="/data/hf-cache/hub",
                        help="HF dataset cache dir (offline)")

    # Legacy arguments are retained. --warmup is now the maximum number of T
    # prompts; --target-chunks can stop earlier using actual tokenizer lengths.
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup-conc", type=int, default=1)
    parser.add_argument("--warmup-mt", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1)

    parser.add_argument("--mode", choices=["pressure", "quiescent"], default="pressure")
    parser.add_argument(
        "--target-chunks",
        type=int,
        default=0,
        help="Cap target LMCache chunks; 96 leaves headroom in a 3 GiB staging pool.",
    )
    parser.add_argument("--evict-count", type=int, default=256)
    parser.add_argument(
        "--evict-blocks",
        type=int,
        default=384,
        help="Minimum actual complete APC blocks in the disjoint E set.",
    )
    parser.add_argument("--evict-concurrency", type=int, default=8)
    parser.add_argument("--evict-mt", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--apc-blocks", type=int, default=0)
    parser.add_argument(
        "--hot-blocks",
        type=int,
        default=0,
        help="Largest hot-cache capacity among compared configurations.",
    )
    parser.add_argument("--evict-margin", type=float, default=1.25)
    parser.add_argument("--min-disk-chunk-ratio", type=float, default=0.90)
    parser.add_argument("--min-external-transfer-ratio", type=float, default=0.90)
    parser.add_argument("--chunk-mib", type=float, default=24.0)
    parser.add_argument("--disk-gib", type=float, default=18.0)
    parser.add_argument("--cpu-gib", type=float, default=0.0)
    parser.add_argument("--staging-gib", type=float, default=0.0)
    parser.add_argument("--disk-headroom", type=float, default=0.80)

    parser.add_argument("--drain-timeout", type=float, default=300.0)
    parser.add_argument("--drain-poll", type=float, default=2.0)
    parser.add_argument("--drain-stable-polls", type=int, default=5)
    parser.add_argument("--expect-disk", action="store_true")
    parser.add_argument("--strict-tier-check", action="store_true")
    parser.add_argument(
        "--ttft-slo-seconds",
        type=float,
        default=0.0,
        help="Optional TTFT SLO used to calculate goodput QPS.",
    )
    parser.add_argument(
        "--latency-slo-seconds",
        type=float,
        default=0.0,
        help="Optional end-to-end latency SLO used to calculate goodput QPS.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="unknown")
    parser.add_argument("--tag", default="diskread")
    parser.add_argument("--out", default="/home/fcqiao/v018-tests/perf_results")
    parser.add_argument("--summary-jsonl", default="")
    args = parser.parse_args()

    if args.rounds != 1:
        parser.error(
            "--rounds must be 1: after the first measured T access, T is hot again. "
            "Use fresh server runs for independent repetitions."
        )
    if args.warmup < 1 or args.evict_count < 1:
        parser.error("--warmup and --evict-count must be positive")
    if args.target_chunks < 0:
        parser.error("--target-chunks cannot be negative")
    if args.ttft_slo_seconds < 0 or args.latency_slo_seconds < 0:
        parser.error("SLO thresholds cannot be negative")
    if not 0 <= args.min_disk_chunk_ratio <= 1:
        parser.error("--min-disk-chunk-ratio must be between 0 and 1")
    if not 0 <= args.min_external_transfer_ratio <= 1:
        parser.error("--min-external-transfer-ratio must be between 0 and 1")

    required_evict_blocks = math.ceil(
        args.evict_margin * max(args.apc_blocks, args.hot_blocks)
    )
    if required_evict_blocks and args.evict_blocks < required_evict_blocks:
        parser.error(
            f"--evict-blocks={args.evict_blocks} is too small; need at least "
            f"{required_evict_blocks} for APC={args.apc_blocks}, "
            f"hot={args.hot_blocks}, margin={args.evict_margin}"
        )

    os.makedirs(args.out, exist_ok=True)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        local_files_only=True,
        trust_remote_code=True,
    )
    train = load_dataset("openai/gsm8k", "main", split="train")

    if args.dataset == "gsm8k":
        rows = load_dataset("openai/gsm8k", "main", split="test")
        builder = lambda row, seed: build_gsm8k_unique(train, row["question"], seed)

        def gold_for(index):
            match = re.search(r"####\s*(-?[\d,]+)", rows[index]["answer"])
            return _norm(match.group(1)) if match else None

        extract_answer = extract_gsm8k_answer
        answer_correct = lambda text, gold: extract_gsm8k_answer(text) == gold
    elif args.dataset == "gpqa":
        rows = load_dataset("ankner/gpqa", split="train")
        builder = lambda row, seed: build_gpqa_unique(train, row["question"], seed)
        gold_for = lambda index: rows[index]["answer"].strip()
        extract_answer = extract_gpqa_answer
        answer_correct = lambda text, gold: extract_gpqa_answer(text) == gold
    else:
        rows = load_dataset(
            "microsoft/SCBench", "scbench_qa_eng", split="test",
            cache_dir=args.cache_dir,
        )
        builder = lambda row, seed: build_scbench_prompt(row, args.ctx_len, tokenizer)
        gold_for = lambda index: rows[index]["multi_turns"][0]["answer"]
        extract_answer = lambda text: text
        answer_correct = scbench_answer_correct

    needed_indices = args.warmup + args.evict_count
    if needed_indices > len(rows):
        raise ValueError(
            f"need {needed_indices} disjoint dataset rows, but {args.dataset} "
            f"only has {len(rows)}; reduce --warmup/--evict-count"
        )

    rng = random.Random(args.seed)
    shuffled_indices = list(range(len(rows)))
    rng.shuffle(shuffled_indices)
    target_candidates = shuffled_indices[: args.warmup]
    evict_candidates = shuffled_indices[args.warmup : needed_indices]

    suffix = "\n\nLet's think step by step.\n" if args.max_tokens >= 64 else "\n\n"
    target_records = make_records(
        args.dataset,
        rows,
        builder,
        target_candidates,
        "target",
        suffix,
        tokenizer,
        args.chunk_size,
        max_prompts=args.warmup,
        max_chunks=args.target_chunks,
    )
    evict_records = make_records(
        args.dataset,
        rows,
        builder,
        evict_candidates,
        "evict",
        suffix,
        tokenizer,
        args.chunk_size,
        max_prompts=args.evict_count,
        min_full_blocks=args.evict_blocks,
    )
    target_summary = records_summary(target_records)
    evict_summary = records_summary(evict_records)

    estimated_mib = (
        target_summary["lmcache_chunks"] + evict_summary["lmcache_chunks"]
    ) * args.chunk_mib
    disk_budget_mib = args.disk_gib * 1024 * args.disk_headroom
    if args.expect_disk and estimated_mib > disk_budget_mib:
        raise RuntimeError(
            f"estimated T+E KV is {estimated_mib / 1024:.2f} GiB, above "
            f"the safe disk budget {disk_budget_mib / 1024:.2f} GiB. "
            "Reduce target/eviction size or increase disk capacity."
        )

    print(
        f"=== [{args.tag}] {args.dataset} mode={args.mode} APC ON; "
        f"T={target_summary['prompts']} prompts/{target_summary['lmcache_chunks']} chunks, "
        f"E={evict_summary['prompts']} prompts/{evict_summary['full_blocks']} full blocks/"
        f"{evict_summary['lmcache_chunks']} chunks; measure conc={args.concurrency}, "
        f"mt={args.max_tokens} ===",
        flush=True,
    )
    print(
        f"  T tokens={target_summary['min_tokens']}..{target_summary['max_tokens']}; "
        f"E tokens={evict_summary['min_tokens']}..{evict_summary['max_tokens']}; "
        f"estimated disk={estimated_mib / 1024:.2f}/{args.disk_gib:.2f} GiB",
        flush=True,
    )

    target_prompts = [record["prompt"] for record in target_records]
    evict_prompts = [record["prompt"] for record in evict_records]
    golds = [gold_for(record["index"]) for record in target_records]

    # T1: persist exact target prompts. Prompt suffix is based on measurement
    # max_tokens, so warm and measured prompts are byte-for-byte identical.
    warm_results = run_batch(
        target_prompts,
        args.endpoint,
        args.model,
        args.warmup_mt,
        args.warmup_conc,
    )
    require_success(warm_results, "target warm")
    print("  T1 target warm done", flush=True)

    disk_root = os.environ.get("SSD_PATH", "/data/lmcache-disk/").replace(
        "file://", ""
    )
    target_drain, target_disk_state, target_peak = wait_disk_stable(
        disk_root,
        "target",
        args.drain_timeout,
        args.drain_poll,
        args.drain_stable_polls,
        args.expect_disk,
    )
    print(
        f"  target drain={target_drain:.1f}s files={target_disk_state['files']} "
        f"bytes={target_disk_state['bytes']}",
        flush=True,
    )

    # E: disjoint prompts whose actual complete blocks exceed APC and the
    # largest hot cache. In pressure mode, intentionally do not drain here.
    evict_results = run_batch(
        evict_prompts,
        args.endpoint,
        args.model,
        args.evict_mt,
        args.evict_concurrency,
    )
    require_success(evict_results, "eviction warm")
    print("  E eviction set done", flush=True)

    if args.mode == "quiescent":
        evict_drain, pre_measure_disk_state, evict_peak = wait_disk_stable(
            disk_root,
            "eviction",
            args.drain_timeout,
            args.drain_poll,
            args.drain_stable_polls,
            args.expect_disk,
        )
    else:
        evict_drain = 0.0
        pre_measure_disk_state = disk_state(disk_root)
        evict_peak = pre_measure_disk_state["files"]

    metrics_before = scrape_metrics(args.endpoint)
    pinned_before = find_metric(metrics_before, "lmcache:pinned_memory_objs_count")
    if args.mode == "pressure":
        if pinned_before is None:
            print(
                "  NOTE: pinned-memory metric is unavailable; pressure overlap "
                "will be verified from E disk-write logs during T",
                flush=True,
            )
        elif pinned_before <= 0:
            print(
                "  NOTE: pinned_memory_objs_count=0; pending disk puts use refcounts, "
                "so final pressure validity will use overlapping disk-write logs",
                flush=True,
            )
        else:
            print(
                f"  pressure gate: pinned_memory_objs_count={pinned_before:g}",
                flush=True,
            )

    server_log = os.environ.get("SERVER_LOG", "")
    # Put the log/disk boundary immediately before T; metrics scraping is kept
    # outside the pressure-overlap window because it can block briefly.
    pre_measure_disk_state = disk_state(disk_root)
    log_offset = os.path.getsize(server_log) if server_log and os.path.exists(server_log) else 0

    # T2: exactly one measured access per target prompt.
    measure_started = time.perf_counter()
    measured_results = run_batch(
        target_prompts,
        args.endpoint,
        args.model,
        args.max_tokens,
        args.concurrency,
    )
    measure_wall_sec = time.perf_counter() - measure_started
    log_text, _ = read_log_from(server_log, log_offset)
    post_measure_disk_state = disk_state(disk_root)
    metrics_after = scrape_metrics(args.endpoint)
    log_analysis = analyze_log(log_text)
    log_events = log_analysis["events"]

    ok_results = [result for result in measured_results if result.ok]
    failures = [result for result in measured_results if not result.ok]
    if not ok_results:
        examples = "; ".join(result.error for result in failures[:3])
        raise RuntimeError(f"measured T: all requests failed: {examples}")

    ttfts = [result.ttft for result in ok_results]
    latencies = [result.latency for result in ok_results]
    decode_latencies = [
        max(0.0, result.latency - result.ttft) for result in ok_results
    ]
    ttft_stats = distribution(ttfts)
    latency_stats = distribution(latencies)
    decode_latency_stats = distribution(decode_latencies)
    mean_ttft = ttft_stats["mean"]
    p50 = ttft_stats["p50"]
    p95 = ttft_stats["p95"]
    p99 = ttft_stats["p99"]

    per_request = []
    successful_prompt_tokens = 0
    successful_output_tokens = 0
    for record, result in zip(target_records, measured_results):
        output_tokens = len(
            tokenizer.encode(result.text, add_special_tokens=False)
        )
        if result.ok:
            successful_prompt_tokens += record["tokens"]
            successful_output_tokens += output_tokens
        per_request.append(
            {
                "dataset_index": record["index"],
                "prompt_tokens": record["tokens"],
                "prompt_full_blocks": record["full_blocks"],
                "output_tokens": output_tokens,
                "ttft_sec": result.ttft,
                "latency_sec": result.latency,
                "decode_after_first_token_sec": max(
                    0.0, result.latency - result.ttft
                ),
                "ok": result.ok,
                "status_code": result.status_code,
                "error": result.error,
            }
        )

    attempted_qps = len(measured_results) / measure_wall_sec
    successful_qps = len(ok_results) / measure_wall_sec
    ttft_good = sum(
        1
        for result in ok_results
        if not args.ttft_slo_seconds or result.ttft <= args.ttft_slo_seconds
    )
    latency_good = sum(
        1
        for result in ok_results
        if not args.latency_slo_seconds
        or result.latency <= args.latency_slo_seconds
    )
    both_good = sum(
        1
        for result in ok_results
        if (not args.ttft_slo_seconds or result.ttft <= args.ttft_slo_seconds)
        and (
            not args.latency_slo_seconds
            or result.latency <= args.latency_slo_seconds
        )
    )
    throughput = {
        "measurement_wall_sec": measure_wall_sec,
        "attempted_requests": len(measured_results),
        "successful_requests": len(ok_results),
        "failed_requests": len(failures),
        "attempted_qps": attempted_qps,
        "successful_qps": successful_qps,
        "prompt_tokens": successful_prompt_tokens,
        "output_tokens": successful_output_tokens,
        "total_tokens": successful_prompt_tokens + successful_output_tokens,
        "prompt_tokens_per_sec": successful_prompt_tokens / measure_wall_sec,
        "output_tokens_per_sec": successful_output_tokens / measure_wall_sec,
        "total_tokens_per_sec": (
            successful_prompt_tokens + successful_output_tokens
        ) / measure_wall_sec,
        "ttft_slo_seconds": args.ttft_slo_seconds or None,
        "latency_slo_seconds": args.latency_slo_seconds or None,
        "ttft_goodput_qps": ttft_good / measure_wall_sec,
        "latency_goodput_qps": latency_good / measure_wall_sec,
        "combined_goodput_qps": both_good / measure_wall_sec,
    }
    accuracy = 0.0
    if args.max_tokens >= 64:
        correct = sum(
            1
            for index, result in enumerate(measured_results)
            if result.ok and answer_correct(result.text, golds[index])
        )
        accuracy = correct / len(measured_results)

    deltas = metric_delta(metrics_before, metrics_after)
    miss_analysis = cache_miss_analysis(deltas)
    allocator_analysis = {
        "hard_alloc_failures": log_events["alloc_fail"],
        "staging_alloc_failures": log_events["staging_alloc_fail"],
        "allocation_retries": log_events["alloc_retry"],
        "no_evict_candidate_events": log_events["no_evict_candidate"],
        "not_busy_looping_events": log_events["not_busy_looping"],
        "sleep_events": log_events["allocator_sleep"],
        "sleep_cumulative_seconds": log_analysis["allocator_sleep_seconds"],
        "local_cpu_evict_count_delta": find_metric(
            deltas, "lmcache:local_cpu_evict_count"
        ),
        "local_cpu_evict_keys_delta": find_metric(
            deltas, "lmcache:local_cpu_evict_keys_count"
        ),
        "local_cpu_evict_failed_delta": find_metric(
            deltas, "lmcache:local_cpu_evict_failed_count"
        ),
    }
    apc_hit_delta = find_metric(deltas, "vllm:prefix_cache_hits")
    external_hit_delta = find_metric(deltas, "vllm:external_prefix_cache_hits")
    lmcache_hit_tokens = find_metric(deltas, "lmcache:num_hit_tokens")
    lookup_hits = find_metric(deltas, "lmcache:num_lookup_hits")
    local_cache_tokens = find_metric(
        deltas, "vllm:prompt_tokens_by_source:local_cache_hit"
    )
    external_transfer_tokens = find_metric(
        deltas, "vllm:prompt_tokens_by_source:external_kv_transfer"
    )
    local_compute_tokens = find_metric(
        deltas, "vllm:prompt_tokens_by_source:local_compute"
    )
    throughput["server_prompt_tokens_by_source"] = {
        "local_cache_hit": local_cache_tokens,
        "external_kv_transfer": external_transfer_tokens,
        "local_compute": local_compute_tokens,
    }
    if local_compute_tokens is not None:
        throughput["local_compute_prompt_tokens_per_sec"] = (
            local_compute_tokens / measure_wall_sec
        )

    capacity_conditioned = (
        not required_evict_blocks
        or evict_summary["full_blocks"] >= required_evict_blocks
    )
    apc_observed_hits = (
        local_cache_tokens if local_cache_tokens is not None else apc_hit_delta
    )
    apc_miss_valid = (
        None if apc_observed_hits is None else apc_observed_hits == 0
    )
    observed_disk_chunks = max(
        log_analysis["disk_prefetch_chunks"],
        log_analysis["disk_read"]["operations"],
    )
    disk_chunk_ratio = observed_disk_chunks / target_summary["lmcache_chunks"]
    target_disk_complete = (
        not args.expect_disk
        or target_disk_state["files"] >= target_summary["lmcache_chunks"]
    )
    disk_path_observed = (
        not args.expect_disk or disk_chunk_ratio >= args.min_disk_chunk_ratio
    )
    if args.mode == "pressure" and args.expect_disk:
        disk_state_changed = pre_measure_disk_state != post_measure_disk_state
        pressure_valid = (
            log_analysis["disk_write"]["operations"] > 0
            or disk_state_changed
        )
    else:
        disk_state_changed = False
        pressure_valid = None
    conditioning_valid = (
        capacity_conditioned
        and apc_miss_valid is True
        and target_disk_complete
        and disk_path_observed
        and len(failures) == 0
        and (pressure_valid is not False)
    )

    print(
        "  metric delta: "
        f"apc_hits={apc_hit_delta} external_hits={external_hit_delta} "
        f"lmcache_hit_tokens={lmcache_hit_tokens} lookup_hits={lookup_hits}",
        flush=True,
    )
    print(
        "  cache miss: "
        f"APC={miss_analysis['apc']['misses']} "
        f"external={miss_analysis['external']['misses']} "
        f"LMCache_retrieve_shortfall="
        f"{miss_analysis['lmcache_retrieve_tokens']['retrieve_shortfall']}; "
        f"disk_prefetch_chunks={log_analysis['disk_prefetch_chunks']}",
        flush=True,
    )
    print(
        f"  log delta: {log_events}; measured failures={len(failures)}/"
        f"{len(measured_results)}",
        flush=True,
    )
    print(
        f"  performance: latency mean={latency_stats['mean']:.3f}s "
        f"p95={latency_stats['p95']:.3f}s; TTFT mean={mean_ttft:.3f}s "
        f"p95={p95:.3f}s; QPS={successful_qps:.3f}; "
        f"prompt_tps={throughput['prompt_tokens_per_sec']:.1f} "
        f"output_tps={throughput['output_tokens_per_sec']:.1f}",
        flush=True,
    )
    print(
        f"  allocator analysis: hard_fail={log_events['alloc_fail']} "
        f"retry={log_events['alloc_retry']} "
        f"no_candidate={log_events['no_evict_candidate']} "
        f"sleep={log_events['allocator_sleep']} events/"
        f"{log_analysis['allocator_sleep_seconds']:.3f}s "
        f"evict_failed_metric="
        f"{allocator_analysis['local_cpu_evict_failed_delta']}",
        flush=True,
    )
    if apc_hit_delta is None:
        print(
            "  WARNING: APC hit metric is unavailable; verify APC miss in the "
            "saved metrics/logs",
            flush=True,
        )
    elif apc_hit_delta != 0:
        print(
            f"  INVALID CONDITIONING: measured T had {apc_hit_delta:g} APC hits",
            flush=True,
        )
    if args.expect_disk and not disk_path_observed:
        print(
            f"  INVALID CONDITIONING: disk coverage {observed_disk_chunks}/"
            f"{target_summary['lmcache_chunks']} chunks is below threshold",
            flush=True,
        )
    if not target_disk_complete:
        print("  INVALID CONDITIONING: target did not fully persist to disk", flush=True)
    if failures:
        print("  INVALID CONDITIONING: measured requests failed", flush=True)
    if args.mode == "pressure" and args.expect_disk and not pressure_valid:
        print(
            "  INVALID PRESSURE: no E disk write overlapped the measured T window",
            flush=True,
        )

    drain_total = target_drain + evict_drain
    alloc_fail = log_events["alloc_fail"]
    server_total_alloc_fail = count_alloc_fail(server_log)
    disk_peak = max(target_peak, evict_peak)
    output = {
        # Legacy output contract.
        "tag": args.tag,
        "config": args.config,
        "dataset": args.dataset,
        "arguments": vars(args),
        "warmup": len(target_records),
        "rounds": args.rounds,
        "mean": mean_ttft,
        "p50": p50,
        "p95": p95,
        "accuracy": accuracy,
        "drain_sec": drain_total,
        "alloc_fail": alloc_fail,
        "server_total_alloc_fail": server_total_alloc_fail,
        "disk_files_peak": disk_peak,
        "per_round_mean": [mean_ttft],
        # T/E/T details.
        "mode": args.mode,
        "p99": p99,
        "ttft_sec": ttft_stats,
        "latency_sec": latency_stats,
        "decode_after_first_token_sec": decode_latency_stats,
        "throughput": throughput,
        "successes": len(ok_results),
        "failures": len(failures),
        "failure_examples": [result.error for result in failures[:10]],
        "target": target_summary,
        "eviction": evict_summary,
        "target_indices": [record["index"] for record in target_records],
        "evict_indices": [record["index"] for record in evict_records],
        "target_token_counts": [record["tokens"] for record in target_records],
        "evict_token_counts": [record["tokens"] for record in evict_records],
        "required_evict_blocks": required_evict_blocks,
        "estimated_disk_gib": estimated_mib / 1024,
        "target_drain_sec": target_drain,
        "evict_drain_sec": evict_drain,
        "target_disk_state": target_disk_state,
        "pre_measure_disk_state": pre_measure_disk_state,
        "post_measure_disk_state": post_measure_disk_state,
        "disk_state_changed_during_measurement": disk_state_changed,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "metric_delta": deltas,
        "cache_miss_analysis": miss_analysis,
        "allocator_analysis": allocator_analysis,
        "log_event_delta": log_events,
        "log_analysis": log_analysis,
        "per_request": per_request,
        "capacity_conditioned": capacity_conditioned,
        "apc_miss_valid": apc_miss_valid,
        "target_disk_complete": target_disk_complete,
        "observed_disk_chunks": observed_disk_chunks,
        "disk_chunk_ratio": disk_chunk_ratio,
        "disk_path_observed": disk_path_observed,
        "pressure_valid": pressure_valid,
        "conditioning_valid": conditioning_valid,
    }

    output_path = os.path.join(
        args.out, f"diskread_{args.dataset}_{args.tag}.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(
        f"=== [{args.tag}] OVERALL: mean={mean_ttft:.3f}s p50={p50:.3f}s "
        f"p95={p95:.3f}s acc={accuracy * 100:.1f}% "
        f"drain={drain_total:.1f}s alloc_fail={alloc_fail} "
        f"latency_mean={latency_stats['mean']:.3f}s qps={successful_qps:.3f} "
        f"valid={conditioning_valid} ===",
        flush=True,
    )

    if args.strict_tier_check and not conditioning_valid:
        raise SystemExit(2)
    if args.summary_jsonl:
        summary_parent = os.path.dirname(args.summary_jsonl)
        if summary_parent:
            os.makedirs(summary_parent, exist_ok=True)
        with open(args.summary_jsonl, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
