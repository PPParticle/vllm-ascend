#!/usr/bin/env python3
"""64 UNIQUE prefixes per batch (different 8-shot examples per request).
10 rounds, each round randomly samples 64 test questions.
APC+hot = avg KV of 64 requests / 2. Supports GSM8K and GPQA.
"""
import argparse, json, os, re, time, statistics, random
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple
import requests


class RequestResult(NamedTuple):
    """Tuple-compatible result; indices 0/1 remain TTFT/text for old callers."""

    ttft: float
    text: str
    ok: bool
    error: str
    status_code: int
    latency: float


def _percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int(len(ordered) * q)))
    return ordered[index]


def _distribution(values):
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None,
                "p99": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values),
            "p50": _percentile(values, 0.50),
            "p95": _percentile(values, 0.95),
            "p99": _percentile(values, 0.99),
            "min": min(values), "max": max(values)}


def _norm(s):
    if s is None: return None
    s = s.replace(",", "").strip()
    try: return float(s)
    except ValueError: return None

def extract_gsm8k_answer(text):
    m = re.findall(r"####\s*(-?[\d,]+)", text)
    if m: return _norm(m[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _norm(nums[-1]) if nums else None

def extract_gpqa_answer(text):
    m = re.search(r'\b([ABCD])\b', text)
    return m.group(1) if m else None

def build_gsm8k_unique(train, test_q, seed):
    """Different 8-shot examples per request → unique prefix."""
    rng = random.Random(seed)
    shot_indices = rng.sample(range(len(train)), 8)
    prefix = ""
    for j in shot_indices:
        prefix += f"Q: {train[j]['question']}\nA: {train[j]['answer']}\n\n"
    return prefix + f"Q: {test_q}\nA:"

def build_gpqa_unique(train, test_q, seed):
    """GPQA: use different GSM8K training examples as padding to create unique ~1200-token prefix + GPQA question."""
    rng = random.Random(seed)
    shot_indices = rng.sample(range(len(train)), 8)
    prefix = ""
    for j in shot_indices:
        prefix += f"Context: {train[j]['question']}\n"
    prefix += f"\nQuestion: {test_q}\nOptions: A/B/C/D\n\nAnswer:"
    return prefix


def request(endpoint, model, prompt, mt, timeout=120):
    """Issue one streaming completion and distinguish failures from TTFT."""
    t0 = time.perf_counter()
    full_text = ""
    ttft = None
    status_code = 0
    try:
        with requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": mt,
                  "temperature": 0, "stream": True},
            stream=True,
            timeout=timeout,
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                text = obj.get("choices", [{}])[0].get("text", "")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    full_text += text
    except requests.RequestException as exc:
        latency = time.perf_counter() - t0
        return RequestResult(latency, full_text, False,
                             f"{type(exc).__name__}: {exc}", status_code,
                             latency)
    except Exception as exc:
        latency = time.perf_counter() - t0
        return RequestResult(latency, full_text, False,
                             f"{type(exc).__name__}: {exc}", status_code,
                             latency)

    if ttft is None:
        latency = time.perf_counter() - t0
        return RequestResult(latency, full_text, False,
                             "stream completed without a non-empty token", status_code,
                             latency)
    latency = time.perf_counter() - t0
    return RequestResult(ttft, full_text, True, "", status_code, latency)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8100")
    ap.add_argument("--model", default="qwen2.5-14b")
    ap.add_argument("--tokenizer", default="/data/models/Qwen2.5-14B-Instruct")
    ap.add_argument("--dataset", choices=["gsm8k", "gpqa"], required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--per-round", type=int, default=64)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="uniq")
    ap.add_argument("--out", default="/home/fcqiao/v018-tests/perf_results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, trust_remote_code=True
    )
    train = load_dataset("openai/gsm8k", "main", split="train")

    rng = random.Random(args.seed)
    N, C, R = args.per_round, args.concurrency, args.rounds

    if args.dataset == "gsm8k":
        test = load_dataset("openai/gsm8k", "main", split="test")
        builder = lambda q, seed: build_gsm8k_unique(train, q, seed)
        gold_fn = lambda ans: _norm(re.search(r"####\s*(-?[\d,]+)", ans).group(1))
        mt = args.max_tokens
    else:
        test = load_dataset("ankner/gpqa", split="train")
        builder = lambda q, seed: build_gpqa_unique(train, q, seed)
        gold_fn = lambda ans: ans.strip()
        mt = args.max_tokens

    print(f"=== [{args.tag}] {args.dataset} {R} rounds x {N} UNIQUE-prefix, conc={C}, mt={mt} ===")

    all_ttfts = []
    all_latencies = []
    per_round_mean = []
    total_correct = 0
    total_reqs = 0
    total_wall = 0.0
    total_attempted = 0
    total_successful = 0
    total_prompt_tokens = 0
    total_output_tokens = 0
    for r in range(R):
        if N <= len(test):
            indices = rng.sample(range(len(test)), N)
        else:
            indices = rng.choices(range(len(test)), k=N)
        prompts = [builder(test[i]["question"] if args.dataset == "gsm8k" else test[i]["question"],
                            seed=r*1000+i) for i in indices]
        suffix = "\n\nLet's think step by step.\n" if mt >= 64 else "\n\n"
        final_prompts = [prompt + suffix for prompt in prompts]

        round_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=C) as ex:
            results = list(ex.map(
                lambda p: request(args.endpoint, args.model, p, mt), final_prompts
            ))
        round_wall = time.perf_counter() - round_started
        ok_results = [result for result in results if result.ok]
        failed = len(results) - len(ok_results)
        if not ok_results:
            errors = [result.error for result in results[:3]]
            raise RuntimeError(f"round {r}: all requests failed: {errors}")
        round_ttfts = [result.ttft for result in ok_results]
        round_latencies = [result.latency for result in ok_results]
        all_ttfts += round_ttfts
        all_latencies += round_latencies
        total_wall += round_wall
        total_attempted += len(results)
        total_successful += len(ok_results)
        for prompt, result in zip(final_prompts, results):
            if result.ok:
                total_prompt_tokens += len(
                    tokenizer.encode(prompt, add_special_tokens=False)
                )
                total_output_tokens += len(
                    tokenizer.encode(result.text, add_special_tokens=False)
                )
        rm = statistics.mean(round_ttfts)
        per_round_mean.append(rm)
        s = sorted(round_ttfts)
        if mt >= 64:
            if args.dataset == "gsm8k":
                correct = sum(1 for i, res in enumerate(results) if extract_gsm8k_answer(res[1]) == gold_fn(test[indices[i]]["answer"]))
            else:
                correct = sum(1 for i, res in enumerate(results) if extract_gpqa_answer(res[1]) == gold_fn(test[indices[i]]["answer"]))
            total_correct += correct
            total_reqs += N
            print(f"  round {r}: TTFT mean={rm:.3f}s p50={s[len(s)//2]:.3f}s "
                  f"p95={s[min(len(s)-1, int(len(s)*0.95))]:.3f}s "
                  f"latency_mean={statistics.mean(round_latencies):.3f}s "
                  f"qps={len(ok_results)/round_wall:.3f} "
                  f"acc={correct}/{N}={correct/N*100:.0f}% failed={failed}")
        else:
            print(f"  round {r}: TTFT mean={rm:.3f}s p50={s[len(s)//2]:.3f}s "
                  f"p95={s[min(len(s)-1, int(len(s)*0.95))]:.3f}s "
                  f"latency_mean={statistics.mean(round_latencies):.3f}s "
                  f"qps={len(ok_results)/round_wall:.3f} failed={failed}")

    s = sorted(all_ttfts)
    p50 = s[len(s)//2]; p95 = s[int(len(s)*0.95)]
    overall = statistics.mean(all_ttfts)
    acc = total_correct/total_reqs if total_reqs > 0 else 0
    ttft_stats = _distribution(all_ttfts)
    latency_stats = _distribution(all_latencies)
    throughput = {
        "measurement_wall_sec": total_wall,
        "attempted_requests": total_attempted,
        "successful_requests": total_successful,
        "failed_requests": total_attempted - total_successful,
        "attempted_qps": total_attempted / total_wall,
        "successful_qps": total_successful / total_wall,
        "prompt_tokens_per_sec": total_prompt_tokens / total_wall,
        "output_tokens_per_sec": total_output_tokens / total_wall,
        "total_tokens_per_sec": (
            total_prompt_tokens + total_output_tokens
        ) / total_wall,
    }
    print(f"=== [{args.tag}] OVERALL: TTFT mean={overall:.3f}s p50={p50:.3f}s "
          f"p95={p95:.3f}s latency_mean={latency_stats['mean']:.3f}s "
          f"qps={throughput['successful_qps']:.3f} acc={acc*100:.1f}% ===")
    with open(f"{args.out}/unique_{args.dataset}_{args.tag}.json", "w") as f:
        json.dump({"tag": args.tag, "dataset": args.dataset, "per_round": N, "rounds": R,
                   "mean": overall, "p50": p50, "p95": p95, "accuracy": acc,
                   "per_round_mean": per_round_mean, "ttft_sec": ttft_stats,
                   "latency_sec": latency_stats, "throughput": throughput},
                  f, indent=2)


if __name__ == "__main__":
    main()
