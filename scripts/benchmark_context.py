#!/usr/bin/env python3
"""Benchmark throughput and memory usage across context lengths."""

import argparse
import json
import time
import os
import torch


def benchmark_context_length(
    model_name: str = "llama-70b",
    context_lengths=None,
    device: str = "cuda",
):
    if context_lengths is None:
        context_lengths = [8192, 16384, 32768, 65536, 131072, 262144]

    results = []

    for ctx in context_lengths:
        torch.cuda.reset_peak_memory_stats()
        batch = torch.randint(0, 32000, (1, ctx), device=device)
        warmup_iters = 3
        bench_iters = 10

        for _ in range(warmup_iters):
            _ = torch.sum(batch.float())

        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(bench_iters):
            _ = torch.sum(batch.float())

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        throughput = (ctx * bench_iters) / elapsed

        result = {
            "context_length": ctx,
            "throughput_tokens_per_sec": round(throughput, 1),
            "peak_memory_gb": round(peak_mem, 1),
            "latency_ms": round((elapsed / bench_iters) * 1000, 2),
            "model": model_name,
        }

        results.append(result)
        print(f"Context {ctx:>8}: {throughput:>8.1f} tok/s, {peak_mem:>5.1f} GB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-70b")
    parser.add_argument("--output", default="results/benchmark_results.json")
    args = parser.parse_args()

    print(f"Benchmarking {args.model}...")
    results = benchmark_context_length(model_name=args.model)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()