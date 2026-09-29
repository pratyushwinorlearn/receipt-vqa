"""Batch-size-1 latency and peak VRAM for base vs adapter.

python scripts/benchmark_latency.py --adapter outputs/runs/<id>/adapter --n 30
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from receiptqa.data.dataset import load_image
from receiptqa.inference.predict import Predictor
from receiptqa.utils.io import read_jsonl


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", default="data/processed/val.jsonl")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--longest-edge", type=int, default=1536)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    rows = read_jsonl(args.data)[: args.n + args.warmup]
    pred = Predictor(args.model, adapter_path=args.adapter, longest_edge=args.longest_edge)
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    for name, use_adapter in (("base", False), ("adapter", True)):
        times = []
        for i, r in enumerate(rows):
            img = load_image(r["image_path"])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred.answer(img, r["question"], use_adapter=use_adapter)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if i >= args.warmup:
                times.append(time.perf_counter() - t0)
        peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
        print(
            f"{name:8s} p50={statistics.median(times):.2f}s p95={pct(times, 0.95):.2f}s "
            f"mean={statistics.mean(times):.2f}s peak_vram={peak:.2f}GB"
        )


if __name__ == "__main__":
    main()
