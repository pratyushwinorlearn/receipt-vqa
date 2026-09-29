"""Evaluate base and fine-tuned models on a split.

    python -m receiptqa.eval.evaluate --config configs/eval.yaml --split val --modes base_plain base_brief
    python -m receiptqa.eval.evaluate --config configs/eval.yaml --split test --adapter outputs/runs/<id>/adapter

Modes: base_plain (B0), base_brief (B1), adapter_plain (M1), adapter_brief.
"""

from __future__ import annotations

import argparse
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from receiptqa.data.dataset import load_image
from receiptqa.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff
from receiptqa.eval.metrics import score_one, summarize
from receiptqa.utils.io import load_yaml, read_jsonl, write_json, write_jsonl
from receiptqa.utils.logging import get_logger

log = get_logger("evaluate")

MODES = {  # mode -> (use_adapter, prompt style)
    "base_plain": (False, "plain"),
    "base_brief": (False, "brief"),
    "adapter_plain": (True, "plain"),
    "adapter_brief": (True, "brief"),
}


def run_mode(predictor: Any, rows: list[dict[str, Any]], mode: str, batch_size: int, max_new_tokens: int):
    use_adapter, style = MODES[mode]
    preds, batch_latencies = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        images = [load_image(r["image_path"]) for r in chunk]
        t0 = time.perf_counter()
        out = predictor.answer_batch(images, [r["question"] for r in chunk], use_adapter, style, max_new_tokens)
        batch_latencies.append((time.perf_counter() - t0) / len(chunk))
        preds.extend(out)
        log.info("[%s] %d/%d", mode, min(i + batch_size, len(rows)), len(rows))
    return preds, batch_latencies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--modes", nargs="+", default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    split = args.split or cfg["split"]
    adapter = args.adapter or cfg.get("adapter_path")
    modes = args.modes or cfg["modes"]
    max_samples = args.max_samples or cfg.get("max_samples")

    if split.startswith("test"):
        log.warning("Evaluating on the TEST split. Do this once per finished model, never for tuning.")
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise SystemExit(f"unknown modes {unknown}, choose from {list(MODES)}")
    if not adapter:
        skipped = [m for m in modes if MODES[m][0]]
        if skipped:
            log.warning("no adapter given, skipping modes: %s", skipped)
        modes = [m for m in modes if not MODES[m][0]]

    rows = read_jsonl(Path(cfg["processed_dir"]) / f"{split}.jsonl")
    if max_samples:
        rows = rows[:max_samples]
    log.info("split=%s questions=%d receipts=%d", split, len(rows), len({r["receipt_id"] for r in rows}))

    from receiptqa.inference.predict import Predictor

    predictor = Predictor(
        cfg["model_id"],
        adapter_path=adapter,
        quantize_4bit=cfg["quantize_4bit"],
        dtype=cfg["dtype"],
        longest_edge=cfg["image"]["longest_edge"],
        do_image_splitting=cfg["image"]["do_image_splitting"],
    )

    run_name = Path(adapter).parent.name if adapter else "baseline"
    run_id = f"{datetime.now():%Y%m%d}-{run_name}"
    out_dir = Path(cfg["results_dir"])
    n_boot, seed = cfg["bootstrap_samples"], cfg["seed"]
    rids = [r["receipt_id"] for r in rows]

    results: dict[str, Any] = {"run_id": run_id, "split": split, "adapter": adapter, "modes": {}}
    all_scores: dict[str, dict[str, list[float]]] = {}
    for mode in modes:
        preds, lat = run_mode(predictor, rows, mode, cfg["batch_size"], cfg["max_new_tokens"])
        records = []
        for r, p in zip(rows, preds, strict=True):
            records.append({**r, "prediction": p, **score_one(p, r["answer"], r["qtype"])})
        write_jsonl(out_dir / f"{run_id}_{mode}_{split}.jsonl", records)

        summary = summarize(records)
        em_m, em_lo, em_hi = bootstrap_ci(rids, [x["em"] for x in records], n_boot, seed)
        an_m, an_lo, an_hi = bootstrap_ci(rids, [x["anls"] for x in records], n_boot, seed)
        summary["em_ci95"] = [em_lo, em_hi]
        summary["anls_ci95"] = [an_lo, an_hi]
        summary["latency_s_per_q"] = {
            "p50": statistics.median(lat),
            "p95": sorted(lat)[max(0, int(0.95 * len(lat)) - 1)],
            "note": f"amortized over batch_size={cfg['batch_size']}",
        }
        results["modes"][mode] = summary
        all_scores[mode] = {"em": [x["em"] for x in records], "anls": [x["anls"] for x in records]}

    if "adapter_plain" in all_scores:
        results["paired_diff"] = {}
        for ref in ("base_brief", "base_plain"):
            if ref in all_scores:
                d, lo, hi = paired_bootstrap_diff(
                    rids, all_scores["adapter_plain"]["em"], all_scores[ref]["em"], n_boot, seed
                )
                results["paired_diff"][f"em: adapter_plain - {ref}"] = {"diff": d, "ci95": [lo, hi]}

    write_json(out_dir / f"{run_id}_{split}.json", results)

    print(f"\n## {run_id} on {split}\n")
    print("| Mode | N | EM | EM 95% CI | ANLS | Numeric acc | Lat p50 (s/q) |")
    print("|------|---|----|-----------|------|-------------|---------------|")
    for mode, s in results["modes"].items():
        lo, hi = s["em_ci95"]
        print(
            f"| {mode} | {s['n']} | {s['em']:.3f} | [{lo:.3f}, {hi:.3f}] | {s['anls']:.3f} | "
            f"{s['numeric_acc']:.3f} | {s['latency_s_per_q']['p50']:.2f} |"
        )
    for name, d in results.get("paired_diff", {}).items():
        print(f"\npaired {name}: {d['diff']:+.3f}  95% CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]")
    print(f"\nfull results -> {out_dir}/{run_id}_{split}.json")


if __name__ == "__main__":
    main()
