"""Make README / LinkedIn figures from an evaluation results JSON.

python scripts/make_figures.py --results outputs/results/<run_id>_test.json --out assets

Writes 4 PNGs to --out: overall_em.png, em_by_type.png, failure_types.png, examples.png.
The last two need the predictions .jsonl files that evaluate.py wrote next to the results JSON
(<run_id>_<mode>_<split>.jsonl). Charts are skipped with a message if those files are missing.
"""

from __future__ import annotations

import argparse
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from receiptqa.eval.error_analysis import tag_failure  # noqa: E402
from receiptqa.utils.io import read_json, read_jsonl  # noqa: E402

ORDER = ["base_plain", "base_brief", "adapter_plain"]
LABELS = {
    "base_plain": "Base (plain prompt)",
    "base_brief": "Base (brief prompt)",
    "adapter_plain": "QLoRA fine-tuned",
}
COLORS = {"base_plain": "#8D99A6", "base_brief": "#C5CCD4", "adapter_plain": "#2F6FED"}
BAD, GOOD = "#C0392B", "#1E8449"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def modes_present(results: dict[str, Any]) -> list[str]:
    return [m for m in ORDER if m in results["modes"]]


def best_baseline(results: dict[str, Any]) -> str | None:
    bases = [m for m in results["modes"] if m.startswith("base")]
    return max(bases, key=lambda m: results["modes"][m]["em"]) if bases else None


def headline(results: dict[str, Any]) -> str:
    """Gain over the BEST zero-shot baseline (the conservative comparison)."""
    best = best_baseline(results)
    d = results.get("paired_diff", {}).get(f"em: adapter_plain - {best}")
    if not d:
        return ""
    lo, hi = d["ci95"]
    return f"{d['diff'] * 100:+.1f} pts vs best zero-shot baseline (95% CI [{lo * 100:+.1f}, {hi * 100:+.1f}])"


def load_preds(results: dict[str, Any], results_dir: Path, mode: str | None) -> list[dict[str, Any]] | None:
    if mode is None:
        return None
    path = results_dir / f"{results['run_id']}_{mode}_{results['split']}.jsonl"
    return read_jsonl(path) if path.exists() else None


def save(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name)
    plt.close(fig)
    print(f"wrote {out / name}")


def fig_overall(results: dict[str, Any], out: Path) -> None:
    modes = modes_present(results)
    m = results["modes"]
    ems = [m[k]["em"] * 100 for k in modes]
    lows = [m[k]["em_ci95"][0] * 100 for k in modes]
    highs = [m[k]["em_ci95"][1] * 100 for k in modes]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.bar(
        range(len(modes)),
        ems,
        color=[COLORS[k] for k in modes],
        yerr=[
            [max(0.0, e - lo) for e, lo in zip(ems, lows, strict=True)],
            [max(0.0, h - e) for h, e in zip(highs, ems, strict=True)],
        ],
        capsize=6,
        width=0.6,
        error_kw={"elinewidth": 1.5},
    )
    for i, (e, h) in enumerate(zip(ems, highs, strict=True)):
        ax.text(i, h + 1.5, f"{e:.1f}%", ha="center", fontweight="bold")
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([LABELS[k] for k in modes])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Exact match (%)")
    n = m[modes[0]]["n"]
    ax.set_title(f"Receipt QA exact match ({results['split']}, n={n} questions)", loc="left", fontweight="bold", pad=24)
    sub = headline(results)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9.5, color="#555")
    save(fig, out, "overall_em.png")


def fig_by_type(results: dict[str, Any], out: Path) -> None:
    modes = modes_present(results)
    ref = results["modes"].get("adapter_plain") or results["modes"][modes[0]]
    types = sorted(ref["by_type"], key=lambda t: ref["by_type"][t]["em"])  # best ends up on top
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(types) + 1.8))
    h = 0.8 / len(modes)
    for j, mode in enumerate(modes):
        ys = [i + (j - (len(modes) - 1) / 2) * h for i in range(len(types))]
        vals = [results["modes"][mode]["by_type"].get(t, {"em": 0.0})["em"] * 100 for t in types]
        ax.barh(ys, vals, height=h * 0.92, color=COLORS[mode], label=LABELS[mode])
        if mode == "adapter_plain":
            for y, v in zip(ys, vals, strict=True):
                ax.text(v + 1, y, f"{v:.0f}", va="center", fontsize=9, fontweight="bold", color=COLORS[mode])
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([f"{t} (n={ref['by_type'][t]['n']})" for t in types])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Exact match (%)")
    ax.set_title(f"Exact match by question type ({results['split']})", loc="left", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    save(fig, out, "em_by_type.png")


def fig_failures(results: dict[str, Any], results_dir: Path, out: Path) -> None:
    base_mode = best_baseline(results)
    base = load_preds(results, results_dir, base_mode)
    tuned = load_preds(results, results_dir, "adapter_plain")
    if base is None or tuned is None:
        print("skip failure_types.png: predictions .jsonl files not found next to the results JSON")
        return

    def counts(rows: list[dict[str, Any]]) -> Counter:
        tags = (tag_failure(r) for r in rows)
        return Counter(t for t in tags if t != "correct")

    cb, ct = counts(base), counts(tuned)
    tags = sorted(set(cb) | set(ct), key=lambda t: cb[t] + ct[t])  # biggest on top
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(tags) + 1.8))
    h = 0.38
    for off, c, mode in ((-h / 2, cb, base_mode), (h / 2, ct, "adapter_plain")):
        ys = [i + off for i in range(len(tags))]
        vals = [c[t] for t in tags]
        ax.barh(ys, vals, height=h * 0.92, color=COLORS[mode], label=LABELS[mode])
        for y, v in zip(ys, vals, strict=True):
            ax.text(v + 0.8, y, str(v), va="center", fontsize=9, color=COLORS[mode] if mode != base_mode else "#555")
    ax.set_yticks(range(len(tags)))
    ax.set_yticklabels([t.replace("_", " ") for t in tags])
    ax.set_xlabel("Failed questions")
    ax.set_title(
        f"Failure types: {sum(cb.values())} (base) vs {sum(ct.values())} (fine-tuned) of {len(base)}",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    save(fig, out, "failure_types.png")


def wrap(s: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(s, width)) or "(empty)"


def pick_examples(base: list[dict], tuned: list[dict], k: int, seed: int) -> list[tuple[dict, dict]]:
    by_id = {r["id"]: r for r in base}
    cands = [
        (by_id[t["id"]], t)
        for t in tuned
        if t["id"] in by_id
        and t["em"] == 1.0
        and by_id[t["id"]]["em"] == 0.0
        and len(by_id[t["id"]]["prediction"]) <= 90
        and Path(t["image_path"]).exists()
    ]
    random.Random(seed).shuffle(cands)
    chosen, types, receipts = [], set(), set()
    for b, t in cands:
        if t["qtype"] in types or t["receipt_id"] in receipts:
            continue
        chosen.append((b, t))
        types.add(t["qtype"])
        receipts.add(t["receipt_id"])
        if len(chosen) == k:
            break
    return chosen


def fig_examples(results: dict[str, Any], results_dir: Path, out: Path, k: int, seed: int) -> None:
    base = load_preds(results, results_dir, best_baseline(results))
    tuned = load_preds(results, results_dir, "adapter_plain")
    if base is None or tuned is None:
        print("skip examples.png: predictions .jsonl files not found next to the results JSON")
        return
    ex = pick_examples(base, tuned, k, seed)
    if not ex:
        print("skip examples.png: no example where base failed and adapter succeeded (or images missing)")
        return
    fig = plt.figure(figsize=(3.3 * len(ex), 6.4))
    gs = fig.add_gridspec(2, len(ex), height_ratios=[4.2, 1.4], hspace=0.05, wspace=0.12)
    for c, (b, t) in enumerate(ex):
        axi = fig.add_subplot(gs[0, c])
        axi.imshow(Image.open(t["image_path"]).convert("RGB"))
        axi.set_xticks([])
        axi.set_yticks([])
        for spine in axi.spines.values():  # light frame so white receipts stay visible
            spine.set_visible(True)
            spine.set_color("#CCCCCC")
        axt = fig.add_subplot(gs[1, c])
        axt.axis("off")
        y = 1.0
        rows = [
            ("Q", t["question"], "#111111"),
            ("Base", b["prediction"], BAD),
            ("Fine-tuned", t["prediction"], GOOD),
            ("Gold", t["answer"], "#444444"),
        ]
        for label, text, color in rows:
            block = wrap(f"{label}: {text}")
            axt.text(
                0, y, block, va="top", fontsize=9.5, color=color, transform=axt.transAxes,
                fontweight="bold" if label == "Q" else "normal",
            )  # fmt: skip
            y -= 0.06 * (block.count("\n") + 1) + 0.04
    note = "Examples selected where the base model failed and the fine-tuned model succeeded (not a random sample)."
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#555555")
    save(fig, out, "examples.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="outputs/results/<run_id>_<split>.json from evaluate.py")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--examples", type=int, default=4, help="number of example receipts")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    results = read_json(args.results)
    results_dir, out = Path(args.results).parent, Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_overall(results, out)
    fig_by_type(results, out)
    fig_failures(results, results_dir, out)
    fig_examples(results, results_dir, out, args.examples, args.seed)


if __name__ == "__main__":
    main()