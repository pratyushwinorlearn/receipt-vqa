"""Dump the worst predictions with rough failure tags to a markdown file.

python -m receiptqa.eval.error_analysis outputs/results/<run>_adapter_plain_test.jsonl --n 40

Tags are heuristics for triage. Confirm by looking at the examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from receiptqa.eval.metrics import NUMERIC_QTYPES, canonical_list, canonicalize, levenshtein
from receiptqa.utils.io import read_jsonl

COUNT_QTYPES = {"item_count", "num_items"}


def tag_failure(rec: dict[str, Any]) -> str:
    pred, gold, qtype = rec["prediction"], rec["answer"], rec["qtype"]
    if not pred.strip():
        return "empty"
    p, g = canonicalize(pred, qtype), canonicalize(gold, qtype)
    if p == g:
        return "correct"

    if qtype in COUNT_QTYPES:
        # counts are tiny numbers, so an off-by-one is a counting/field error, not a misread digit
        return "wrong_count" if p else "no_number"

    if qtype in NUMERIC_QTYPES:  # amounts
        if not p:
            return "no_number"
        if len(g) >= 3 and len(p) >= 3 and levenshtein(p, g) == 1:
            return "misread_digit"  # e.g. 45,000 vs 45,500: likely a reading/resolution problem
        return "wrong_value"  # wrong field, hallucination, or a rate instead of an amount

    if qtype == "item_list":
        gold_has_digits = any(ch.isdigit() for ch in gold)
        if any(ch.isdigit() for ch in pred) and not gold_has_digits:
            return "copied_line_with_numbers"  # dumped prices/totals instead of item names
        ps, gs = set(canonical_list(pred)), set(canonical_list(gold))
        if ps < gs:
            return "missing_items"
        if ps > gs:
            return "extra_items"
        return "wrong_items"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = read_jsonl(args.predictions)
    for r in recs:
        r["tag"] = tag_failure(r)
    fails = sorted((r for r in recs if r["tag"] != "correct"), key=lambda r: r["anls"])

    out = Path(args.out or Path(args.predictions).with_suffix(".errors.md"))
    lines = [f"# Error analysis: {Path(args.predictions).name}", ""]
    lines += [f"{len(fails)} failures out of {len(recs)}.", "", "| Tag | Count |", "|-----|-------|"]
    lines += [f"| {t} | {c} |" for t, c in Counter(r["tag"] for r in fails).most_common()]

    lines += ["", "## Failures by question type and tag", "", "| Type | Tag | Count |", "|------|-----|-------|"]
    by_type_tag = Counter((r["qtype"], r["tag"]) for r in fails)
    lines += [f"| {qt} | {tag} | {c} |" for (qt, tag), c in sorted(by_type_tag.items(), key=lambda kv: -kv[1])]

    lines += ["", "## Worst cases (lowest similarity first, NOT a representative sample)", ""]
    for r in fails[: args.n]:
        lines += [
            f"### {r['id']} ({r['qtype']}, {r['tag']})",
            f"![]({Path(r['image_path']).resolve().as_posix()})",
            f"- **Q:** {r['question']}",
            f"- **Gold:** {r['answer']}",
            f"- **Pred:** {r['prediction']}",
            "",
        ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out}")
    print("\n".join(lines[: 6 + len(Counter(r["tag"] for r in fails))]))


if __name__ == "__main__":
    main()