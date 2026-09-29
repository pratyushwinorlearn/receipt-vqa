"""Dump the worst predictions with rough failure tags to a markdown file.

python -m receiptqa.eval.error_analysis outputs/results/<run>_adapter_plain_test.jsonl --n 40
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from receiptqa.eval.metrics import NUMERIC_QTYPES, canonical_list, canonicalize, levenshtein
from receiptqa.utils.io import read_jsonl


def tag_failure(rec: dict[str, Any]) -> str:
    """Heuristic tags. Confirm by eye, these are for triage."""
    pred, gold, qtype = rec["prediction"], rec["answer"], rec["qtype"]
    if not pred.strip():
        return "empty"
    p, g = canonicalize(pred, qtype), canonicalize(gold, qtype)
    if p == g:
        return "correct"
    if qtype in NUMERIC_QTYPES:
        if not p:
            return "no_number"
        if levenshtein(p, g) == 1:
            return "misread_digit"
        return "wrong_value"
    if qtype == "item_list":
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
    lines += ["", "## Worst cases", ""]
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


if __name__ == "__main__":
    main()
