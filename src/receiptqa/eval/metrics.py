"""Metrics: Exact Match, ANLS, numeric accuracy (see docs/EVALUATION.md).

Both prediction and gold are first canonicalized per question type:
- amount / count types: keep digits only ("Rp 45.000" == "45,000" == "45000"). Separators are dropped
  because CORD mixes "." and "," as thousands separators.
- item_list: split on , ; newline, lowercase each item, sort, compare as a set.
- everything else: lowercase, collapse whitespace, strip surrounding punctuation.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

NUMERIC_QTYPES = {"total", "subtotal", "tax", "item_price", "payment", "change", "item_count", "num_items"}


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s.strip(" .;:")


def digits_only(s: str) -> str:
    d = "".join(ch for ch in str(s) if ch.isdigit())
    return (d.lstrip("0") or "0") if d else ""


def canonical_list(s: str) -> list[str]:
    parts = [normalize_text(p) for p in re.split(r"[,;\n]+", str(s))]
    return sorted(p for p in parts if p)


def canonicalize(text: str, qtype: str) -> str:
    if qtype in NUMERIC_QTYPES:
        return digits_only(text)
    if qtype == "item_list":
        return " | ".join(canonical_list(text))
    return normalize_text(text)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def anls_score(pred: str, gold: str, tau: float = 0.5) -> float:
    """Average Normalized Levenshtein Similarity for one pair (DocVQA convention, tau=0.5)."""
    if not pred and not gold:
        return 1.0
    nl = levenshtein(pred, gold) / max(len(pred), len(gold))
    return 1.0 - nl if nl < tau else 0.0


def score_one(pred: str, gold: str, qtype: str) -> dict[str, float]:
    p, g = canonicalize(pred, qtype), canonicalize(gold, qtype)
    return {"em": float(bool(g) and p == g), "anls": anls_score(p, g)}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """records: dicts with at least qtype, em, anls."""
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_type[r["qtype"]].append(r)
    numeric = [r["em"] for r in records if r["qtype"] in NUMERIC_QTYPES]
    return {
        "n": len(records),
        "em": _mean([r["em"] for r in records]),
        "anls": _mean([r["anls"] for r in records]),
        "numeric_acc": _mean(numeric),
        "by_type": {
            qt: {"n": len(rs), "em": _mean([r["em"] for r in rs]), "anls": _mean([r["anls"] for r in rs])}
            for qt, rs in sorted(by_type.items())
        },
    }
