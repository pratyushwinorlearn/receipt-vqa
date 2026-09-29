"""Split helpers. Splits are by receipt id, never by question (see docs/DECISIONS.md ADR-004)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from receiptqa.utils.io import read_jsonl


def receipt_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    return {r["receipt_id"] for r in rows}


def assert_disjoint(named_sets: dict[str, set[str]]) -> None:
    """Raise if any receipt id appears in more than one split."""
    names = list(named_sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = named_sets[a] & named_sets[b]
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(f"Leakage: {len(overlap)} receipts in both '{a}' and '{b}', e.g. {sample}")


def check_files_disjoint(paths: dict[str, str | Path]) -> None:
    assert_disjoint({name: receipt_ids(read_jsonl(p)) for name, p in paths.items()})
