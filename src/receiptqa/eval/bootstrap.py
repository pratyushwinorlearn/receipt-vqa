"""Receipt-level bootstrap. Questions from the same receipt are correlated, so we resample receipts."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _per_receipt(receipt_ids: list[str], scores: list[float]) -> tuple[np.ndarray, np.ndarray]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for rid, s in zip(receipt_ids, scores, strict=True):
        sums[rid] += s
        counts[rid] += 1
    keys = sorted(sums)
    return np.array([sums[k] for k in keys]), np.array([counts[k] for k in keys])


def bootstrap_ci(
    receipt_ids: list[str], scores: list[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (mean, low, high) for the mean score, resampling receipts with replacement."""
    sums, counts = _per_receipt(receipt_ids, scores)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(sums.sum() / counts.sum()), float(low), float(high)


def paired_bootstrap_diff(
    receipt_ids: list[str],
    scores_a: list[float],
    scores_b: list[float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (mean_diff, low, high) for mean(a) - mean(b) on the same questions."""
    sums_a, counts = _per_receipt(receipt_ids, scores_a)
    sums_b, _ = _per_receipt(receipt_ids, scores_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    diffs = (sums_a[idx].sum(axis=1) - sums_b[idx].sum(axis=1)) / counts[idx].sum(axis=1)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float((sums_a.sum() - sums_b.sum()) / counts.sum()), float(low), float(high)
