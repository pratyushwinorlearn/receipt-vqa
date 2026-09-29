import pytest

np = pytest.importorskip("numpy")

from receiptqa.eval.bootstrap import bootstrap_ci, paired_bootstrap_diff  # noqa: E402


def test_ci_brackets_mean_and_is_deterministic():
    rids = [f"r{i // 4}" for i in range(400)]
    scores = [float(i % 3 == 0) for i in range(400)]
    m, lo, hi = bootstrap_ci(rids, scores, n_boot=500, seed=1)
    assert lo <= m <= hi
    assert (m, lo, hi) == bootstrap_ci(rids, scores, n_boot=500, seed=1)


def test_paired_diff_positive_when_a_better():
    rids = [f"r{i // 2}" for i in range(200)]
    a = [1.0] * 200
    b = [float(i % 2) for i in range(200)]
    d, lo, hi = paired_bootstrap_diff(rids, a, b, n_boot=500, seed=0)
    assert d == pytest.approx(0.5) and lo > 0
