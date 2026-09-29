import pytest

from receiptqa.eval.metrics import (
    anls_score,
    canonicalize,
    levenshtein,
    score_one,
    summarize,
)


def test_numeric_separators_and_currency_are_ignored():
    assert score_one("Rp 45.000", "45,000", "total")["em"] == 1.0
    assert score_one("45000", "45,000", "total")["em"] == 1.0
    assert score_one("The total is 45,000.", "45,000", "total")["em"] == 1.0


def test_numeric_mismatch():
    assert score_one("45,500", "45,000", "total")["em"] == 0.0


def test_count_with_x_suffix():
    assert score_one("2x", "2", "item_count")["em"] == 1.0


def test_item_list_is_order_and_case_insensitive():
    gold = "Nasi Goreng, Es Teh, Ayam Bakar"
    assert score_one("ayam bakar; es teh; nasi goreng", gold, "item_list")["em"] == 1.0
    assert score_one("nasi goreng, es teh", gold, "item_list")["em"] == 0.0


def test_levenshtein_and_anls():
    assert levenshtein("kitten", "sitting") == 3
    assert anls_score("kitten", "sitting") == pytest.approx(1 - 3 / 7)
    assert anls_score("abc", "xyz") == 0.0  # nl = 1.0 >= tau
    assert anls_score("", "") == 1.0


def test_empty_prediction_scores_zero():
    s = score_one("", "45,000", "total")
    assert s["em"] == 0.0 and s["anls"] == 0.0


def test_canonicalize_leading_zeros():
    assert canonicalize("007", "total") == "7"
    assert canonicalize("0", "total") == "0"


def test_summarize_by_type_and_numeric_acc():
    recs = [
        {"qtype": "total", "em": 1.0, "anls": 1.0},
        {"qtype": "total", "em": 0.0, "anls": 0.2},
        {"qtype": "item_list", "em": 1.0, "anls": 1.0},
    ]
    s = summarize(recs)
    assert s["n"] == 3
    assert s["em"] == pytest.approx(2 / 3)
    assert s["numeric_acc"] == pytest.approx(0.5)
    assert s["by_type"]["total"]["n"] == 2
    assert s["by_type"]["item_list"]["em"] == 1.0
