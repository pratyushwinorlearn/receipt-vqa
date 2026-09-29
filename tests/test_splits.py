import pytest

from receiptqa.data.splits import assert_disjoint, receipt_ids


def test_disjoint_ok():
    assert_disjoint({"train": {"a", "b"}, "val": {"c"}, "test": {"d"}})


def test_leakage_raises():
    with pytest.raises(ValueError, match="Leakage"):
        assert_disjoint({"train": {"a", "b"}, "test": {"b"}})


def test_receipt_ids():
    assert receipt_ids([{"receipt_id": "x"}, {"receipt_id": "x"}, {"receipt_id": "y"}]) == {"x", "y"}
