import pytest

from receiptqa.data.build_qa import (
    build_candidates,
    generate_for_receipt,
    get_items,
    load_templates,
)

TEMPLATES = load_templates("configs/templates.yaml")

GT = {
    "menu": [
        {"nm": "Nasi Goreng", "cnt": "2", "price": "40,000"},
        {"nm": "Es Teh", "cnt": "1", "price": "5,000"},
        {"nm": "Es Teh", "cnt": "1", "price": "5,000"},  # duplicate name -> ambiguous, must be skipped
        {"nm": "Ayam Bakar", "cnt": "1x", "price": "25,000"},  # non-digit count -> no item_count Q
    ],
    "sub_total": {"subtotal_price": "75,000", "tax_price": "7,500"},
    "total": {"total_price": "82,500", "cashprice": "100,000", "changeprice": "17,500"},
}
RECEIPT = {"receipt_id": "cord_train_0001", "image_path": "data/raw/cord/train/cord_train_0001.png", "gt_parse": GT}


def test_single_dict_menu_is_handled():
    items = get_items({"menu": {"nm": "Coffee", "cnt": "1", "price": "3,000"}})
    assert [i["name"] for i in items] == ["Coffee"]


def test_candidates_skip_ambiguous_items():
    singles, items = build_candidates(GT)
    types = {c["qtype"] for c in singles}
    assert {"total", "subtotal", "tax", "payment", "change", "num_items", "item_list"} <= types
    priced = {c["item"] for c in items if c["qtype"] == "item_price"}
    assert "Es Teh" not in priced and {"Nasi Goreng", "Ayam Bakar"} <= priced
    counted = {c["item"] for c in items if c["qtype"] == "item_count"}
    assert counted == {"Nasi Goreng"}


def test_missing_fields_are_skipped():
    singles, items = build_candidates({"total": {"total_price": "10"}})
    assert [c["qtype"] for c in singles] == ["total"] and items == []


def test_generation_is_deterministic_and_capped():
    a = generate_for_receipt(RECEIPT, "train", TEMPLATES, "train", seed=42, max_qa=8)
    b = generate_for_receipt(RECEIPT, "train", TEMPLATES, "train", seed=42, max_qa=8)
    assert a == b and 0 < len(a) <= 8
    assert len({r["id"] for r in a}) == len(a)


def test_answers_match_annotation():
    rows = generate_for_receipt(RECEIPT, "train", TEMPLATES, "train", seed=1, max_qa=10)
    by_type = {r["qtype"]: r["answer"] for r in rows if r["qtype"] in ("total", "tax", "change", "num_items")}
    assert by_type["total"] == "82,500" and by_type["tax"] == "7,500"
    assert by_type["change"] == "17,500" and by_type["num_items"] == "4"


def test_heldout_asks_same_questions_with_unseen_phrasing():
    train = generate_for_receipt(RECEIPT, "test", TEMPLATES, "train", seed=42, max_qa=10)
    held = generate_for_receipt(RECEIPT, "test", TEMPLATES, "heldout", seed=42, max_qa=10)
    assert [(r["qtype"], r["answer"]) for r in train] == [(r["qtype"], r["answer"]) for r in held]
    train_phrasings = {p for pools in TEMPLATES.values() for p in pools["train"]}
    for r in held:
        assert r["question"] not in train_phrasings


def test_item_placeholder_is_filled():
    rows = generate_for_receipt(RECEIPT, "train", TEMPLATES, "train", seed=3, max_qa=10)
    for r in rows:
        assert "{item}" not in r["question"]


def test_template_pools_are_disjoint():
    for qtype, pools in TEMPLATES.items():
        assert not set(pools["train"]) & set(pools["heldout"]), qtype


def test_overlapping_template_pools_raise(tmp_path):
    p = tmp_path / "t.yaml"
    p.write_text("total:\n  train: ['a?']\n  heldout: ['a?']\n")
    with pytest.raises(ValueError):
        load_templates(p)
