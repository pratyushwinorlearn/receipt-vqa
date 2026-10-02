from receiptqa.eval.error_analysis import tag_failure


def rec(qtype, gold, pred):
    return {"qtype": qtype, "answer": gold, "prediction": pred}


def test_count_off_by_one_is_wrong_count_not_misread():
    assert tag_failure(rec("num_items", "1", "2")) == "wrong_count"
    assert tag_failure(rec("item_count", "1", "16,000.")) == "wrong_count"


def test_amount_one_digit_off_is_misread_digit():
    assert tag_failure(rec("total", "45,000", "45,500")) == "misread_digit"


def test_amount_wrong_field_or_rate_is_wrong_value():
    assert tag_failure(rec("tax", "1,818", "10%")) == "wrong_value"
    assert tag_failure(rec("payment", "50.000", "23.00")) == "wrong_value"


def test_item_list_copying_prices():
    gold = "REAL GANACHE, EGG TART, PIZZA TOAST"
    pred = "REAL GANACHE 16,500 EGG TART 13,000 PIZZA TOAST 16,000 TOTAL 45,500"
    assert tag_failure(rec("item_list", gold, pred)) == "copied_line_with_numbers"


def test_item_list_missing_and_extra():
    assert tag_failure(rec("item_list", "a, b, c", "a, b")) == "missing_items"
    assert tag_failure(rec("item_list", "a, b", "a, b, c")) == "extra_items"


def test_correct_and_empty():
    assert tag_failure(rec("total", "45,000", "45.000")) == "correct"
    assert tag_failure(rec("total", "45,000", "  ")) == "empty"