# Data

## 1. Source

| Dataset | Use | Notes |
|---------|-----|-------|
| CORD v2 (`naver-clova-ix/cord-v2` on HF) | Train / val / test | ~800 train, 100 val, 100 test receipts with structured annotations. Verify counts and license on load. |
| SROIE (optional) | Out-of-distribution test | Different layouts and fields. Only if time allows. |

Swapping the domain (charts, plant disease, etc.) only requires rewriting `build_qa.py` and `templates.yaml`.

## 2. Raw annotation

Each CORD sample has an image and a `ground_truth` JSON string whose parsed content (`gt_parse`) contains fields like `menu` (list of items with name, count, price), `sub_total`, and `total`. **Print a few samples and confirm exact field names before writing the builder.** Some fields are missing on some receipts, so every template must check that its field exists.

## 3. QA generation

Questions are generated programmatically from the structured annotation, so answers are exact by construction.

| Type | Example question | Answer source |
|------|------------------|---------------|
| `total` | What is the total amount? | `total.total_price` |
| `subtotal` | What is the subtotal? | `sub_total.subtotal_price` |
| `tax` | How much tax was charged? | `sub_total.tax_price` |
| `item_price` | What is the price of <item>? | `menu[i].price` |
| `item_count` | How many <item> were ordered? | `menu[i].cnt` |
| `num_items` | How many different items are on this receipt? | `len(menu)` |
| `item_list` | List all items on the receipt. | joined `menu[*].nm` |
| `payment` | How much was paid in cash? | `total.cashprice` |
| `change` | How much change was given? | `total.changeprice` |

Rules:
- Each type has **5-10 paraphrased phrasings** in `configs/templates.yaml`, sampled randomly.
- **Reserve 2-3 phrasings per type as a held-out set**, never used in training. This tests whether the model learned the task or the template.
- Skip a question if its answer field is missing, empty, or ambiguous (e.g. duplicate item names for `item_price`).
- Answers are short and normalized (see `EVALUATION.md`).
- Target: ~5-10k training QA pairs total, capped per receipt so no receipt dominates.

## 4. JSONL format

```json
{
  "id": "cord_train_0012_q3",
  "receipt_id": "cord_train_0012",
  "image_path": "data/raw/cord/train/0012.png",
  "question": "How much was the total?",
  "answer": "45,000",
  "qtype": "total",
  "template_id": "total_04",
  "split": "train"
}
```

Files written by `build_qa.py`: `train.jsonl`, `val.jsonl`, `test.jsonl`, `val_heldout.jsonl`, `test_heldout.jsonl` (same questions as val/test, unseen phrasings) and `stats.json`. Ids look like `cord_train_0012_q3` (`_h3` for heldout).

## 5. Splits

- **Split by `receipt_id`**, never by question. Two questions about the same receipt must never land in different splits.
- Keep CORD's official train/val/test as-is. If re-splitting, use a fixed seed from `configs/data.yaml`.
- `tests/test_splits.py` asserts the receipt-id sets are disjoint.

## 6. Quality checks (run before training)

- [ ] Random-sample 50 QA pairs and verify by eye against the image
- [ ] Answer-length histogram (no giant outliers)
- [ ] Question-type distribution is reasonably balanced
- [ ] No duplicate `id`s
- [ ] No empty answers
- [ ] Held-out phrasings absent from train

## 7. Contamination check

Read the SmolVLM model card / training-data notes and check whether receipt or document QA data overlapped with its training mix. If unsure, say so in the write-up. The zero-shot baseline is still valid, it just needs an honest caveat.

## 8. Privacy and licensing

- Check the CORD license and cite it in the README and model card.
- Receipts may contain store names and addresses. Do not add real personal receipts to the repo or the public demo.

## 9. Dataset statistics (fill after build)

| Split | Receipts | QA pairs | Types covered |
|-------|----------|----------|---------------|
| train | TBD | TBD | TBD |
| val | TBD | TBD | TBD |
| test | TBD | TBD | TBD |
