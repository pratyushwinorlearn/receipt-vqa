# Data

## 1. Source

| Dataset | Use | Status / notes |
|---------|-----|----------------|
| CORD v2 (`naver-clova-ix/cord-v2` on HF) | Train / val / test | Main dataset. Approximately 800 train receipts, 100 validation receipts, and 100 test receipts with structured annotations. |
| SROIE | Out-of-distribution test | Optional future extension; not part of the reported final results. |

The domain can be swapped by rewriting `build_qa.py` and `templates.yaml`.

## 2. Raw annotation

Each CORD sample contains an image and a `ground_truth` JSON string whose parsed content (`gt_parse`) includes fields such as `menu`, `sub_total`, and `total`. Some fields are missing on some receipts, so every QA template checks that its source field exists before creating a question.

## 3. QA generation

Questions are generated programmatically from structured annotations, so answers are exact by construction.

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

- Each type has multiple paraphrased phrasings in `configs/templates.yaml`.
- Held-out phrasings are reserved and never used in training.
- A question is skipped when its answer field is missing, empty, or ambiguous.
- Answers are short and normalized before evaluation.
- Questions are capped per receipt so one receipt does not dominate training.

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

`build_qa.py` writes:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `val_heldout.jsonl`
- `test_heldout.jsonl`
- `stats.json`

Held-out records use the same receipts as the corresponding validation/test split but use unseen question phrasings.

## 5. Splits

- **Split by `receipt_id`, never by question.**
- All questions from a receipt stay in the same split.
- CORD's official train/validation/test organization is retained.
- `tests/test_splits.py` checks that receipt-id sets are disjoint.

## 6. Dataset statistics

The final reported evaluation sizes are known exactly:

| Split | Receipts | QA pairs | Types covered |
|-------|---------:|---------:|---------------|
| train | ≈800 | ≈6.6k | 9 |
| val | 100 | 822 | 9 |
| test | 100 | 809 | 9 |

The successful main training run used 411 optimizer steps per epoch with per-device batch size 2 and gradient accumulation 8, corresponding to approximately 6.6k training QA examples.

Validation and test QA counts come directly from the recorded evaluation runs.

## 7. Quality checks

Automated project checks include:

- unique QA ids
- non-empty answers
- split disjointness by receipt
- held-out phrasing separation
- question-type generation tests
- metric tests
- collator label-masking tests

The repository test suite currently passes **31/31 tests**.

For the benchmark, qualitative error analysis was also run on the final test predictions.

## 8. Contamination check

Potential overlap between CORD and the SmolVLM training mixture has **not been independently verified**. The final write-up should state this as an evaluation caveat rather than assuming either overlap or no overlap.

## 9. Privacy and licensing

- Check and cite the CORD license in the repository and model card.
- Do not add real personal receipts to the repository.
- A public demo should include a visible warning not to upload sensitive personal receipts.
