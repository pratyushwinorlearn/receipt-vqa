# Experiment Log

One row per run. Never delete failed runs, they are data.

Setup for all runs: RTX 4060 Laptop GPU (8GB), Windows, seed 42.

| Run ID | Date | Model | Change vs previous | Res. | LoRA r | LR | Epochs | Peak VRAM | Val EM | Val ANLS | Notes / verdict |
|--------|------|-------|--------------------|------|--------|----|--------|-----------|--------|----------|-----------------|
| 20260930-256m-debug-overfit | 2026-09-30 | 256M (bf16, no 4-bit) | Pipeline sanity test, 16 questions | 512, no split | 16 | 2e-4 | 100 | 0.73 GB | n/a | n/a | PASS. Loss reached near zero; sample predictions matched gold. Collator masking and training loop verified on the real SmolVLM processor. |
| B0-zeroshot | 2026-09-30 | 500M (4-bit) | Baseline, plain prompt | 1536, split | - | - | - | - | 0.535 (CI 0.496-0.574) | 0.665 | Numeric acc 0.596. |
| B1-prompted | 2026-09-30 | 500M (4-bit) | Baseline, "answer briefly" prompt | 1536, split | - | - | - | - | 0.557 (CI 0.519-0.597) | 0.696 | Numeric acc 0.614. Validation result only; prompt effect was small relative to the CIs. |
| 20261001-500m-r16-e2 | 2026-10-01 | 500M QLoRA | Main run attempt | 1536, split | 16 | 2e-4 | 2 | 3.13 GB | n/a | n/a | INTERRUPTED at step 486/822 by host-RAM NumPy/DataLoader MemoryError while preprocessing an image. No final evaluation from this attempt. |
| 20261002-500m-r16-e2 | 2026-10-02 | 500M QLoRA | Resumed from checkpoint-400 with `num_workers: 0` | 1536, split | 16 | 2e-4 | 2 | 3.22 GB | 0.858 | 0.908 | PASS. Training completed; final validation, held-out phrasing, and final test evaluation completed. |

## Baseline detail (validation, 822 questions)

EM by question type:

| Type | B0 plain | B1 brief |
|------|----------|----------|
| subtotal (67) | 0.75 | 0.75 |
| item_count (115) | 0.73 | 0.73 |
| item_price (167) | 0.74 | 0.71 |
| total (98) | 0.67 | 0.69 |
| change (64) | 0.69 | 0.59 |
| payment (64) | 0.48 | 0.45 |
| num_items (100) | 0.18 | 0.39 |
| tax (47) | 0.30 | 0.34 |
| item_list (100) | 0.10 | 0.15 |

Failure tags for B1 (364 failures out of 822, heuristic tags):

| Tag | Count | Meaning |
|-----|------:|---------|
| misread_digit | 99 | Amount 3+ digits, one edit off |
| wrong_count | 91 | Wrong number for a count question |
| wrong_value | 88 | Wrong field, rate instead of amount, or hallucination |
| copied_line_with_numbers | 50 | Item list answered with whole receipt line including numbers |
| wrong_items | 31 | Item list wrong |
| extra_items | 4 | Item list has extra items |
| no_number | 1 | No number found |

## Main run

### 20261002-500m-r16-e2

- **Hypothesis:** Fine-tuning should substantially improve field selection, counting, and item extraction over the zero-shot baseline.
- **Config:** `configs/train_500m_qlora.yaml`
- **Resolution:** 1536, image splitting enabled
- **LoRA:** r=16, alpha=32, dropout=0.05
- **Learning rate:** 2e-4
- **Epochs:** 2
- **Per-device batch:** 2
- **Gradient accumulation:** 8
- **Peak VRAM:** 3.22 GB
- **Training result:** train loss 0.03131, final validation loss 0.1029
- **Validation:** EM 0.858, ANLS 0.908, numeric accuracy 0.909
- **Held-out phrasing:** EM 0.813, ANLS 0.879, numeric accuracy 0.855
- **Final test:** EM 0.844, ANLS 0.911, numeric accuracy 0.891
- **Verdict:** PASS. QLoRA produced a large improvement over both zero-shot baselines while remaining comfortably within the 8GB VRAM target.

### Training interruption note

The first main-run attempt (`20261001-500m-r16-e2`) was interrupted at step 486/822 by a host-RAM `MemoryError` in a DataLoader worker. The underlying NumPy allocation failed during image preprocessing for a receipt split into seven 512x512 regions. Peak GPU VRAM was about 3.1 GB, so this was not a CUDA OOM.

The successful run resumed from `checkpoint-400` with `num_workers: 0`. The resumed run completed successfully. The result was not verified to be bit-identical to a single uninterrupted run.

## Final test result

Final test split: 100 held-out receipts, 809 questions.

| Model | EM | 95% CI | ANLS | Numeric accuracy | Latency p50 |
|-------|----|--------|------|------------------|-------------|
| Base plain | 0.538 | [0.501, 0.575] | 0.678 | 0.597 | 0.43 s/q |
| Base brief | 0.520 | [0.480, 0.561] | 0.668 | 0.578 | 0.35 s/q |
| QLoRA adapter | 0.844 | [0.813, 0.875] | 0.911 | 0.891 | 0.41 s/q |

Paired EM differences:

- Adapter vs base plain: **+0.307**, 95% CI **[+0.265, +0.350]**
- Adapter vs base brief: **+0.324**, 95% CI **[+0.284, +0.369]**

For the headline comparison, use the best zero-shot baseline on the same test split: base plain, giving a **+30.7 point** improvement.

## Final test EM by question type

| Type | N | EM |
|------|---:|---:|
| item_count | 110 | 0.973 |
| change | 56 | 0.946 |
| payment | 65 | 0.938 |
| subtotal | 64 | 0.922 |
| item_price | 177 | 0.910 |
| total | 95 | 0.874 |
| tax | 42 | 0.786 |
| num_items | 100 | 0.750 |
| item_list | 100 | 0.510 |

## Final error analysis

125 failures out of 809 test questions.

| Tag | Count |
|-----|------:|
| wrong_items | 37 |
| wrong_value | 33 |
| wrong_count | 28 |
| misread_digit | 15 |
| copied_line_with_numbers | 7 |
| missing_items | 3 |
| extra_items | 2 |

The dominant remaining weakness is item-list extraction, followed by number-of-items questions.

## Generalization check

Held-out validation phrasings:

- EM: 0.813
- ANLS: 0.879
- Numeric accuracy: 0.855

Ordinary validation EM: 0.858.

Held-out phrasing EM drop: 0.0447, or **4.47 percentage points**.

## Conclusions

- Fine-tuning substantially improved receipt VQA over both zero-shot baselines.
- Scalar extraction tasks improved most strongly.
- `item_list` remains the main failure mode.
- `num_items` is another remaining weakness.
- Held-out question phrasing performance remained close to ordinary validation performance.
- The model stayed well within the 8GB GPU constraint.
- The test set was evaluated once after selecting the final configuration.
