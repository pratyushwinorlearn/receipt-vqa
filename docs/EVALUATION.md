# Evaluation

## 1. Principles

- Test set is touched **once per finished model**. Use val for tuning.
- Every number is tied to a run id and a config.
- Report confidence intervals. With ~100 test receipts, differences of a few points can be noise.

## 2. Metrics

| Metric | Definition | Use |
|--------|-----------|-----|
| Exact Match (EM) | Prediction equals gold after normalization | Headline metric |
| ANLS | Average Normalized Levenshtein Similarity with threshold 0.5 (standard in DocVQA) | Tolerates small OCR-style typos |
| Numeric accuracy | For amount answers: parsed numbers equal | Money must be exactly right |
| Per-type EM | EM broken down by `qtype` | Shows where the model is weak |
| Latency | Seconds per question (p50/p95), hardware stated | Practical cost |

### Normalization (apply to both prediction and gold)

- Lowercase, strip whitespace
- Numeric types (amounts, counts): keep digits only, so `Rp 45.000`, `45,000` and `45000` all match (CORD mixes `.` and `,` as thousands separators). Implemented in `eval/metrics.py`
- Collapse repeated whitespace
- For lists: split on commas/newlines, sort, compare as sets

Document any normalization change in `DECISIONS.md`, because it changes every number.

## 3. Baselines

| ID | Setup |
|----|-------|
| B0 | Base model, zero-shot, plain question |
| B1 | Base model, zero-shot, prompt asking for a short exact answer |
| B2 | (Optional) Larger open VLM zero-shot as a reference ceiling |
| M1 | Base + QLoRA adapter (main result) |

B1 exists so the improvement can't be dismissed as "just a better prompt".

## 4. Statistical rigor

- **Bootstrap CI:** resample test receipts (not questions) 1000 times, report 95% CI for EM and ANLS.
- **Paired comparison:** compare M1 vs B0/B1 on the same questions, report the paired difference with CI.
- Report the count of questions per type next to per-type scores.

## 5. Generalization checks

| Test | What it shows |
|------|---------------|
| Held-out phrasings | Learned the task, not the template |
| Unseen question style (hand-written 30 questions) | Robustness to real user phrasing |
| SROIE receipts (optional) | Out-of-distribution layouts |
| Image perturbations (blur, rotation, JPEG) | Robustness |

## 6. Error analysis

`error_analysis.py` dumps the worst N predictions with the image, question, gold, and prediction. Tag each failure:

- Misread digits
- Wrong field (answered subtotal instead of total)
- Hallucinated item
- Format error (right value, wrong format)
- Missing or truncated list
- Refusal / empty output

Include the tag counts and 5-8 examples in the write-up. Showing failures makes the results more credible.

## 7. Reporting template

| Model | EM | 95% CI | ANLS | Num. acc | Held-out phrasing EM | Latency p50 |
|-------|----|--------|------|----------|----------------------|-------------|
| B0 zero-shot | | | | | | |
| B1 prompted | | | | | | |
| M1 QLoRA | | | | | | |

Per-type table and 10-15 side-by-side qualitative examples go in the README or a `results/` page.

## 8. Checklist before publishing numbers

- [ ] Split verified disjoint by receipt
- [ ] Test set not used for tuning
- [ ] Same generation params for all models
- [ ] CI computed
- [ ] Hardware and versions stated
- [ ] Failure cases included
