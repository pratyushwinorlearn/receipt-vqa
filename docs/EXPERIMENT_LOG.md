# Experiment Log

One row per run. Never delete failed runs, they are data.

| Run ID | Date | Model | Change vs previous | Res. | LoRA r | LR | Epochs | Peak VRAM | Val EM | Val ANLS | Notes / verdict |
|--------|------|-------|--------------------|------|--------|----|--------|-----------|--------|----------|-----------------|
| B0-zeroshot | | 500M | baseline, plain prompt | | - | - | - | | | | |
| B1-prompted | | 500M | baseline, "answer briefly" prompt | | - | - | - | | | | |
| | | | | | | | | | | | |

## Notes per run (optional)

### <run_id>
- **Hypothesis:**
- **Config file:**
- **Result:**
- **What I'd try next:**

## First-run checklist
- [ ] `make test` passes
- [ ] `make data` finishes, `data/processed/stats.json` looks sane
- [ ] `python scripts/token_stats.py` run, resolution chosen (ADR-008)
- [ ] `make overfit` reaches near-zero loss and reproduces the gold answers
- [ ] `make baseline` recorded as B0 / B1 above
