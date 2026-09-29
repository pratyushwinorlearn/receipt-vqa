# CLAUDE.md (context for AI coding assistants)

## Project
ReceiptQA: QLoRA fine-tune of SmolVLM-500M for receipt VQA. Single 8GB GPU. See `docs/PRD.md` and `docs/ARCHITECTURE.md` before making structural changes.

## Commands
- `make data` / `make baseline` / `make train` / `make eval` / `make app`
- `pytest -q` runs unit tests (metrics, QA builder, collator masking)
- `ruff check . && ruff format .` before committing

## Conventions
- Python 3.10+, type hints on public functions, no wildcard imports.
- All hyperparameters live in `configs/*.yaml`. Never hardcode them in scripts.
- Seeds are set via `receiptqa.utils.seed.set_seed`. Every run must be reproducible from its config.
- Splits are by **receipt id**, never by question. Do not change this.
- Loss is computed on **answer tokens only**. If you touch the collator, run `tests/test_collator.py`.
- `data/` and `outputs/` are git-ignored. Never commit checkpoints or raw data.

## Do not
- Do not add dependencies without noting why in `docs/DECISIONS.md`.
- Do not report a metric without the run id from `docs/EXPERIMENT_LOG.md`.
- Do not evaluate on the validation or train split and call it a test result.

## Gotchas
- Receipts are text-heavy: image resolution is the main quality/VRAM tradeoff.
- Use `padding_side="left"` for generation, and mask pad tokens in labels for training.
- Base vs fine-tuned comparison uses the same loaded model with `model.disable_adapter()`.
