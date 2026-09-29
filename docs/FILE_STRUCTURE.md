# File Structure

```
receipt-vqa/
├── README.md                  # Landing page: demo, results, quickstart
├── CLAUDE.md                  # Context for AI coding assistants
├── pyproject.toml             # Package + deps (or requirements.txt)
├── Makefile                   # data / baseline / train / eval / app shortcuts
├── .gitignore                 # data/, outputs/, .venv, wandb/
├── .env.example               # HF_TOKEN=, WANDB_API_KEY= (never commit .env)
│
├── docs/
│   ├── PRD.md
│   ├── ARCHITECTURE.md
│   ├── FILE_STRUCTURE.md
│   ├── DATA.md
│   ├── TRAINING.md
│   ├── EVALUATION.md
│   ├── ROADMAP.md
│   ├── DECISIONS.md
│   ├── LEARNING.md
│   ├── EXPERIMENT_LOG.md
│   └── MODEL_CARD.md
│
├── configs/
│   ├── data.yaml              # dataset ids, split seed, question templates path
│   ├── train_500m_qlora.yaml  # main run
│   ├── train_256m_debug.yaml  # fast iteration run
│   ├── eval.yaml              # eval settings, generation params
│   └── templates.yaml         # question paraphrase templates
│
├── src/receiptqa/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py        # fetch CORD (and SROIE)
│   │   ├── build_qa.py        # annotations -> QA pairs
│   │   ├── splits.py          # split by receipt id
│   │   ├── dataset.py         # torch Dataset over JSONL
│   │   └── collator.py        # chat template + label masking
│   ├── models/
│   │   ├── load.py            # 4-bit base loader, processor
│   │   └── lora.py            # LoRA config, adapter save/load
│   ├── train/
│   │   ├── train.py           # entrypoint: python -m receiptqa.train.train --config ...
│   │   └── callbacks.py       # VRAM logging, sample generations
│   ├── eval/
│   │   ├── metrics.py         # EM, ANLS, numeric acc, normalization
│   │   ├── evaluate.py        # entrypoint
│   │   ├── bootstrap.py       # confidence intervals
│   │   └── error_analysis.py  # dump worst cases with images
│   ├── inference/
│   │   └── predict.py         # answer(image, question, use_adapter=True)
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── io.py
│
├── app/
│   ├── app.py                 # Gradio demo (base vs fine-tuned)
│   ├── requirements.txt       # minimal deps for HF Space
│   └── examples/              # 3-5 sample receipts (no personal data)
│
├── scripts/
│   ├── token_stats.py         # tokens per receipt at each resolution setting
│   ├── push_to_hub.py         # upload adapter + model card
│   ├── make_demo_gif.sh
│   └── benchmark_latency.py
│
├── notebooks/
│   ├── 01_explore_cord.ipynb
│   ├── 02_token_counts_vs_resolution.ipynb
│   └── 03_error_analysis.ipynb
│
├── tests/
│   ├── conftest.py
│   ├── test_build_qa.py       # QA generation from a fixture annotation
│   ├── test_splits.py         # no receipt id appears in two splits
│   ├── test_collator.py       # only answer tokens have labels != -100
│   └── test_metrics.py        # EM / ANLS on known cases
│
├── data/                      # git-ignored
│   ├── raw/
│   └── processed/             # train.jsonl, val.jsonl, test.jsonl
├── outputs/                   # git-ignored
│   ├── runs/<run_id>/         # config copy, adapter, logs
│   └── results/<run_id>.json
└── assets/                    # committed: demo.gif, result charts, sample outputs
```

## Conventions

- **One responsibility per module.** Data code never imports training code.
- **Entry points** are runnable as `python -m receiptqa.<area>.<script> --config configs/<file>.yaml`.
- **Configs over constants.** No hyperparameters in Python files.
- **Run ids** follow `YYYYMMDD-<model>-<desc>` and appear in `outputs/`, `EXPERIMENT_LOG.md`, and any reported number.
- **Git:** small commits, conventional prefixes (`feat:`, `fix:`, `docs:`, `exp:`). Tag the commit behind the published results (`v0.1-results`).
- **Never commit:** checkpoints, raw datasets, `.env`, personal receipts.

## Build order

1. `data/` + tests -> 2. `models/` -> 3. baseline `eval/` -> 4. `train/` -> 5. full `eval/` -> 6. `app/` -> 7. docs polish
