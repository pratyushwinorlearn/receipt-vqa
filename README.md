# ReceiptQA: Domain VQA with a Small VLM + QLoRA

Fine-tune **SmolVLM-500M** with **QLoRA** to answer questions about receipts using a single 8GB laptop GPU, and measure the improvement over the zero-shot base model.

> Status: ✅ completed. The final test evaluation was run once after model selection.

> Domain: receipts (CORD v2). See `docs/DATA.md` for the dataset and QA generation pipeline.

## Results

The final model uses SmolVLM-500M-Instruct with QLoRA (LoRA rank 16), 1536px longest-edge image processing, image splitting, and 2 training epochs.

| Model | Exact Match | ANLS | Numeric Accuracy | Latency p50 |
|-------|-------------|------|------------------|-------------|
| SmolVLM-500M zero-shot | 53.8% | 67.8% | 59.7% | 0.43 s/q |
| SmolVLM-500M prompted | 52.0% | 66.8% | 57.8% | 0.35 s/q |
| **SmolVLM-500M + QLoRA** | **84.4%** | **91.1%** | **89.1%** | **0.41 s/q** |

Final test set: **809 questions** from 100 held-out receipts, split by receipt rather than by question.

The QLoRA model improved Exact Match by **30.7 percentage points** over the plain zero-shot baseline. The paired 95% bootstrap CI for this improvement is **[+26.5, +35.0] points**.

95% bootstrap confidence intervals and paired comparisons are documented in `docs/EVALUATION.md` and `docs/EXPERIMENT_LOG.md`.

## Validation and generalization

| Evaluation | EM | ANLS | Numeric Accuracy |
|------------|---:|------:|-----------------:|
| Validation | 85.8% | 90.8% | 90.9% |
| Held-out phrasings | 81.3% | 87.9% | 85.5% |
| Final test | 84.4% | 91.1% | 89.1% |

The held-out phrasing EM was **4.47 percentage points** below ordinary validation EM.

## Final test performance by question type

| Type | N | Exact Match |
|------|---:|------------:|
| item_count | 110 | 97.3% |
| change | 56 | 94.6% |
| payment | 65 | 93.8% |
| subtotal | 64 | 92.2% |
| item_price | 177 | 91.0% |
| total | 95 | 87.4% |
| tax | 42 | 78.6% |
| num_items | 100 | 75.0% |
| item_list | 100 | 51.0% |

`item_list` remains the main weakness, while scalar extraction tasks are substantially stronger.

## Demo

Local Gradio demo:

```bash
python app/app.py
```

A hosted Hugging Face Space will be added only after the local demo is verified.

For privacy, do not upload personal receipts to a public demo. Receipts can contain names, addresses, payment details, and other sensitive information.

## Quickstart

Create and activate a virtual environment, then install the package:

```bash
git clone https://github.com/pratyushwinorlearn/receipt-vqa
cd receipt-vqa

python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
```

Project entry points are available as Python modules. The repository also contains a `Makefile` for environments where GNU Make is available.

Typical workflow:

```bash
python -m pytest -q

python scripts/token_stats.py

python -m receiptqa.train.train --config configs/train_500m_qlora.yaml

python -m receiptqa.eval.evaluate --config configs/eval.yaml --split test --adapter outputs/runs/<run_id>/adapter --modes base_plain base_brief adapter_plain

python scripts/make_figures.py --results outputs/results/<run_id>_test.json --out assets
```

## Code status

| Part | Status |
|------|--------|
| Data builder, splits, metrics, bootstrap, error tagging | Implemented and unit-tested |
| Collator label masking | Implemented and tested |
| Model loading, QLoRA training, evaluation | Implemented and run on the target GPU |
| Gradio app | Implemented; local verification is the next demo step |
| Result figures | Generated from the final test results |

All repository tests currently pass.

## Hardware

Developed and trained on:

- NVIDIA RTX 4060 Laptop GPU
- 8GB VRAM
- Windows
- bf16 compute
- 4-bit NF4 quantization
- gradient checkpointing
- paged 8-bit AdamW

Measured peak VRAM during the final training run: **3.22 GB**.

## Main findings

- QLoRA substantially improved receipt VQA over the zero-shot baselines.
- Final test EM increased from **53.8% to 84.4%**.
- Numeric accuracy reached **89.1%** on the final test set.
- Item counts and monetary fields became much more reliable.
- `item_list` remained the hardest task at **51.0% test EM**.
- The held-out phrasing result stayed within the project's planned <5-point EM drop.
- The final training run remained comfortably within the 8GB VRAM constraint.

## Limitations

- The model was trained on approximately 800 CORD receipts from one source.
- Performance on other receipt layouts, regions, languages, or real-world photos is not established by the benchmark.
- Complete item-list extraction is substantially weaker than scalar field extraction.
- The model can still misread digits or select the wrong numeric field.
- Exact Match for lists is strict; small formatting or naming differences can count as incorrect.
- Overlap between CORD and the SmolVLM pretraining/training mixture has not been independently verified.

## Training interruption

The main training attempt was interrupted once by a host-RAM `MemoryError` in a DataLoader worker during image preprocessing. This was not a GPU OOM; peak GPU VRAM was approximately 3.2 GB. The run was resumed from the step-400 checkpoint after setting `num_workers: 0`, and the run then completed successfully. The resumed result was not verified to be bit-identical to an uninterrupted run.

## Docs

| Doc | What's in it |
|-----|--------------|
| [PRD](docs/PRD.md) | What and why, scope, success metrics |
| [Architecture](docs/ARCHITECTURE.md) | System design, model, memory budget |
| [File structure](docs/FILE_STRUCTURE.md) | Repo layout and conventions |
| [Data](docs/DATA.md) | Dataset, QA generation, splits |
| [Training](docs/TRAINING.md) | Hyperparameters, commands, troubleshooting |
| [Evaluation](docs/EVALUATION.md) | Metrics, baselines, ablations |
| [Roadmap](docs/ROADMAP.md) | Milestones and checklist |
| [Decisions](docs/DECISIONS.md) | Why project design choices were made |
| [Learning](docs/LEARNING.md) | Concepts to learn and resources |
| [Experiment log](docs/EXPERIMENT_LOG.md) | Every run and final results |
| [Model card](docs/MODEL_CARD.md) | HF-style model card |

## License

Code: MIT. Base model and dataset licenses apply separately; see `docs/MODEL_CARD.md`.

## Acknowledgements

SmolVLM and Hugging Face, CORD (Naver Clova), PEFT, bitsandbytes, Transformers, PyTorch, and Gradio.
