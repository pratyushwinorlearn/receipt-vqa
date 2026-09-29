# Roadmap

Target: ~3 weeks part-time. Adjust dates to your calendar.

## Phase 0: Setup (day 1)
- [ ] Create repo, `.gitignore`, `.env.example`, `pyproject.toml`
- [ ] Verify GPU, CUDA, `bitsandbytes` import and a 4-bit model load
- [ ] Load SmolVLM-256M and answer one image question (smoke test)
- [ ] Set up W&B / TensorBoard

## Phase 1: Data (days 2-4)
- [ ] Explore CORD, print annotation fields (`01_explore_cord.ipynb`)
- [ ] Write `build_qa.py`, `templates.yaml`, `splits.py`
- [ ] Unit tests for QA builder and split disjointness
- [ ] Manually verify 50 QA pairs
- [ ] Fill dataset stats in `DATA.md`

## Phase 2: Baseline (days 4-6)
- [ ] Write `metrics.py` + tests
- [ ] Token count vs resolution analysis (`02_token_counts_vs_resolution.ipynb`)
- [ ] Run B0 and B1 on val and record in `EXPERIMENT_LOG.md`
- [ ] Choose image resolution setting, write `DECISIONS.md` entry

## Phase 3: Training (days 7-12)
- [ ] Write collator + label-masking test
- [ ] Overfit-on-16 sanity test passes
- [ ] Debug run on 256M
- [ ] Main run on 500M QLoRA
- [ ] Ablations: rank, resolution, connector, data size
- [ ] Pick the best config by **val** metrics

## Phase 4: Evaluation (days 13-15)
- [ ] Evaluate final model once on test
- [ ] Bootstrap CIs, per-type breakdown, held-out phrasing test
- [ ] Error analysis + tagged failure table
- [ ] Latency benchmark

## Phase 5: Demo (days 16-18)
- [ ] Gradio app with side-by-side base vs fine-tuned
- [ ] Sample receipts in `app/examples/`
- [ ] Push adapter to HF Hub, deploy Space
- [ ] Record demo GIF

## Phase 6: Publish (days 19-21)
- [ ] Fill README results and limitations
- [ ] Finalize `MODEL_CARD.md`
- [ ] Tag release `v0.1-results`
- [ ] LinkedIn post + resume bullet
- [ ] Optional: short blog write-up

## Definition of done

1. Fresh clone reproduces results with the documented commands.
2. README has results table, demo GIF, and live link.
3. Test numbers have CIs and come from a single final evaluation.
4. Limitations are written down.

## LinkedIn post checklist

- [ ] Hook line with one number (e.g. "EM 31% -> 68% on a laptop GPU")
- [ ] Demo GIF or 30-second video as the first media
- [ ] 3 bullets: what, how (QLoRA, 8GB), what I learned
- [ ] One honest limitation
- [ ] Links: GitHub, HF Space
- [ ] Tags: #VLM #LoRA #QLoRA #HuggingFace #ComputerVision

## Stretch ideas

- Add SROIE to training and test OOD gains
- Distill answers from a larger VLM for extra training data
- Structured JSON extraction mode (all fields in one answer)
- Quantize the merged model to GGUF/ONNX and benchmark CPU latency
