# Product Requirements Document: ReceiptQA

| Field | Value |
|-------|-------|
| Owner | <your name> |
| Status | Draft v0.1 |
| Target completion | 3 weeks from start |

## 1. Problem

General-purpose small VLMs are weak at reading dense, small-print documents like receipts. Bigger models fix this but cost more, need more VRAM, and are overkill for a narrow task. Question: **how far can a small VLM be pushed on one domain with cheap, local fine-tuning?**

## 2. Goals

1. Fine-tune a small VLM (SmolVLM-500M) with QLoRA on receipt QA using only a consumer 8GB GPU.
2. Show a measurable, statistically credible improvement over the zero-shot base model.
3. Ship a public demo (Gradio on HF Spaces) with base vs fine-tuned answers side by side.
4. Publish a reproducible repo with honest evaluation and limitations.

## 3. Non-goals

- Not a production receipt parser or accounting tool.
- Not training a VLM from scratch.
- Not multilingual or multi-region in v1.
- Not beating large proprietary models.

## 4. Audience

| Persona | Need |
|---------|------|
| Recruiter / hiring manager | Understand in 30 seconds what was built and see numbers |
| Technical reviewer | Check rigor: splits, baselines, CIs, ablations, reproducibility |
| Me (learner) | Hands-on experience with VLM fine-tuning, PEFT, evaluation |

## 5. User stories

- As a reviewer, I upload a receipt and ask "What is the total?" and see both models answer side by side.
- As a reviewer, I open the README and see the results table and a one-line summary of the improvement.
- As a developer, I reproduce the results by running 5 `make` commands.
- As a reader, I find the failure cases and limitations documented, not hidden.

## 6. Functional requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Download CORD and build QA pairs (JSONL) from structured annotations | Must |
| FR2 | Split by receipt into train/val/test with fixed seed | Must |
| FR3 | Zero-shot baseline eval of the base model | Must |
| FR4 | QLoRA training driven by a YAML config | Must |
| FR5 | Evaluation with EM, ANLS, numeric accuracy, per-question-type breakdown, bootstrap CI | Must |
| FR6 | Gradio demo: image + question -> base and fine-tuned answers | Must |
| FR7 | Error analysis script that dumps worst failures with images | Should |
| FR8 | Ablations: LoRA rank, image resolution, 256M vs 500M | Should |
| FR9 | Held-out paraphrased question set to test template overfitting | Should |
| FR10 | Second dataset (SROIE) as out-of-distribution test | Could |

## 7. Non-functional requirements

- **Hardware:** training and inference fit in 8GB VRAM.
- **Reproducibility:** fixed seeds, pinned dependencies, config saved with every run.
- **Runtime:** a full training run finishes in under ~6 hours on the target GPU (to be verified).
- **Demo latency:** a few seconds per question on GPU; documented for CPU.
- **Privacy:** no real personal receipts committed or uploaded to the public Space.

## 8. Success metrics

Targets are hypotheses; revise after the baseline run.

| Metric | Target |
|--------|--------|
| Exact Match gain over zero-shot | >= +15 points |
| ANLS on test | Report with 95% CI, improvement must exceed the CI |
| Numeric accuracy (totals, prices) | Report per type |
| Paraphrase generalization | Drop of < 5 points vs template questions |
| Reproducibility | Fresh clone -> results in 5 commands |
| Portfolio | README + demo GIF + LinkedIn post + HF Space live |

## 9. Scope

**MVP:** FR1-FR6.
**Stretch:** FR7-FR10.
**Cut first if time runs out:** FR10, then ablations beyond LoRA rank.

## 10. Risks and mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Receipt text unreadable at low image resolution | High | Treat resolution as a primary hyperparameter; test image splitting early |
| OOM on 8GB | Medium | Batch 1 + grad accumulation, gradient checkpointing, 8-bit paged optimizer, smaller resolution |
| Model memorizes question templates | Medium | Many paraphrases, held-out template test |
| Small test set gives noisy metrics | High | Bootstrap CIs, report per-type counts |
| Base model already saw similar data | Medium | Check SmolVLM's training data notes, report zero-shot honestly |
| Free HF Space too slow for 500M | Medium | Use 256M or ZeroGPU, or ship a recorded demo video |

## 11. Open questions

- Is 800 training receipts enough, or should SROIE be added to training?
- Does training the connector layer help beyond LoRA on the text decoder?
- Which image resolution is the best VRAM/accuracy tradeoff?
