# Product Requirements Document: ReceiptQA

| Field | Value |
|-------|-------|
| Owner | Pratyush |
| Status | Core training and evaluation complete; demo/deployment pending |
| Target completion | 3 weeks from start |

## 1. Problem

General-purpose small VLMs can struggle with dense, small-print documents such as receipts. Larger models may improve document understanding but require more compute and memory. The project asks:

**How far can a small VLM be pushed on one domain with cheap, local fine-tuning?**

## 2. Goals

1. Fine-tune a small VLM (SmolVLM-500M) with QLoRA on receipt QA using a consumer 8GB GPU.
2. Show a measurable and statistically supported improvement over zero-shot baselines.
3. Provide a demo in which a receipt and question produce base and fine-tuned answers side by side.
4. Publish a reproducible repo with honest evaluation and limitations.

## 3. Non-goals

- Not a production receipt parser or accounting tool.
- Not training a VLM from scratch.
- Not multilingual or multi-region in v1.
- Not a comparison against large proprietary models.

## 4. Audience

| Persona | Need |
|---------|------|
| Recruiter / hiring manager | Understand in 30 seconds what was built and see the key result |
| Technical reviewer | Check splits, baselines, confidence intervals, reproducibility, and failure analysis |
| Me (learner) | Gain hands-on experience with VLM fine-tuning, PEFT, and evaluation |

## 5. User stories

- As a reviewer, I upload a receipt and ask "What is the total?" and see base and fine-tuned answers side by side.
- As a reviewer, I open the README and see the final results and a concise explanation.
- As a developer, I can reproduce the data, training, and evaluation pipeline from documented commands.
- As a reader, I can inspect failures and limitations rather than seeing only successful examples.

## 6. Functional requirements

| ID | Requirement | Priority | Current status |
|----|-------------|----------|---------------|
| FR1 | Download CORD and build QA pairs (JSONL) from structured annotations | Must | Complete |
| FR2 | Split by receipt into train/val/test with fixed seed | Must | Complete |
| FR3 | Zero-shot baseline evaluation of the base model | Must | Complete |
| FR4 | QLoRA training driven by a YAML config | Must | Complete |
| FR5 | Evaluation with EM, ANLS, numeric accuracy, per-question-type breakdown, bootstrap CI | Must | Complete |
| FR6 | Gradio demo: image + question -> base and fine-tuned answers | Must | Implemented; local/demo verification pending |
| FR7 | Error analysis script that dumps worst failures with images | Should | Complete |
| FR8 | Ablations: LoRA rank, image resolution, 256M vs 500M | Should | 256M debug and resolution analysis completed; full ablation matrix not required for final result |
| FR9 | Held-out paraphrased question set to test template overfitting | Should | Complete |
| FR10 | Second dataset (SROIE) as out-of-distribution test | Could | Not run; future work |

## 7. Non-functional requirements

- **Hardware:** training and inference should fit within an 8GB GPU target.
- **Reproducibility:** fixed seeds, pinned dependencies, and configuration saved with each run.
- **Runtime:** the successful main training run completed in about 2 hours 6 minutes after resuming from the step-400 checkpoint; total project training time also included the interrupted attempt.
- **Demo latency:** final adapter benchmark p50 was approximately 0.41 seconds per question on the evaluation setup.
- **Privacy:** no personal receipts are included in the repository or benchmark.

## 8. Final results against success metrics

| Metric | Project target | Observed result |
|--------|----------------|-----------------|
| Exact Match gain over zero-shot | >= +15 points | **+30.7 points** on final test vs base_plain |
| ANLS | Report with 95% CI | **91.1%** test ANLS; EM CI **[81.3%, 87.5%]** |
| Numeric accuracy | Report per type | **89.1%** test numeric accuracy |
| Paraphrase generalization | Drop < 5 points | **4.47-point EM drop** (85.8% val -> 81.3% held-out phrasing) |
| Reproducibility | Documented pipeline | 31/31 repository tests passing; final run recorded |
| Portfolio | README + visuals + demo + publishing | README/results visuals complete; demo/deployment and publishing remain |

Final test comparison:

- Base plain: **53.8% EM**
- Base prompted: **52.0% EM**
- QLoRA adapter: **84.4% EM**

The conservative headline improvement is **+30.7 percentage points** over the best zero-shot baseline on the final test split.

## 9. Scope status

**Core MVP:** FR1-FR6.  
**Current state:** FR1-FR5 complete; FR6 implemented and ready for local/demo verification.

**Stretch:** FR7-FR10.  
FR7 and FR9 are complete. FR10 remains future work. Full ablations beyond the main configuration are not required for the final result.

## 10. Risks and mitigations

| Risk | Observed / status | Mitigation |
|------|-------------------|------------|
| Receipt text unreadable at low resolution | Real risk for numeric/OCR-style errors | 1536 chosen for main run; 2048 remains a future ablation |
| OOM on 8GB | GPU stayed below 3.22 GB; one host-RAM DataLoader error occurred | `num_workers: 0`, batch 2, gradient accumulation, checkpointing |
| Model memorizes question templates | Held-out phrasing was evaluated | Multiple paraphrases + held-out phrasing set |
| Small test set gives noisy metrics | 809 questions from 100 receipts | Bootstrap CIs and per-type reporting |
| Base model may have seen related data | Not independently verified | State the contamination caveat explicitly |
| Free hosted demo may be slow or unavailable | Not yet resolved | Verify local Gradio first; use compatible hosted GPU tier or recorded demo |

## 11. Open / future questions

- Does 2048 resolution improve difficult digit or item-list cases enough to justify additional compute?
- Does training the connector improve performance beyond LM-only LoRA?
- Would SROIE or another receipt distribution improve out-of-domain generalization?
- Can item-list extraction be improved with better list-specific training examples or structured decoding?
