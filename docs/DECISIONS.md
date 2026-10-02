# Decision Log (lightweight ADRs)

Format: Context -> Decision -> Alternatives -> Consequences. Add a new entry whenever you make a choice that would be annoying to reverse or explain later.

---

## ADR-001: Domain = receipts (CORD)

- **Status:** Accepted
- **Context:** Need a niche where the base model is weak, data is easy, and answers are objectively checkable.
- **Decision:** Receipts, using CORD's structured annotations to generate QA programmatically.
- **Alternatives:** Charts (base models already trained heavily on them), plant diseases (labels are classes, questions become templated), road signs (little QA data).
- **Consequences:** Exact answers by construction. Small dataset (~800 receipts). Single-region layouts limit generalization.

## ADR-002: SmolVLM-500M primary, 256M for debugging

- **Status:** Accepted
- **Context:** 8GB VRAM.
- **Decision:** Iterate fast on 256M, report the main result on 500M with QLoRA.
- **Consequences:** 256M can be used for fast debugging and sanity tests; the main reported model is the 500M QLoRA configuration.

## ADR-003: QLoRA on the language model only; vision encoder frozen

- **Status:** Accepted
- **Decision:** LoRA on LM attention + MLP projections; vision encoder and connector frozen.
- **Alternatives:** Also adapt the connector or the vision encoder.
- **Consequences:** Low memory use and a clear, reproducible main experiment. Vision-side adaptation remains a possible future ablation.

## ADR-004: Split by receipt, not by question

- **Status:** Accepted, non-negotiable
- **Decision:** All questions from a receipt stay in the same split.
- **Consequences:** Prevents receipt-level leakage that could inflate metrics.

## ADR-005: Programmatic QA with held-out phrasings

- **Status:** Accepted
- **Decision:** Generate template-based questions with multiple paraphrases; reserve unseen phrasings for held-out evaluation.
- **Consequences:** Answers remain exact while the held-out set tests whether the model learned the task rather than a single question wording.

## ADR-006: Training stack = transformers.Trainer + custom collator

- **Status:** Accepted
- **Context:** TRL's VLM support changes often and can hide batching/masking details that matter for this project.
- **Decision:** Plain `transformers.Trainer` with `VQACollator` using chat-template batching and answer-only loss.
- **Alternatives:** TRL SFTTrainer.
- **Consequences:** More explicit code, fewer moving parts, and direct control over answer-token masking.

## ADR-007: Demo = Gradio with one model load and adapter toggle

- **Status:** Accepted
- **Decision:** Base answer with the adapter disabled; fine-tuned answer with the adapter enabled.
- **Consequences:** One base model load keeps the comparison fair and avoids loading two separate copies of the model. A hosted demo will be attempted only after local verification.

## ADR-008: Resolution = 1536 with image splitting

- **Status:** Accepted
- **Context:** Receipt text is small and dense, so resolution affects readability, visual-token count, and memory. Token analysis measured approximately 484 tokens/receipt at 1536 and 881 at 2048.
- **Decision:** Use longest-edge 1536 with image splitting enabled for the main 500M QLoRA run.
- **Alternatives:** Lower resolution, 2048 resolution, image splitting disabled.
- **Consequences:** The chosen setting achieved 84.4% test EM with 3.22 GB peak training VRAM. The final evaluation does not establish whether 2048 would improve the remaining item-list/digit errors enough to justify its additional cost.

## ADR-009: DataLoader workers = 0 on Windows

- **Status:** Accepted for the final reproducibility configuration
- **Context:** The first main training attempt failed with a host-RAM NumPy/DataLoader `MemoryError` while preprocessing an image. Peak GPU VRAM was only about 3.1 GB. The Windows environment used multiple DataLoader worker processes.
- **Decision:** Set `num_workers: 0` for the successful run and resume from `checkpoint-400`.
- **Alternatives:** Keep multiple workers, lower image resolution, or disable image splitting.
- **Consequences:** Training completed successfully with 3.22 GB peak VRAM. This fixes the observed host-memory failure without changing the model architecture or image-resolution setting. The resumed run was not verified to be bit-identical to an uninterrupted run.

---
