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
- **Decision:** Iterate fast on 256M, report main results on 500M with QLoRA.
- **Consequences:** 256M could be fully fine-tuned, so it also enables a QLoRA-vs-full comparison as an ablation.

## ADR-003: QLoRA on the language model only; vision encoder frozen
- **Status:** Accepted (revisit after ablation)
- **Decision:** LoRA on LM attention + MLP projections, vision encoder and connector frozen.
- **Alternatives:** Also adapt the connector or the vision encoder.
- **Consequences:** Lowest VRAM. May cap gains if the vision side is the bottleneck. Connector ablation planned.

## ADR-004: Split by receipt, not by question
- **Status:** Accepted, non-negotiable
- **Decision:** All questions from a receipt stay in the same split. Enforced by test.
- **Consequences:** Prevents leakage that would inflate every metric.

## ADR-005: Programmatic QA with held-out phrasings
- **Status:** Accepted
- **Decision:** Template-generated questions with paraphrases, some phrasings never seen in training.
- **Consequences:** Answers are exact. Requires the held-out test to show it isn't just memorizing templates.

## ADR-006: Training stack = transformers.Trainer + custom collator
- **Status:** Accepted
- **Context:** TRL's SFTTrainer VLM support changes often and hides the batching that matters here.
- **Decision:** Plain `transformers.Trainer` with `VQACollator` (image + chat-template batching, answer-only loss).
- **Alternatives:** TRL SFTTrainer.
- **Consequences:** More explicit code, fewer moving parts, and the loss masking is unit-tested.

## ADR-007: Demo = Gradio on HF Spaces, one model load with adapter toggle
- **Status:** Accepted
- **Decision:** Base answer via `disable_adapter()`, fine-tuned answer with adapter on.
- **Consequences:** Half the VRAM, fair comparison. If the free tier is too slow, ship 256M or a recorded demo.

## ADR-008: Resolution setting
- **Status:** Pending (decide after token-count analysis)
- **Options:** image splitting on/off, longest-edge sizes.
- **Decision:** TBD, record measured VRAM and val EM here.

---

## Template

## ADR-XXX: Title
- **Status:** Proposed / Accepted / Superseded
- **Context:**
- **Decision:**
- **Alternatives:**
- **Consequences:**
