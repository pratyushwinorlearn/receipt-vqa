# Learning Notes

What to understand while building this. Tick a box only when you can explain it out loud without notes.

## 1. Concepts checklist

### Vision-language models
- [ ] How a VLM works: vision encoder -> connector/projector -> language model
- [ ] What "visual tokens" are and why their count drives memory and speed
- [ ] Why image resolution matters for text-heavy images (OCR-like tasks)
- [ ] Chat templates and how images are inserted into the prompt

### Fine-tuning
- [ ] Full fine-tuning vs parameter-efficient fine-tuning (PEFT)
- [ ] LoRA: low-rank update `W + (alpha/r) * B @ A`, what `r` and `alpha` do
- [ ] Which layers to adapt and why (attention vs MLP)
- [ ] QLoRA: NF4 quantization, double quantization, paged optimizers, bf16 compute
- [ ] Why the base is frozen and only adapters get gradients
- [ ] Supervised fine-tuning (SFT) with loss masking on answer tokens only

### Training mechanics
- [ ] Gradient accumulation and effective batch size
- [ ] Gradient checkpointing: memory vs compute tradeoff
- [ ] bf16 vs fp16, and why bf16 is safer
- [ ] Learning-rate schedules and warmup
- [ ] Overfitting, validation loss, early stopping
- [ ] Reading a VRAM breakdown: weights, activations, gradients, optimizer state

### Evaluation
- [ ] Exact Match vs ANLS, when each is appropriate
- [ ] Why you never tune on the test set
- [ ] Data leakage (splitting by receipt, not by question)
- [ ] Bootstrap confidence intervals
- [ ] Paired comparisons between models
- [ ] Template overfitting and how to test for it

### Engineering
- [ ] Config-driven experiments and reproducibility (seeds, pinned versions)
- [ ] Experiment tracking (W&B / TensorBoard)
- [ ] Packaging and pushing adapters to the HF Hub
- [ ] Building and deploying a Gradio app to HF Spaces

## 2. Reading list

Papers (search by title):
- **LoRA: Low-Rank Adaptation of Large Language Models** (Hu et al., 2021)
- **QLoRA: Efficient Finetuning of Quantized LLMs** (Dettmers et al., 2023)
- **SmolVLM** technical report / blog from Hugging Face
- **Idefics3** paper (architecture that SmolVLM builds on)
- **SigLIP** (vision encoder family used in many small VLMs)
- **Document Visual Question Answering** (Mathew et al., DocVQA) for the ANLS metric
- **CORD: A Consolidated Receipt Dataset for Post-OCR Parsing** (Park et al., 2019)

Docs to keep open:
- Hugging Face Transformers: model docs for SmolVLM/Idefics3, chat templating
- PEFT docs: LoRA config, `disable_adapter`, saving/loading adapters
- Transformers `Trainer` docs: custom data collators, TrainingArguments
- bitsandbytes docs: 4-bit quantization config
- Gradio docs: Blocks, image inputs, Spaces deployment

Also check the Hugging Face cookbook for VLM fine-tuning notebooks and read one end to end before writing your own.

## 3. Suggested learning order

| Week | Focus |
|------|-------|
| 1 | VLM basics, run inference, token counts, dataset exploration |
| 2 | LoRA/QLoRA theory + first training run, debugging the collator |
| 3 | Evaluation rigor, error analysis, demo and write-up |

## 4. Self-check questions

1. Why does receipt QA need higher image resolution than natural-image VQA?
2. Roughly how much smaller is a LoRA adapter than the full model, and why?
3. What would happen if the loss were computed on the prompt tokens too?
4. Why split by receipt id instead of by question?
5. What does `r=16` change vs `r=64`, in memory and in capacity?
6. Why can two models with 3 points EM difference still be statistically indistinguishable?
7. How does `disable_adapter()` let you compare base vs fine-tuned with one loaded model?
8. What are two ways the model could "cheat" and score well without reading the receipt?

## 5. Learning log (append as you go)

| Date | What I learned | What confused me | Follow-up |
|------|----------------|------------------|-----------|
|      |                |                  |           |
