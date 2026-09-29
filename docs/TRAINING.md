# Training Guide

## 1. Environment

- Python 3.10+, CUDA-capable GPU (>= 8GB VRAM), bf16 support recommended (RTX 30/40/50 series).
- Core libs: `torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`, `pillow`, `gradio`.
- Pin exact versions after the first successful run and record them in `requirements.txt`.
- Windows note: `bitsandbytes` can be fiddly. If it fails, use WSL2 or a recent wheel with Windows support.

## 2. Starting hyperparameters

Starting points, not truths. Tune with validation loss/metrics.

| Param | Value |
|-------|-------|
| Base model | SmolVLM-500M-Instruct |
| Quantization | 4-bit NF4, double quant, bf16 compute |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| Learning rate | 2e-4 (LoRA typical), cosine schedule |
| Warmup | ~3% of steps |
| Epochs | 2-3 |
| Per-device batch size | 1-2 |
| Gradient accumulation | 8-16 (effective batch ~16) |
| Optimizer | `paged_adamw_8bit` |
| Weight decay | 0.0 - 0.01 |
| Max sequence length | set from token-count analysis |
| Gradient checkpointing | on |
| Seed | 42 |

## 3. Commands

```bash
# quick sanity run on 256M, ~50 steps
python -m receiptqa.train.train --config configs/train_256m_debug.yaml

# main run
python -m receiptqa.train.train --config configs/train_500m_qlora.yaml

# resume
python -m receiptqa.train.train --config configs/train_500m_qlora.yaml --resume outputs/runs/<run_id>
```

## 4. Before the first real run

1. **Overfit test:** train on 16 QA pairs for ~100 steps. Loss should approach ~0 and the model should reproduce those answers. If not, the pipeline is broken.
2. **Print one collated batch:** decode `input_ids` and confirm the labels are `-100` everywhere except the answer.
3. **Check VRAM:** log peak memory after 20 steps (`torch.cuda.max_memory_allocated()`).
4. **Generate after N steps:** sample 5 val questions every epoch via a callback, eyeball the outputs.

## 5. Experiment tracking

- Weights & Biases or TensorBoard, one project `receiptqa`.
- Log: train loss, val loss, VRAM, tokens/sec, learning rate, sample generations.
- Every run gets a row in `docs/EXPERIMENT_LOG.md`.

## 6. Planned ablations

| Ablation | Values |
|----------|--------|
| LoRA rank | 8, 16, 32 |
| Image resolution / splitting | low, medium, high |
| Connector trainable | off vs on |
| Model size | 256M vs 500M |
| Train set size | 25%, 50%, 100% (learning curve) |
| Paraphrase diversity | 1 vs many phrasings per type |

One change per run. Keep everything else fixed, same seed.

## 7. Troubleshooting

| Symptom | Likely fix |
|---------|-----------|
| CUDA OOM | Lower image resolution, batch size 1, more accumulation, confirm gradient checkpointing is on, shorten max length |
| Loss is NaN | Use bf16 not fp16, lower LR, check for empty answers |
| Loss drops but outputs are junk | Chat-template mismatch between train and inference, wrong padding side |
| Model repeats the question | Labels not masked correctly or EOS missing from labels |
| No improvement over baseline | Resolution too low, LR too low, too few steps, or labels wrong (do the overfit test) |
| Very slow steps | Too many image tokens, dataloader bottleneck, try `num_workers` > 0 |
| Adapter loads but behaves like base | Adapter not enabled, or merged into wrong dtype |

## 8. Saving

- Save the LoRA adapter only (`model.save_pretrained`), plus the config and tokenizer/processor files.
- Optionally merge into a 16-bit base for easier serving (needs more VRAM/RAM than 4-bit, do on CPU if necessary).
