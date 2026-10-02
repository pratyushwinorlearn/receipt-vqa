---
base_model: HuggingFaceTB/SmolVLM-500M-Instruct
library_name: peft
license: apache-2.0
datasets:
  - naver-clova-ix/cord-v2
tags:
  - vision-language
  - vqa
  - receipts
  - qlora
  - peft
---

# ReceiptQA-SmolVLM-500M (QLoRA adapter)

A LoRA adapter for SmolVLM-500M-Instruct fine-tuned with QLoRA to answer questions about receipt images.

## Intended use

- Research and demos of small-VLM domain adaptation.
- Answering simple questions about receipts: totals, item prices, counts, item lists, payment, and change.

## Out of scope

- Accounting, tax, auditing, or any financial decision.
- Receipts from regions, layouts, or languages far from the training data.
- Processing personal receipts without consent.

## How to use

The adapter is loaded on top of the base SmolVLM-500M-Instruct model.

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

base_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
adapter_id = "PATH_OR_HF_REPO_TO_YOUR_ADAPTER"

processor = AutoProcessor.from_pretrained(base_id)
model = AutoModelForVision2Seq.from_pretrained(
    base_id,
    dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_id)

image = Image.open("receipt.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is the total amount?"},
        ],
    }
]

prompt = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

inputs = processor(
    text=prompt,
    images=[image],
    return_tensors="pt",
).to(model.device)

out = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,
)

answer = processor.batch_decode(
    out[:, inputs["input_ids"].shape[1]:],
    skip_special_tokens=True,
)[0]

print(answer)
```

## Training

| Setting | Value |
|---|---|
| Method | QLoRA (4-bit NF4, double quant, bf16 compute) |
| Base model | SmolVLM-500M-Instruct |
| Adapter | LoRA r=16, alpha=32, dropout 0.05 |
| Target modules | LM attention + MLP projections |
| Vision encoder | Frozen |
| Connector | Frozen |
| Data | CORD v2 receipts converted to QA pairs |
| Hardware | Single RTX 4060 Laptop GPU, 8GB VRAM |
| Image resolution | Longest edge 1536, image splitting enabled |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Per-device batch | 2 |
| Gradient accumulation | 8 |
| Optimizer | paged_adamw_8bit |
| Gradient checkpointing | Enabled |
| Seed | 42 |
| Peak training VRAM | 3.22 GB |

## Evaluation

Final test split: 100 held-out receipts, 809 questions. The test set was evaluated once after model selection. Confidence intervals use the project's bootstrap procedure.

| Model | EM | ANLS | Numeric acc |
|---|---:|---:|---:|
| Base zero-shot | 53.8% | 67.8% | 59.7% |
| Base prompted | 52.0% | 66.8% | 57.8% |
| **+ this adapter** | **84.4%** | **91.1%** | **89.1%** |

Adapter vs base zero-shot:

- EM improvement: **+30.7 points**
- 95% CI: **[+26.5, +35.0]**

Adapter vs prompted baseline:

- EM improvement: **+32.4 points**
- 95% CI: **[+28.4, +36.9]**

## Test performance by question type

| Type | N | EM |
|---|---:|---:|
| item_count | 110 | 97.3% |
| change | 56 | 94.6% |
| payment | 65 | 93.8% |
| subtotal | 64 | 92.2% |
| item_price | 177 | 91.0% |
| total | 95 | 87.4% |
| tax | 42 | 78.6% |
| num_items | 100 | 75.0% |
| item_list | 100 | 51.0% |

## Generalization

Held-out validation phrasings:

- EM: 81.3%
- ANLS: 87.9%
- Numeric accuracy: 85.5%

Ordinary validation EM: 85.8%.

The held-out phrasing EM drop was 4.47 percentage points.

## Known failure modes

The final test error analysis identified:

- `wrong_items`: 37
- `wrong_value`: 33
- `wrong_count`: 28
- `misread_digit`: 15
- `copied_line_with_numbers`: 7
- `missing_items`: 3
- `extra_items`: 2

The primary remaining weakness is complete item-list extraction.

## Limitations and biases

- Small training set of approximately 800 receipts from one source.
- CORD has region/layout characteristics that may not represent other receipt distributions.
- Item-list extraction is substantially weaker than scalar field extraction.
- The model can still misread digits or select the wrong numeric field.
- Exact Match is strict, especially for item lists.
- The evaluation does not establish performance on arbitrary real-world receipts.
- Performance on Indian GST bills, handwritten receipts, or other unseen receipt styles has not been measured in this benchmark.
- Overlap between CORD and the base model's training mixture has not been independently verified.

## Privacy

CORD is a public research dataset, but real receipts can contain sensitive information. Do not upload personal receipts to shared demos or public repositories.

## Training interruption

The main run was interrupted once by a host-RAM `MemoryError` in a DataLoader worker during image preprocessing. This was not a GPU OOM; peak GPU VRAM was approximately 3.2 GB. Training was resumed from checkpoint-400 after setting `num_workers: 0`, and the run then completed successfully. The resumed result was not verified to be bit-identical to an uninterrupted run.

## Citation

```bibtex
@misc{receiptqa2026,
  title  = {ReceiptQA: QLoRA fine-tuning of a small VLM for receipt VQA},
  author = {Pratyush},
  year   = {2026}
}
```

Also cite the SmolVLM and CORD publications/resources used by the project.
