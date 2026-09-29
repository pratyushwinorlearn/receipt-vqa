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

<!-- Verify the base model repo id and licenses of the base model and dataset before publishing. -->

# ReceiptQA-SmolVLM-500M (QLoRA adapter)

A LoRA adapter for SmolVLM-500M-Instruct fine-tuned with QLoRA to answer questions about receipt images.

## Intended use

- Research and demos of small-VLM domain adaptation.
- Answering simple questions about receipts (totals, item prices, counts, item lists).

## Out of scope

- Accounting, tax, auditing, or any financial decision.
- Receipts from regions/layouts/languages far from the training data.
- Processing personal receipts without consent.

## How to use

```python
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from peft import PeftModel
from PIL import Image

base_id = "HuggingFaceTB/SmolVLM-500M-Instruct"
adapter_id = "<your-username>/receiptqa-smolvlm-500m"

processor = AutoProcessor.from_pretrained(base_id)
model = AutoModelForVision2Seq.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, adapter_id)

image = Image.open("receipt.png")
messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What is the total amount?"}]}]
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(text=prompt, images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0])
```

## Training

| | |
|---|---|
| Method | QLoRA (4-bit NF4, double quant, bf16 compute) |
| Adapter | LoRA r=16, alpha=32, dropout 0.05 on LM attention + MLP |
| Data | CORD v2 receipts converted to QA pairs (see repo `docs/DATA.md`) |
| Hardware | Single 8GB laptop GPU |
| Epochs / LR / batch | TBD |
| Seed | 42 |

## Evaluation

Test split of 100 receipts, split by receipt. 95% bootstrap CIs.

| Model | EM | ANLS | Numeric acc |
|-------|----|------|-------------|
| Base zero-shot | TBD | TBD | TBD |
| + this adapter | TBD | TBD | TBD |

## Limitations and biases

- Small training set (about 800 receipts) from one source.
- Can misread digits, especially on blurry or small-print receipts.
- May confuse similar fields (subtotal vs total).
- Not evaluated on non-Latin or handwritten receipts.

## Privacy

Training data contains store names and item names from a public dataset. Do not upload personal receipts to shared demos.

## Citation

```
@misc{receiptqa2026,
  title  = {ReceiptQA: QLoRA fine-tuning of a small VLM for receipt VQA},
  author = {<your name>},
  year   = {2026}
}
```

Also cite SmolVLM and CORD (Park et al., 2019).
