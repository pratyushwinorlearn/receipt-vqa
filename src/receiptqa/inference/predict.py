"""Inference helpers shared by evaluation, the training sample callback, the app and benchmarks."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from PIL import Image

from receiptqa.models.load import load_base_model, load_processor
from receiptqa.models.lora import load_adapter

BRIEF_SUFFIX = "\nAnswer with only the exact value, as briefly as possible."


def build_messages(question: str, style: str = "plain") -> list[dict[str, Any]]:
    text = question + (BRIEF_SUFFIX if style == "brief" else "")
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


@torch.inference_mode()
def generate_answers(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    questions: list[str],
    style: str = "plain",
    max_new_tokens: int = 64,
) -> list[str]:
    """Greedy generation for a batch. Uses left padding, restores the tokenizer state afterwards."""
    tok = processor.tokenizer
    prev_side = tok.padding_side
    tok.padding_side = "left"
    try:
        prompts = [
            processor.apply_chat_template(build_messages(q, style), add_generation_prompt=True) for q in questions
        ]
        inputs = processor(
            text=prompts,
            images=[[im.convert("RGB")] for im in images],
            return_tensors="pt",
            padding=True,
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
        new_tokens = out[:, inputs["input_ids"].shape[1] :]
        return [t.strip() for t in processor.batch_decode(new_tokens, skip_special_tokens=True)]
    finally:
        tok.padding_side = prev_side


class Predictor:
    """One loaded model that can answer with the adapter on or off (fair base vs fine-tuned comparison)."""

    def __init__(
        self,
        model_id: str,
        adapter_path: str | None = None,
        quantize_4bit: bool = True,
        dtype: str = "bfloat16",
        longest_edge: int | None = None,
        do_image_splitting: bool = True,
    ):
        self.processor = load_processor(model_id, longest_edge, do_image_splitting)
        model = load_base_model(model_id, quantize_4bit=quantize_4bit, dtype=dtype)
        self.has_adapter = adapter_path is not None
        if self.has_adapter:
            model = load_adapter(model, adapter_path)  # type: ignore[arg-type]
        self.model = model.eval()

    def answer_batch(
        self,
        images: list[Image.Image],
        questions: list[str],
        use_adapter: bool = True,
        style: str = "plain",
        max_new_tokens: int = 64,
    ) -> list[str]:
        ctx = self.model.disable_adapter() if (self.has_adapter and not use_adapter) else nullcontext()
        with ctx:
            return generate_answers(self.model, self.processor, images, questions, style, max_new_tokens)

    def answer(self, image: Image.Image, question: str, use_adapter: bool = True, style: str = "plain") -> str:
        return self.answer_batch([image], [question], use_adapter, style)[0]
