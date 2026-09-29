from __future__ import annotations

from typing import Any

import torch
from transformers import TrainerCallback

from receiptqa.data.dataset import load_image
from receiptqa.inference.predict import generate_answers
from receiptqa.utils.logging import get_logger

log = get_logger("callbacks")


class VRAMCallback(TrainerCallback):
    """Log peak GPU memory with every trainer log line."""

    def on_log(self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any) -> None:
        if torch.cuda.is_available() and logs and "loss" in logs:
            peak = torch.cuda.max_memory_allocated() / 1024**3
            log.info("step=%d loss=%.4f peak_vram=%.2fGB", state.global_step, logs["loss"], peak)


class SampleGenerationCallback(TrainerCallback):
    """At the end of each epoch, answer a few validation questions so you can eyeball progress."""

    def __init__(self, processor: Any, val_rows: list[dict[str, Any]], n: int = 5, max_new_tokens: int = 48):
        self.processor = processor
        self.rows = val_rows[:n]
        self.max_new_tokens = max_new_tokens

    def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
        if model is None or not self.rows:
            return
        was_training = model.training
        model.eval()
        try:
            images = [load_image(r["image_path"]) for r in self.rows]
            preds = generate_answers(
                model, self.processor, images, [r["question"] for r in self.rows], max_new_tokens=self.max_new_tokens
            )
            log.info("--- sample generations (epoch %.1f) ---", state.epoch or 0)
            for r, p in zip(self.rows, preds, strict=True):
                log.info("Q: %s | gold: %s | pred: %s", r["question"], r["answer"], p)
        finally:
            if was_training:
                model.train()
