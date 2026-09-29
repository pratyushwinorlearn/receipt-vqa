from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from receiptqa.utils.io import read_jsonl


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class QADataset:
    """Plain list-backed dataset over a QA jsonl. Images are loaded lazily in the collator."""

    def __init__(self, path: str | Path, max_samples: int | None = None):
        rows = read_jsonl(path)
        self.rows = rows[:max_samples] if max_samples else rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]
