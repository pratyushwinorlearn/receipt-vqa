"""Download CORD v2 from the Hugging Face Hub and store images + parsed annotations locally.

Output (per split): data/raw/cord/<split>/<receipt_id>.png and data/raw/cord/<split>.jsonl with
{"receipt_id", "image_path", "gt_parse"}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from receiptqa.utils.io import load_yaml, write_jsonl
from receiptqa.utils.logging import get_logger

log = get_logger("download")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    from datasets import load_dataset
    from tqdm import tqdm

    raw_dir = Path(cfg["raw_dir"])
    ds = load_dataset(cfg["dataset_id"])
    for hf_split, split in cfg["split_map"].items():
        img_dir = raw_dir / split
        img_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for i, ex in enumerate(tqdm(ds[hf_split], desc=split)):
            receipt_id = f"cord_{split}_{i:04d}"
            path = img_dir / f"{receipt_id}.png"
            if not path.exists():
                ex["image"].convert("RGB").save(path)
            gt = json.loads(ex["ground_truth"])
            rows.append(
                {
                    "receipt_id": receipt_id,
                    "image_path": path.as_posix(),
                    "gt_parse": gt.get("gt_parse", {}),
                }
            )
        n = write_jsonl(raw_dir / f"{split}.jsonl", rows)
        log.info("split=%s receipts=%d", split, n)


if __name__ == "__main__":
    main()
