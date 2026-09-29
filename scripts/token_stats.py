"""How many tokens does one receipt cost at each resolution setting? Run this before training.

python scripts/token_stats.py --model HuggingFaceTB/SmolVLM-500M-Instruct --n 20
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from receiptqa.data.dataset import load_image
from receiptqa.inference.predict import build_messages
from receiptqa.models.load import load_processor
from receiptqa.utils.io import read_jsonl

SETTINGS = [(512, False), (512, True), (1024, True), (1536, True), (2048, True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    ap.add_argument("--data", default="data/processed/train.jsonl")
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    rows = read_jsonl(args.data)[: args.n]
    print(f"{'longest_edge':>12} {'splitting':>9} {'tokens mean':>12} {'tokens max':>11}")
    for edge, split in SETTINGS:
        proc = load_processor(args.model, edge, split)
        lens = []
        for r in rows:
            prompt = proc.apply_chat_template(build_messages(r["question"]), add_generation_prompt=True)
            ids = proc(text=[prompt], images=[[load_image(Path(r["image_path"]))]], return_tensors="pt")["input_ids"]
            lens.append(ids.shape[1])
        print(f"{edge:>12} {str(split):>9} {statistics.mean(lens):>12.0f} {max(lens):>11}")


if __name__ == "__main__":
    main()
