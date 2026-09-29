"""Push the LoRA adapter + model card to the Hugging Face Hub.

    python scripts/push_to_hub.py --adapter-dir outputs/runs/<id>/adapter --repo-id <user>/receiptqa-smolvlm-500m
Requires `huggingface-cli login` or HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--model-card", default="docs/MODEL_CARD.md")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    create_repo(args.repo_id, exist_ok=True, private=args.private)
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(args.adapter_dir, tmp, dirs_exist_ok=True)
        shutil.copy(args.model_card, Path(tmp) / "README.md")
        HfApi().upload_folder(folder_path=tmp, repo_id=args.repo_id, commit_message="Upload adapter and model card")
    print(f"pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
