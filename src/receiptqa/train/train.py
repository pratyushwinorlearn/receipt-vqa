"""QLoRA fine-tuning entrypoint.

python -m receiptqa.train.train --config configs/train_500m_qlora.yaml
python -m receiptqa.train.train --config configs/train_256m_debug.yaml --overfit 16
"""

from __future__ import annotations

import argparse
import math
import shutil
from datetime import datetime
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from receiptqa.data.collator import VQACollator
from receiptqa.data.dataset import QADataset
from receiptqa.models.load import load_base_model, load_processor
from receiptqa.models.lora import attach_lora, prepare_model
from receiptqa.train.callbacks import SampleGenerationCallback, VRAMCallback
from receiptqa.utils.io import load_yaml, write_json
from receiptqa.utils.logging import get_logger
from receiptqa.utils.seed import set_seed

log = get_logger("train")


def make_run_id(cfg: dict, overfit: int | None) -> str:
    short = cfg["model"]["model_id"].split("/")[-1].lower().replace("smolvlm-", "").replace("-instruct", "")
    run_id = f"{datetime.now():%Y%m%d}-{short}-{cfg['run_name']}"
    return run_id + "-overfit" if overfit else run_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="path to a checkpoint dir to resume from")
    ap.add_argument("--overfit", type=int, default=None, help="sanity test: overfit on N training examples")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    run_id = make_run_id(cfg, args.overfit)
    run_dir = Path(cfg["output"]["runs_dir"]) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    log.info("run_id=%s", run_id)

    mcfg, icfg, lcfg, dcfg, tcfg = cfg["model"], cfg["image"], cfg["lora"], cfg["data"], cfg["train"]

    processor = load_processor(mcfg["model_id"], icfg["longest_edge"], icfg["do_image_splitting"])
    model = load_base_model(mcfg["model_id"], mcfg["quantize_4bit"], mcfg["dtype"])
    model = prepare_model(model, mcfg["quantize_4bit"], tcfg["gradient_checkpointing"])
    model = attach_lora(model, lcfg["r"], lcfg["alpha"], lcfg["dropout"], lcfg["target_regex"])

    train_ds = QADataset(dcfg["train"], dcfg.get("max_train_samples"))
    val_ds = QADataset(dcfg["val"], dcfg.get("max_val_samples"))

    epochs, accum, eval_strategy, save_strategy = tcfg["epochs"], tcfg["grad_accum"], "epoch", "epoch"
    if args.overfit:
        train_ds.rows = train_ds.rows[: args.overfit]
        val_ds = train_ds
        epochs, accum, eval_strategy, save_strategy = max(epochs, 100), 1, "no", "no"
        log.info("OVERFIT MODE on %d examples: loss should go to ~0 and outputs should match gold", len(train_ds))

    # version-proof warmup: newer transformers removed `warmup_ratio`, `warmup_steps` works everywhere
    steps_per_epoch = math.ceil(len(train_ds) / (tcfg["per_device_batch_size"] * accum))
    warmup_steps = max(1, int(tcfg["warmup_ratio"] * steps_per_epoch * epochs))
    log.info("steps/epoch=%d total~%d warmup_steps=%d", steps_per_epoch, steps_per_epoch * epochs, warmup_steps)

    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and mcfg["dtype"] == "bfloat16" and torch.cuda.is_bf16_supported()
    use_fp16 = use_cuda and mcfg["dtype"] == "float16"

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        run_name=run_id,
        seed=cfg["seed"],
        num_train_epochs=epochs,
        learning_rate=float(tcfg["lr"]),
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg["per_device_batch_size"],
        gradient_accumulation_steps=accum,
        warmup_steps=warmup_steps,
        weight_decay=tcfg["weight_decay"],
        lr_scheduler_type=tcfg["lr_scheduler"],
        optim=tcfg["optim"] if use_cuda else "adamw_torch",
        gradient_checkpointing=tcfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=tcfg["logging_steps"],
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_total_limit=tcfg["save_total_limit"],
        report_to=tcfg["report_to"],
        remove_unused_columns=False,
        dataloader_num_workers=tcfg["num_workers"],
        label_names=["labels"],
    )

    scfg = cfg["sample_generation"]
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=VQACollator(processor),
        callbacks=[
            VRAMCallback(),
            SampleGenerationCallback(processor, val_ds.rows, scfg["n"], scfg["max_new_tokens"]),
        ],
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=args.resume)

    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    summary = {
        "run_id": run_id,
        "train_metrics": result.metrics,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if use_cuda else None,
        "adapter_dir": adapter_dir.as_posix(),
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    write_json(run_dir / "summary.json", summary)
    log.info("done. adapter saved to %s (peak VRAM %s GB)", adapter_dir, summary["peak_vram_gb"])
    log.info("next: make eval ADAPTER=%s", adapter_dir.as_posix())


if __name__ == "__main__":
    main()