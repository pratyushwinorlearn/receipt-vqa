"""LoRA / QLoRA helpers."""

from __future__ import annotations

from typing import Any

from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

GC_KWARGS = {"use_reentrant": False}


def prepare_model(model: Any, quantized: bool, gradient_checkpointing: bool = True) -> Any:
    """Make the base model ready for adapter training."""
    if quantized:
        return prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing, gradient_checkpointing_kwargs=GC_KWARGS
        )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=GC_KWARGS)
        model.enable_input_require_grads()
    return model


def attach_lora(model: Any, r: int, alpha: int, dropout: float, target_regex: str) -> Any:
    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_regex, bias="none")
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_adapter(model: Any, adapter_path: str) -> Any:
    return PeftModel.from_pretrained(model, adapter_path)
