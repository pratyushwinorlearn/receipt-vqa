"""Load the processor and the (optionally 4-bit) SmolVLM base model."""

from __future__ import annotations

from typing import Any

import torch

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def get_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def load_processor(model_id: str, longest_edge: int | None = None, do_image_splitting: bool = True) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, do_image_splitting=do_image_splitting)
    if longest_edge:
        # resolution knob: N * 512. Verify the effect with scripts/token_stats.py
        processor.image_processor.size = {"longest_edge": int(longest_edge)}
    return processor


def load_base_model(model_id: str, quantize_4bit: bool = True, dtype: str = "bfloat16") -> Any:
    """Load the base VLM. With quantize_4bit=True this is the QLoRA setup (NF4 + double quant)."""
    from transformers import BitsAndBytesConfig

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # older transformers
        from transformers import AutoModelForVision2Seq as AutoVLM

    torch_dtype = get_dtype(dtype)
    bnb_config = None
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    return AutoVLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        quantization_config=bnb_config,
        device_map="auto",
    )
