"""Batch collation for SmolVLM/Idefics3-style processors with answer-only loss."""

from __future__ import annotations

from typing import Any

from receiptqa.data.dataset import load_image

IGNORE_INDEX = -100


def build_labels(
    input_ids: list[int],
    attention_mask: list[int],
    prompt_len: int,
    image_token_id: int | None,
) -> list[int]:
    """Labels are the input ids, except everything before the answer, padding, and image tokens
    is set to -100 so it never contributes to the loss. Assumes right padding."""
    labels = list(input_ids)
    for i in range(len(labels)):
        is_prompt = i < prompt_len
        is_pad = attention_mask[i] == 0
        is_image = image_token_id is not None and labels[i] == image_token_id
        if is_prompt or is_pad or is_image:
            labels[i] = IGNORE_INDEX
    return labels


def _user_messages(question: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]


def _full_messages(question: str, answer: str) -> list[dict[str, Any]]:
    return _user_messages(question) + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]


class VQACollator:
    def __init__(self, processor: Any, verify_prefix: bool = True):
        self.processor = processor
        self.processor.tokenizer.padding_side = "right"
        image_token_id = getattr(processor, "image_token_id", None)
        if image_token_id is None:
            image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
        self.image_token_id = image_token_id
        self._verify_pending = verify_prefix

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        images, full_texts, prompt_lens = [], [], []
        first_prompt_ids = None
        for ex in batch:
            image = load_image(ex["image_path"])
            prompt = self.processor.apply_chat_template(_user_messages(ex["question"]), add_generation_prompt=True)
            full = self.processor.apply_chat_template(
                _full_messages(ex["question"], ex["answer"]), add_generation_prompt=False
            )
            if not full.startswith(prompt):
                raise ValueError(
                    "Chat template: full conversation does not start with the generation prompt. "
                    "Answer-only masking would be wrong.\n"
                    f"prompt={prompt!r}\nfull={full!r}"
                )
            # tokenize the prompt with the same image so the (expanded) image tokens are counted
            p_ids = self.processor(text=[prompt], images=[[image]], return_tensors="pt")["input_ids"][0]
            prompt_lens.append(int(p_ids.shape[0]))
            if first_prompt_ids is None:
                first_prompt_ids = p_ids
            images.append([image])
            full_texts.append(full)

        enc = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)

        if self._verify_pending and first_prompt_ids is not None:
            n = prompt_lens[0]
            if not torch.equal(enc["input_ids"][0][:n], first_prompt_ids):
                raise ValueError("Prompt tokens are not a prefix of the full sequence; masking would be wrong.")
            self._verify_pending = False

        labels = [
            build_labels(ids.tolist(), mask.tolist(), plen, self.image_token_id)
            for ids, mask, plen in zip(enc["input_ids"], enc["attention_mask"], prompt_lens, strict=True)
        ]
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return dict(enc)
