import pytest

from receiptqa.data.collator import IGNORE_INDEX, build_labels

IMG = 99


def test_only_answer_tokens_keep_labels():
    #        prompt (5, incl. 2 image tokens)   answer  eos  pad
    ids = [1, IMG, IMG, 2, 3, 10, 11, 12, 0, 0]
    mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    labels = build_labels(ids, mask, prompt_len=5, image_token_id=IMG)
    assert labels[:5] == [IGNORE_INDEX] * 5
    assert labels[5:8] == [10, 11, 12]
    assert labels[8:] == [IGNORE_INDEX] * 2


def test_image_tokens_never_labeled_even_in_answer_region():
    labels = build_labels([1, 2, IMG, 3], [1, 1, 1, 1], prompt_len=1, image_token_id=IMG)
    assert labels == [IGNORE_INDEX, 2, IGNORE_INDEX, 3]


def test_full_pipeline_with_fake_processor(tmp_path):
    torch = pytest.importorskip("torch")
    from PIL import Image

    from receiptqa.data.collator import VQACollator

    class Tok:
        padding_side = "left"

        def convert_tokens_to_ids(self, t):
            return IMG

    class FakeProcessor:
        tokenizer = Tok()
        image_token_id = IMG

        def apply_chat_template(self, messages, add_generation_prompt=False):
            q = messages[0]["content"][1]["text"]
            s = f"USER:<image>{q}\nASSISTANT:"
            if not add_generation_prompt:
                s += " " + messages[1]["content"][0]["text"] + "<eos>"
            return s

        def __call__(self, text, images, return_tensors="pt", padding=False):
            seqs = []
            for t in text:
                ids = [IMG, IMG] + [ord(c) % 50 + 1 for c in t.replace("<image>", "")]
                seqs.append(ids)
            width = max(map(len, seqs))
            input_ids = torch.tensor([s + [0] * (width - len(s)) for s in seqs])
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs])
            return {"input_ids": input_ids, "attention_mask": mask}

    img_path = tmp_path / "r.png"
    Image.new("RGB", (8, 8)).save(img_path)
    batch = [
        {"image_path": str(img_path), "question": "total?", "answer": "45,000"},
        {"image_path": str(img_path), "question": "tax?", "answer": "5"},
    ]
    proc = FakeProcessor()
    out = VQACollator(proc)(batch)
    assert proc.tokenizer.padding_side == "right"
    for i, ex in enumerate(batch):
        labeled = out["labels"][i][out["labels"][i] != IGNORE_INDEX]
        # answer chars + "<eos>" chars are labeled, the prompt is not
        assert len(labeled) == len(" " + ex["answer"] + "<eos>")
