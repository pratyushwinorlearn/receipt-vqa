"""Gradio demo: base vs fine-tuned answers side by side.

    ADAPTER_PATH=outputs/runs/<run_id>/adapter python app/app.py

Env vars: MODEL_ID, ADAPTER_PATH (local path or HF Hub id), LONGEST_EDGE, SPLITTING, QUANTIZE_4BIT.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from receiptqa.inference.predict import Predictor

MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolVLM-500M-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH") or None
LONGEST_EDGE = int(os.getenv("LONGEST_EDGE", "1536"))
SPLITTING = os.getenv("SPLITTING", "1") == "1"
QUANTIZE_4BIT = os.getenv("QUANTIZE_4BIT", "1") == "1"

EXAMPLE_QUESTIONS = [
    "What is the total amount?",
    "What is the subtotal?",
    "How much tax was charged?",
    "How many different items are on this receipt?",
    "List all items on the receipt.",
]

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(
            MODEL_ID,
            adapter_path=ADAPTER_PATH,
            quantize_4bit=QUANTIZE_4BIT,
            longest_edge=LONGEST_EDGE,
            do_image_splitting=SPLITTING,
        )
    return _predictor


def run(image, question: str):
    if image is None or not question.strip():
        return "Upload a receipt and ask a question.", "Upload a receipt and ask a question."
    p = get_predictor()
    # base gets the "brief" prompt, so the comparison is against a fairly prompted baseline
    base = p.answer(image, question, use_adapter=False, style="brief")
    tuned = p.answer(image, question, use_adapter=True) if p.has_adapter else "(no adapter loaded, set ADAPTER_PATH)"
    return base, tuned


with gr.Blocks(title="ReceiptQA") as demo:
    gr.Markdown(
        "# ReceiptQA\nSmolVLM fine-tuned with QLoRA to answer questions about receipts. "
        "Research demo only, not for accounting or financial decisions. "
        "Do not upload receipts with personal information."
    )
    with gr.Row():
        with gr.Column():
            image = gr.Image(type="pil", label="Receipt")
            question = gr.Textbox(label="Question", value=EXAMPLE_QUESTIONS[0])
            gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[question], label="Example questions")
            btn = gr.Button("Ask", variant="primary")
        with gr.Column():
            base_out = gr.Textbox(label="Base model (zero-shot, brief prompt)", lines=3)
            tuned_out = gr.Textbox(label="Fine-tuned (QLoRA adapter)", lines=3)
    btn.click(run, inputs=[image, question], outputs=[base_out, tuned_out])
    question.submit(run, inputs=[image, question], outputs=[base_out, tuned_out])

    sample_imgs = sorted(str(p) for p in Path(__file__).parent.joinpath("examples").glob("*.[pj][np]g"))
    if sample_imgs:
        gr.Examples(examples=[[p] for p in sample_imgs], inputs=[image], label="Sample receipts")

if __name__ == "__main__":
    demo.queue().launch()
