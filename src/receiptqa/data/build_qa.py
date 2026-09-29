"""Build QA pairs from CORD annotations.

Reads data/raw/cord/<split>.jsonl and writes data/processed/{train,val,test}.jsonl plus
{val,test}_heldout.jsonl (same questions, phrasings never seen in training) and stats.json.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Any

from receiptqa.data.splits import assert_disjoint, receipt_ids
from receiptqa.utils.io import load_yaml, read_jsonl, write_json, write_jsonl
from receiptqa.utils.logging import get_logger

log = get_logger("build_qa")

# qtype -> (section in gt_parse, key)
SINGLE_FIELDS = {
    "total": ("total", "total_price"),
    "subtotal": ("sub_total", "subtotal_price"),
    "tax": ("sub_total", "tax_price"),
    "payment": ("total", "cashprice"),
    "change": ("total", "changeprice"),
}
MAX_LIST_ITEMS = 12


def load_templates(path: str | Path) -> dict[str, dict[str, list[str]]]:
    templates = load_yaml(path)
    for qtype, pools in templates.items():
        overlap = set(pools["train"]) & set(pools["heldout"])
        if overlap:
            raise ValueError(f"Template leakage for '{qtype}': {sorted(overlap)}")
    return templates


def _clean(value: Any) -> str | None:
    if not isinstance(value, (str, int, float)):
        return None
    s = str(value).strip()
    return s or None


def get_items(gt_parse: dict[str, Any]) -> list[dict[str, str | None]]:
    """CORD stores `menu` as a list, or as a single dict when the receipt has one item."""
    menu = gt_parse.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    if not isinstance(menu, list):
        return []
    items = []
    for m in menu:
        if not isinstance(m, dict):
            continue
        name = _clean(m.get("nm"))
        if name and len(name) >= 2:
            items.append({"name": name, "price": _clean(m.get("price")), "cnt": _clean(m.get("cnt"))})
    return items


def unique_items(items: list[dict[str, str | None]]) -> list[dict[str, str | None]]:
    """Drop items whose name appears more than once (the question would be ambiguous)."""
    counts = Counter(i["name"].lower() for i in items)  # type: ignore[union-attr]
    return [i for i in items if counts[i["name"].lower()] == 1]  # type: ignore[union-attr]


def build_candidates(gt_parse: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Return (single-answer candidates, item-level candidates)."""
    singles: list[dict] = []
    for qtype, (section, key) in SINGLE_FIELDS.items():
        sec = gt_parse.get(section)
        if isinstance(sec, dict):
            ans = _clean(sec.get(key))
            if ans:
                singles.append({"qtype": qtype, "answer": ans})

    items = get_items(gt_parse)
    if items:
        singles.append({"qtype": "num_items", "answer": str(len(items))})
        if len(items) <= MAX_LIST_ITEMS:
            singles.append({"qtype": "item_list", "answer": ", ".join(str(i["name"]) for i in items)})

    item_cands: list[dict] = []
    for it in unique_items(items):
        if it["price"]:
            item_cands.append({"qtype": "item_price", "answer": it["price"], "item": it["name"]})
        if it["cnt"] and it["cnt"].isdigit():
            item_cands.append({"qtype": "item_count", "answer": it["cnt"], "item": it["name"]})
    return singles, item_cands


def select_candidates(singles: list[dict], item_cands: list[dict], rng: random.Random, max_qa: int) -> list[dict]:
    """Prefer receipt-level questions, fill the remaining budget with item-level ones."""
    if len(singles) >= max_qa:
        return rng.sample(singles, max_qa)
    budget = max_qa - len(singles)
    return singles + rng.sample(item_cands, min(len(item_cands), budget))


def generate_for_receipt(
    receipt: dict[str, Any],
    split: str,
    templates: dict[str, dict[str, list[str]]],
    pool: str,
    seed: int,
    max_qa: int,
) -> list[dict[str, Any]]:
    """pool is 'train' or 'heldout'. Question selection does not depend on the pool, so the
    heldout set asks the same questions as the normal set, only phrased differently."""
    rid = receipt["receipt_id"]
    singles, item_cands = build_candidates(receipt["gt_parse"])
    chosen = select_candidates(singles, item_cands, random.Random(f"{seed}-{rid}-select"), max_qa)
    phrase_rng = random.Random(f"{seed}-{rid}-{pool}")
    tag = "h" if pool == "heldout" else "q"
    rows = []
    for n, cand in enumerate(chosen):
        phrasings = templates[cand["qtype"]][pool]
        idx = phrase_rng.randrange(len(phrasings))
        question = phrasings[idx].replace("{item}", cand.get("item", ""))
        rows.append(
            {
                "id": f"{rid}_{tag}{n}",
                "receipt_id": rid,
                "image_path": receipt["image_path"],
                "question": question,
                "answer": cand["answer"],
                "qtype": cand["qtype"],
                "template_id": f"{cand['qtype']}_{tag}{idx:02d}",
                "split": split,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    raw_dir, out_dir = Path(cfg["raw_dir"]), Path(cfg["processed_dir"])
    templates = load_templates(cfg["templates_path"])
    seed, max_qa = cfg["seed"], cfg["max_qa_per_receipt"]

    stats: dict[str, Any] = {}
    ids_by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        receipts = read_jsonl(raw_dir / f"{split}.jsonl")
        ids_by_split[split] = receipt_ids(receipts)
        variants = [("train", split)] + ([("heldout", f"{split}_heldout")] if split != "train" else [])
        for pool, name in variants:
            rows = [r for rec in receipts for r in generate_for_receipt(rec, split, templates, pool, seed, max_qa)]
            write_jsonl(out_dir / f"{name}.jsonl", rows)
            stats[name] = {
                "receipts": len({r["receipt_id"] for r in rows}),
                "qa_pairs": len(rows),
                "by_type": dict(Counter(r["qtype"] for r in rows)),
            }
            log.info("%s: %d QA pairs from %d receipts", name, len(rows), stats[name]["receipts"])

    assert_disjoint(ids_by_split)
    write_json(out_dir / "stats.json", stats)
    log.info("splits are disjoint by receipt id. stats -> %s", out_dir / "stats.json")


if __name__ == "__main__":
    main()
