"""Helpers shared by evaluation.py and ablation.py: loading an index, gold chunks, metrics."""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np

from src.config import SAMPLE_PDF
from src.pipeline import build_index
from src.retrieval import Index, index_chunks


def add_index_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pdf", default=str(SAMPLE_PDF), help="PDF to index (default: the sample report)")
    parser.add_argument("--pickles", help="Folder with child_chunks.pkl and parent_chunks.pkl from the "
                                          "notebooks. Use it to compare against the notebook numbers.")
    parser.add_argument("--auto-ocr", action="store_true",
                        help="OCR only pages without a text layer. Default OCRs every page, like the notebook.")


def load_index(args: argparse.Namespace) -> Index:
    if args.pickles:
        folder = Path(args.pickles)
        # These pickles are my own files from the notebooks. Never unpickle files you do not trust.
        with open(folder / "child_chunks.pkl", "rb") as f:
            children = pickle.load(f)
        with open(folder / "parent_chunks.pkl", "rb") as f:
            parents = pickle.load(f)
        children = [c for c in children if c.page_content and c.page_content.strip()]
        return index_chunks(parents, children)
    return build_index([args.pdf], force_ocr=not args.auto_ocr)


def norm_text(text) -> str:
    text = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def norm_numbers(text) -> str:
    """norm_text plus number words -> digits, so 'two months' matches '2 months'."""
    pattern = r"\b(" + "|".join(NUMBER_WORDS) + r")\b"
    return re.sub(pattern, lambda m: NUMBER_WORDS[m.group(1)], norm_text(text))


def count_anchor_hits(text: str, anchors: list[str]) -> int:
    text_n = norm_text(text)
    return sum(norm_text(a) in text_n for a in anchors)


def gold_ids_any(children, anchors) -> set[int]:
    """Original rule from the notebook: a chunk is gold if it contains ANY anchor."""
    return {i for i, c in enumerate(children) if count_anchor_hits(c.page_content, anchors) >= 1}


def gold_ids_strict(children, anchors) -> tuple[set[int], bool]:
    """Stricter rule: at least 2 anchors (or the only one). Falls back to 'any' if nothing matches."""
    needed = min(2, len(anchors))
    strict = {i for i, c in enumerate(children) if count_anchor_hits(c.page_content, anchors) >= needed}
    if strict:
        return strict, False
    return gold_ids_any(children, anchors), True


def child_ids(rows: list[dict]) -> list[int]:
    return [row["child_index"] for row in rows if row.get("child_index") is not None]


def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        return np.nan
    return len(set(ranked_ids[:k]) & gold_ids) / len(gold_ids)


def reciprocal_rank(ranked_ids, gold_ids):
    if not gold_ids:
        return np.nan
    for rank, child_id in enumerate(ranked_ids, start=1):
        if child_id in gold_ids:
            return 1.0 / rank
    return 0.0