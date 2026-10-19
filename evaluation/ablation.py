"""Retrieval-only ablation plus a refusal-threshold calibration. No LLM needed.

    python -m evaluation.ablation
    python -m evaluation.ablation --with-rewrite   # also tests query rewriting (needs GROQ_API_KEY)

Queries are the raw questions (no decomposition), so multi-part questions are a little harder here
than in the full pipeline. Compare rows with each other, not with evaluation.py.
"""
import argparse

import numpy as np
import pandas as pd

from evaluation import common
from evaluation.eval_dataset import EVAL_DATASET
from evaluation.eval_heldout import HELDOUT_DATASET, UNANSWERABLE
from src.config import CANDIDATE_K, RESULTS_DIR
from src.query_processing import rewrite_query
from src.reranking import rerank
from src.retrieval import bm25_search, dense_search, hybrid_retrieval


def build_methods(index, with_rewrite: bool) -> dict:
    methods = {
        "dense only": lambda q: dense_search(index, q, top_k=10),
        "bm25 only": lambda q: bm25_search(index, q, top_k=10),
        "hybrid (RRF)": lambda q: hybrid_retrieval(index, q, top_k=10),
        "hybrid + rerank": lambda q: rerank(q, hybrid_retrieval(index, q, top_k=CANDIDATE_K), top_k=10),
    }
    if with_rewrite:
        methods["rewrite + hybrid + rerank"] = lambda q: rerank(
            rewrite_query(q), hybrid_retrieval(index, rewrite_query(q), top_k=CANDIDATE_K), top_k=10)
    return methods


def run_ablation(index, dataset, methods) -> pd.DataFrame:
    rows = []
    for name, search in methods.items():
        stats = {"recall@5": [], "MRR": [], "strict_recall@5": [], "strict_MRR": []}
        for item in dataset:
            ids = common.child_ids(search(item["question"]))
            gold = common.gold_ids_any(index.children, item["anchors"])
            strict, _ = common.gold_ids_strict(index.children, item["anchors"])
            stats["recall@5"].append(common.recall_at_k(ids, gold, 5))
            stats["MRR"].append(common.reciprocal_rank(ids, gold))
            stats["strict_recall@5"].append(common.recall_at_k(ids, strict, 5))
            stats["strict_MRR"].append(common.reciprocal_rank(ids, strict))
        rows.append({"method": name, **{k: round(float(np.nanmean(v)), 4) for k, v in stats.items()}})
    return pd.DataFrame(rows)


def calibrate_threshold(index, answerable, unanswerable) -> tuple[pd.DataFrame, float, float]:
    """Best reranker score per question, then the cut-off that separates the two groups best."""
    def top_score(question):
        ranked = rerank(question, hybrid_retrieval(index, question, top_k=CANDIDATE_K), top_k=1)
        return ranked[0]["reranker_score"] if ranked else float("-inf")

    scores = pd.DataFrame(
        [{"id": q["id"], "answerable": True, "top_score": top_score(q["question"])} for q in answerable]
        + [{"id": q["id"], "answerable": False, "top_score": top_score(q["question"])} for q in unanswerable])

    yes = scores.loc[scores["answerable"], "top_score"].to_numpy()
    no = scores.loc[~scores["answerable"], "top_score"].to_numpy()
    best_threshold, best_balanced = None, -1.0
    for t in sorted(scores["top_score"].unique()):
        balanced = ((yes >= t).mean() + (no < t).mean()) / 2  # right on answerable and on unanswerable
        if balanced > best_balanced:
            best_threshold, best_balanced = float(t), float(balanced)
    return scores, best_threshold, best_balanced


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_index_args(parser)
    parser.add_argument("--with-rewrite", action="store_true")
    args = parser.parse_args()

    index = common.load_index(args)
    answerable = EVAL_DATASET + HELDOUT_DATASET

    table = run_ablation(index, answerable, build_methods(index, args.with_rewrite))
    print("\n=== Ablation (%d questions) ===\n%s" % (len(answerable), table.to_string(index=False)))

    scores, threshold, balanced = calibrate_threshold(index, answerable, UNANSWERABLE)
    print("\n=== Top reranker score per question ===\n", scores.round(3).to_string(index=False))
    print(f"\nBest cut-off: {threshold:.3f} (balanced accuracy {balanced:.2f}) "
          f"on {len(answerable)} answerable and {len(UNANSWERABLE)} unanswerable questions.")
    print("That is a small sample. Treat it as a starting point and set MIN_RERANK_SCORE in src/config.py yourself.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "ablation.csv", index=False)
    scores.to_csv(RESULTS_DIR / "threshold_calibration.csv", index=False)


if __name__ == "__main__":
    main()