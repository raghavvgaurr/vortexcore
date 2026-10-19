"""End-to-end evaluation: runs the full pipeline on the questions and writes CSVs.

    python -m evaluation.evaluation                  # the original 10 questions
    python -m evaluation.evaluation --heldout        # + held-out and unanswerable questions
    python -m evaluation.evaluation --pickles DIR    # use the notebook's chunk pickles
"""
import argparse
import time

import numpy as np
import pandas as pd

from evaluation import common
from evaluation.eval_dataset import EVAL_DATASET
from evaluation.eval_heldout import HELDOUT_DATASET, UNANSWERABLE
from src.config import CANDIDATE_K, MIN_RERANK_SCORE, RESULTS_DIR, TOP_K
from src.pipeline import rag_pipeline

# Phrases the answer prompt tends to use when it refuses. This is a heuristic.
REFUSAL_HINTS = ["not available in the provided context", "not provided in the", "does not contain",
                 "no information", "not mentioned", "cannot be determined", "not specified"]


def looks_like_refusal(answer: str) -> bool:
    return any(hint in common.norm_text(answer) for hint in REFUSAL_HINTS)


def run_benchmark(index, dataset, candidate_k=CANDIDATE_K, top_k=TOP_K):
    runs = []
    for item in dataset:
        print(f"[{item['id']}] {item['question']}")
        start = time.perf_counter()
        try:
            result, error = rag_pipeline(item["question"], index, candidate_k, top_k, min_score=MIN_RERANK_SCORE), None
        except Exception as exc:  # keep going so one failure does not hide the rest
            result, error = None, repr(exc)
        latency = time.perf_counter() - start
        runs.append({
            "id": item["id"], "question": item["question"], "latency_sec": latency, "error": error,
            "result": result,
            "mode": result["mode"] if result else None,
            "answer": result["answer"] if result else "",
            "abstained": result["abstained"] if result else None,
        })
        print(f"    mode={runs[-1]['mode']} latency={latency:.2f}s")
    return runs


def ranked_lists(result):
    if result is None:
        return []
    if result["mode"] == "single":
        return [{"sub_query": result["sub_queries"][0], "candidates": result["candidates"], "ranked": result["ranked"]}]
    return [{"sub_query": r["sub_query"], "candidates": r["candidates"], "ranked": r["ranked"]}
            for r in result["results"]]


def retrieval_table(index, dataset, runs) -> pd.DataFrame:
    rows = []
    for item, run in zip(dataset, runs):
        gold_any = common.gold_ids_any(index.children, item["anchors"])
        gold_strict, fallback = common.gold_ids_strict(index.children, item["anchors"])
        for number, sub in enumerate(ranked_lists(run["result"]), start=1):
            cand, ranked = common.child_ids(sub["candidates"]), common.child_ids(sub["ranked"])
            row = {"id": item["id"], "sub_query_index": number, "sub_query": sub["sub_query"],
                   "gold_chunks": len(gold_any), "strict_gold_chunks": len(gold_strict),
                   "strict_fell_back_to_any": fallback}
            for prefix, gold in (("", gold_any), ("strict_", gold_strict)):
                for k in (1, 5, 10):
                    row[f"{prefix}candidate_recall@{k}"] = common.recall_at_k(cand, gold, k)
                row[f"{prefix}candidate_MRR"] = common.reciprocal_rank(cand, gold)
                for k in (1, 5, 10):
                    row[f"{prefix}reranked_recall@{k}"] = common.recall_at_k(ranked, gold, k)
                row[f"{prefix}reranked_MRR"] = common.reciprocal_rank(ranked, gold)
            rows.append(row)
    return pd.DataFrame(rows)


def generation_table(dataset, runs) -> pd.DataFrame:
    rows = []
    for item, run in zip(dataset, runs):
        facts = item["required_facts"]
        original = [f for f in facts if common.norm_text(f) in common.norm_text(run["answer"])]
        relaxed = [f for f in facts if common.norm_numbers(f) in common.norm_numbers(run["answer"])]
        rows.append({"id": item["id"], "fact_coverage": round(len(original) / len(facts), 3),
                     "fact_coverage_numbers_normalized": round(len(relaxed) / len(facts), 3),
                     "facts_found": original, "required_facts": facts, "answer": run["answer"]})
    return pd.DataFrame(rows)


def diagnose(dataset, runs, retrieval_df, generation_df) -> pd.DataFrame:
    rows = []
    for item, run in zip(dataset, runs):
        group = retrieval_df[retrieval_df["id"] == item["id"]]
        coverage = generation_df.loc[generation_df["id"] == item["id"], "fact_coverage"].iloc[0]
        if run["result"] is None:
            text = "Pipeline error"
        elif group.empty:
            text = "No retrieval metrics"
        elif group["candidate_recall@10"].fillna(0).max() == 0:
            text = "Candidate-pool miss"
        elif group["reranked_recall@5"].fillna(0).max() == 0:
            text = "Evidence in candidate pool but not top-5 after reranking"
        elif coverage < 1.0:
            text = "Retrieval reached evidence; answer missed required fact(s)"
        else:
            text = "Pass on current diagnostic checks"
        rows.append({"id": item["id"], "diagnosis": text})
    return pd.DataFrame(rows)


def latency_table(runs) -> pd.DataFrame:
    values = [r["latency_sec"] for r in runs]
    return pd.DataFrame([{
        "num_questions": len(values), "mean_sec": np.mean(values), "median_sec": np.median(values),
        "p95_sec": np.percentile(values, 95), "min_sec": np.min(values), "max_sec": np.max(values),
    }]).round(4)


def refusal_report(index, answerable_runs, unanswerable_runs) -> pd.DataFrame:
    refused_bad = [looks_like_refusal(r["answer"]) for r in unanswerable_runs]
    refused_good = [looks_like_refusal(r["answer"]) for r in answerable_runs]
    return pd.DataFrame([{
        "unanswerable_correctly_refused": f"{sum(refused_bad)}/{len(refused_bad)}",
        "answerable_wrongly_refused": f"{sum(refused_good)}/{len(refused_good)}",
        "note": "refusal detected by phrase matching on the answer text",
    }])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_index_args(parser)
    parser.add_argument("--heldout", action="store_true", help="also run held-out and unanswerable questions")
    args = parser.parse_args()

    index = common.load_index(args)
    print(f"Indexed {len(index.children)} child chunks.\n")

    dataset = EVAL_DATASET + (HELDOUT_DATASET if args.heldout else [])
    runs = run_benchmark(index, dataset)

    retrieval_df = retrieval_table(index, dataset, runs)
    generation_df = generation_table(dataset, runs)
    diagnosis_df = diagnose(dataset, runs, retrieval_df, generation_df)
    latency_df = latency_table(runs)

    metric_cols = [c for c in retrieval_df.columns if "recall" in c or "MRR" in c]
    summary = retrieval_df[metric_cols].mean().round(4).rename("score").to_frame()
    print("\n=== Retrieval summary (mean over sub-queries) ===\n", summary.to_string())
    print("\nMean fact coverage:", round(generation_df["fact_coverage"].mean(), 4),
          "| numbers normalized:", round(generation_df["fact_coverage_numbers_normalized"].mean(), 4))
    print("\n", latency_df.to_string(index=False))
    print("\n", diagnosis_df.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(runs).drop(columns=["result"]).to_csv(RESULTS_DIR / "evaluation_results.csv", index=False)
    retrieval_df.to_csv(RESULTS_DIR / "retrieval_metrics.csv", index=False)
    summary.to_csv(RESULTS_DIR / "retrieval_summary.csv")
    generation_df.to_csv(RESULTS_DIR / "generation_diagnostics.csv", index=False)
    latency_df.to_csv(RESULTS_DIR / "latency_summary.csv", index=False)
    diagnosis_df.to_csv(RESULTS_DIR / "failure_diagnosis.csv", index=False)

    if args.heldout:
        unanswerable_runs = run_benchmark(index, UNANSWERABLE)
        report = refusal_report(index, runs, unanswerable_runs)
        print("\n=== Refusals ===\n", report.to_string(index=False))
        report.to_csv(RESULTS_DIR / "refusal_report.csv", index=False)

    print(f"\nSaved CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()