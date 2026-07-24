"""
evaluate.py
-----------
Run the retrieval eval set and report accuracy metrics.
Compares retrieval quality across embedding methods.

Usage:
    python src/evaluate.py
"""
import json
from pathlib import Path

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval_set.json"


def run_eval():
    from retrieve import Retriever

    eval_data = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    retriever = Retriever()

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []
    results_detail = []

    for item in eval_data:
        question = item["question"]
        expected_title = item.get("expected_chunk_title")
        qtype = item.get("type", "exact")

        search_results = retriever.search(question, top_k=3)
        retrieved_titles = [r["chunk"]["title"] for r in search_results]

        if expected_title is None:
            # Negative question — we just check the top score is low
            top_score = search_results[0]["score"] if search_results else 0
            status = "LOW_CONFIDENCE" if top_score < 0.15 else "FALSE_POSITIVE"
            results_detail.append({
                "question": question,
                "type": qtype,
                "status": status,
                "top_score": round(top_score, 3),
                "top_result": retrieved_titles[0] if retrieved_titles else None,
            })
            continue

        # Check if expected chunk is in top-k results (partial title match)
        rank = None
        for i, title in enumerate(retrieved_titles):
            if expected_title.lower() in title.lower() or title.lower() in expected_title.lower():
                rank = i + 1
                break

        if rank == 1:
            hit_at_1 += 1
        if rank is not None:
            hit_at_3 += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        results_detail.append({
            "question": question,
            "type": qtype,
            "expected": expected_title[:60],
            "got": retrieved_titles[0][:60] if retrieved_titles else "NONE",
            "rank": rank,
            "scores": [round(r["score"], 3) for r in search_results],
        })

    # Compute metrics (excluding negative questions)
    n_questions = len([i for i in eval_data if i.get("expected_chunk_title")])
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0

    print("=" * 70)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)

    for r in results_detail:
        if r.get("rank") is not None:
            mark = "PASS" if r["rank"] == 1 else f"rank={r['rank']}"
        elif r.get("status"):
            mark = r["status"]
        else:
            mark = "MISS"
        print(f"  [{r['type']:>12}] {mark:>15}  {r['question'][:55]}")

    print("-" * 70)
    print(f"  Hit@1: {hit_at_1}/{n_questions} ({hit_at_1/n_questions*100:.0f}%)")
    print(f"  Hit@3: {hit_at_3}/{n_questions} ({hit_at_3/n_questions*100:.0f}%)")
    print(f"  MRR:   {mrr:.3f}")
    print("=" * 70)


if __name__ == "__main__":
    run_eval()
