"""
cli.py
------
Step 5: minimal interface. Type a question, see the answer, see which
chunks were used, and see citation verification results.

Usage:
    python src/cli.py
    python src/cli.py --top-k 5
    python src/cli.py --question "How many Ayushman Arogya Mandirs are functional?"
    python src/cli.py --question "Compare AB-PMJAY and PM-ABHIM" --rerank
"""
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from generate import call_llm
from retrieve import Retriever
from verify import verify_citations


def answer_question(retriever: Retriever, question: str, top_k: int, rerank: bool = False):
    if rerank:
        results = retriever.search_with_rerank(question, top_k=top_k)
    else:
        # Use multi-hop aware search by default
        results = retriever.search_multihop(question, top_k=top_k)
    answer = call_llm(question, results)
    return answer, results


def print_answer(question: str, answer: str, results: list):
    print("\n" + "=" * 70)
    print(f"Q: {question}")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    print("Sources used:")
    for r in results:
        snippet = r["chunk"]["text"][:120].replace("\n", " ")
        score_str = f"{r['score']:.3f}"
        if "rerank_score" in r:
            score_str += f" (rerank: {r['rerank_score']:.0f}/10)"
        print(f"  [{score_str}] {r['chunk']['title']}")
        print(f"          \"{snippet}...\"")

    # Citation verification
    citation_result = verify_citations(answer, results)
    if citation_result["total"] > 0:
        print("-" * 70)
        n_verified = len(citation_result["verified"])
        n_total = citation_result["total"]
        print(f"Citation check: {n_verified}/{n_total} claims verified in source")
        if citation_result["unverified"]:
            print(f"  ⚠ Unverified: {', '.join(citation_result['unverified'])}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="RAG Q&A over India's Health Transformation (PIB)")
    parser.add_argument("--question", "-q", type=str, help="Ask a single question and exit")
    parser.add_argument("--top-k", "-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--rerank", "-r", action="store_true", help="Enable LLM-based re-ranking")
    args = parser.parse_args()

    try:
        retriever = Retriever()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.question:
        try:
            answer, results = answer_question(retriever, args.question, args.top_k, args.rerank)
        except (EnvironmentError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        print_answer(args.question, answer, results)
        return

    print("RAG Q&A - India's Health Transformation (PIB backgrounder)")
    print("Type a question, or 'exit' to quit.\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        try:
            answer, results = answer_question(retriever, question, args.top_k, args.rerank)
        except (EnvironmentError, RuntimeError) as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        print_answer(question, answer, results)


if __name__ == "__main__":
    main()
