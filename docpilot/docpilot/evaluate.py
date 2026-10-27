"""Evaluate retrieval quality and abstention on a labelled question set.

Metrics
  * Hit@k         : fraction of questions where a retrieved passage contains the
                    answer keyword(s) AND comes from the right page.
  * MRR           : mean reciprocal rank of the first correct passage.
  * Answer rate   : fraction of answerable questions the system chooses to answer.
  * Abstain rate  : fraction of unanswerable questions the system correctly
                    refuses to answer (higher is better, no hallucination).
Answer rate and abstain rate are measured without an API key (extractive
mode), so the numbers are free and reproducible.

Usage
    python evaluate.py                       # compares keyword / vector / hybrid
    python evaluate.py --k 3 --chunk-size 500
    python evaluate.py --no-embeddings       # keyword only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.generator import answer_question
from rag.ingest import build_chunks
from rag.retriever import HybridRetriever


def is_correct(hit, item) -> bool:
    text = hit.chunk.text.lower()
    return hit.chunk.page == item["page"] and all(k.lower() in text for k in item["keywords"])


def evaluate(retriever, questions, k, mode):
    hits_at_k, rr = 0, 0.0
    for item in questions["questions"]:
        results = retriever.search(item["q"], k=k, mode=mode)
        rank = next((r for r, h in enumerate(results, 1) if is_correct(h, item)), None)
        if rank:
            hits_at_k += 1
            rr += 1 / rank
    n = len(questions["questions"])
    return hits_at_k / n, rr / n


def answer_and_abstain(retriever, questions, k, mode):
    """Return (answer rate on answerable, abstain rate on unanswerable)."""
    answered = sum(
        answer_question(i["q"], retriever.search(i["q"], k=k, mode=mode), use_llm=False).found
        for i in questions["questions"]
    )
    refused = sum(
        not answer_question(q, retriever.search(q, k=k, mode=mode), use_llm=False).found
        for q in questions["unanswerable"]
    )
    return answered / len(questions["questions"]), refused / len(questions["unanswerable"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=250)
    ap.add_argument("--no-embeddings", action="store_true")
    args = ap.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    chunks = build_chunks(questions["document"], Path(questions["document"]).name,
                          chunk_size=args.chunk_size)
    retriever = HybridRetriever(chunks, use_embeddings=not args.no_embeddings)

    modes = ["keyword"] + (["vector", "hybrid"] if retriever.has_embeddings else [])
    print(f"\n{len(chunks)} chunks · chunk size {args.chunk_size} · k={args.k}\n")
    print(f"{'mode':<10} {'Hit@' + str(args.k):<9} {'MRR':<7} {'Answered':<10} {'Abstained':<10}")
    print("-" * 48)
    for mode in modes:
        hit, mrr = evaluate(retriever, questions, args.k, mode)
        ans, abst = answer_and_abstain(retriever, questions, args.k, mode)
        print(f"{mode:<10} {hit:<9.0%} {mrr:<7.3f} {ans:<10.0%} {abst:<10.0%}")
    n_a, n_u = len(questions["questions"]), len(questions["unanswerable"])
    print(f"\nAnswered = share of {n_a} answerable questions answered; "
          f"Abstained = share of {n_u} unanswerable questions correctly refused.\n")


if __name__ == "__main__":
    main()
