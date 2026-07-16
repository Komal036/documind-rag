"""
DocuMind RAGAS Evaluation
---------------------------
Runs the RAG pipeline against a fixed question set (data/eval/hr_policy_qa.json)
and scores it with RAGAS: faithfulness (no hallucination) and answer relevancy
(does the answer address the question).

This establishes the v1 baseline BEFORE any Self-RAG / multi-hop work, so that
v2 can be measured against real numbers instead of "it feels better."

Usage:
    python scripts/evaluate_ragas.py --email your@email.com

Requires: the user account already has documents ingested (e.g. sample_hr_policy.txt).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from openai import AsyncOpenAI
from ragas.embeddings.huggingface_provider import HuggingFaceEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import AnswerRelevancy, Faithfulness

from src.db.connection import SessionLocal
from src.db.models import User
from src.pipeline import RAGPipeline
from src.utils.config import get_settings

EVAL_SET_PATH = Path(__file__).parent.parent / "data" / "eval" / "hr_policy_qa.json"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval" / "results"


def clean_for_scoring(answer: str) -> str:
    """
    Strip the '[N] filename, page X' citation footer before scoring.
    That footer is pure formatting, not semantic content — leaving it in
    dilutes AnswerRelevancy's embedding-similarity comparison.
    """
    marker = "\n\nSources:"
    idx = answer.find(marker)
    return answer[:idx].strip() if idx != -1 else answer.strip()


REFUSAL_MARKERS = ("couldn't find", "don't have enough information", "no relevant information")


def is_refusal(answer: str) -> bool:
    """AnswerRelevancy can't meaningfully score a refusal — there's no real
    content to reverse-engineer a synthetic question from, so the score
    tanks toward 0 even though the refusal is the *correct* behavior."""
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def get_user_id(email: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user is None:
            raise SystemExit(f"No user found with email '{email}'. Sign up first, or pass the right --email.")
        return user.id
    finally:
        db.close()


def run_pipeline(pipeline: RAGPipeline, question: str, user_id) -> dict:
    """Retrieve + rerank + generate, returning full (untruncated) context texts."""
    candidates = pipeline._retriever.retrieve(question, user_id=user_id)
    final_chunks = pipeline._reranker.rerank(question, candidates)
    result = pipeline._generator.generate(question, final_chunks)

    return {
        "answer": result["answer"],
        "contexts": [doc.page_content for doc, _score in final_chunks],
    }


async def score_sample(judge_llm, judge_embeddings, question: str, answer: str, contexts: list[str]) -> dict:
    faithfulness = Faithfulness(llm=judge_llm)
    relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings)

    faithfulness_result = await faithfulness.ascore(
        user_input=question, response=answer, retrieved_contexts=contexts
    )
    relevancy_result = await relevancy.ascore(user_input=question, response=answer)
    return {
        "faithfulness": faithfulness_result.value,
        "answer_relevancy": relevancy_result.value,
    }


async def main(email: str):
    settings = get_settings()
    qa_pairs = json.loads(EVAL_SET_PATH.read_text())

    print(f"Loaded {len(qa_pairs)} evaluation questions from {EVAL_SET_PATH.name}")
    user_id = get_user_id(email)

    print("Initialising RAG pipeline…")
    pipeline = RAGPipeline()
    pipeline.initialize()

    # Judge LLM: reuse Groq via its OpenAI-compatible endpoint, so no extra API key needed.
    judge_client = AsyncOpenAI(
        api_key=settings.llm.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    judge_llm = llm_factory(settings.llm.groq_model, client=judge_client)

    # Judge embeddings: reuse the same local model DocuMind already uses for retrieval,
    # so answer-relevancy scoring doesn't need a separate embedding API/key.
    judge_embeddings = HuggingFaceEmbeddings(model=settings.embedding.embedding_model_name)

    rows = []
    for i, pair in enumerate(qa_pairs, 1):
        question = pair["question"]
        print(f"[{i}/{len(qa_pairs)}] {question}")

        gen = run_pipeline(pipeline, question, user_id)
        clean_answer = clean_for_scoring(gen["answer"])
        refusal = is_refusal(gen["answer"])

        scores = await score_sample(judge_llm, judge_embeddings, question, clean_answer, gen["contexts"])

        rows.append(
            {
                "question": question,
                "ground_truth": pair["ground_truth"],
                "answer": gen["answer"],
                "num_contexts": len(gen["contexts"]),
                "is_refusal": refusal,
                **scores,
            }
        )
        flag = " (refusal — excluded from relevancy avg)" if refusal else ""
        print(f"    faithfulness={scores['faithfulness']:.3f}  answer_relevancy={scores['answer_relevancy']:.3f}{flag}")

    avg_faithfulness = sum(r["faithfulness"] for r in rows) / len(rows)
    scorable_relevancy = [r["answer_relevancy"] for r in rows if not r["is_refusal"]]
    avg_relevancy = sum(scorable_relevancy) / len(scorable_relevancy) if scorable_relevancy else float("nan")
    num_refusals = sum(1 for r in rows if r["is_refusal"])

    print("\n" + "=" * 50)
    print(f"Average faithfulness (all {len(rows)} questions):        {avg_faithfulness:.3f}")
    print(f"Average answer relevancy ({len(scorable_relevancy)} non-refusal questions): {avg_relevancy:.3f}")
    if num_refusals:
        print(f"({num_refusals} refusal(s) excluded from relevancy average — see note in script)")
    print("=" * 50)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ragas_baseline_{timestamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "num_questions": len(rows),
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevancy": avg_relevancy,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved full results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", required=True, help="Email of the user whose indexed documents to evaluate against")
    args = parser.parse_args()
    asyncio.run(main(args.email))