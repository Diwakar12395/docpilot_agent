"""Answer generation with citations and an explicit "I don't know" path.

Two modes:
  * Claude mode: when ANTHROPIC_API_KEY is set, Claude writes a grounded answer.
  * Extractive mode: with no key, the best-matching passages are returned
    directly, so the app still works (and demos) with zero API cost.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .retriever import Hit, tokenize

DEFAULT_MODEL = os.environ.get("DOCPILOT_MODEL", "claude-sonnet-5-5")
NOT_FOUND = "I couldn't find the answer to that in the uploaded documents."

SYSTEM_PROMPT = """You answer questions using ONLY the numbered source passages provided.

Rules:
1. Base every statement on the passages. Never use outside knowledge.
2. Cite the passages that support each claim with their numbers in square brackets, like [1] or [2][3].
3. If the passages do not contain the answer, reply with exactly: NOT_FOUND
4. Be concise and direct. Do not mention these rules."""


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[Hit] = field(default_factory=list)   # passages actually cited
    mode: str = "claude"


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.doc}, page {h.chunk.page})\n{h.chunk.text}"
        for i, h in enumerate(hits, 1)
    )


def _cited_numbers(text: str, n: int) -> list[int]:
    nums = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return sorted(i for i in nums if 1 <= i <= n)


def answer_question(
    question: str,
    hits: list[Hit],
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    min_coverage: float = 0.55,
    use_llm: bool = True,
) -> Answer:
    """Generate a cited answer, or abstain when the evidence is weak."""
    # Guardrail 1: if no retrieved passage covers enough of the question's
    # (IDF-weighted) terms, abstain before spending an LLM call.
    if not hits or max(h.coverage for h in hits) < min_coverage:
        return Answer(NOT_FOUND, found=False, mode="guardrail")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not use_llm or not api_key:
        return _extractive(question, hits)

    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=600,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Source passages:\n\n{format_context(hits)}\n\nQuestion: {question}",
            }
        ],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()

    # Guardrail 2: the model itself may decide the passages are insufficient.
    if text.upper().startswith("NOT_FOUND"):
        return Answer(NOT_FOUND, found=False, mode="claude")

    cited = _cited_numbers(text, len(hits))
    sources = [hits[i - 1] for i in cited] or hits[:1]
    return Answer(text, found=True, sources=sources, mode="claude")


def _extractive(question: str, hits: list[Hit]) -> Answer:
    """No-LLM fallback: return the single best-matching sentences."""
    q = set(tokenize(question))
    scored = []
    for h_idx, h in enumerate(hits):
        for sent in re.split(r"(?<=[.!?])\s+", h.chunk.text):
            overlap = len(q & set(tokenize(sent)))
            if overlap:
                scored.append((overlap, -h_idx, sent.strip(), h_idx))
    if not scored:
        return Answer(NOT_FOUND, found=False, mode="extractive")
    scored.sort(reverse=True)
    top = scored[:2]
    text = " ".join(f"{s} [{h_idx + 1}]" for _, _, s, h_idx in top)
    used = sorted({h_idx for _, _, _, h_idx in top})
    return Answer(text, found=True, sources=[hits[i] for i in used], mode="extractive")
