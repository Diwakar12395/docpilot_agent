"""Hybrid retriever: BM25 keyword search + dense embeddings, fused with
Reciprocal Rank Fusion (RRF).

Keyword search is strong on exact terms (names, numbers, acronyms); embeddings
are strong on meaning and paraphrase. Fusing both beats either alone, which
`evaluate.py` measures on your own question set.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from .ingest import Chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "was", "were",
    "for", "on", "at", "by", "with", "what", "which", "who", "how", "does",
    "do", "did", "it", "this", "that", "be", "as", "from", "when", "where",
    "can", "must", "should", "will", "get", "gets", "there", "their", "they",
    "many", "much", "often", "soon", "early", "long", "any", "have", "has",
    "i", "my", "me", "we", "our", "you", "your", "if", "than", "then", "into",
}


def _stem(token: str) -> str:
    """Very light stemming so 'passwords' matches 'password'."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    keyword_rank: int | None = None
    vector_rank: int | None = None
    coverage: float = 0.0   # IDF-weighted share of question terms found in this passage


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], use_embeddings: bool = True):
        if not chunks:
            raise ValueError("No chunks to index.")
        self.chunks = chunks
        tokenized = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
        self._df = Counter(t for toks in tokenized for t in set(toks))
        self.embeddings: np.ndarray | None = None
        self._model = None
        if use_embeddings:
            try:
                self._model = _load_model()
                self.embeddings = self._model.encode(
                    [c.text for c in chunks],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:  # model download blocked, etc.
                print(f"[retriever] embeddings unavailable, keyword-only mode: {exc}")
                self._model = None

    @property
    def has_embeddings(self) -> bool:
        return self.embeddings is not None

    # -- evidence strength --------------------------------------------------
    def coverage(self, query: str, text: str) -> float:
        """IDF-weighted fraction of query terms present in `text`.

        Rare terms count for more. A term that appears nowhere in the corpus
        gets the maximum weight, so asking about something the documents never
        mention (e.g. "CEO") drags coverage down and triggers abstention.
        """
        q_terms = set(tokenize(query))
        if not q_terms:
            return 0.0
        n = len(self.chunks)
        idf = lambda t: math.log((n + 1) / (self._df.get(t, 0) + 0.5))
        present = set(tokenize(text))
        total = sum(idf(t) for t in q_terms)
        return sum(idf(t) for t in q_terms if t in present) / total if total else 0.0

    # -- individual rankers -------------------------------------------------
    def keyword_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def vector_search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self.embeddings is None:
            return []
        q = self._model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ q
        order = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in order]

    # -- fusion --------------------------------------------------------------
    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        """mode is one of 'keyword', 'vector' or 'hybrid'."""
        pool = max(k * 4, 20)
        kw = self.keyword_search(query, pool) if mode in ("keyword", "hybrid") else []
        vec = self.vector_search(query, pool) if mode in ("vector", "hybrid") else []

        if mode == "keyword":
            hits = [Hit(self.chunks[i], s, keyword_rank=r) for r, (i, s) in enumerate(kw[:k], 1)]
            return self._with_coverage(query, hits)
        if mode == "vector":
            hits = [Hit(self.chunks[i], s, vector_rank=r) for r, (i, s) in enumerate(vec[:k], 1)]
            return self._with_coverage(query, hits)

        # Reciprocal Rank Fusion
        rrf_k = 60
        fused: dict[int, float] = {}
        kw_rank = {i: r for r, (i, _) in enumerate(kw, 1)}
        vec_rank = {i: r for r, (i, _) in enumerate(vec, 1)}
        for ranks in (kw_rank, vec_rank):
            for i, r in ranks.items():
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + r)
        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        hits = [
            Hit(self.chunks[i], score, keyword_rank=kw_rank.get(i), vector_rank=vec_rank.get(i))
            for i, score in top
        ]
        return self._with_coverage(query, hits)

    def _with_coverage(self, query: str, hits: list[Hit]) -> list[Hit]:
        for h in hits:
            h.coverage = self.coverage(query, h.chunk.text)
        return hits


_MODEL_CACHE = None


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        from sentence_transformers import SentenceTransformer

        _MODEL_CACHE = SentenceTransformer(EMBED_MODEL)
    return _MODEL_CACHE
