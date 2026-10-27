"""
Tokenizer, BM25 helpers, and reciprocal rank fusion for hybrid search.

Public API:
    tokenize(text)                      -- lowercase word tokens, stopwords removed
    build_bm25(corpus)                  -- build a BM25Okapi index over a list of strings
    bm25_search(index, corpus_ids, query, top_k)  -- return ranked (chunk_id, score) pairs
    reciprocal_rank_fusion(rankings, k) -- fuse multiple ranked lists into one
"""

import re
import logging
from typing import Any

from rank_bm25 import BM25Okapi  # type: ignore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopwords (small, pragmatic set; not exhaustive)
# ---------------------------------------------------------------------------

_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "it", "its", "be", "as", "was",
        "are", "were", "been", "has", "have", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can", "that",
        "this", "these", "those", "not", "no", "nor",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens with stopwords removed.

    Splits on non-alphanumeric characters, lowercases, and filters out
    stopwords and single-character tokens.

    Args:
        text: Input string (any case).

    Returns:
        List of cleaned tokens.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def build_bm25(corpus: list[str]) -> BM25Okapi:
    """Build a BM25Okapi index from a list of document strings.

    Each string is tokenized before indexing.

    Args:
        corpus: List of document strings (e.g., "heading > subheading\nchunk text").

    Returns:
        A fitted BM25Okapi instance.
    """
    tokenized = [tokenize(doc) for doc in corpus]
    return BM25Okapi(tokenized)


def bm25_search(
    index: BM25Okapi,
    corpus_ids: list[str],
    query: str,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Score all documents against a query and return the top-k results.

    Args:
        index: A fitted BM25Okapi index.
        corpus_ids: List of chunk IDs corresponding 1:1 with the BM25 corpus.
        query: The search query string.
        top_k: Maximum number of results to return.

    Returns:
        List of (chunk_id, bm25_score) tuples, sorted descending by score.
        Chunks with a score of zero are excluded.
    """
    tokens = tokenize(query)
    if not tokens:
        return []

    scores = index.get_scores(tokens)
    ranked = sorted(
        zip(corpus_ids, scores), key=lambda x: x[1], reverse=True
    )
    return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0.0]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists with Reciprocal Rank Fusion (RRF).

    RRF score for item d across rankings: sum_r 1 / (k + rank_r(d))
    where rank_r(d) is the 1-based position of d in ranking r, or 0 if absent.

    Args:
        rankings: List of ranked lists of chunk IDs. Items earlier in each
            list are ranked higher. Items absent from a list contribute 0.
        k: RRF smoothing constant (default 60, standard in the literature).

    Returns:
        List of (chunk_id, rrf_score) tuples sorted descending by score.
    """
    scores: dict[str, float] = {}
    for ranked_list in rankings:
        for rank, item in enumerate(ranked_list, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
