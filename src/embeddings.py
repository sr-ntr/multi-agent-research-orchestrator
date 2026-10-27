"""
Local embedding generator using FastEmbed (ONNX Runtime) with an SQLite cache.

FastEmbed runs 100% locally on CPU without API calls, rate limits, or network latency.
Default model: BAAI/bge-small-en-v1.5 (384 dimensions, L2-normalized).

Public API:
    embed_texts(texts, task_type)   -- embed a list of strings, returns list[list[float]]
    embed_query(text)               -- convenience: single query embedding
    embed_documents(texts)          -- convenience: batch document embeddings

Cache:
    Key = sha1(model + ":" + task_type + ":" + text).
    Stored in .cache/embeddings.db.
"""

import hashlib
import json
import logging
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, Optional

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"]

_BATCH_SIZE = 64

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_CACHE_PATH: Path = Path(
    os.environ.get("EMBEDDING_CACHE_PATH", ".cache/embeddings.db")
)


def _get_conn() -> sqlite3.Connection:
    """Open (and initialise) the sqlite cache, creating directories as needed."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_CACHE_PATH), check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS embeddings "
        "(key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def _cache_key(text: str, task_type: str) -> str:
    """Compute a deterministic cache key for a (model, task_type, text) triple."""
    raw = f"{EMBEDDING_MODEL}:{task_type}:{text}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_from_cache(
    conn: sqlite3.Connection, keys: list[str]
) -> dict[str, list[float]]:
    """Return cached vectors for the given keys (those that exist)."""
    if not keys:
        return {}
    placeholders = ",".join("?" * len(keys))
    rows = conn.execute(
        f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", keys
    ).fetchall()
    return {row[0]: json.loads(row[1]) for row in rows}


def _save_to_cache(
    conn: sqlite3.Connection, mapping: dict[str, list[float]]
) -> None:
    """Persist new (key, vector) pairs to the cache."""
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
        [(k, json.dumps(v)) for k, v in mapping.items()],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def _l2_normalize(vec: list[float]) -> list[float]:
    """L2-normalise a vector into native Python float objects."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if norm == 0.0:
        return [float(x) for x in vec]
    return [float(x) / norm for x in vec]


# ---------------------------------------------------------------------------
# FastEmbed Model Singleton
# ---------------------------------------------------------------------------

_FASTEMBED_INSTANCE: Optional[Any] = None


def _get_fastembed_model() -> Any:
    """Lazy initialization of the FastEmbed TextEmbedding model."""
    global _FASTEMBED_INSTANCE
    if _FASTEMBED_INSTANCE is None:
        try:
            from fastembed import TextEmbedding  # type: ignore

            model_name = EMBEDDING_MODEL
            # If a Gemini model name was left in config, default to BGE small
            if "gemini" in model_name.lower() or "text-embedding" in model_name.lower():
                model_name = "BAAI/bge-small-en-v1.5"

            logger.info("Initializing FastEmbed model: %s", model_name)
            _FASTEMBED_INSTANCE = TextEmbedding(model_name=model_name)
        except ImportError:
            raise ImportError(
                "FastEmbed is not installed. Please install it:\n"
                "  pip install fastembed\n"
            )
    return _FASTEMBED_INSTANCE


# ---------------------------------------------------------------------------
# Embedding function
# ---------------------------------------------------------------------------


def _call_embed_api(texts: list[str], task_type: str) -> list[list[float]]:
    """Compute embeddings locally via FastEmbed."""
    model = _get_fastembed_model()
    # FastEmbed's model.embed() yields numpy arrays; convert each element to native float
    embeddings_iter = model.embed(texts)
    return [[float(x) for x in vec] for vec in embeddings_iter]


def embed_texts(texts: list[str], task_type: TaskType = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts using local FastEmbed model.

    Results are fetched from SQLite cache when available. Unseen texts are
    embedded locally via ONNX Runtime and cached. All vectors are L2-normalised.

    Args:
        texts: List of strings to embed.
        task_type: Kept for compatibility ("RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY").

    Returns:
        List of float vectors, each of length EMBEDDING_DIM.
    """
    if not texts:
        return []

    conn = _get_conn()

    keys = [_cache_key(t, task_type) for t in texts]
    cached = _load_from_cache(conn, keys)

    results: list[list[float]] = [[] for _ in texts]
    to_embed: list[tuple[int, str, str]] = []

    for i, (text, key) in enumerate(zip(texts, keys)):
        if key in cached:
            results[i] = cached[key]
        else:
            to_embed.append((i, text, key))

    if to_embed:
        logger.debug(
            "embed_texts: %d total, %d cached, %d to embed locally via FastEmbed",
            len(texts),
            len(texts) - len(to_embed),
            len(to_embed),
        )

        new_cache: dict[str, list[float]] = {}
        for batch_start in range(0, len(to_embed), _BATCH_SIZE):
            batch = to_embed[batch_start : batch_start + _BATCH_SIZE]
            batch_texts = [t for _, t, _ in batch]

            raw_embeddings = _call_embed_api(batch_texts, task_type)

            for (idx, _text, key), vec in zip(batch, raw_embeddings):
                normed = _l2_normalize(vec)
                results[idx] = normed
                new_cache[key] = normed

        _save_to_cache(conn, new_cache)

    conn.close()
    return results


def embed_query(text: str) -> list[float]:
    """Embed a single query string.

    Args:
        text: Search query string.

    Returns:
        L2-normalised float vector.
    """
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunk strings.

    Args:
        texts: List of chunk strings.

    Returns:
        List of L2-normalised float vectors.
    """
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
