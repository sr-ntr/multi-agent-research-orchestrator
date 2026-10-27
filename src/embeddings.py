"""
Gemini embedding helper with a sqlite3 on-disk cache.

Public API:
    embed_texts(texts, task_type)   -- embed a list of strings, returns list[list[float]]
    embed_query(text)               -- convenience: single query embedding
    embed_documents(texts)          -- convenience: batch document embeddings

Cache:
    Key = sha1(model + ":" + task_type + ":" + text).
    Stored in .cache/embeddings.db (configurable via EMBEDDING_CACHE_PATH env var).
    Cache is transparent: callers never see cache misses.

Notes:
    - Vectors are L2-normalised after fetching (required for cosine correctness when
      using reduced dimensionality, and good practice regardless).
    - Batching: the Gemini embedding API accepts up to 100 texts per request.
      We batch in chunks of 50 to stay well within the limit.
    - Task types used: RETRIEVAL_DOCUMENT for chunks, RETRIEVAL_QUERY for queries.
"""

import hashlib
import logging
import math
import os
import sqlite3
import json
from pathlib import Path
from typing import Literal

from src.config import EMBEDDING_DIM, EMBEDDING_MODEL, GOOGLE_API_KEY

logger = logging.getLogger(__name__)

TaskType = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY", "SEMANTIC_SIMILARITY"]

_BATCH_SIZE = 50

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
    """L2-normalise a vector in-place equivalent, returning a new list."""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------


def _call_embed_api(texts: list[str], task_type: str) -> list[list[float]]:
    """Invoke the Gemini embedding API using google-genai, google.generativeai, or langchain."""
    # 1. Try google-genai (v1.0+)
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
        client = genai.Client(api_key=GOOGLE_API_KEY)
        model_name = EMBEDDING_MODEL if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL
        config = types.EmbedContentConfig(task_type=task_type) if hasattr(types, "EmbedContentConfig") else None
        res = client.models.embed_content(
            model=model_name,
            contents=texts,
            config=config,
        )
        if hasattr(res, "embeddings") and res.embeddings:
            return [list(emb.values) for emb in res.embeddings]
        elif hasattr(res, "embedding") and res.embedding:
            return [list(res.embedding.values)]
    except (ImportError, Exception) as exc:
        logger.debug("google-genai embedding attempt: %s", exc)

    # 2. Try google-generativeai (legacy)
    try:
        import google.generativeai as legacy_genai  # type: ignore
        legacy_genai.configure(api_key=GOOGLE_API_KEY)
        model_name = f"models/{EMBEDDING_MODEL}" if not EMBEDDING_MODEL.startswith("models/") else EMBEDDING_MODEL
        api_result = legacy_genai.embed_content(
            model=model_name,
            content=texts,
            task_type=task_type,
        )
        raw = api_result["embedding"]
        if isinstance(raw[0], float):
            return [raw]
        return raw
    except (ImportError, Exception) as exc:
        logger.debug("google.generativeai embedding attempt: %s", exc)

    # 3. Fallback to langchain-google-genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # type: ignore
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    return embedder.embed_documents(texts)


def embed_texts(texts: list[str], task_type: TaskType = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """Embed a list of texts using the Gemini embedding model.

    Results are fetched from the sqlite cache when available; new results are
    saved back to the cache before returning.  All vectors are L2-normalised.

    Args:
        texts: List of strings to embed. Empty strings are accepted and will
            produce a zero vector (normalisation is skipped for zero vectors).
        task_type: Gemini task type. Use RETRIEVAL_DOCUMENT for corpus chunks
            and RETRIEVAL_QUERY for search queries.

    Returns:
        List of float vectors, one per input text, each of length EMBEDDING_DIM.
    """
    if not texts:
        return []

    conn = _get_conn()

    keys = [_cache_key(t, task_type) for t in texts]
    cached = _load_from_cache(conn, keys)

    results: list[list[float]] = [[] for _ in texts]
    to_embed: list[tuple[int, str, str]] = []  # (index, text, key)

    for i, (text, key) in enumerate(zip(texts, keys)):
        if key in cached:
            results[i] = cached[key]
        else:
            to_embed.append((i, text, key))

    logger.debug(
        "embed_texts: %d total, %d cached, %d to embed",
        len(texts),
        len(texts) - len(to_embed),
        len(to_embed),
    )

    # Batch the un-cached texts
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
    """Embed a single query string (RETRIEVAL_QUERY task type).

    Args:
        text: The query string.

    Returns:
        A single L2-normalised float vector.
    """
    return embed_texts([text], task_type="RETRIEVAL_QUERY")[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks (RETRIEVAL_DOCUMENT task type).

    Args:
        texts: List of chunk strings.

    Returns:
        List of L2-normalised float vectors.
    """
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")
