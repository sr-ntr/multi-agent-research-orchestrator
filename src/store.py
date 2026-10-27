"""
KnowledgeBase: Chroma vector store + BM25 index with hybrid search.

Public API:
    KnowledgeBase(persist_dir)  -- create or open a knowledge base
    kb.add_chunks(chunks)       -- upsert a list of Chunk dicts
    kb.delete_source(source)    -- remove all chunks from a source file
    kb.list_documents()         -- return list of {source, chunk_count, doc_hash}
    kb.get_chunk(chunk_id)      -- return a single chunk dict or None
    kb.get_neighbors(chunk_id, window)  -- adjacent chunks for context
    kb.search(query, top_k, source, mode)  -- hybrid/vector/bm25 search

Chunk dict schema:
    {
        "chunk_id":    str,    # "<filename-slug>:<zero-padded-index>"
        "source":      str,    # original filename (basename)
        "heading":     str,    # heading path, e.g. "Risk Factors > Competition"
        "chunk_index": int,    # 0-based index within the source document
        "doc_hash":    str,    # sha1 of file content (for idempotency)
        "text":        str,    # plain chunk text (displayed to users)
        "embed_text":  str,    # context header + text (what gets embedded)
    }

Search result dict schema:
    {
        "chunk_id": str,
        "source":   str,
        "heading":  str,
        "text":     str,
        "score":    float,
    }
"""

import logging
import uuid
from pathlib import Path
from typing import Literal, Optional

import chromadb  # type: ignore

from src.config import VECTOR_FETCH, BM25_FETCH, TOP_K_DEFAULT, RRF_K
from src.embeddings import embed_documents, embed_query
from src.retrieval import build_bm25, bm25_search, reciprocal_rank_fusion

logger = logging.getLogger(__name__)

SearchMode = Literal["vector", "bm25", "hybrid"]


class KnowledgeBase:
    """Hybrid knowledge base combining Chroma (vector) and BM25 (keyword) search.

    Supports both persistent (disk) and ephemeral (in-memory) Chroma clients.
    Ephemeral collections are given unique names to avoid test collisions.

    Args:
        persist_dir: Path to a directory for persistent Chroma storage.
            Pass None for an ephemeral (in-memory) instance.
        collection_name: Override the Chroma collection name. Only set this
            when you need a specific name; omit for unique auto-naming.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection_name = collection_name or "doc_research"
        else:
            self._client = chromadb.EphemeralClient()
            # Unique name so parallel tests never share state
            self._collection_name = collection_name or f"doc_research_{uuid.uuid4().hex[:8]}"

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 is rebuilt from the current Chroma state on each instantiation.
        self._rebuild_bm25()

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _rebuild_bm25(self) -> None:
        """Rebuild the in-memory BM25 index from all chunks in Chroma."""
        result = self._collection.get(include=["documents", "metadatas"])
        ids = result["ids"]
        docs = result["documents"] or []
        metadatas = result["metadatas"] or []

        if not ids:
            self._bm25_ids: list[str] = []
            self._bm25_corpus: list[str] = []
            self._bm25_metas: list[dict] = []
            self._bm25_index = None
            return

        # BM25 corpus = heading + text (richer signal than text alone)
        corpus = []
        for doc, meta in zip(docs, metadatas):
            heading = (meta or {}).get("heading", "")
            corpus.append(f"{heading}\n{doc}" if heading else doc)

        self._bm25_ids = ids
        self._bm25_corpus = corpus
        self._bm25_metas = metadatas
        self._bm25_index = build_bm25(corpus)
        logger.debug("BM25 index rebuilt with %d documents", len(ids))

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[dict]) -> int:
        """Upsert a list of chunk dicts into the knowledge base.

        Embedding is computed on the ``embed_text`` field (context header +
        chunk text). The ``text`` field (plain chunk) is stored as the Chroma
        document for display. All other fields go into metadata.

        Args:
            chunks: List of chunk dicts as described in the module docstring.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        embed_texts_list = [c["embed_text"] for c in chunks]
        vectors = embed_documents(embed_texts_list)

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [
            {
                "source": c["source"],
                "heading": c.get("heading", ""),
                "chunk_index": c["chunk_index"],
                "doc_hash": c["doc_hash"],
            }
            for c in chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )

        self._rebuild_bm25()
        logger.info("Upserted %d chunks", len(chunks))
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a given source file.

        Args:
            source: The source filename (basename) as stored in metadata.

        Returns:
            Number of chunks deleted.
        """
        result = self._collection.get(where={"source": source}, include=[])
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
            self._rebuild_bm25()
            logger.info("Deleted %d chunks from source=%s", len(ids), source)
        return len(ids)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """Return a summary of all ingested documents.

        Returns:
            List of dicts with keys: source, chunk_count, doc_hash.
        """
        result = self._collection.get(include=["metadatas"])
        metadatas = result["metadatas"] or []

        docs: dict[str, dict] = {}
        for meta in metadatas:
            source = (meta or {}).get("source", "unknown")
            doc_hash = (meta or {}).get("doc_hash", "")
            if source not in docs:
                docs[source] = {"source": source, "chunk_count": 0, "doc_hash": doc_hash}
            docs[source]["chunk_count"] += 1

        return list(docs.values())

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a single chunk by its ID.

        Args:
            chunk_id: The chunk_id string.

        Returns:
            A chunk dict or None if not found.
        """
        result = self._collection.get(
            ids=[chunk_id], include=["documents", "metadatas"]
        )
        if not result["ids"]:
            return None
        meta = (result["metadatas"] or [{}])[0] or {}
        return {
            "chunk_id": chunk_id,
            "source": meta.get("source", ""),
            "heading": meta.get("heading", ""),
            "chunk_index": meta.get("chunk_index", 0),
            "doc_hash": meta.get("doc_hash", ""),
            "text": (result["documents"] or [""])[0],
        }

    def get_neighbors(self, chunk_id: str, window: int = 1) -> list[dict]:
        """Return adjacent chunks (context window) for a given chunk.

        Fetches chunks with chunk_index in [index-window, index+window] from
        the same source file, excluding the chunk itself.

        Args:
            chunk_id: The central chunk ID.
            window: Number of chunks to fetch on each side.

        Returns:
            List of chunk dicts sorted by chunk_index.
        """
        center = self.get_chunk(chunk_id)
        if center is None:
            return []

        source = center["source"]
        center_idx = int(center["chunk_index"])
        min_idx = max(0, center_idx - window)
        max_idx = center_idx + window

        result = self._collection.get(
            where={
                "$and": [
                    {"source": {"$eq": source}},
                    {"chunk_index": {"$gte": min_idx}},
                    {"chunk_index": {"$lte": max_idx}},
                ]
            },
            include=["documents", "metadatas"],
        )

        neighbors = []
        for cid, doc, meta in zip(
            result["ids"],
            result["documents"] or [],
            result["metadatas"] or [],
        ):
            if cid == chunk_id:
                continue
            meta = meta or {}
            neighbors.append(
                {
                    "chunk_id": cid,
                    "source": meta.get("source", ""),
                    "heading": meta.get("heading", ""),
                    "chunk_index": meta.get("chunk_index", 0),
                    "doc_hash": meta.get("doc_hash", ""),
                    "text": doc,
                }
            )

        return sorted(neighbors, key=lambda c: c["chunk_index"])

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _resolve_source(self, source: Optional[str]) -> Optional[str]:
        """Resolve a query source string to an actual stored source filename."""
        if not source:
            return None
        cleaned = source.strip().lower()
        docs = self.list_documents()
        known_sources = [d["source"] for d in docs]

        # Exact match
        for s in known_sources:
            if s.lower() == cleaned:
                return s

        # Stem / slug match
        c_stem = Path(cleaned).stem.replace("_", "-")
        for s in known_sources:
            s_stem = Path(s.lower()).stem.replace("_", "-")
            if s_stem == c_stem:
                return s

        # Substring match
        for s in known_sources:
            if cleaned in s.lower() or s.lower() in cleaned:
                return s

        return source

    def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        source: Optional[str] = None,
        mode: SearchMode = "hybrid",
    ) -> list[dict]:
        """Search the knowledge base and return ranked chunks.

        Args:
            query: The search query string.
            top_k: Number of results to return after fusion/ranking.
            source: If set, restrict results to this source file.
            mode: "vector", "bm25", or "hybrid" (default).

        Returns:
            List of result dicts: {chunk_id, source, heading, text, score}.
            Sorted descending by score.
        """
        canonical_source = self._resolve_source(source)
        where = {"source": canonical_source} if canonical_source else None

        vector_ids: list[str] = []
        bm25_ids: list[str] = []

        if mode in ("vector", "hybrid"):
            vector_ids = self._vector_search(query, VECTOR_FETCH, where)

        if mode in ("bm25", "hybrid"):
            bm25_ids = self._bm25_search(query, BM25_FETCH, canonical_source)

        if mode == "vector":
            ranked = [(cid, 1.0 / (RRF_K + rank)) for rank, cid in enumerate(vector_ids, 1)]
        elif mode == "bm25":
            ranked = [(cid, 1.0 / (RRF_K + rank)) for rank, cid in enumerate(bm25_ids, 1)]
        else:
            ranked = reciprocal_rank_fusion([vector_ids, bm25_ids], k=RRF_K)

        top_ids = [cid for cid, _ in ranked[:top_k]]
        score_map = {cid: score for cid, score in ranked}

        if not top_ids:
            return []

        fetch_result = self._collection.get(
            ids=top_ids, include=["documents", "metadatas"]
        )

        results = []
        id_to_idx = {cid: i for i, cid in enumerate(fetch_result["ids"])}

        for chunk_id in top_ids:
            if chunk_id not in id_to_idx:
                continue
            i = id_to_idx[chunk_id]
            meta = (fetch_result["metadatas"] or [{}])[i] or {}
            text = (fetch_result["documents"] or [""])[i]
            results.append(
                {
                    "chunk_id": chunk_id,
                    "source": meta.get("source", ""),
                    "heading": meta.get("heading", ""),
                    "text": text,
                    "score": score_map.get(chunk_id, 0.0),
                }
            )

        return results

    def _vector_search(
        self, query: str, n: int, where: Optional[dict]
    ) -> list[str]:
        """Run a vector (cosine) search and return ranked chunk IDs."""
        total = self._collection.count()
        if total == 0:
            return []

        query_vec = embed_query(query)
        kwargs: dict = {
            "query_embeddings": [query_vec],
            "n_results": min(n, total),
            "include": ["distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._collection.query(**kwargs)
        return (result["ids"] or [[]])[0]

    def _bm25_search(self, query: str, n: int, source: Optional[str]) -> list[str]:
        """Run BM25 search with optional source filtering."""
        if self._bm25_index is None:
            return []

        if source:
            target = source.lower()
            target_stem = Path(target).stem.replace("_", "-")
            # Filter BM25 corpus to the source, then search
            filtered = []
            for cid, corp, meta in zip(self._bm25_ids, self._bm25_corpus, getattr(self, "_bm25_metas", [])):
                m_src = (meta or {}).get("source", "").lower()
                m_stem = Path(m_src).stem.replace("_", "-")
                cid_slug = cid.split(":")[0].lower()
                if (
                    m_src == target
                    or target in m_src
                    or m_src in target
                    or target_stem == m_stem
                    or target_stem == cid_slug
                    or target in cid.lower()
                ):
                    filtered.append((cid, corp))

            if not filtered:
                return []
            # Re-run BM25 on filtered subset
            filtered_ids = [f[0] for f in filtered]
            filtered_corpus = [f[1] for f in filtered]
            filtered_index = build_bm25(filtered_corpus)
            results = bm25_search(filtered_index, filtered_ids, query, top_k=n)
        else:
            results = bm25_search(self._bm25_index, self._bm25_ids, query, top_k=n)

        return [cid for cid, _ in results]
