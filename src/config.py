"""
Configuration for the Agentic Document Research Assistant.

All model names, hard limits, and feature toggles live here.
No model name or limit constant should appear outside this file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from src/)
_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")


# ---------------------------------------------------------------------------
# Model identifiers (Groq Cloud + FastEmbed)
# ---------------------------------------------------------------------------

MODEL_FAST: str = os.environ.get("MODEL_FAST", "openai/gpt-oss-20b")
MODEL_STRONG: str = os.environ.get("MODEL_STRONG", "openai/gpt-oss-120b")
MODEL_JUDGE: str = os.environ.get("MODEL_JUDGE", MODEL_STRONG)

EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "384"))


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------

GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY", "")
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "")


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

WEB_SEARCH_PROVIDER: str = os.environ.get("WEB_SEARCH_PROVIDER", "ddgs")
WEB_ENABLED: bool = os.environ.get("WEB_ENABLED", "0").strip() in ("1", "true", "True")


# ---------------------------------------------------------------------------
# LLM concurrency and retry
# ---------------------------------------------------------------------------

LLM_MAX_CONCURRENCY: int = int(os.environ.get("LLM_MAX_CONCURRENCY", "3"))


# ---------------------------------------------------------------------------
# Hard limits (all agent behaviour is bounded by these)
# ---------------------------------------------------------------------------

MAX_SUBQUESTIONS: int = 4
MAX_TOOL_CALLS_PER_RESEARCHER: int = 6
MAX_REVISIONS: int = 1
QUESTION_TIMEOUT_S: int = 120


# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

CHUNK_TARGET_CHARS: int = 800        # target chunk size in characters
CHUNK_OVERLAP_CHARS: int = 120       # overlap between adjacent paragraph chunks
CHUNK_MIN_CHARS: int = 40            # drop chunks shorter than this
CHUNK_MAX_CHARS: int = 950           # hard cap before forced split


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

TOP_K_DEFAULT: int = 6
RRF_K: int = 60                      # reciprocal rank fusion constant
VECTOR_FETCH: int = 20               # candidates fetched per modality before fusion
BM25_FETCH: int = 20


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

WEB_SNIPPET_MAX_CHARS: int = 500
WEB_MAX_RESULTS: int = 5


# ---------------------------------------------------------------------------
# Session / UI limits
# ---------------------------------------------------------------------------

MAX_FILES_PER_SESSION: int = 10
MAX_FILE_BYTES: int = 1_048_576      # 1 MB
MAX_QUESTIONS_PER_SESSION: int = 15


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------

ABSTAIN_TEXT: str = (
    "The available documents do not contain enough information to answer this question."
)
