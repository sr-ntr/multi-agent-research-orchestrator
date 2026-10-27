# Agentic Document Research Assistant

A complete, production-grade retrieval-augmented generation (RAG) and multi-agent document research system powered by Google Gemini.

The system ingests markdown and plain text files, indexes them using hybrid retrieval (dense vector embeddings fused with sparse BM25 keyword matching), and coordinates an orchestrated team of specialized LLM agents to plan, research, write, and verify grounded answers with verifiable citations.

---

## Architecture Overview

```
 Upload (.txt/.md)
      |
      v
  Ingestion --> Heading-aware chunking --> Gemini Embeddings (SQLite cache)
                                                    |
                                                    v
                                      Chroma DB (Cosine) + BM25 Index
                                                    ^
                                                    | Read-only tools:
                                                    |   search_documents (Hybrid + RRF)
                                                    |   get_chunk_context (Adjacent chunks)
                                                    |   list_documents
                                                    |   calculate (Safe AST)
                                                    |   web_search (DuckDuckGo / Tavily)
                                                    |
 User Question --> [Planner] --fan-out--> [Parallel Researchers] --> [Writer] --> [Verifier]
                      |                         |                         ^             |
                      | Decomposes complex      | Bounded tool loop;      |             |
                      | questions into          | captures evidence into  +-- revise ---+ (max 1)
                      | sub-questions with      | shared LEDGER                         |
                      | scope (doc / web)                                               v
                                                                             Cited Answer + Trace
```

### Core Components

1. **Heading-Aware Ingestion (`src/ingest.py`)**:
   - Parses Markdown hierarchy (`#` to `###`) to preserve section context (e.g., `Risk Factors > Competition`).
   - Splits on paragraph boundaries with character budgeting (700-900 chars, 100-150 char overlap).
   - Prepends heading context to embeddings while storing display text separately.
   - Computes SHA1 file hashes for idempotent upserts.

2. **Hybrid Retrieval with Reciprocal Rank Fusion (`src/store.py`, `src/retrieval.py`)**:
   - Computes dense vector embeddings with `gemini-embedding-004` (L2-normalized) cached in SQLite.
   - Chroma vector store (cosine distance) retrieves top candidates.
   - `BM25Okapi` index matches exact terminology, product codes, and acronyms.
   - Reciprocal Rank Fusion (RRF, k=60) merges rankings to maximize retrieval recall.

3. **Multi-Agent Orchestration with LangGraph (`src/agents/`)**:
   - **Planner (`planner_node`)**: Categorizes complexity and decomposes queries into independent `SubQuestion` items with scope (`documents`, `web`, or `both`).
   - **Parallel Researchers (`researcher_node`)**: Hand-written, bounded tool loop (max 6 calls). Appends retrieved passages to an immutable `Evidence` ledger.
   - **Writer (`writer_node`)**: Synthesizes findings using exclusively the retrieved evidence IDs. Appends inline `[chunk_id]` citations to every factual sentence.
   - **Verifier (`verifier_node`)**: Deterministically catches hallucinated or un-retrieved citations; runs an independent judge to verify that cited text entails every draft claim.
   - **Finalize (`finalize_node`)**: Honest abstention if no facts were discovered; appends unverified disclaimers if claims fail verification.

---

## Manual Setup and Run Instructions

### 1. Prerequisites

- Python 3.10 or 3.11
- A Google Gemini API Key from Google AI Studio

### 2. Environment Setup

```bash
# Clone the repository
cd /path/to/doc-research-agent

# Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. API Key Configuration

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and set your key:
```env
GOOGLE_API_KEY=AIzaSyYourGeminiApiKeyHere
MODEL_FAST=gemini-2.0-flash-lite
MODEL_STRONG=gemini-2.0-flash
MODEL_JUDGE=gemini-2.0-flash
EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIM=768
WEB_SEARCH_PROVIDER=ddgs
WEB_ENABLED=0
LLM_MAX_CONCURRENCY=3
```

### 4. Smoke Test

Verify your Gemini API key and model connectivity:

```bash
python -m scripts.check_gemini
```
All five checks (model list, chat, embeddings, structured output, tool calls) must report `PASS`.

### 5. Running Offline Tests

Run the complete offline test suite (no API key required):

```bash
python -m pytest
```

### 6. Ingesting Documents

Ingest the sample corpus into a persistent Chroma collection (`.chroma/`):

```bash
python -m src.ingest evals/corpus --persist .chroma
```

### 7. CLI Usage

Ask questions via command-line interface:

- **Single Researcher (fast lookup)**:
  ```bash
  python -m src.cli ask "What was Arcturus Technologies' total revenue in 2023?" --mode single --persist .chroma
  ```

- **Full Multi-Agent Pipeline with Verification**:
  ```bash
  python -m src.cli ask "How did CloudSync Pro product revenue change between 2022 and 2023?" --mode multi --persist .chroma
  ```

- **Honest Abstention (Unanswerable Question)**:
  ```bash
  python -m src.cli ask "What was Arcturus Technologies' stock price on December 31, 2023?" --mode multi --persist .chroma
  ```

- **Web-Augmented Research**:
  ```bash
  python -m src.cli ask "What does GDPR stand for and how does it relate to Arcturus policy?" --mode multi --web on --persist .chroma
  ```

### 8. Streamlit Web Interface

Launch the interactive UI:

```bash
streamlit run app.py
```

Features in the UI:
- Sidebar API key override and configuration switches.
- Drag-and-drop document uploader with chunking status.
- Expandable live execution trace (`st.status`).
- Verified source cards showing exact cited passages and provenance.

---

## Evaluation Harness

The evaluation harness measures retrieval quality, agent faithfulness, correctness against hand-verified references, and architectural ablation.

### 1. Retrieval Benchmark (Vector vs BM25 vs Hybrid)

Evaluates retrieval hit rate at k (k=3, 6) and Mean Reciprocal Rank (MRR) over all answerable core questions:

```bash
python -m evals.run_retrieval_eval --persist .chroma
```
Results are saved to `evals/results/retrieval_eval.json` and `evals/results/retrieval_eval.md`.

### 2. Agent Evaluation (Dry Run)

Run a 3-question dry run to verify the judge models and metric calculations:

```bash
python -m evals.run_evals --suite core --mode single --limit 3 --allow-unverified
```

### 3. Full Architectural Ablation

Runs all 18 core questions across all three configurations:
- `single`: Direct researcher without planning or independent verification.
- `multi_noverify`: Planner with parallel researchers; no verification.
- `multi`: Full pipeline with planner, parallel researchers, and independent verifier.

```bash
python -m evals.run_evals --suite core --mode all --allow-unverified
```

### 4. Web Search Suite

Evaluates external web routing accuracy:

```bash
python -m evals.run_evals --suite web --mode multi --allow-unverified
```

### 5. Generate Summary and Failure Analysis

Compile the results into the final ablation table:

```bash
python -m evals.report
```
Outputs are written to `evals/results/SUMMARY.md` and `evals/results/FAILURE_ANALYSIS.md`.

---

## Evaluation Results

*Evaluation runs are executed locally on the user's Gemini API quota. Run `python -m evals.run_evals` followed by `python -m evals.report` to populate this section from `evals/results/SUMMARY.md`.*

---

## Future Work

The following features were intentionally excluded from the initial scope and represent future directions:
- Guardrails and safety harness (prompt-injection test cases, output filters, PII redaction)
- Re-ranking pipeline (cross-encoder or LLM reranker) evaluated in the benchmark tables
- MCP (Model Context Protocol) tool exposure
- Deep web page fetching with content sanitisation
- Native Google Search grounding via Gemini SDK
- PDF and DOCX document ingestion parsers
- Production tracing with Langfuse or OpenTelemetry
- CI/CD automated eval regression gates on pull requests
- Conversation history and multi-turn session memory
- Multi-annotator golden evaluation sets with statistical confidence intervals
