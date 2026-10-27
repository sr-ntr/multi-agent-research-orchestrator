# Agentic Document Research Assistant

A portfolio project demonstrating RAG, multi-agent orchestration, and evaluation,
powered by Google Gemini.

## What it does

Upload `.txt` or `.md` files, ask a question, and receive a cited answer. A team of
LLM agents plans the work, searches the documents (and optionally the web) with tools,
writes an answer where every factual sentence carries a citation, and an independent
verifier checks each claim against the retrieved evidence.

## Architecture

- **Ingestion**: heading-aware chunking, Gemini embeddings (cached), Chroma + BM25 hybrid search
- **Agents**: Planner, parallel Researchers, Writer, Verifier (LangGraph)
- **Evaluation**: deterministic retrieval metrics, LLM judges, three-way ablation

## Local setup

```bash
# 1. Clone
git clone <repo-url>
cd doc-research-agent

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Smoke test (verify Gemini connectivity)
python -m scripts.check_gemini

# 6. Ingest documents
python -m src.ingest evals/corpus --persist .chroma

# 7. Ask a question (CLI)
python -m src.cli ask "What is the main product?" --persist .chroma

# 8. Launch the UI
streamlit run app.py
```

## Evaluation

```bash
# Retrieval metrics (no agent calls)
python -m evals.run_retrieval_eval

# Agent eval (dry run, 3 questions)
python -m evals.run_evals --suite core --mode single --limit 3

# Full ablation
python -m evals.run_evals --suite core --mode all

# Generate summary report
python -m evals.report
```

## Results

*Not yet run*

## Failure analysis

*Not yet run*

## Future work

- Guardrails and safety harness (prompt-injection test cases, output filters, PII redaction)
- A reranker (cross-encoder or LLM) measured in the same eval tables
- MCP exposure of the tools
- `fetch_url` with sanitisation
- Gemini built-in Google Search grounding as an alternative web tool
- PDF and DOCX ingestion
- Tracing (Langfuse or OpenTelemetry)
- CI job that runs evals on pull requests
- Conversation memory
- Larger, multi-annotator golden sets with confidence intervals via repeated runs
