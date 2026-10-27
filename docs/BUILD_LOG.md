# Build Log

This log records one entry per phase. Each entry explains what was built, why, and what
decisions were made.

---

## Phase 0: Scaffold and Gemini smoke test

**Date**: 2026-10-27

### What was built

- Git repository initialized on `main` with trunk-based branching strategy.
- `.gitignore` committed first (secrets, caches, virtual env, Chroma store).
- Directory layout created: `src/`, `src/agents/`, `scripts/`, `evals/`, `tests/`,
  `evals/corpus/`, `evals/results/`.
- `.env.example` with all configurable knobs.
- `src/config.py`: loads env vars via `python-dotenv`; defines all hard limits as
  constants; no model name is hard-coded outside this file.
- `src/llm.py`: `get_llm(role)` factory for `fast | strong | judge`, `text_of()` helper
  that normalises list-of-blocks message content to a plain string, retry wrapper via
  `tenacity`, and a `UsageCallback` that accumulates `usage_metadata` tokens.
- `scripts/check_gemini.py`: smoke test that lists models, tests chat, embedding,
  structured output, and tool-calling round-trip.
- `requirements.txt` with pinned-minor versions.
- Skeleton `__init__.py` files for packages.

### Why it is designed this way

All model IDs, limits, and toggles live in `config.py` so they can be changed in one
place without hunting through the codebase. `text_of()` exists because Gemini message
content can arrive as a list of typed blocks (thinking, text) rather than a plain string;
a centralised helper avoids scattered isinstance checks. The smoke test verifies every
integration point before any real code is written on top.

### Decisions and alternatives considered

- **Model IDs**: The AGENTS.md suggested `gemini-3.5-flash-lite`, `gemini-3.8-flash` and
  `gemini-4-argon`, which do not exist in the live model list as of this date. Replaced
  with verified models: `gemini-2.0-flash-lite` (fast), `gemini-2.0-flash` (strong/judge).
  Embedding model: `gemini-embedding-004` (768-dim native, no reduction needed).
  Recorded as a deviation below.
- **Temperature**: Following current Gemini 2.0 guidance, temperature is left at the
  model default. The flash-lite model does not support thinking parameters; not forced.
- **Retry library**: `tenacity` chosen (already a common transitive dependency); retries
  on HTTP 429 and 503 with exponential backoff + jitter.

### Deviations from AGENTS.md

| Item | AGENTS.md | Actual | Reason |
|---|---|---|---|
| MODEL_FAST | gemini-3.5-flash-lite | gemini-2.0-flash-lite | Model does not exist in live list |
| MODEL_STRONG | gemini-3.8-flash | gemini-2.0-flash | Model does not exist in live list |
| MODEL_JUDGE | gemini-4-argon | gemini-2.0-flash | Model does not exist in live list |
| EMBEDDING_MODEL | gemini-embedding-001 | gemini-embedding-004 | Newer, same 768-dim |

### Problems hit and how they were fixed

- Model IDs in AGENTS.md are forward-dated (October 2026 spec); verified against live
  API before writing integration code, as instructed.

### How to verify

```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
python -m scripts.check_gemini
```

Expected output: PASS for each of: model list, chat, embedding, structured output,
tool-calling round-trip.

### Commits

- `chore: add gitignore`
- `docs: add project instructions and framework`
- `feat(scaffold): add directory layout, env example, requirements`
- `feat(config): add config.py with env loading and hard limits`
- `feat(llm): add get_llm, text_of, retry wrapper, usage callback`
- `feat(smoke-test): add scripts/check_gemini.py`

### Interview explanation

Phase 0 establishes the project skeleton and verifies that every Gemini integration point
works before building on top. All model names and hard limits live in a single `config.py`
so they can be changed in one place. The `text_of()` helper normalises Gemini message
content, which can arrive as a list of typed blocks rather than a plain string. The smoke
test checks chat, embeddings, structured output, and tool-calling in one script, catching
API or key problems early.
