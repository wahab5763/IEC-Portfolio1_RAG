# Production RAG Service (`IEC-Portfolio1_RAG`)

A production-grade, code-first Retrieval-Augmented Generation (RAG) service engineered with **FastAPI**, **Neon Serverless Postgres (`pgvector` HNSW + `tsvector` GIN)**, **Cohere Embed v4.0 & Rerank v5.0-fast**, **Groq (`llama-3.3-70b-versatile`)**, multi-layer security guardrails, a zero-build Phase 1 web frontend with a **Retrieved Chunks Inspector**, automated `pytest` suites, and a reproducible evaluation harness.

![Production RAG Service — System Architecture](docs/architecture.png)

---

## Key Capabilities

- **Universal Multi-Format Ingestion (6 Formats)**: Pluggable `BaseExtractor` + `ExtractorRegistry` supporting `.pdf`, `.docx`, `.txt`, `.md`, `.pptx`, and `.csv` with magic-byte/OOXML/UTF-8 signature validation (`415`), size cap (`413`), SHA-256 deduplication (`409`), and scanned-PDF/header-only CSV rejection (`400`).
- **Format-Aware Chunking**: Recursive token splitting (`tiktoken cl100k_base`), atomic/header-repeated Markdown tables, header-repeated CSV row batching (`Column: Value` records for wide CSVs > 12 columns), and contextual metadata header injection (`[Document | Section | Page/Slide/Rows]`).
- **Single-Store Hybrid Retrieval**: Neon Postgres stores relational metadata, 1024-dim dense vectors (`HNSW` `vector_cosine_ops`), and sparse full-text vectors (`GIN` `tsvector('simple')`) in one ACID database (`ON DELETE CASCADE`, `WHERE user_id = $1`). Combines dense + sparse via **Reciprocal Rank Fusion (`k=60`)** and **Cohere `rerank-v5.0-fast`**.
- **Grounded Generation & Multi-Layer Guardrails**: Input length/prompt-injection/jailbreak screening, `<retrieved_context>` XML document fencing with breakout escaping, strict inline citations (`[Chunk ID: <id> | <filename>, p. <page>]`), contradiction reporting, deterministic refusals, outbound citation ID verification, and secret scrubbing.
- **Web Frontend + Retrieved Chunks Inspector**: Served directly at `/` with drag-and-drop multi-format upload, paginated document list/delete, Q&A chat with Config A vs Config B toggle, inline citations, and per-chunk Cosine/RRF/Rerank score inspection.

---

## Project Structure

```text
IEC-Portfolio1_RAG/
├── .env.example                 # Secret-free environment variable contract
├── requirements.txt             # Pinned Python dependencies (==)
├── pyproject.toml               # pytest-asyncio & Ruff configuration
├── README.md                    # Setup, run, test, and evaluation guide
├── DESIGN.md                    # Architecture & justified design decisions
├── EVAL.md                      # Golden set, Config A vs B benchmarks & failure analysis
├── docs/
│   ├── architecture.png         # End-to-end system architecture blueprint
│   └── plan_build.html          # Master architecture & build plan
├── backend/app/
│   ├── config.py                # Pydantic BaseSettings + Neon asyncpg URL normalizer
│   ├── schemas.py               # Strict Pydantic v2 request/response contracts
│   ├── guardrails.py            # Input, XML fencing & output citation/secret guardrails
│   ├── retrieval.py             # Hybrid HNSW + TSVector + RRF + Cohere Rerank
│   ├── generation.py            # Grounded Groq LLM generator & refusal logic
│   ├── observability.py         # X-Request-ID middleware & JSON structured logging
│   ├── main.py                  # FastAPI app, global error handlers & static frontend mount
│   ├── db/                      # schema.sql, models.py, session.py, migrate.py
│   ├── ingestion/               # validator.py, extractors/, chunker.py, embedder.py, pipeline.py
│   └── api/v1/                  # Thin HTTP routes: health.py, documents.py, ask.py, router.py
├── frontend/                    # Phase 1 HTML/CSS/JS UI + Retrieved Chunks Inspector
├── notebooks/                   # Executed Milestone 3 validation notebooks (01, 02, 03)
├── data/                        # 8-file multi-format corpus & 4 edge-case fixtures
├── evals/                       # golden_dataset.json (30 cases), run_eval.py, results/
└── tests/                       # pytest unit & httpx.AsyncClient integration tests
```

---

## Setup & Configuration

### 1. Create Virtual Environment & Install Pinned Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and populate your Neon Postgres, Cohere, and Groq credentials (`.env` is gitignored):

```powershell
Copy-Item .env.example .env
```

Required variables for live execution:
- `DATABASE_URL` — Neon Serverless Postgres connection string (`postgresql://...` is automatically normalized to `postgresql+asyncpg://` with TLS)
- `COHERE_API_KEY` — Cohere v2 API key (`embed-v4.0` + `rerank-v5.0-fast`)
- `GROQ_API_KEY` — Groq API key (`llama-3.3-70b-versatile`)

### 3. Provision Schema & Indexes on Neon Postgres

```powershell
.\.venv\Scripts\python.exe -m backend.app.db.migrate
```

This idempotently enables `pgvector`, creates `users`, `documents`, and `chunks`, seeds the default single-tenant user, creates all 5 indexes (`HNSW`, `GIN`, and 3 B-Trees), and verifies them against PostgreSQL catalogs.

---

## Running the Service

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

- **Web Frontend & Chunk Inspector**: `http://127.0.0.1:8000/`
- **Interactive Swagger API Docs**: `http://127.0.0.1:8000/docs`

### API Endpoints

| Method | Path | Success | Error Codes | Purpose |
| --- | --- | --- | --- | --- |
| `GET` | `/api/v1/health` | `200 OK` | `503` | Verify Postgres & `pgvector` health |
| `POST` | `/api/v1/documents` | `201 Created` | `400`, `409`, `413`, `415`, `503` | Validate, extract, chunk, embed & store |
| `GET` | `/api/v1/documents` | `200 OK` | `422` | Paginated list of uploaded documents |
| `DELETE` | `/api/v1/documents/{id}` | `204 No Content` | `404` | Delete document & cascade all chunks |
| `POST` | `/api/v1/ask` | `200 OK` | `400`, `422`, `429`, `503` | Guardrailed hybrid retrieval & cited Q&A |

---

## Running Tests, Notebooks & Evaluation

```powershell
# Regenerate validation corpus & edge-case fixtures (optional; already committed)
.\.venv\Scripts\python.exe -m scripts.generate_corpus

# Run linter (0 errors)
.\.venv\Scripts\python.exe -m ruff check backend evals scripts tests

# Run full pytest suite with coverage (37 tests · 83% backend coverage)
.\.venv\Scripts\python.exe -m pytest --cov=backend/app --cov-report=term-missing -v

# Run reproducible evaluation harness (Config A vs Config B)
.\.venv\Scripts\python.exe -m evals.run_eval

# Launch JupyterLab to inspect/re-run Milestone 3 validation notebooks
.\.venv\Scripts\python.exe -m jupyter lab
```

See [DESIGN.md](DESIGN.md) for all architecture justifications and [EVAL.md](EVAL.md) for measured benchmark results and failure analysis.
