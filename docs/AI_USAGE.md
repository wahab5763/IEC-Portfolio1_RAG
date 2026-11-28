# AI Usage Transparency Documentation (`docs/AI_USAGE.md`)

In accordance with **Section 8 (AI Usage Policy)** and **Submission Requirement #9** of `PD.pdf`, this document transparently records all AI tools, models, configurations, and the complete chronological list of prompts and chat interactions used to plan, build, test, document, and deploy the **Production RAG Service** (`IEC-Portfolio1_RAG`).

---

## 1. AI Tools & Models Used

### Development & Engineering Assistance
| Tool / Assistant | Model / Capability | Role in Project |
| --- | --- | --- |
| **Google Antigravity IDE** | Gemini (High & Medium reasoning modes) | Architecture review, rubric validation, milestone-by-milestone code implementation, debugging, test authoring, documentation, and deployment setup |
| **Antigravity Image Generator Subagent** | Image Generation | Created the system architecture blueprint diagram (`docs/architecture.png`) |

### Application Runtime AI Models (Inside the RAG Service)
| Component | Provider & Model | Purpose |
| --- | --- | --- |
| **Embeddings** | Cohere `embed-v4.0` (`output_dimension=1024`) | Asymmetric multilingual (`search_document` / `search_query`) 1024-dim vector generation |
| **Cross-Encoder Reranker** | Cohere `rerank-v5.0-fast` | Joint `(query, chunk)` reranking (Top-20 RRF candidates → Top-5) and relevance gating |
| **Grounded Answer LLM** | Groq `openai/gpt-oss-120b` | Cited answer generation, contradiction reporting, multilingual responses, and strict refusal |
| **Evaluation Judge** | RAGAS (`langchain-groq` + `langchain-cohere`) | Faithfulness and Answer Correctness evaluation at `temperature=0.0` |

---

## 2. Relevant Configurations & Settings

- **Runtime Environment**: Windows 11, Python `3.12.10` (`.venv`), Node.js `v24.21.0`, npm `11.19.0`.
- **Linting & Testing (`pyproject.toml`)**:
  - `pytest` (`9.2.0`) + `pytest-asyncio` (`asyncio_mode = "auto"`, `asyncio_default_fixture_loop_scope = "function"`) + `pytest-cov` (`83%` backend coverage across `42` tests).
  - `ruff` (`0.17.6`, `target-version = "py312"`, `line-length = 100`, rules `E, F, I, UP, B, RUF`, ignoring `E501, B008`).
- **Core RAG Hyperparameters (`.env.example` / `backend/app/config.py`)**:
  - Chunking (Config B): `CHUNK_SIZE_TOKENS=400`, `CHUNK_OVERLAP_TOKENS=60` (15%), `tiktoken cl100k_base`, contextual header injection enabled.
  - Retrieval: `RETRIEVAL_CANDIDATE_POOL=20`, `RETRIEVAL_TOP_K=5`, `RRF_K=60`, `SIMILARITY_THRESHOLD=0.35` (Config A), `RERANK_THRESHOLD=0.25` (Config B).
  - Generation & Resilience: `GROQ_TEMPERATURE=0.0`, `GROQ_MAX_TOKENS=1024`, `API_TIMEOUT_SECONDS=30.0`, `API_MAX_RETRIES=3`, `EMBEDDING_BATCH_SIZE=32`, `EMBEDDING_MAX_CONCURRENCY=4`.
  - Security & Multi-Tenancy: `AUTH_REQUIRED=true`, `JWT_ALGORITHM=HS256`, `JWT_EXPIRE_MINUTES=60`, `MAX_FILE_SIZE_MB=20`, `MAX_QUERY_LENGTH=1000`, `MIN_PDF_CHARS_PER_PAGE=20`, `RATE_LIMIT_PER_MINUTE=30`.
- **Cloud Deployment (`Dockerfile`, `render.yaml`, `requirements-prod.txt`)**:
  - 2-stage Docker container (`node:22-alpine` React build → `python:3.12-slim` FastAPI runtime with `libmagic1` and pre-warmed `tiktoken` cache) deployed on Render Free Tier (`https://production-rag-service.onrender.com/`) backed by Neon Serverless Postgres (`pgvector 0.8.7`).

---

## 3. Complete Chronological Prompt Log & Chat History Summary

Every user prompt from the development session is listed verbatim below in chronological order, paired with the exact engineering outcome and verification performed.

### Phase 0: Architecture Design, Universal Format Planning & Rubric Validation

1. **Prompt 1:**
   > `make the project archetecure visually appealing based on our build plan`
   - **Action & Outcome**: Inspected `docs/plan_build.html`, `README.md`, and workspace layout to map all system components.

2. **Prompt 2:**
   > `make the project architecture image in docs folder, should be visually appealing based on our build plan`
   - **Action & Outcome**: Generated a 16:9 widescreen modern architecture diagram (`docs/architecture.png`, `1376×768`) covering Client/FastAPI Gateway, Async Ingestion Pipeline, Neon Postgres + `pgvector` Single-Store Hub, Hybrid Retrieval + RRF + Rerank + Groq LLM, Multi-Layer Guardrails, and Observability/RAGAS/Testing footer. Visually inspected for zero typos.

3. **Prompt 3:**
   > `our corpus can be any document so please design in a way that the system can handle any document like pdf, txt, md, pptx, docs, csv. This is just for our build plan, right now don't code until we final the plan`
   - **Action & Outcome**: Updated `docs/plan_build.html` (Sections 1, 3, 4, 5, 6, 8, 9) with a pluggable `BaseExtractor` + `ExtractorRegistry` architecture supporting all 6 formats in Phase 1 (`.pdf`, `.docx`, `.txt`, `.md`, `.pptx`, `.csv`), including header-repeated CSV row batching and wide-CSV `Column: Value` serialization. Regenerated `docs/architecture.png` and embedded it in `docs/plan_build.html`. Zero application code written.

4. **Prompt 4:**
   > `for testing what we are using? pytest or unit test?`
   - **Action & Outcome**: Confirmed `pytest` (`pytest`, `pytest-asyncio`, `pytest-cov`, `httpx.AsyncClient`) is used to author both unit tests and async API integration tests.

5. **Prompt 5:**
   > `validate the plan`
   - **Action & Outcome**: Audited `docs/plan_build.html` and `requirements.txt` against `PD.pdf` and live Cohere v2 / Groq API specs. Verified `embed-v4.0` (`output_dimension=1024`) and `rerank-v5.0-fast`. Identified 6 concrete gaps/traps: (1) unpinned `requirements.txt`, (2) missing `Recall@k` + LLM-judge reliability check + separate multilingual reporting, (3) explicit off-topic guardrail mapping, (4) missing Swagger screenshots / AI usage / demo video items in Milestone 10, (5) Config A cosine vs Config B rerank score threshold mismatch, and (6) explicit `output_dimension=1024` enforcement + DB rollback on embed failure.

6. **Prompt 6:**
   > `yes patch these 6 items`
   - **Action & Outcome**: Patched all 6 validated items into `docs/plan_build.html`.

7. **Prompt 7:**
   > `for evaluation what are you using? RAGAS?`
   - **Action & Outcome**: Confirmed the 3-part evaluation stack: custom deterministic retrieval harness (`Hit Rate@5`, `Recall@5`, `MRR`, `Refusal Accuracy`), RAGAS LLM-as-a-judge (`Faithfulness`, `Answer Correctness` at `temperature=0`), and Langfuse observability.

---

### Phase 1: Core Mandatory MVP (Milestones 1–10)

8. **Prompt 8:**
   > `Act as a Senior Full-Stack AI/LLM Engineer. Build this project strictly following the build plan. C:\Users\Abdul Wahab\Desktop\AI Portfolio-1\IEC-Portfolio1_RAG\docs\plan_build.html [Rules: Phase 1 First, Production Quality, Full Comments, Step-by-Step] Start with Milestone 1 & 2 (or tell me if you need anything first).`
   - **Action & Outcome**: Installed dependencies into `.venv` and pinned all 41 packages (`==`) in `requirements.txt`. Created `.env.example`, `pyproject.toml`, `scripts/generate_corpus.py` (generating 8 multi-format corpus files in `data/corpus/` + 4 edge-case fixtures in `data/edge_cases/`), `backend/app/config.py` (with automatic Neon `postgresql://` → `postgresql+asyncpg://` + TLS normalization), `backend/app/db/schema.sql`, `models.py`, `session.py`, `migrate.py`, and `tests/test_milestone1_2.py`. Verified **6/6 pytest passed**, **0 Ruff errors**.

9. **Prompt 9:**
   > `please proceed to Milestone 3 (Validation Notebooks)`
   - **Action & Outcome**: Created and executed all 3 Jupyter notebooks (`notebooks/01_extraction_chunking.ipynb`, `02_embeddings_retrieval.ipynb`, `03_generation_eval_prototype.ipynb`) via `jupyter nbconvert --execute --inplace`, validating 6-format extraction, edge-case rejections (`400`/`415`), Config A vs B token stats, RRF ranking, XML fencing, citation stripping, and retrieval metrics.

10. **Prompt 10 & 11:**
    > `please proceed to Milestone 4` / `please continue building the project based on build plan`
    - **Action & Outcome**: Built Milestone 4 (`backend/app/ingestion/`: `exceptions.py`, `validator.py`, pluggable `extractors/` for all 6 formats, `chunker.py`, `embedder.py`, `pipeline.py`, `tests/test_ingestion.py`) and Milestone 5 (`guardrails.py`, `retrieval.py`, `generation.py`, `tests/test_guardrails.py`, `tests/test_retrieval.py`). Verified **29/29 pytest passed**, **0 Ruff errors**.

11. **Prompt 12:**
    > `yes please continue`
    - **Action & Outcome**: Built Milestone 6 (`schemas.py`, `observability.py`, `api/v1/dependencies.py`, `health.py`, `documents.py`, `ask.py`, `router.py`, `main.py`, `tests/conftest.py`, `tests/test_api.py`). Verified **33/33 pytest passed**, **0 Ruff errors**.

12. **Prompt 13:**
    > `Yes move on to next Milestone`
    - **Action & Outcome**: Built Milestone 7 Phase 1 zero-build HTML/CSS/JS frontend (`frontend/index.html`, `styles.css`, `app.js`) with multi-format upload, paginated corpus manager, Q&A chat with citations, and Retrieved Chunks Inspector. Verified **34/34 pytest passed**.

13. **Prompt 14:**
    > `Yes move on to next Milestone`
    - **Action & Outcome**: Built Milestones 8 & 9: 30-question labeled golden set (`evals/golden_dataset.json`: 18 English, 6 unanswerable, 6 Urdu/Arabic multilingual), reproducible benchmark runner (`evals/run_eval.py`), and `tests/test_evals.py`. Executed benchmark showing Config B (`Hit@5=1.0000`, `Recall@5=0.9722`, `MRR=1.0000`) beating Config A (`Hit@5=0.9444`, `MRR=0.9074`). Verified **37/37 pytest passed** (`83%` backend coverage).

14. **Prompt 15:**
    > `Yes move on to next Milestone`
    - **Action & Outcome**: Completed Milestone 10 documentation (`README.md`, `DESIGN.md`, `EVAL.md`).

15. **Prompt 16 & 17:**
    > `lets run the first phase, if we found no issue in it then will move to the next phase` / `I have set the db url and api keys for cohere and groq in .env. lets run the app`
    - **Action & Outcome**: Created `.env` from `.env.example`, ran live Neon Postgres migration (`pgvector 0.8.7`, 3 tables, 5 indexes verified), started FastAPI on port 8000, and executed live end-to-end smoke tests across all 8 corpus files, edge cases (`400`/`409`/`415`), and live Q&A. **Live bug caught & fixed**: `llama-3.3-70b-versatile` returned 404 on the Groq account; queried live `/models` and switched to Groq's active `openai/gpt-oss-120b`. All live factual, contradiction, CSV, refusal, and prompt-injection checks passed.

16. **Prompt 18:**
    > `push the phase1 commit to our git repo`
    - **Action & Outcome**: Verified `.env` excluded by `.gitignore`, committed Phase 1 (`7ea2ac8`, 75 files), and pushed to `origin/dev`.

---

### Phase 2: Extended Production Capabilities, UI Upgrade, Documentation & Cloud Deployment

17. **Prompt 19:**
    > `what components are included in phase2`
    - **Action & Outcome**: Summarized all 8 Phase 2 components mapped to Bonus A–H and noted which were already built in Phase 1.

18. **Prompt 20:**
    > `start developing phase2 component excluding Enterprise Format Plugins & OCR, Advanced Document Guardrails, and deployment righ now`
    - **Action & Outcome**: Built JWT Auth & Multi-Tenancy (`backend/app/auth.py`, `api/v1/auth.py`, `HTTPBearer` dependency enforcing `401`), live operational telemetry & Langfuse tracing (`GET /api/v1/metrics` + `MetricsTracker` + `record_rag_trace`), React + TypeScript + Vite + Tailwind SPA (`frontend/`), and Phase 2 tests (`tests/test_phase2_auth_observability.py`, **42/42 pytest passed**). Also fixed Neon idle-connection handling and added `--rerank-delay 6.2` in `evals/run_eval.py` after catching Cohere's 10 req/min Trial limit live.

19. **Prompt 21:**
    > `Make UI more professional, intuitive and impressive`
    - **Action & Outcome**: Refactored the React SPA into modular components (`AuthScreen.tsx`, `CorpusSidebar.tsx`, `ChatStage.tsx`, `ChunksInspector.tsx`, `App.tsx`) with split-screen auth, multi-file drag-and-drop, clickable inline citation pills linked to highlighted chunk cards, Cosine/Rerank progress bars, animated thinking state, and a live Telemetry popover.

20. **Prompt 22:**
    > `working fine, why documents are not showing which are stored in corpus for other users?`
    - **Action & Outcome**: Explained that Phase 2 JWT multi-tenant isolation (`WHERE user_id = :authenticated_user_id`) intentionally isolates each user's workspace from the Phase 1 system user's corpus per `PD.pdf` Bonus A.

21. **Prompt 23:**
    > `make a button Load Sample Corpus, user can load and ask the sample questions that are pre written as placeholder for testing. the sample questions should not be visible if user does not load sample data because they are of not use if user donot want sample corpus.`
    - **Action & Outcome**: Added `POST /api/v1/documents/load-sample-corpus` (cloning pre-computed embeddings into the logged-in user's isolated `user_id` workspace when cached), added the **Load Sample Corpus** button in `CorpusSidebar.tsx`, and made the 6 sample test prompt cards in `ChatStage.tsx` appear only when sample corpus documents are present.

22. **Prompt 24:**
    > `commit and push the code on git`
    - **Action & Outcome**: Verified `.env` and `node_modules/` excluded, committed Phase 2 (`8d7d055`, 32 files), and pushed to `origin/dev`.

23. **Prompt 25:**
    > `create a doc app_detail pdf where explain each features and file/code/library etc. why we use and how it is better for our use case. make it easy and short explaination for each topic`
    - **Action & Outcome**: Generated `docs/app_detail.pdf` with embedded architecture diagram, feature cards, library comparison table, file-by-file code map, and HTTP status code cheat sheet.

24. **Prompt 26, 27 & 28:**
    > `guide me how we can deploy our project on cloud as a free tier` / `anyother simple options for deployment like hostinger, I have domain available for the project, the deploment should not be costly` / `lets use render, prepare the requirements and push code to dev branch`
    - **Action & Outcome**: Compared Render Free, Koyeb, Vercel+Render, and Hostinger VPS. Created `Dockerfile`, `.dockerignore`, `requirements-prod.txt`, and `render.yaml`, committed (`bf5d976`), and pushed to `origin/dev`.

25. **Prompt 29 & 30:**
    > `[Render Blueprint Settings: IEC-RAG, branch dev]` / `[Failed deploy notification]`
    - **Action & Outcome**: Diagnosed the failed Render Docker build immediately: Python's default `.gitignore` rule `lib/` had silently ignored `frontend/src/lib/api.ts`. Added `!frontend/src/lib/` to `.gitignore`, committed `frontend/src/lib/api.ts` (`8a46bd2`), and pushed to `origin/dev` — Render deploy succeeded and went live at `https://production-rag-service.onrender.com/`.

26. **Prompt 31 & 32:**
    > `in app_detail.pdf, mention the check list at the end, what features are completed and what features are remaining based on the portfolio project requirements.` / `our app is live now on https://production-rag-service.onrender.com/ so update the read me, check list and other parts where required, push the code and merge in the main branch`
    - **Action & Outcome**: Added Section 6 (Completed vs Remaining checklist) to `docs/app_detail.pdf`, updated `README.md` with the live production URLs, updated `render.yaml` to track `main`, committed (`ca888c3`), pushed `dev`, and fast-forward merged `dev` → `main` on `origin/main`.

27. **Prompt 33:**
    > `create a AI Usage Documentation in docs folder, capture all the prompts and document it`
    - **Action & Outcome**: Created this complete AI Usage Transparency report at `docs/AI_USAGE.md`.

28. **Prompt 34:**
    > `now I have added swagger_ss folder in docs atfter saving the screenshots, video is also uploaded on drive, link: https://drive.google.com/file/d/1nQ9cIdGgdxLQNrUzLraaYitz-vItPVuc/view?usp=sharing so update the checklist and other parts where required, then commit and push to main and dev`
    - **Action & Outcome**: Linked all 9 Swagger/Neon screenshots (`docs/swagger_ss/`), `docs/AI_USAGE.md`, and the Project Demo Video Drive URL across `README.md` and `docs/app_detail.pdf` (marking 100% of Mandatory, Bonus A–H, and Submission checklist items as Done), committed, and pushed cleanly to both `main` and `dev`.

---

## 4. Human Review & Understanding Statement

All generated code, architectural choices, and configurations were directed milestone-by-milestone by the project author, verified through:
- Local execution of all 3 Jupyter validation notebooks (`notebooks/01..03`)
- Full automated test suite execution (`42/42` passing `pytest` tests, `83%` coverage) and zero `ruff` lint errors
- Live end-to-end testing against Neon Postgres + Cohere + Groq (local & cloud deployment on Render)
- Manual UI testing across authentication, multi-tenant isolation, 6-format ingestion, Load Sample Corpus, clickable inline citations, Retrieved Chunks Inspector, refusal behavior, and guardrails.
