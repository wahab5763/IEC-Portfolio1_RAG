-- ==============================================================================
-- Production RAG Service — PostgreSQL + pgvector Schema Migration (Idempotent)
-- Target: Neon Serverless Postgres (PostgreSQL 15+ with pgvector extension)
-- Design: Single-store relational + dense vector (HNSW) + sparse FTS (GIN)
-- ==============================================================================

-- 1. Enable pgvector extension for native vector storage and cosine HNSW search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Users Table (Supports Phase 1 single-tenant default & Phase 2 multi-tenant JWT)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Seed default system user for Phase 1 single-tenant mode (idempotent)
INSERT INTO users (id, email, password_hash)
VALUES ('00000000-0000-0000-0000-000000000000', 'system@local.internal', 'DISABLED')
ON CONFLICT (email) DO NOTHING;

-- 3. Documents Table (Metadata, SHA-256 deduplication, cascade parent)
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    file_hash CHAR(64) NOT NULL,
    file_type TEXT NOT NULL,              -- Normalized format: pdf | docx | txt | md | pptx | csv
    file_size_bytes BIGINT NOT NULL,
    page_count INT DEFAULT 1,             -- Pages (PDF), slides (PPTX), row batches (CSV), or 1
    chunk_count INT DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_user_file_hash UNIQUE (user_id, file_hash)
);

-- 4. Chunks Table (Text, 1024-dim vector, generated stored tsvector, JSONB provenance)
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_number INT DEFAULT 1,            -- Page (PDF), slide (PPTX), batch index (CSV), or 1
    section_title TEXT,                   -- Heading breadcrumb, slide title, or CSV row range
    content TEXT NOT NULL,
    token_count INT NOT NULL,
    embedding vector(1024),               -- Enforces Cohere embed-v4.0 output_dimension=1024
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- 5. Indexes for Vector, Full-Text, and Multi-Tenant Relational Queries

-- HNSW Index for Dense Vector Search (Cosine distance operator class)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw
ON chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- GIN Index for Sparse Full-Text Search ('simple' preserves exact codes/multilingual tokens)
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin
ON chunks USING gin (tsv);

-- Relational B-Tree Indexes for Cascade Deletion & Multi-Tenant Filtering
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_user_id ON chunks (user_id);
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents (user_id);
