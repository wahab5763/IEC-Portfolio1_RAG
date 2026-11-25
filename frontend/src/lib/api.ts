/**
 * Typed REST API client for the Production RAG Service (`/api/v1`).
 * Automatically attaches JWT Bearer tokens for multi-tenant isolation.
 */

export interface HealthData {
  status: string;
  database: string;
  pgvector_enabled: boolean;
  pgvector_version: string;
  request_id: string;
}

export interface MetricsData {
  total_requests: number;
  total_errors: number;
  error_rate: number;
  ask_requests: number;
  p50_latency_ms: number;
  p95_latency_ms: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  estimated_cost_usd: number;
  langfuse_enabled: boolean;
  request_id: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user_id: string;
  email: string;
  request_id: string;
}

export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  page_count: number;
  chunk_count: number;
  status: string;
  created_at: string;
}

export interface DocumentListResponse {
  items: DocumentItem[];
  page: number;
  limit: number;
  total: number;
  request_id: string;
}

export interface CitationItem {
  chunk_id: string;
  filename: string;
  page_number: number;
  section_title: string | null;
}

export interface RetrievedChunkItem {
  chunk_id: string;
  document_id: string;
  filename: string;
  page_number: number;
  section_title: string | null;
  content: string;
  similarity_score: number;
  rrf_score: number | null;
  rerank_score: number | null;
  metadata: Record<string, unknown>;
}

export interface AskResponse {
  request_id: string;
  answer: string;
  citations: CitationItem[];
  retrieved_chunks: RetrievedChunkItem[];
  retrieved_count: number;
  refused: boolean;
  refusal_reason: string | null;
  guardrail_violations: string[];
  retrieval_latency_ms: number;
  generation_latency_ms: number;
  total_latency_ms: number;
  model: string;
}

export interface ApiError {
  status: number;
  error_code: string;
  message: string;
  request_id?: string;
}

const API_BASE = "/api/v1";

function authHeaders(token: string | null): HeadersInit {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function parseOrThrow<T>(res: Response): Promise<T> {
  if (res.status === 204) {
    return undefined as T;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw {
      status: res.status,
      error_code: data.error_code || `HTTP_${res.status}`,
      message: data.message || res.statusText || "Request failed",
      request_id: data.request_id,
    } as ApiError;
  }
  return data as T;
}

export async function fetchHealth(): Promise<HealthData> {
  return parseOrThrow(await fetch(`${API_BASE}/health`));
}

export async function fetchMetrics(): Promise<MetricsData> {
  return parseOrThrow(await fetch(`${API_BASE}/metrics`));
}

export async function authSignup(email: string, password: string): Promise<AuthResponse> {
  return parseOrThrow(
    await fetch(`${API_BASE}/auth/signup`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })
  );
}

export async function authLogin(email: string, password: string): Promise<AuthResponse> {
  return parseOrThrow(
    await fetch(`${API_BASE}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })
  );
}

export async function fetchDocuments(
  token: string | null,
  page = 1,
  limit = 10
): Promise<DocumentListResponse> {
  return parseOrThrow(
    await fetch(`${API_BASE}/documents?page=${page}&limit=${limit}`, {
      headers: authHeaders(token),
    })
  );
}

export async function uploadDocument(token: string | null, file: File) {
  const form = new FormData();
  form.append("file", file);
  return parseOrThrow<{
    document_id: string;
    filename: string;
    chunk_count: number;
    latency_ms: number;
  }>(
    await fetch(`${API_BASE}/documents`, {
      method: "POST",
      headers: authHeaders(token),
      body: form,
    })
  );
}

export async function deleteDocument(token: string | null, docId: string): Promise<void> {
  return parseOrThrow(
    await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
      headers: authHeaders(token),
    })
  );
}

export async function loadSampleCorpus(token: string | null) {
  return parseOrThrow<{
    loaded_count: number;
    skipped_count: number;
    total_chunks: number;
    filenames: string[];
    request_id: string;
  }>(
    await fetch(`${API_BASE}/documents/load-sample-corpus`, {
      method: "POST",
      headers: authHeaders(token),
    })
  );
}

export async function askQuestion(
  token: string | null,
  question: string,
  top_k: number,
  mode: "hybrid_rerank" | "dense_only"
): Promise<AskResponse> {
  return parseOrThrow(
    await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...authHeaders(token),
      },
      body: JSON.stringify({ question, top_k, mode }),
    })
  );
}
