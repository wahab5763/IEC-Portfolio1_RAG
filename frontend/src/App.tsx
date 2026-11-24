import React, { useCallback, useEffect, useState } from "react";
import {
  Activity,
  ExternalLink,
  LogOut,
  Moon,
  Sparkles,
  Sun,
} from "lucide-react";
import { AuthScreen } from "./components/AuthScreen";
import { ChatMessage, ChatStage } from "./components/ChatStage";
import { ChunksInspector } from "./components/ChunksInspector";
import { CorpusSidebar } from "./components/CorpusSidebar";
import {
  ApiError,
  AuthResponse,
  DocumentItem,
  HealthData,
  MetricsData,
  RetrievedChunkItem,
  askQuestion,
  deleteDocument,
  fetchDocuments,
  fetchHealth,
  fetchMetrics,
  loadSampleCorpus,
  uploadDocument,
} from "./lib/api";

const SAMPLE_CORPUS_MARKERS = new Set([
  "01_company_hr_policy.pdf",
  "02_engineering_handbook.pdf",
  "03_remote_work_policy_2025.docx",
  "04_remote_work_addendum_2026.docx",
  "05_incident_runbook.md",
  "06_security_guidelines.txt",
  "07_q3_product_roadmap.pptx",
  "08_regional_sales_2025.csv",
]);

export default function App() {
  const [dark, setDark] = useState(() => localStorage.getItem("rag_ui_theme") === "dark");
  const [token, setToken] = useState<string | null>(() => localStorage.getItem("rag_jwt"));
  const [userEmail, setUserEmail] = useState<string | null>(() => localStorage.getItem("rag_email"));

  const [health, setHealth] = useState<HealthData | null>(null);
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [showMetrics, setShowMetrics] = useState(false);

  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [docTotal, setDocTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [uploading, setUploading] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);
  const [sampleLoadedFlag, setSampleLoadedFlag] = useState(false);
  const [uploadBanner, setUploadBanner] = useState<{
    type: "ok" | "err" | "info";
    text: string;
  } | null>(null);

  const [mode, setMode] = useState<"hybrid_rerank" | "dense_only">("hybrid_rerank");
  const [topK, setTopK] = useState(5);
  const [asking, setAsking] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [inspectedChunks, setInspectedChunks] = useState<RetrievedChunkItem[]>([]);
  const [inspectedMeta, setInspectedMeta] = useState<{
    count: number;
    model: string;
    requestId: string;
  } | null>(null);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
    localStorage.setItem("rag_ui_theme", dark ? "dark" : "light");
  }, [dark]);

  const handleLogout = useCallback(() => {
    localStorage.removeItem("rag_jwt");
    localStorage.removeItem("rag_email");
    setToken(null);
    setUserEmail(null);
    setDocs([]);
    setDocTotal(0);
    setSampleLoadedFlag(false);
    setMessages([]);
    setInspectedChunks([]);
    setInspectedMeta(null);
    setHighlightedChunkId(null);
  }, []);

  const handleAuthenticated = useCallback((res: AuthResponse) => {
    localStorage.setItem("rag_jwt", res.access_token);
    localStorage.setItem("rag_email", res.email);
    setToken(res.access_token);
    setUserEmail(res.email);
  }, []);

  const refreshTelemetry = useCallback(async () => {
    try {
      const [h, m] = await Promise.all([fetchHealth(), fetchMetrics()]);
      setHealth(h);
      setMetrics(m);
    } catch {
      setHealth(null);
    }
  }, []);

  const refreshDocs = useCallback(async () => {
    if (!token) return;
    try {
      const data = await fetchDocuments(token, page, 10);
      setDocs(data.items);
      setDocTotal(data.total);
    } catch (err) {
      if ((err as ApiError).status === 401) handleLogout();
    }
  }, [token, page, handleLogout]);

  useEffect(() => {
    refreshTelemetry();
  }, [refreshTelemetry]);

  useEffect(() => {
    if (token) refreshDocs();
  }, [token, refreshDocs]);

  const handleUploadFiles = async (files: File[]) => {
    if (!token) return;
    setUploading(true);
    let ingestedCount = 0;
    let totalChunks = 0;

    try {
      for (const file of files) {
        setUploadBanner({
          type: "info",
          text: `Ingesting ${file.name} (${ingestedCount + 1}/${files.length})...`,
        });
        try {
          const res = await uploadDocument(token, file);
          ingestedCount += 1;
          totalChunks += res.chunk_count;
        } catch (err) {
          const apiErr = err as ApiError;
          if (apiErr.status === 401) {
            handleLogout();
            return;
          }
          setUploadBanner({
            type: "err",
            text: `${file.name} → HTTP ${apiErr.status} [${apiErr.error_code}]: ${apiErr.message}`,
          });
          await Promise.all([refreshDocs(), refreshTelemetry()]);
          return;
        }
      }
      setUploadBanner({
        type: "ok",
        text:
          files.length === 1
            ? `Ingested ${files[0].name} (${totalChunks} chunks).`
            : `Ingested ${ingestedCount} documents (${totalChunks} total chunks).`,
      });
      setPage(1);
      await Promise.all([refreshDocs(), refreshTelemetry()]);
    } finally {
      setUploading(false);
    }
  };

  const handleLoadSampleCorpus = async () => {
    if (!token) return;
    setLoadingSample(true);
    setUploadBanner({ type: "info", text: "Loading 6-format sample corpus into your workspace..." });
    try {
      const res = await loadSampleCorpus(token);
      setSampleLoadedFlag(true);
      setUploadBanner({
        type: "ok",
        text:
          res.loaded_count > 0
            ? `Loaded ${res.loaded_count} sample documents (${res.total_chunks} chunks). Sample test prompts unlocked!`
            : "Sample corpus is already loaded in your workspace.",
      });
      setPage(1);
      await Promise.all([refreshDocs(), refreshTelemetry()]);
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 401) handleLogout();
      setUploadBanner({
        type: "err",
        text: `Failed to load sample corpus (${apiErr.status}): ${apiErr.message}`,
      });
    } finally {
      setLoadingSample(false);
    }
  };

  const handleDelete = async (docId: string) => {
    if (!token) return;
    try {
      await deleteDocument(token, docId);
      setUploadBanner({ type: "ok", text: "Document & all associated vector chunks deleted (204)." });
      await Promise.all([refreshDocs(), refreshTelemetry()]);
    } catch (err) {
      const apiErr = err as ApiError;
      setUploadBanner({ type: "err", text: `Delete failed (${apiErr.status}): ${apiErr.message}` });
    }
  };

  const handleSelectMessage = (msg: ChatMessage) => {
    if (!msg.response) return;
    setActiveMessageId(msg.id);
    setInspectedChunks(msg.response.retrieved_chunks);
    setInspectedMeta({
      count: msg.response.retrieved_count,
      model: msg.response.model,
      requestId: msg.response.request_id,
    });
  };

  const handleSubmitQuestion = async (trimmed: string) => {
    if (!token) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", text: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setAsking(true);
    setHighlightedChunkId(null);

    try {
      const res = await askQuestion(token, trimmed, topK, mode);
      const assistantId = crypto.randomUUID();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        text: res.answer,
        response: res,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      setActiveMessageId(assistantId);
      setInspectedChunks(res.retrieved_chunks);
      setInspectedMeta({
        count: res.retrieved_count,
        model: res.model,
        requestId: res.request_id,
      });
      await refreshTelemetry();
    } catch (err) {
      const apiErr = err as ApiError;
      if (apiErr.status === 401) handleLogout();
      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "error", text: apiErr.message, error: apiErr },
      ]);
      setInspectedChunks([]);
      setInspectedMeta(null);
      await refreshTelemetry();
    } finally {
      setAsking(false);
    }
  };

  const totalPages = Math.max(1, Math.ceil(docTotal / 10));
  const hasSampleCorpus =
    (sampleLoadedFlag && docTotal > 0) || docs.some((d) => SAMPLE_CORPUS_MARKERS.has(d.filename));

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 dark:bg-[#090d16] text-slate-900 dark:text-slate-100">
      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-30 border-b border-slate-200/80 dark:border-slate-800/90 bg-white/85 dark:bg-slate-900/85 backdrop-blur-md px-5 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white grid place-items-center shadow-sm">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h1 className="text-sm font-extrabold tracking-tight leading-none">
              Production RAG Service
            </h1>
            <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
              Neon pgvector · Cohere Embed &amp; Rerank · Groq LLM
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs relative">
          {/* DB Status */}
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-semibold ${
              health?.pgvector_enabled
                ? "bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border border-emerald-500/20"
                : "bg-red-500/10 text-red-700 dark:text-red-300 border border-red-500/20"
            }`}
          >
            <span className="w-1.5 h-1.5 rounded-full bg-current" />
            {health ? `pgvector ${health.pgvector_version}` : "DB Offline"}
          </span>

          {/* Telemetry Popover Button */}
          {metrics && (
            <div className="relative">
              <button
                type="button"
                onClick={() => setShowMetrics((s) => !s)}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 hover:bg-slate-200/80 dark:hover:bg-slate-700 text-slate-700 dark:text-slate-200 font-semibold transition-colors"
              >
                <Activity className="w-3.5 h-3.5 text-blue-500" />
                <span>Telemetry</span>
              </button>

              {showMetrics && (
                <div className="absolute right-0 mt-2 w-72 p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xl z-40 space-y-2.5 animate-fade-up">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold">Live Observability</span>
                    <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-500">
                      Langfuse: {metrics.langfuse_enabled ? "ON" : "Local"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px] font-mono">
                    <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-slate-400">p50 Latency</div>
                      <div className="font-bold text-sm">{metrics.p50_latency_ms}ms</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-slate-400">p95 Latency</div>
                      <div className="font-bold text-sm">{metrics.p95_latency_ms}ms</div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-slate-400">Error Rate</div>
                      <div className="font-bold text-sm">
                        {(metrics.error_rate * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-2 rounded-lg bg-slate-50 dark:bg-slate-950 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-slate-400">Est. Cost</div>
                      <div className="font-bold text-sm">
                        ${metrics.estimated_cost_usd.toFixed(4)}
                      </div>
                    </div>
                  </div>
                  <div className="text-[11px] text-slate-500 font-mono flex justify-between pt-1 border-t border-slate-200/70 dark:border-slate-800">
                    <span>Reqs: {metrics.total_requests}</span>
                    <span>
                      Tokens: {metrics.total_prompt_tokens + metrics.total_completion_tokens}
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}

          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-blue-500 font-semibold transition-colors"
          >
            API Docs <ExternalLink className="w-3 h-3" />
          </a>

          <button
            type="button"
            onClick={() => setDark((d) => !d)}
            className="p-1.5 rounded-lg border border-slate-200 dark:border-slate-800 hover:border-blue-500 transition-colors"
            title="Toggle Dark / Light mode"
          >
            {dark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
          </button>

          {token && userEmail && (
            <button
              type="button"
              onClick={handleLogout}
              className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-red-50 dark:hover:bg-red-950/60 hover:text-red-600 dark:hover:text-red-400 font-semibold transition-colors"
              title="Sign out"
            >
              <span className="max-w-[140px] truncate">{userEmail}</span>
              <LogOut className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </header>

      {/* Main Content */}
      {!token ? (
        <AuthScreen health={health} onAuthenticated={handleAuthenticated} />
      ) : (
        <main className="flex-1 grid grid-cols-1 lg:grid-cols-[310px_minmax(0,1.4fr)_370px] gap-4 p-4 max-w-[1720px] w-full mx-auto lg:h-[calc(100vh-55px)]">
          <CorpusSidebar
            docs={docs}
            total={docTotal}
            page={page}
            totalPages={totalPages}
            uploading={uploading}
            loadingSample={loadingSample}
            hasSampleCorpus={hasSampleCorpus}
            banner={uploadBanner}
            onUploadFiles={handleUploadFiles}
            onLoadSampleCorpus={handleLoadSampleCorpus}
            onDelete={handleDelete}
            onRefresh={refreshDocs}
            onPageChange={setPage}
          />

          <ChatStage
            messages={messages}
            activeMessageId={activeMessageId}
            hasSampleCorpus={hasSampleCorpus}
            hasAnyDocs={docTotal > 0}
            mode={mode}
            topK={topK}
            asking={asking}
            onModeChange={setMode}
            onTopKChange={setTopK}
            onSubmitQuestion={handleSubmitQuestion}
            onSelectMessage={handleSelectMessage}
            onHighlightChunk={(cid) => setHighlightedChunkId(cid)}
            onClearChat={() => {
              setMessages([]);
              setActiveMessageId(null);
              setInspectedChunks([]);
              setInspectedMeta(null);
              setHighlightedChunkId(null);
            }}
          />

          <ChunksInspector
            chunks={inspectedChunks}
            meta={inspectedMeta}
            highlightedChunkId={highlightedChunkId}
          />
        </main>
      )}
    </div>
  );
}
