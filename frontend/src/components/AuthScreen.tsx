import React, { useState } from "react";
import {
  Database,
  Eye,
  EyeOff,
  FileStack,
  Lock,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { ApiError, AuthResponse, HealthData, authLogin, authSignup } from "../lib/api";

interface AuthScreenProps {
  health: HealthData | null;
  onAuthenticated: (res: AuthResponse) => void;
}

export function AuthScreen({ health, onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPass, setShowPass] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res =
        mode === "signup"
          ? await authSignup(email.trim(), password)
          : await authLogin(email.trim(), password);
      onAuthenticated(res);
    } catch (err) {
      const apiErr = err as ApiError;
      setError(`${apiErr.error_code}: ${apiErr.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex-1 relative overflow-hidden flex items-center justify-center p-6">
      {/* Subtle ambient gradient backdrop */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,_rgba(37,99,235,0.12),_transparent_50%),radial-gradient(ellipse_at_bottom_right,_rgba(124,58,237,0.10),_transparent_50%)]"
      />

      <div className="relative z-10 w-full max-w-5xl grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        {/* Left Brand & Architecture Showcase */}
        <div className="hidden lg:flex flex-col gap-6 pr-4">
          <div className="inline-flex items-center gap-2 self-start px-3 py-1 rounded-full text-xs font-semibold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
            <Sparkles className="w-3.5 h-3.5" />
            Production RAG · Hybrid Retrieval + Guardrails
          </div>

          <div>
            <h2 className="text-3xl font-extrabold tracking-tight leading-tight text-slate-900 dark:text-white">
              Ask anything across your documents — with verifiable citations.
            </h2>
            <p className="mt-3 text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
              Ingest PDFs, Word docs, slide decks, Markdown, plain text, and CSV datasets into a
              multi-tenant Neon Postgres + pgvector workspace with live chunk score inspection.
            </p>
          </div>

          <div className="space-y-3">
            <div className="flex items-start gap-3 p-3.5 rounded-xl bg-white/75 dark:bg-slate-900/75 border border-slate-200/80 dark:border-slate-800 backdrop-blur-sm">
              <div className="p-2 rounded-lg bg-teal-500/10 text-teal-600 dark:text-teal-400">
                <FileStack className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold">Universal 6-Format Ingestion</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  PDF, DOCX, TXT, MD, PPTX &amp; CSV with header-repeated tables and SHA-256 deduplication.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3.5 rounded-xl bg-white/75 dark:bg-slate-900/75 border border-slate-200/80 dark:border-slate-800 backdrop-blur-sm">
              <div className="p-2 rounded-lg bg-blue-500/10 text-blue-600 dark:text-blue-400">
                <Database className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold">Hybrid HNSW + FTS + Cohere Rerank</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  Dense 1024-dim semantic vectors fused with sparse keyword search via RRF (k=60) and reranked.
                </div>
              </div>
            </div>

            <div className="flex items-start gap-3 p-3.5 rounded-xl bg-white/75 dark:bg-slate-900/75 border border-slate-200/80 dark:border-slate-800 backdrop-blur-sm">
              <div className="p-2 rounded-lg bg-amber-500/10 text-amber-600 dark:text-amber-400">
                <ShieldCheck className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold">Multi-Layer Guardrails &amp; Tenant Isolation</div>
                <div className="text-xs text-slate-500 dark:text-slate-400">
                  JWT Bearer isolation, XML prompt fencing, strict refusal gates, and citation ID verification.
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <span
              className={`w-2 h-2 rounded-full ${
                health?.pgvector_enabled ? "bg-emerald-500" : "bg-red-500"
              }`}
            />
            {health?.pgvector_enabled
              ? `Connected to Neon Serverless Postgres (pgvector ${health.pgvector_version})`
              : "Connecting to Neon Postgres..."}
          </div>
        </div>

        {/* Right Auth Card */}
        <div className="w-full max-w-md mx-auto bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border border-slate-200/90 dark:border-slate-800 rounded-2xl p-7 shadow-xl shadow-slate-900/5 dark:shadow-black/30">
          <div className="flex items-center gap-2.5 mb-2">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-600 to-indigo-600 text-white grid place-items-center shadow-sm">
              <Lock className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-lg font-bold leading-tight">
                {mode === "login" ? "Welcome back" : "Create your workspace"}
              </h2>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                {mode === "login"
                  ? "Sign in to access your private document corpus"
                  : "Passwords are secured with Argon2id hashing"}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-1.5 p-1 rounded-xl bg-slate-100 dark:bg-slate-800/80 my-5 text-xs font-semibold">
            <button
              type="button"
              onClick={() => {
                setMode("login");
                setError(null);
              }}
              className={`py-2 rounded-lg transition-all ${
                mode === "login"
                  ? "bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 shadow-sm"
                  : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
            >
              Sign In
            </button>
            <button
              type="button"
              onClick={() => {
                setMode("signup");
                setError(null);
              }}
              className={`py-2 rounded-lg transition-all ${
                mode === "signup"
                  ? "bg-white dark:bg-slate-900 text-blue-600 dark:text-blue-400 shadow-sm"
                  : "text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
              }`}
            >
              Create Account
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                Email Address
              </label>
              <input
                type="email"
                required
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@company.com"
                className="w-full px-3.5 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 transition-all"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 mb-1.5">
                Password {mode === "signup" && <span className="font-normal text-slate-400">(min 8 chars)</span>}
              </label>
              <div className="relative">
                <input
                  type={showPass ? "text" : "password"}
                  required
                  minLength={mode === "signup" ? 8 : 1}
                  autoComplete={mode === "signup" ? "new-password" : "current-password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full pl-3.5 pr-10 py-2.5 text-sm rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-950 focus:outline-none focus:ring-2 focus:ring-blue-500/30 focus:border-blue-500 transition-all"
                />
                <button
                  type="button"
                  onClick={() => setShowPass((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
                  aria-label="Toggle password visibility"
                >
                  {showPass ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {error && (
              <div className="p-3 rounded-xl bg-red-50 dark:bg-red-950/60 border border-red-200 dark:border-red-900 text-red-700 dark:text-red-300 text-xs font-medium">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 text-white text-sm font-semibold shadow-md shadow-blue-600/20 transition-all"
            >
              {loading
                ? "Authenticating..."
                : mode === "signup"
                  ? "Create Account & Enter Workspace"
                  : "Sign In to Workspace"}
            </button>
          </form>
        </div>
      </div>
    </main>
  );
}
