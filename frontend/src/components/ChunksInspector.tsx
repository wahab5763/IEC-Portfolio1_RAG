import React, { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronUp, Layers } from "lucide-react";
import { RetrievedChunkItem } from "../lib/api";

interface ChunksInspectorProps {
  chunks: RetrievedChunkItem[];
  meta: { count: number; model: string; requestId: string } | null;
  highlightedChunkId: string | null;
}

function ScoreBar({
  label,
  value,
  colorClass,
}: {
  label: string;
  value: number;
  colorClass: string;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(value * 100)));
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[10.5px] font-mono">
        <span className="text-slate-500 dark:text-slate-400">{label}</span>
        <span className="font-bold text-slate-800 dark:text-slate-200">{value.toFixed(4)}</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
        <div className={`h-full rounded-full transition-all ${colorClass}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ChunksInspector({
  chunks,
  meta,
  highlightedChunkId,
}: ChunksInspectorProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const cardRefs = useRef<Record<string, HTMLElement | null>>({});

  useEffect(() => {
    if (highlightedChunkId && cardRefs.current[highlightedChunkId]) {
      cardRefs.current[highlightedChunkId]?.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
      setExpanded((prev) => ({ ...prev, [highlightedChunkId]: true }));
    }
  }, [highlightedChunkId]);

  return (
    <aside className="bg-white dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800 rounded-2xl flex flex-col min-h-0 overflow-hidden shadow-sm">
      <div className="px-4 py-3.5 border-b border-slate-200/80 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-blue-600 dark:text-blue-400" />
          <h2 className="text-sm font-bold tracking-tight">Retrieved Chunks</h2>
        </div>
        <span className="text-[11px] font-bold px-2.5 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400">
          {chunks.length} fetched
        </span>
      </div>

      {meta && (
        <div className="px-4 py-2 border-b border-slate-200/80 dark:border-slate-800 bg-slate-50/80 dark:bg-slate-950/60 text-[11px] font-mono text-slate-500 flex flex-wrap items-center justify-between gap-2">
          <span className="truncate">{meta.model}</span>
          <span className="text-slate-400">{meta.requestId}</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-3 space-y-2.5">
        {chunks.length === 0 ? (
          <div className="text-center py-12 px-4">
            <Layers className="w-8 h-8 text-slate-300 dark:text-slate-700 mx-auto mb-2" />
            <p className="text-xs font-medium text-slate-500 leading-relaxed">
              Submit a question to inspect retrieved chunks, dense cosine similarity, RRF fusion,
              and Cohere rerank scores.
            </p>
          </div>
        ) : (
          chunks.map((c, idx) => {
            const isHighlighted = highlightedChunkId === c.chunk_id;
            const isExpanded = expanded[c.chunk_id] ?? idx === 0;
            return (
              <article
                key={c.chunk_id}
                ref={(el) => {
                  cardRefs.current[c.chunk_id] = el;
                }}
                className={`p-3 rounded-xl border transition-all space-y-2.5 ${
                  isHighlighted
                    ? "border-blue-500 bg-blue-50/40 dark:bg-blue-950/30 ring-2 ring-blue-500/25"
                    : "border-slate-200/80 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60"
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="w-5 h-5 rounded-md bg-blue-600 text-white text-[10px] font-bold grid place-items-center shrink-0">
                        {idx + 1}
                      </span>
                      <span className="text-xs font-bold truncate" title={c.filename}>
                        {c.filename}
                      </span>
                    </div>
                    <div className="text-[11px] text-slate-500 mt-0.5 truncate">
                      Page/Unit {c.page_number} · {c.section_title || "General"}
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() =>
                      setExpanded((prev) => ({ ...prev, [c.chunk_id]: !isExpanded }))
                    }
                    className="p-1 rounded-md text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 shrink-0"
                    aria-label="Toggle chunk text"
                  >
                    {isExpanded ? (
                      <ChevronUp className="w-4 h-4" />
                    ) : (
                      <ChevronDown className="w-4 h-4" />
                    )}
                  </button>
                </div>

                {/* Visual Score Bars */}
                <div className="grid grid-cols-2 gap-2.5">
                  <ScoreBar
                    label="Cosine"
                    value={c.similarity_score}
                    colorClass="bg-emerald-500"
                  />
                  {c.rerank_score !== null ? (
                    <ScoreBar
                      label="Rerank"
                      value={c.rerank_score}
                      colorClass="bg-blue-500"
                    />
                  ) : (
                    <div className="text-[10.5px] font-mono text-slate-400 flex items-end justify-end">
                      Dense-only mode
                    </div>
                  )}
                </div>

                {c.rrf_score !== null && (
                  <div className="text-[10.5px] font-mono text-slate-400">
                    RRF (k=60): {c.rrf_score.toFixed(5)} · ID: {c.chunk_id.slice(0, 8)}
                  </div>
                )}

                {isExpanded && (
                  <pre className="text-[11.5px] leading-relaxed whitespace-pre-wrap break-words max-h-48 overflow-y-auto p-2.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/80 dark:border-slate-800 text-slate-700 dark:text-slate-300">
                    {c.content}
                  </pre>
                )}
              </article>
            );
          })
        )}
      </div>
    </aside>
  );
}
