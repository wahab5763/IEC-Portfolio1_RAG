import React, { useMemo, useState } from "react";
import {
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  FileText,
  FolderPlus,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  XCircle,
} from "lucide-react";
import { DocumentItem } from "../lib/api";

interface CorpusSidebarProps {
  docs: DocumentItem[];
  total: number;
  page: number;
  totalPages: number;
  uploading: boolean;
  loadingSample: boolean;
  hasSampleCorpus: boolean;
  banner: { type: "ok" | "err" | "info"; text: string } | null;
  onUploadFiles: (files: File[]) => Promise<void>;
  onLoadSampleCorpus: () => Promise<void>;
  onDelete: (docId: string) => Promise<void>;
  onRefresh: () => void;
  onPageChange: (nextPage: number) => void;
}

const FORMAT_STYLES: Record<string, string> = {
  pdf: "bg-rose-100 text-rose-700 dark:bg-rose-950/80 dark:text-rose-300",
  docx: "bg-blue-100 text-blue-700 dark:bg-blue-950/80 dark:text-blue-300",
  csv: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/80 dark:text-emerald-300",
  pptx: "bg-amber-100 text-amber-700 dark:bg-amber-950/80 dark:text-amber-300",
  md: "bg-violet-100 text-violet-700 dark:bg-violet-950/80 dark:text-violet-300",
  txt: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function CorpusSidebar({
  docs,
  total,
  page,
  totalPages,
  uploading,
  loadingSample,
  hasSampleCorpus,
  banner,
  onUploadFiles,
  onLoadSampleCorpus,
  onDelete,
  onRefresh,
  onPageChange,
}: CorpusSidebarProps) {
  const [dragOver, setDragOver] = useState(false);
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const filteredDocs = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return docs;
    return docs.filter(
      (d) => d.filename.toLowerCase().includes(q) || d.file_type.toLowerCase().includes(q)
    );
  }, [docs, search]);

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) await onUploadFiles(files);
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    e.target.value = "";
    if (files.length > 0) await onUploadFiles(files);
  };

  const handleDeleteClick = async (id: string) => {
    setDeletingId(id);
    try {
      await onDelete(id);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <aside className="bg-white dark:bg-slate-900/90 border border-slate-200/80 dark:border-slate-800 rounded-2xl flex flex-col min-h-0 overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-4 py-3.5 border-b border-slate-200/80 dark:border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-bold tracking-tight">Knowledge Base</h2>
          <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-600 dark:text-blue-400">
            {total}
          </span>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          title="Refresh corpus"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Multi-File Drag & Drop Zone */}
      <div className="p-3.5 border-b border-slate-200/80 dark:border-slate-800 space-y-2.5">
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`group border-2 border-dashed rounded-xl p-4 text-center cursor-pointer flex flex-col items-center gap-1.5 transition-all ${
            dragOver
              ? "border-blue-500 bg-blue-500/10 scale-[1.01]"
              : "border-slate-200 dark:border-slate-700/80 bg-slate-50/70 dark:bg-slate-950/50 hover:border-blue-400 dark:hover:border-blue-600"
          }`}
        >
          {uploading ? (
            <Loader2 className="w-6 h-6 text-blue-600 animate-spin" />
          ) : (
            <div className="w-9 h-9 rounded-xl bg-blue-500/10 text-blue-600 dark:text-blue-400 grid place-items-center group-hover:scale-105 transition-transform">
              <UploadCloud className="w-5 h-5" />
            </div>
          )}
          <span className="text-xs font-semibold text-slate-800 dark:text-slate-200">
            {uploading ? "Ingesting & Embedding..." : "Drop files or click to ingest"}
          </span>
          <span className="text-[11px] text-slate-500 dark:text-slate-400">
            PDF · DOCX · TXT · MD · PPTX · CSV (max 20MB)
          </span>
          <input
            type="file"
            multiple
            disabled={uploading}
            accept=".pdf,.docx,.txt,.md,.pptx,.csv"
            className="hidden"
            onChange={handleFileChange}
          />
        </label>

        {/* Optional 1-click Sample Corpus Loader */}
        <button
          type="button"
          disabled={loadingSample || uploading || hasSampleCorpus}
          onClick={onLoadSampleCorpus}
          className={`w-full py-2 px-3 rounded-xl text-xs font-semibold inline-flex items-center justify-center gap-1.5 border transition-all ${
            hasSampleCorpus
              ? "border-emerald-200 dark:border-emerald-900 bg-emerald-50/70 dark:bg-emerald-950/40 text-emerald-700 dark:text-emerald-300 cursor-default"
              : "border-blue-200 dark:border-blue-900 bg-blue-50/70 dark:bg-blue-950/40 hover:bg-blue-100/80 dark:hover:bg-blue-950/70 text-blue-700 dark:text-blue-300 disabled:opacity-50"
          }`}
        >
          {loadingSample ? (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              Loading Sample Corpus...
            </>
          ) : hasSampleCorpus ? (
            <>
              <CheckCircle2 className="w-3.5 h-3.5" />
              Sample Corpus Loaded (8 files)
            </>
          ) : (
            <>
              <FolderPlus className="w-3.5 h-3.5" />
              Load Sample Corpus (6 formats)
            </>
          )}
        </button>

        {banner && (
          <div
            className={`flex items-start gap-2 p-2.5 rounded-xl text-xs font-medium animate-fade-up ${
              banner.type === "ok"
                ? "bg-emerald-50 text-emerald-800 border border-emerald-200/70 dark:bg-emerald-950/50 dark:text-emerald-300 dark:border-emerald-900"
                : banner.type === "err"
                  ? "bg-red-50 text-red-800 border border-red-200/70 dark:bg-red-950/50 dark:text-red-300 dark:border-red-900"
                  : "bg-blue-50 text-blue-800 border border-blue-200/70 dark:bg-blue-950/50 dark:text-blue-300 dark:border-blue-900"
            }`}
          >
            {banner.type === "ok" ? (
              <CheckCircle2 className="w-4 h-4 shrink-0 mt-0.5" />
            ) : banner.type === "err" ? (
              <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
            ) : (
              <Loader2 className="w-4 h-4 shrink-0 mt-0.5 animate-spin" />
            )}
            <span className="leading-snug">{banner.text}</span>
          </div>
        )}

        {/* Filter input */}
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter documents..."
            className="w-full pl-8 pr-3 py-1.5 text-xs rounded-lg border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950 focus:outline-none focus:border-blue-500 transition-colors"
          />
        </div>
      </div>

      {/* Document List */}
      <ul className="flex-1 overflow-y-auto p-3 space-y-2">
        {filteredDocs.length === 0 ? (
          <li className="text-center py-10 px-4">
            <FileText className="w-8 h-8 text-slate-300 dark:text-slate-700 mx-auto mb-2" />
            <p className="text-xs font-medium text-slate-500">
              {docs.length === 0
                ? "Upload documents above to start querying."
                : "No documents match your filter."}
            </p>
          </li>
        ) : (
          filteredDocs.map((doc) => (
            <li
              key={doc.id}
              className="group p-2.5 rounded-xl border border-slate-200/80 dark:border-slate-800 bg-slate-50/70 dark:bg-slate-950/60 hover:border-blue-300 dark:hover:border-blue-800 transition-all"
            >
              <div className="flex items-center justify-between gap-2 mb-1.5">
                <span
                  className="text-xs font-semibold truncate text-slate-800 dark:text-slate-200"
                  title={doc.filename}
                >
                  {doc.filename}
                </span>
                <span
                  className={`text-[10px] font-bold uppercase px-1.5 py-0.5 rounded-md shrink-0 ${
                    FORMAT_STYLES[doc.file_type] || FORMAT_STYLES.txt
                  }`}
                >
                  {doc.file_type}
                </span>
              </div>
              <div className="flex items-center justify-between text-[11px] text-slate-500 dark:text-slate-400">
                <span>
                  {doc.chunk_count} chunks · {formatBytes(doc.file_size_bytes)}
                </span>
                <button
                  type="button"
                  disabled={deletingId === doc.id}
                  onClick={() => handleDeleteClick(doc.id)}
                  className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-slate-400 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-950/60 dark:hover:text-red-400 transition-colors disabled:opacity-50"
                  title="Delete document & cascade vectors"
                >
                  {deletingId === doc.id ? (
                    <Loader2 className="w-3 h-3 animate-spin" />
                  ) : (
                    <Trash2 className="w-3 h-3" />
                  )}
                  <span>Delete</span>
                </button>
              </div>
            </li>
          ))
        )}
      </ul>

      {/* Pagination */}
      <div className="px-3.5 py-2.5 border-t border-slate-200/80 dark:border-slate-800 flex items-center justify-between text-xs">
        <button
          type="button"
          disabled={page <= 1}
          onClick={() => onPageChange(page - 1)}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-35 transition-colors"
        >
          <ChevronLeft className="w-3.5 h-3.5" /> Prev
        </button>
        <span className="text-[11px] font-medium text-slate-500">
          Page {page} of {totalPages}
        </span>
        <button
          type="button"
          disabled={page >= totalPages}
          onClick={() => onPageChange(page + 1)}
          className="inline-flex items-center gap-1 px-2 py-1 rounded-lg border border-slate-200 dark:border-slate-800 hover:bg-slate-100 dark:hover:bg-slate-800 disabled:opacity-35 transition-colors"
        >
          Next <ChevronRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </aside>
  );
}
