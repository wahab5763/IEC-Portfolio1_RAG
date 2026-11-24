/**
 * Production RAG Service — Phase 1 Vanilla JS Client
 * Consumes /api/v1/health, /api/v1/documents, and /api/v1/ask over clean REST contracts.
 */

const API_BASE = "/api/v1";

const state = {
  page: 1,
  limit: 10,
  total: 0,
  selectedFile: null,
};

// DOM References
const healthBadge = document.getElementById("healthBadge");
const healthText = document.getElementById("healthText");
const themeToggle = document.getElementById("themeToggle");

const uploadForm = document.getElementById("uploadForm");
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const selectedFileRow = document.getElementById("selectedFileRow");
const selectedFileName = document.getElementById("selectedFileName");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const docTotalBadge = document.getElementById("docTotalBadge");
const docList = document.getElementById("docList");
const refreshDocsBtn = document.getElementById("refreshDocsBtn");
const prevPageBtn = document.getElementById("prevPageBtn");
const nextPageBtn = document.getElementById("nextPageBtn");
const pageIndicator = document.getElementById("pageIndicator");

const chatFeed = document.getElementById("chatFeed");
const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const charCounter = document.getElementById("charCounter");
const askBtn = document.getElementById("askBtn");
const modeSelect = document.getElementById("modeSelect");
const topKSelect = document.getElementById("topKSelect");

const chunkCountBadge = document.getElementById("chunkCountBadge");
const inspectorMeta = document.getElementById("inspectorMeta");
const chunksContainer = document.getElementById("chunksContainer");

/* --- Theme Persistence --- */
const savedTheme = localStorage.getItem("rag_ui_theme");
if (savedTheme === "dark") {
  document.documentElement.setAttribute("data-theme", "dark");
  themeToggle.textContent = "Light Mode";
}
themeToggle.addEventListener("click", () => {
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  if (isDark) {
    document.documentElement.removeAttribute("data-theme");
    localStorage.setItem("rag_ui_theme", "light");
    themeToggle.textContent = "Dark Mode";
  } else {
    document.documentElement.setAttribute("data-theme", "dark");
    localStorage.setItem("rag_ui_theme", "dark");
    themeToggle.textContent = "Light Mode";
  }
});

/* --- Helpers --- */
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text ?? "";
  return div.innerHTML;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function showUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.className = `status-banner status-${type}`;
  uploadStatus.classList.remove("hidden");
}

/* --- Health Check --- */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    if (res.ok) {
      healthBadge.className = "health-badge health-ok";
      healthText.textContent = `Postgres + pgvector ${data.pgvector_version}`;
    } else {
      healthBadge.className = "health-badge health-err";
      healthText.textContent = data.message || "DB Unavailable (503)";
    }
  } catch {
    healthBadge.className = "health-badge health-err";
    healthText.textContent = "API Offline";
  }
}

/* --- Document List & Deletion --- */
async function loadDocuments() {
  try {
    const res = await fetch(`${API_BASE}/documents?page=${state.page}&limit=${state.limit}`);
    const data = await res.json();
    if (!res.ok) {
      docList.innerHTML = `<li class="empty-state">${escapeHtml(data.message || "Failed to load documents")}</li>`;
      return;
    }

    state.total = data.total;
    docTotalBadge.textContent = `${data.total} doc${data.total === 1 ? "" : "s"}`;

    const totalPages = Math.max(1, Math.ceil(data.total / state.limit));
    pageIndicator.textContent = `Page ${state.page} / ${totalPages}`;
    prevPageBtn.disabled = state.page <= 1;
    nextPageBtn.disabled = state.page >= totalPages;

    if (!data.items.length) {
      docList.innerHTML = `<li class="empty-state">No documents uploaded yet.</li>`;
      return;
    }

    docList.innerHTML = data.items
      .map(
        (doc) => `
      <li class="doc-item">
        <div class="doc-item-top">
          <span class="doc-name truncate" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
          <span class="fmt-badge">${escapeHtml(doc.file_type)}</span>
        </div>
        <div class="doc-meta">
          <span>${doc.chunk_count} chunks &middot; ${formatBytes(doc.file_size_bytes)}</span>
          <button class="btn-danger" type="button" data-delete-id="${escapeHtml(doc.id)}">Delete</button>
        </div>
      </li>`
      )
      .join("");
  } catch {
    docList.innerHTML = `<li class="empty-state">Unable to reach API server.</li>`;
  }
}

docList.addEventListener("click", async (evt) => {
  const btn = evt.target.closest("[data-delete-id]");
  if (!btn) return;
  const docId = btn.getAttribute("data-delete-id");
  btn.disabled = true;
  btn.textContent = "Deleting...";
  try {
    const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, {
      method: "DELETE",
    });
    if (res.status === 204) {
      showUploadStatus("Document and all associated vector chunks deleted (204).", "ok");
      await loadDocuments();
    } else {
      const err = await res.json();
      showUploadStatus(`Delete failed (${res.status}): ${err.message}`, "err");
      btn.disabled = false;
      btn.textContent = "Delete";
    }
  } catch {
    showUploadStatus("Network error while deleting document.", "err");
    btn.disabled = false;
    btn.textContent = "Delete";
  }
});

refreshDocsBtn.addEventListener("click", loadDocuments);
prevPageBtn.addEventListener("click", () => {
  if (state.page > 1) {
    state.page -= 1;
    loadDocuments();
  }
});
nextPageBtn.addEventListener("click", () => {
  state.page += 1;
  loadDocuments();
});

/* --- File Upload --- */
function setSelectedFile(file) {
  state.selectedFile = file;
  if (file) {
    selectedFileName.textContent = `${file.name} (${formatBytes(file.size)})`;
    selectedFileRow.classList.remove("hidden");
  } else {
    selectedFileRow.classList.add("hidden");
  }
}

fileInput.addEventListener("change", () => setSelectedFile(fileInput.files[0] || null));

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) {
    setSelectedFile(e.dataTransfer.files[0]);
  }
});

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.selectedFile) return;

  uploadBtn.disabled = true;
  uploadBtn.textContent = "Ingesting...";
  showUploadStatus(`Extracting, chunking & embedding ${state.selectedFile.name}...`, "info");

  const formData = new FormData();
  formData.append("file", state.selectedFile);

  try {
    const res = await fetch(`${API_BASE}/documents`, { method: "POST", body: formData });
    const data = await res.json();
    if (res.status === 201) {
      showUploadStatus(
        `Ingested ${data.filename}: ${data.chunk_count} chunks in ${data.latency_ms} ms (201).`,
        "ok"
      );
      setSelectedFile(null);
      fileInput.value = "";
      state.page = 1;
      await loadDocuments();
    } else {
      showUploadStatus(
        `HTTP ${res.status} [${data.error_code}]: ${data.message} (${data.request_id})`,
        "err"
      );
    }
  } catch {
    showUploadStatus("Network error uploading document.", "err");
  } finally {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Ingest";
  }
});

/* --- Q&A Chat & Retrieved Chunks Inspector --- */
questionInput.addEventListener("input", () => {
  charCounter.textContent = `${questionInput.value.length} / 1000`;
});

document.querySelectorAll(".chip[data-q]").forEach((chip) => {
  chip.addEventListener("click", () => {
    questionInput.value = chip.getAttribute("data-q");
    charCounter.textContent = `${questionInput.value.length} / 1000`;
    questionInput.focus();
  });
});

function appendUserMessage(text) {
  const el = document.createElement("div");
  el.className = "msg msg-user";
  el.innerHTML = `<div class="msg-body">${escapeHtml(text)}</div>`;
  chatFeed.appendChild(el);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendAssistantResponse(data) {
  const el = document.createElement("div");
  el.className = "msg msg-assistant";

  const refusalHtml = data.refused
    ? `<span class="refusal-tag">REFUSED &middot; ${escapeHtml(data.refusal_reason || "insufficient_evidence")}</span>`
    : "";

  const citationsHtml =
    data.citations && data.citations.length
      ? `<div class="citations-bar">${data.citations
          .map(
            (c) =>
              `<span class="cite-pill">${escapeHtml(c.filename)} &middot; p.${c.page_number}</span>`
          )
          .join("")}</div>`
      : "";

  el.innerHTML = `
    ${refusalHtml}
    <div class="msg-body">${escapeHtml(data.answer)}</div>
    ${citationsHtml}
    <div class="msg-meta">
      <span>Chunks: <strong>${data.retrieved_count}</strong></span>
      <span>Retrieval: ${data.retrieval_latency_ms} ms</span>
      <span>LLM: ${data.generation_latency_ms} ms</span>
      <span>Total: ${data.total_latency_ms} ms</span>
      <span>ID: ${escapeHtml(data.request_id)}</span>
    </div>
  `;
  chatFeed.appendChild(el);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function appendErrorMessage(status, data) {
  const el = document.createElement("div");
  el.className = "msg msg-error";
  el.innerHTML = `
    <strong>HTTP ${status} &middot; ${escapeHtml(data.error_code || "ERROR")}</strong>
    <div class="msg-body">${escapeHtml(data.message || "Request failed.")}</div>
    <div class="msg-meta"><span>Request ID: ${escapeHtml(data.request_id || "n/a")}</span></div>
  `;
  chatFeed.appendChild(el);
  chatFeed.scrollTop = chatFeed.scrollHeight;
}

function renderChunksInspector(data) {
  const chunks = data.retrieved_chunks || [];
  chunkCountBadge.textContent = `${chunks.length} chunk${chunks.length === 1 ? "" : "s"}`;

  inspectorMeta.classList.remove("hidden");
  inspectorMeta.innerHTML = `
    <span>Fetched: <strong>${data.retrieved_count}</strong></span>
    <span>Model: <strong>${escapeHtml(data.model)}</strong></span>
    <span>Req: <code>${escapeHtml(data.request_id)}</code></span>
  `;

  if (!chunks.length) {
    chunksContainer.innerHTML = `<div class="empty-state">No chunks cleared the relevance threshold for this query.</div>`;
    return;
  }

  chunksContainer.innerHTML = chunks
    .map(
      (c, idx) => `
    <article class="chunk-card">
      <div class="chunk-head">
        <div>
          <div class="chunk-title">#${idx + 1} &middot; ${escapeHtml(c.filename)}</div>
          <div class="chunk-sub">Page/Unit ${c.page_number} &middot; ${escapeHtml(c.section_title || "General")}</div>
        </div>
      </div>
      <div class="score-row">
        <span class="score-pill score-primary">Cosine: ${Number(c.similarity_score).toFixed(4)}</span>
        ${c.rerank_score !== null && c.rerank_score !== undefined ? `<span class="score-pill score-primary">Rerank: ${Number(c.rerank_score).toFixed(4)}</span>` : ""}
        ${c.rrf_score !== null && c.rrf_score !== undefined ? `<span class="score-pill">RRF: ${Number(c.rrf_score).toFixed(5)}</span>` : ""}
      </div>
      <pre class="chunk-text">${escapeHtml(c.content)}</pre>
    </article>`
    )
    .join("");
}

askForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  appendUserMessage(question);
  questionInput.value = "";
  charCounter.textContent = "0 / 1000";
  askBtn.disabled = true;
  askBtn.textContent = "Thinking...";

  try {
    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_k: Number(topKSelect.value),
        mode: modeSelect.value,
      }),
    });
    const data = await res.json();
    if (res.ok) {
      appendAssistantResponse(data);
      renderChunksInspector(data);
    } else {
      appendErrorMessage(res.status, data);
      chunkCountBadge.textContent = "0 chunks";
      chunksContainer.innerHTML = `<div class="empty-state">Request blocked or failed (${res.status}).</div>`;
    }
  } catch {
    appendErrorMessage(0, { error_code: "NETWORK_ERROR", message: "Could not reach /api/v1/ask." });
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask Question";
  }
});

// Initial bootstrap
checkHealth();
loadDocuments();
