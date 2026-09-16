import { useState } from "react";
import { searchCorpus } from "./api";

/**
 * Knowledge-base browser.
 *
 * Every figure comes from /search: the relevance percentage is the fusion
 * rank score the retriever actually returned, the timing is measured
 * server-side, and the viewer shows the real indexed chunk text. Nothing on
 * this page is illustrative.
 */
export default function DocumentsPage({ token, role }) {
  const [q, setQ] = useState("");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [err, setErr] = useState(null);

  async function run() {
    const term = q.trim();
    if (!term || !token) return;
    setBusy(true);
    setErr(null);
    try {
      const d = await searchCorpus(token, term, 8);
      setRes(d);
      setSelected(d.results?.[0] || null);
    } catch (e) {
      setErr(e.message);
      setRes(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page">
      <div className="kb-search" style={{ flex: "1 1 auto", marginBottom: 18 }}>
        <span style={{ color: "var(--text-faint)" }}>⌕</span>
        <input
          value={q}
          placeholder="Search knowledge base..."
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
        />
        {busy && <span className="spinner" />}
      </div>

      {err && <div className="empty">Search failed: {err}</div>}

      {res && (
        <>
          <div className="found-line">
            <span>
              Found {res.results.length} source{res.results.length === 1 ? "" : "s"}
              {" · "}{res.took_ms}ms
              {res.confidence != null && ` · best dense match ${res.confidence.toFixed(3)}`}
            </span>
            <span className="verified-badge">⊘ Access scoped to {role || res.role}</span>
          </div>

          {res.results.length === 0 && (
            <div className="empty">
              Nothing in this corpus is readable by role “{res.role}” for that query.
            </div>
          )}

          {res.results.map((r) => (
            <div
              key={r.chunk_id}
              className={`result-card ${selected?.chunk_id === r.chunk_id ? "active" : ""}`}
              onClick={() => setSelected(r)}
            >
              <div className="result-top">
                <div style={{ minWidth: 0 }}>
                  <div className="result-title">{r.file_name || "Untitled"}</div>
                  <div className="result-src">
                    {r.department}{r.sub_department ? ` › ${r.sub_department}` : ""}
                    {r.sensitivity ? ` · ${r.sensitivity}` : ""}
                  </div>
                </div>
                <div className="result-score">
                  <div className="result-pct">{(r.score * 100).toFixed(1)}%</div>
                  <div className="result-conf">fusion rank</div>
                </div>
              </div>
              <div className="result-meta">
                {[
                  r.page != null && `Page ${r.page}`,
                  r.chunk_index != null && `Chunk ${r.chunk_index}`,
                  r.fiscal_year && `FY${r.fiscal_year}${r.fiscal_quarter ? ` Q${r.fiscal_quarter}` : ""}`,
                  r.doc_date && `Dated ${r.doc_date}`,
                ].filter(Boolean).join(" · ")}
              </div>
            </div>
          ))}

          {selected && (
            <div className="viewer">
              <div className="viewer-bar">
                <span>Viewing: {selected.file_name}</span>
                <span>{selected.source_path}</span>
              </div>
              <div className="viewer-body">{selected.text}</div>
            </div>
          )}
        </>
      )}

      {!res && !busy && (
        <div className="empty">
          Search the corpus. Results are filtered to what role “{role}” may read.
        </div>
      )}
    </div>
  );
}
