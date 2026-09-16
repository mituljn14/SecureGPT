import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  getRoles, getCapabilities, tokenClaims, fetchStats, uploadFile, sendQuery,
  downloadUrl, acceptAttr, kindOf, agentFromTrajectory, isFullyLocal,
  listSessions, getSession, deleteSession,
} from "./api";
import SecurityPage from "./SecurityPage";
import DocumentsPage from "./DocumentsPage";
import AdminPage from "./AdminPage";
import Markdown from "./Markdown";
import {
  Logo, IconPencil, IconSearch, IconSidebar, IconBook, IconCode, IconImage, IconMic,
  IconShield, IconSliders, IconPlus, IconArrowUp, IconStop, IconChevronDown, IconUpDown,
  IconTrash, IconDots, IconPaperclip, IconCheck, IconCopy, IconClock, IconSparkle,
  IconDownload, IconX, IconFile, IconChevronRight, IconLogout,
} from "./icons";

/* ------------------------------------------------------------------ */
/* Static structure only. Every catalogue (roles, models, departments,  */
/* accepted file types, runtime targets, chat history) is fetched from  */
/* the backend that enforces it -- nothing below is a copy of data.     */
/* ------------------------------------------------------------------ */

const NAV = [
  { id: "documents", label: "Documents", Icon: IconBook },
  { id: "coding", label: "Coding", Icon: IconCode },
  { id: "vision", label: "Vision", Icon: IconImage },
  { id: "meetings", label: "Meetings", Icon: IconMic },
  { id: "security", label: "Security", Icon: IconShield },
  { id: "admin", label: "Admin", Icon: IconSliders },
];
const CHAT_VIEWS = new Set(["chat", "coding", "vision", "meetings"]);

const AUTO_MODEL = "Auto (Supervisor)";
const STAGE_LABELS = { fast: "fast path", crag: "CRAG", multi_agent: "multi-agent" };

/* ---------- small helpers ---------- */

const words = (s = "") => s.replace(/_/g, " ");

/** Two-letter avatar from a role name: "Regional_Managing_Directors" -> "RM". */
function initials(role = "") {
  const parts = role.split(/[_\s]+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return role.slice(0, 2).toUpperCase();
}

/** ChatGPT-style date buckets from the server's updated_at (epoch seconds). */
function groupSessions(sessions) {
  const now = new Date();
  const startOf = (d) => new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const today = startOf(now);
  const day = 86400000;
  const buckets = [
    { label: "Today", test: (t) => t >= today },
    { label: "Yesterday", test: (t) => t >= today - day },
    { label: "Previous 7 days", test: (t) => t >= today - 7 * day },
    { label: "Previous 30 days", test: (t) => t >= today - 30 * day },
    { label: "Older", test: () => true },
  ].map((b) => ({ ...b, items: [] }));
  for (const s of sessions) {
    const t = (s.updated_at || s.created_at || 0) * 1000;
    buckets.find((b) => b.test(t)).items.push(s);
  }
  return buckets.filter((b) => b.items.length);
}

/** Close a popover on outside click / Escape. */
function useDismiss(open, onClose) {
  useEffect(() => {
    if (!open) return;
    const onDown = (e) => { if (!e.target.closest?.("[data-pop]")) onClose(); };
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => { document.removeEventListener("mousedown", onDown); document.removeEventListener("keydown", onKey); };
  }, [open, onClose]);
}

/* ================================================================== */

export default function Workbench({ token, onSignOut }) {
  const [roles, setRoles] = useState([]);
  const [caps, setCaps] = useState(null);
  const [authError, setAuthError] = useState(null);

  // Identity comes from the signed token the backend issued at login; the
  // backend re-validates it on every request, so this is display only.
  const claims = useMemo(() => tokenClaims(token) || {}, [token]);
  const role = claims.role || null;
  const username = claims.sub || null;
  const [stats, setStats] = useState(null);

  const [view, setView] = useState("chat");
  const [mode, setMode] = useState("chat");          // "chat" | "trace"
  const [sideOpen, setSideOpen] = useState(true);
  const [searchOpen, setSearchOpen] = useState(false);
  const [search, setSearch] = useState("");

  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState([]);
  const [attachments, setAttachments] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(AUTO_MODEL);
  const [sessions, setSessions] = useState([]);
  const [sessionId, setSessionId] = useState(null);

  const [roleMenu, setRoleMenu] = useState(false);
  const [modelMenu, setModelMenu] = useState(false);
  const [rowMenu, setRowMenu] = useState(null);      // session id with its "..." open

  const fileRef = useRef(null);
  const endRef = useRef(null);
  const abortRef = useRef(null);

  useDismiss(roleMenu, useCallback(() => setRoleMenu(false), []));
  useDismiss(modelMenu, useCallback(() => setModelMenu(false), []));
  useDismiss(rowMenu != null, useCallback(() => setRowMenu(null), []));

  // Catalogues are retried on a backoff: the backend needs ~90s to import
  // on a cold boot, and a page opened during that window must simply come
  // alive when the server does rather than sit dead until a refresh.
  useEffect(() => {
    let cancelled = false, attempt = 0, timer = null;
    const load = () => {
      attempt += 1;
      getRoles()
        .then((d) => {
          if (cancelled) return;
          setRoles(d.roles || []);
          setAuthError(null);
        })
        .catch((e) => {
          if (cancelled) return;
          const wait = Math.min(2000 * attempt, 10000);
          setAuthError(`Backend not reachable (${e.message}) - retrying in ${wait / 1000}s`);
          timer = setTimeout(load, wait);
        });
      getCapabilities().then((d) => !cancelled && setCaps(d)).catch(() => {});
    };
    load();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    const onErr = (e) => { if (!cancelled && isAuthError(e)) onSignOut(); };
    fetchStats(token).then((s) => !cancelled && setStats(s)).catch(onErr);
    listSessions(token).then((d) => !cancelled && setSessions(d.sessions || [])).catch(onErr);
    return () => { cancelled = true; };
  }, [token]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const fileTypes = caps?.file_types || [];
  const models = stats?.models || [];
  const roleEntry = roles.find((r) => r.role === role) || (claims.department ? { department: claims.department } : null);

  /* ---- sessions ---- */

  async function openSession(id) {
    if (!token) return;
    try {
      const d = await getSession(token, id);
      setMessages(d.messages || []);
      setSessionId(id);
      setView("chat");
    } catch (e) {
      setAuthError(`Could not open session: ${e.message}`);
    }
  }

  async function removeSession(id) {
    setRowMenu(null);
    if (!token) return;
    try {
      await deleteSession(token, id);
      setSessions((prev) => prev.filter((s) => s.id !== id));
      if (sessionId === id) { setSessionId(null); setMessages([]); }
    } catch { /* leave the row if the delete failed */ }
  }

  function newChat(nextView = "chat") {
    setMessages([]); setAttachments([]); setSessionId(null); setView(nextView);
  }


  const visibleSessions = useMemo(() => {
    const q = search.trim().toLowerCase();
    return q ? sessions.filter((s) => (s.title || "").toLowerCase().includes(q)) : sessions;
  }, [sessions, search]);
  const grouped = useMemo(() => groupSessions(visibleSessions), [visibleSessions]);

  /* ---- attachments & send ---- */

  const onPickFiles = useCallback(async (files) => {
    const list = Array.from(files || []);
    if (!list.length) return;
    setUploading(true);
    try {
      for (const f of list) {
        const { file_path } = await uploadFile(f);
        setAttachments((prev) => [...prev, { name: f.name, path: file_path, kind: kindOf(f.name, fileTypes) }]);
      }
    } catch (e) {
      setAttachments((prev) => [...prev, { name: `Upload failed: ${e.message}`, path: null }]);
    } finally {
      setUploading(false);
    }
  }, [fileTypes]);

  async function submit() {
    const q = query.trim();
    if ((!q && !attachments.length) || loading || !token) return;
    const paths = attachments.filter((a) => a.path).map((a) => a.path);
    const shown = attachments.map((a) => a.name);

    setMessages((m) => [...m, { role: "user", text: q, files: shown }]);
    setQuery(""); setAttachments([]); setLoading(true);

    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const d = await sendQuery(token, {
        query: q || "Analyse the attached file.",
        uploadedFiles: paths, modelOverride: model, sessionId, signal: controller.signal,
      });
      if (d.session_id && d.session_id !== sessionId) setSessionId(d.session_id);
      setMessages((m) => [...m, {
        role: "agent",
        text: d.response,
        trajectory: d.trajectory || [],
        artifacts: d.artifacts || (d.download_url
          ? [{ name: d.download_url.split("/").pop(), url: d.download_url, kind: "file" }] : []),
        code: d.generated_code || null,
        structured: d.structured || null,
        elapsedMs: d.elapsed_ms ?? null,
        model,
      }]);
      listSessions(token).then((r) => setSessions(r.sessions || [])).catch(() => {});
    } catch (e) {
      const stopped = e.name === "AbortError";
      if (isAuthError(e)) { onSignOut(); return; }
      setMessages((m) => [...m, {
        role: "agent",
        text: stopped
          ? "Stopped waiting for this query. The backend is still finishing it — the result will be in the server log."
          : `Request failed: ${e.message}`,
        trajectory: [],
      }]);
    } finally {
      abortRef.current = null;
      setLoading(false);
    }
  }

  const stop = () => abortRef.current?.abort();

  /* ---- derived indicators (from real trajectories, never asserted) ---- */

  const events = messages.filter((m) => m.role === "agent").flatMap((m) => m.trajectory || []);
  const external = events.filter((e) => e.is_local === false).length;
  const llmTarget = caps?.runtimes?.find((r) => r.key === "llm")?.target;

  /* ---- placeholder derived from what the backend accepts ---- */
  const labelsFor = (agentName) =>
    fileTypes.filter((t) => (t.agent || "").toLowerCase().includes(agentName)).map((t) => t.label.toLowerCase());
  const placeholder = authError
    ? authError
    : view === "vision" && labelsFor("vision").length
      ? `Attach ${labelsFor("vision").join(", ")} and ask about it`
      : view === "meetings" && labelsFor("meeting").length
        ? `Attach ${labelsFor("meeting").join(", ")} to generate minutes`
        : view === "coding"
          ? "Ask for a calculation, an analysis or a file"
          : "Ask anything";

  const composer = (
    <Composer
      {...{ query, setQuery, submit, stop, loading, uploading, attachments, setAttachments,
            fileRef, onPickFiles, token, fileTypes, placeholder }}
    />
  );

  const chatView = CHAT_VIEWS.has(view);

  return (
    <div className={`app ${sideOpen ? "" : "side-closed"}`}>
      {/* ============================ sidebar ============================ */}
      <aside className="sb">
        <div className="sb-top">
          <button className="sb-logo" onClick={() => newChat()} title="SecureGPT"><Logo /></button>
          <div className="sb-top-actions">
            <button className="ic" title="Search chats" onClick={() => { setSearchOpen((o) => !o); setSearch(""); }}><IconSearch size={18} /></button>
            <button className="ic" title="Close sidebar" onClick={() => setSideOpen(false)}><IconSidebar size={18} /></button>
          </div>
        </div>

        {searchOpen && (
          <div className="sb-search">
            <IconSearch size={15} />
            <input autoFocus value={search} placeholder="Search chats" onChange={(e) => setSearch(e.target.value)} />
            <button className="ic sm" onClick={() => { setSearchOpen(false); setSearch(""); }}><IconX size={14} /></button>
          </div>
        )}

        <button className={`sb-row primary ${view === "chat" && !sessionId ? "active" : ""}`} onClick={() => newChat()}>
          <IconPencil size={18} /><span>New chat</span>
        </button>

        <nav className="sb-nav">
          {NAV.map(({ id, label, Icon }) => (
            <button
              key={id}
              className={`sb-row ${view === id ? "active" : ""}`}
              onClick={() => (CHAT_VIEWS.has(id) ? newChat(id) : setView(id))}
            >
              <Icon size={18} /><span>{label}</span>
              {id === "documents" && stats?.chunks_visible_to_role != null && (
                <span className="sb-tag">{compact(stats.chunks_visible_to_role)}</span>
              )}
              {id === "security" && events.length > 0 && (
                <span className={`sb-tag ${external ? "warn" : "ok"}`}>{external ? "EXTERNAL" : "LOCAL"}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sb-scroll">
          {grouped.length === 0 && token && (
            <div className="sb-empty">{search ? "No chats match." : "Your chats will appear here."}</div>
          )}
          {grouped.map((g) => (
            <div key={g.label}>
              <div className="sb-section">{g.label}</div>
              {g.items.map((s) => (
                <div key={s.id} className={`sb-chat ${sessionId === s.id ? "active" : ""}`} onClick={() => openSession(s.id)}>
                  <span className="sb-chat-title">{s.title || "Untitled"}</span>
                  <span className="sb-chat-actions" data-pop onClick={(e) => e.stopPropagation()}>
                    <button className="ic sm" title="Options" onClick={() => setRowMenu(rowMenu === s.id ? null : s.id)}>
                      <IconDots size={16} />
                    </button>
                    {rowMenu === s.id && (
                      <div className="menu right">
                        <div className="menu-meta">{words(s.role)} · {s.message_count} messages</div>
                        <button className="menu-item danger" onClick={() => removeSession(s.id)}>
                          <IconTrash size={16} /> Delete
                        </button>
                      </div>
                    )}
                  </span>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div className="sb-user-wrap" data-pop>
          {roleMenu && (
            <div className="menu up role-menu">
              <div className="menu-meta">
                <span><b>{username}</b></span>
                {roleEntry?.department && <span>{words(roleEntry.department)}</span>}
                {roleEntry?.chunks != null && (
                  <span>{roleEntry.chunks.toLocaleString()} chunks readable</span>
                )}
              </div>
              <div className="menu-sep" />
              <button className="menu-item" onClick={() => { setRoleMenu(false); onSignOut(); }}>
                <IconLogout size={16} /> Sign out
              </button>
            </div>
          )}
          <button className="sb-user" onClick={() => setRoleMenu((o) => !o)}>
            <span className="avatar">{role ? initials(role) : "…"}</span>
            <span className="sb-user-text">
              <span className="sb-user-name">{role ? words(role) : username}</span>
              <span className="sb-user-sub">{roleEntry?.department ? words(roleEntry.department) : (authError ? "offline" : username)}</span>
            </span>
            <IconUpDown size={16} />
          </button>
        </div>
      </aside>

      {/* ============================ main ============================ */}
      <main className="main">
        <header className="topbar">
          <div className="topbar-left">
            {!sideOpen && (
              <button className="ic" title="Open sidebar" onClick={() => setSideOpen(true)}><IconSidebar size={18} /></button>
            )}
            <div className="model-wrap" data-pop>
              <button className="model-btn" onClick={() => setModelMenu((o) => !o)}>
                <span className="model-name">SecureGPT</span>
                <span className="model-sub">{model === AUTO_MODEL ? "Auto" : model}</span>
                <IconChevronDown size={16} />
              </button>
              {modelMenu && (
                <div className="menu">
                  <div className="menu-meta">{llmTarget ? `Inference · ${llmTarget}` : "Inference"}</div>
                  <button className={`menu-item ${model === AUTO_MODEL ? "current" : ""}`} onClick={() => { setModel(AUTO_MODEL); setModelMenu(false); }}>
                    <span className="menu-item-main">
                      <span>Auto</span>
                      <span className="menu-item-sub">Supervisor picks the model per agent</span>
                    </span>
                    {model === AUTO_MODEL && <IconCheck size={16} />}
                  </button>
                  {models.map((m) => (
                    <button key={m} className={`menu-item ${model === m ? "current" : ""}`} onClick={() => { setModel(m); setModelMenu(false); }}>
                      <span className="menu-item-main"><span>{m}</span></span>
                      {model === m && <IconCheck size={16} />}
                    </button>
                  ))}
                  {models.length === 0 && <div className="menu-meta">No models reported yet.</div>}
                </div>
              )}
            </div>
          </div>

          <div className="segment">
            <button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>Chat</button>
            <button className={mode === "trace" ? "active" : ""} onClick={() => setMode("trace")}>
              <IconSparkle size={14} /> Trace
            </button>
          </div>

          <div className="topbar-right">
            <button
              className={`perimeter ${events.length === 0 ? "" : external ? "warn" : "ok"}`}
              title={events.length === 0
                ? `Deployment: ${stats?.deployment || caps?.deployment || "…"}`
                : external ? `${external} step(s) left the local perimeter` : `All ${events.length} steps stayed local`}
              onClick={() => setView("security")}
            >
              <IconShield size={18} />
              <span>{events.length === 0 ? (stats?.deployment || "…") : external ? "External" : "Local"}</span>
            </button>
          </div>
        </header>

        <section className={`body ${chatView && messages.length ? "in-chat" : ""}`}>
          {view === "security" ? (
            <SecurityPage stats={stats} caps={caps} messages={messages} token={token} />
          ) : view === "documents" ? (
            <DocumentsPage token={token} role={role} />
          ) : view === "admin" ? (
            <AdminPage stats={stats} caps={caps} roles={roles} />
          ) : messages.length === 0 ? (
            <div className="hero">
              <h1>What&rsquo;s on your mind today?</h1>
              {composer}
              {fileTypes.length > 0 && (
                <div className="hero-hint">
                  Attach {fileTypes.map((t) => t.label.toLowerCase()).join(", ")} — routed to the right agent automatically.
                </div>
              )}
            </div>
          ) : (
            <>
              <div className="thread-scroll">
                <div className="thread">
                  {messages.map((m, i) =>
                    m.role === "user"
                      ? <UserMsg key={i} msg={m} />
                      : <AgentMsg key={i} msg={m} traceMode={mode === "trace"} />
                  )}
                  {loading && (
                    <div className="a-msg">
                      <div className="a-avatar"><Logo size={18} /></div>
                      <div className="a-body"><div className="thinking"><span className="pulse" /> Working locally…</div></div>
                    </div>
                  )}
                  <div ref={endRef} />
                </div>
              </div>
              <div className="dock">
                {composer}
                <div className="footnote">
                  SecureGPT runs {stats?.deployment ? words(stats.deployment) : "on-premise"}. Every step is logged — open Trace to verify an answer.
                </div>
              </div>
            </>
          )}
        </section>
      </main>
    </div>
  );
}

/** Documents (and, for the final context, chunk identities) behind one trace step. */
const SOURCE_LABELS = {
  retrieved: "Retrieved", pool: "Pool", selected: "Best window", candidates: "Candidates",
  reranked: "Reranked", verified: "Verified", context: "In context",
};

function TraceSources({ sources }) {
  return (
    <div className="trace-sources">
      {Object.entries(sources).map(([key, docs]) => (
        <div className="trace-src-group" key={key}>
          <div className="trace-src-label">
            {SOURCE_LABELS[key] || key}
            <span className="trace-src-total">{docs.reduce((n, d) => n + d.chunks, 0)} chunks · {docs.length} doc{docs.length === 1 ? "" : "s"}</span>
          </div>
          {docs.length === 0 && <div className="trace-src-doc muted">none</div>}
          {docs.map((d) => (
            <div className="trace-src-doc" key={d.doc}>
              <span className="trace-src-name" title={d.doc}>{d.doc}</span>
              <span className="trace-src-count">{d.chunks}</span>
              {d.ids && (
                <div className="trace-src-ids">
                  {d.ids.map((c) => (
                    <span className="trace-src-id" key={c.chunk_id} title={c.chunk_id || ""}>
                      #{c.chunk_index ?? "?"}{c.where ? ` · ${c.where}` : ""} · {c.score.toFixed(2)}
                      {c.chunk_id && <em> {String(c.chunk_id).slice(0, 8)}</em>}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/** A 401 from the backend: token expired or revoked -- back to sign-in. */
const isAuthError = (e) => /credentials|401/i.test(String(e?.message || ""));

/** 120324 -> "120k" for sidebar tags. */
function compact(n) {
  if (n == null) return "";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${Math.round(n / 1e3)}k`;
  return String(n);
}

/* ================================================================== */

function Composer(props) {
  const { query, setQuery, submit, stop, loading, uploading, attachments, setAttachments,
          fileRef, onPickFiles, token, fileTypes, placeholder } = props;
  const [plusOpen, setPlusOpen] = useState(false);
  const taRef = useRef(null);
  useDismiss(plusOpen, useCallback(() => setPlusOpen(false), []));

  const accept = acceptAttr(fileTypes);
  const canSend = !uploading && token && (query.trim() || attachments.length);

  // Grow with the text like ChatGPT's box, capped so the thread stays visible.
  useEffect(() => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 220)}px`;
  }, [query]);

  return (
    <div className="composer">
      {attachments.length > 0 && (
        <div className="attach-row">
          {attachments.map((a, i) => (
            <span className="attach-chip" key={i}>
              <span className="attach-tag">{a.kind?.tag || "FILE"}</span>
              <span className="attach-name">{a.name}</span>
              {a.kind && <span className="attach-agent">{a.kind.agent}</span>}
              <button className="ic sm" onClick={() => setAttachments((p) => p.filter((_, j) => j !== i))}><IconX size={13} /></button>
            </span>
          ))}
        </div>
      )}

      <div className="composer-row">
        <div className="plus-wrap" data-pop>
          <button className="round" title="Add files" onClick={() => setPlusOpen((o) => !o)} disabled={loading || uploading}>
            {uploading ? <span className="pulse" /> : <IconPlus size={20} />}
          </button>
          {plusOpen && (
            <div className="menu up">
              <button className="menu-item" onClick={() => { setPlusOpen(false); fileRef.current?.click(); }}>
                <IconPaperclip size={17} /> Add files
              </button>
              {fileTypes.length > 0 && (
                <>
                  <div className="menu-sep" />
                  {fileTypes.map((t) => (
                    <div className="menu-row" key={t.kind}>
                      <span>{t.label}</span>
                      <span className="menu-row-sub">{t.agent}</span>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}
        </div>

        <input
          ref={fileRef} type="file" multiple {...(accept ? { accept } : {})} style={{ display: "none" }}
          onChange={(e) => { onPickFiles(e.target.files); e.target.value = ""; }}
        />

        <textarea
          ref={taRef}
          rows={1}
          value={query}
          placeholder={placeholder}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }}
          disabled={loading}
        />

        {loading ? (
          <button className="send stop" onClick={stop} title="Stop waiting"><IconStop size={18} /></button>
        ) : (
          <button className="send" onClick={submit} disabled={!canSend} title="Send"><IconArrowUp size={20} /></button>
        )}
      </div>
    </div>
  );
}

/* ================================================================== */

function UserMsg({ msg }) {
  return (
    <div className="u-msg">
      <div className="u-bubble">
        {msg.files?.length > 0 && (
          <div className="u-files">
            {msg.files.map((f, i) => <span key={i} className="u-file"><IconFile size={13} /> {f}</span>)}
          </div>
        )}
        {msg.text}
      </div>
    </div>
  );
}

function AgentMsg({ msg, traceMode }) {
  const [showTrace, setShowTrace] = useState(false);
  const [showCode, setShowCode] = useState(false);
  const [copied, setCopied] = useState(false);

  const agent = agentFromTrajectory(msg.trajectory);
  const local = isFullyLocal(msg.trajectory);
  const artifacts = msg.artifacts || [];
  const images = artifacts.filter((a) => a.kind === "image");
  const files = artifacts.filter((a) => a.kind !== "image");
  const steps = msg.trajectory?.length || 0;
  const traceOpen = traceMode || showTrace;

  // Which cascade stage the context came from -- stamped by the backend on
  // the "Built Context" event, never inferred from which nodes appear.
  const built = (msg.trajectory || []).find((t) => t.retrieval_stage != null);
  const stage = built?.retrieval_stage ? STAGE_LABELS[built.retrieval_stage] || built.retrieval_stage : null;
  const path = built?.retrieval_path || [];

  // Metric cards come only from the sandbox's verified {answer, details}
  // JSON -- a card can never show a value the agent did not compute.
  const details = msg.structured?.details;
  const metrics = details && typeof details === "object" && !Array.isArray(details)
    ? Object.entries(details).filter(([, v]) => v !== null && typeof v !== "object").slice(0, 4)
    : [];

  function copy() {
    navigator.clipboard?.writeText(msg.text || "").then(() => {
      setCopied(true); setTimeout(() => setCopied(false), 1400);
    }).catch(() => {});
  }

  return (
    <div className="a-msg">
      <div className="a-avatar"><Logo size={18} /></div>
      <div className="a-body">
        <div className="a-head">
          <span className="a-agent">{agent.name}</span>
          {steps > 0 && <span className={`pill ${local ? "ok" : "warn"}`}>{local ? "local" : "external"}</span>}
          {stage && (
            <span
              className="pill stage"
              title={path.map((p) => `${STAGE_LABELS[p.stage] || p.stage}${p.score != null ? ` ${p.score.toFixed(2)}` : ""} ${p.accepted ? "✓" : "✗"}`).join(" → ")}
            >
              via {stage}
            </span>
          )}
          {msg.model && msg.model !== AUTO_MODEL && <span className="pill">{msg.model}</span>}
        </div>

        <Markdown text={msg.text || ""} />

        {metrics.length > 0 && (
          <div className="metric-cards">
            {metrics.map(([k, v]) => (
              <div className="metric-card" key={k}>
                <div className="metric-label">{words(k)}</div>
                <div className="metric-value">{typeof v === "number" ? v.toLocaleString() : String(v)}</div>
              </div>
            ))}
          </div>
        )}

        {images.map((a) => (
          <figure key={a.url} className="a-figure">
            <a href={downloadUrl(a.url)} target="_blank" rel="noreferrer">
              <img src={downloadUrl(a.url)} alt={a.name} />
            </a>
            <figcaption>{a.name}</figcaption>
          </figure>
        ))}
        {files.length > 0 && (
          <div className="a-files">
            {files.map((a) => (
              <a key={a.url} className="file-card" href={downloadUrl(a.url)} target="_blank" rel="noreferrer">
                <span className="file-ico"><IconFile size={18} /></span>
                <span className="file-name">{a.name}</span>
                <IconDownload size={16} />
              </a>
            ))}
          </div>
        )}

        <div className="a-actions">
          <button className="act" onClick={copy} title="Copy">{copied ? <IconCheck size={15} /> : <IconCopy size={15} />}</button>
          {msg.elapsedMs != null && <span className="act-info"><IconClock size={14} /> {(msg.elapsedMs / 1000).toFixed(1)}s</span>}
          {steps > 0 && !traceMode && (
            <button className={`act text ${showTrace ? "on" : ""}`} onClick={() => setShowTrace((o) => !o)}>
              <IconChevronRight size={14} className={`chev ${showTrace ? "open" : ""}`} /> {steps} steps
            </button>
          )}
          {msg.code && (
            <button className={`act code-btn ${showCode ? "on" : ""}`} onClick={() => setShowCode((o) => !o)} title="Python the coding agent wrote and ran">
              <IconCode size={15} /> {showCode ? "Hide code" : "View code"}
            </button>
          )}
        </div>

        {showCode && msg.code && (
          <div className="code-panel">
            <div className="code-panel-bar">
              <span>python · written by the coding agent, run in the sandbox</span>
              <button className="act" onClick={() => navigator.clipboard?.writeText(msg.code).catch(() => {})} title="Copy code">
                <IconCopy size={14} /> Copy
              </button>
            </div>
            <pre className="md-code"><code>{msg.code}</code></pre>
          </div>
        )}

        {traceOpen && steps > 0 && (
          <div className="trace">
            {msg.trajectory.map((t, i) => (
              <div className="trace-row" key={i}>
                <div className="trace-idx">{String(i + 1).padStart(2, "0")}</div>
                <div className="trace-main">
                  <div className="trace-action">
                    {t.node && <span className="trace-node">{t.node}</span>}
                    {t.action}
                    <span className={`pill ${t.is_local === false ? "warn" : "ok"}`}>{t.is_local === false ? "external" : "local"}</span>
                  </div>
                  <div className="trace-detail">{t.details}</div>
                  {t.sources && <TraceSources sources={t.sources} />}
                  {t.network_target && <div className="trace-target">{t.network_target}</div>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
