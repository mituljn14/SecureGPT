/**
 * Security posture view.
 *
 * Nothing here is asserted by the UI. The posture cards come from
 * /capabilities (real runtime targets), the counts from /stats, and the audit
 * stream from the trajectory of queries actually run in this session. A panel
 * that printed "SYSTEM SECURE" as static copy would keep printing it after a
 * genuine external call, which is worse than having no panel.
 */
import { useEffect, useState } from "react";
import { fetchAudit } from "./api";

const AUDIT_LIMIT = 100;

const TONES = {
  mint: "var(--mint)",
  red: "var(--red)",
  amber: "var(--amber)",
  violet: "var(--violet)",
};

export default function SecurityPage({ stats, caps, messages = [], token }) {
  // The audit stream is the signed-in user's last AUDIT_LIMIT trace events
  // across every chat they own, served by /audit from the same stored
  // trajectories the Trace tab shows. It is re-fetched whenever a turn
  // completes in this tab. Until it arrives (or if it fails), the events of
  // the chat currently open are shown so the page is never blank.
  const [audit, setAudit] = useState(null);
  const [auditErr, setAuditErr] = useState(null);
  const agentTurns = messages.filter((m) => m.role === "agent").length;

  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    fetchAudit(token, AUDIT_LIMIT)
      .then((d) => { if (!cancelled) { setAudit(d); setAuditErr(null); } })
      .catch((e) => { if (!cancelled) setAuditErr(e.message); });
    return () => { cancelled = true; };
  }, [token, agentTurns]);

  const local = messages.filter((m) => m.role === "agent").flatMap((m) => m.trajectory || []).reverse();
  const events = audit?.events || local;

  const external = events.filter((e) => e.is_local === false);
  const secure = external.length === 0;
  const recent = events.slice(0, AUDIT_LIMIT);

  const runtimes = caps?.runtimes || [];
  const models = stats?.models || [];
  const fileTypes = caps?.file_types || [];

  return (
    <div className="page">
      <h1 className="page-title" style={{ color: secure ? "var(--mint)" : "var(--red)" }}>
        {secure ? "SYSTEM SECURE" : "EXTERNAL CALL DETECTED"}
      </h1>
      <p className="page-sub">
        {events.length === 0
          ? "No agent activity recorded for this account yet. Run a query to populate the audit stream."
          : audit
            ? `Last ${events.length} step(s) for ${stats?.username || "this account"} across ${audit.sessions} chat(s) · ${external.length} outside the local perimeter.`
            : `${events.length} step(s) in the open chat · ${external.length} outside the local perimeter.`}
        {auditErr && ` (history unavailable: ${auditErr})`}
      </p>

      {runtimes.length > 0 && (
        <div className="grid grid-3" style={{ marginBottom: 26 }}>
          {runtimes.map((r) => {
            const color = TONES[r.tone] || "var(--text-dim)";
            return (
              <div className="card" key={r.key}
                   style={{ borderColor: `color-mix(in srgb, ${color} 32%, transparent)` }}>
                <div className="card-top">
                  <span className="card-icon">⬡</span>
                  <span className="badge" style={{ color, border: `1px solid ${color}`, background: "transparent" }}>
                    {r.tag}
                  </span>
                </div>
                <div className="card-title">{r.label}</div>
                <div className="card-note">{r.note}</div>
                <div className="trace-target" style={{ marginTop: 6 }}>{r.target}</div>
              </div>
            );
          })}
        </div>
      )}

      <div className="grid grid-2">
        <div className="card">
          <div className="card-top">
            <div className="card-title">Real-time Audit Stream</div>
            <span style={{ fontFamily: "var(--mono)", fontSize: ".6rem", color: "var(--text-faint)" }}>
              {audit ? `last ${events.length} of this account` : `${events.length} events`}
            </span>
          </div>
          {recent.length === 0 ? (
            <div className="empty" style={{ padding: "26px 10px" }}>Awaiting agent activity.</div>
          ) : (
            <div className="audit-list">
            {recent.map((e, i) => (
              <div className={`audit ${e.is_local === false ? "blocked" : ""}`} key={i}>
                <span className="audit-time">
                  {e.timestamp ? new Date(e.timestamp * 1000).toLocaleString([], { dateStyle: "short", timeStyle: "medium" }) : "--:--:--"}
                </span>{" "}
                {e.node && <span className="audit-node">{e.node}</span>}{" "}
                <span className="audit-kind">[{(e.action || "step").toUpperCase()}]</span>{" "}
                {e.details}
                {e.network_target ? ` → ${e.network_target}` : ""}
                {e.session_title && (
                  <div className="audit-session" title={e.session_id}>chat: {e.session_title}</div>
                )}
              </div>
            ))}
            </div>
          )}
        </div>

        <div>
          <div className="card" style={{ marginBottom: 13 }}>
            <div className="card-top"><div className="card-title">Local Runtimes</div></div>
            {models.length === 0 ? (
              <div className="empty" style={{ padding: "22px 10px" }}>No models reported.</div>
            ) : (
              models.map((m) => (
                <div className="runtime-row" key={m}>
                  <div>
                    <div className="runtime-name">{m}</div>
                    <div className="runtime-tag">ollama · local</div>
                  </div>
                  <div className="ready"><span className="dot" /> Ready</div>
                </div>
              ))
            )}
          </div>

          <div className="card" style={{ marginBottom: 13 }}>
            <div className="card-top"><div className="card-title">Corpus</div></div>
            <div className="kv"><span>Collection</span><span>{stats?.collection || "—"}</span></div>
            <div className="kv"><span>Total chunks</span><span>{(stats?.chunks_total ?? 0).toLocaleString()}</span></div>
            <div className="kv"><span>Readable by role</span><span>{(stats?.chunks_visible_to_role ?? 0).toLocaleString()}</span></div>
            <div className="kv"><span>Role</span><span>{stats?.role || "—"}</span></div>
            {(stats?.department_names || []).length > 0 && (
              <div className="kv">
                <span>Departments</span>
                <span>{stats.department_names.length}</span>
              </div>
            )}
          </div>

          {fileTypes.length > 0 && (
            <div className="card">
              <div className="card-top"><div className="card-title">Accepted Inputs</div></div>
              {fileTypes.map((t) => (
                <div className="kv" key={t.kind}>
                  <span>{t.label} <span style={{ opacity: .55 }}>→ {t.agent}</span></span>
                  <span>{t.exts.join(" ")}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
