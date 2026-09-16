/**
 * Deployment view.
 *
 * Every row is served by /stats, /capabilities or /roles. There is no static
 * inventory in this file: if the backend is reconfigured, this page follows
 * without a code change.
 */
export default function AdminPage({ stats, caps, roles }) {
  const runtimes = caps?.runtimes || [];
  const fileTypes = caps?.file_types || [];
  const depts = stats?.department_names || [];

  return (
    <div className="page">
      <h1 className="page-title">Deployment</h1>
      <p className="page-sub">Configuration reported by this backend at runtime.</p>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-top"><div className="card-title">Runtime targets</div></div>
          {runtimes.length === 0 ? (
            <div className="empty" style={{ padding: "22px 10px" }}>Capabilities unavailable.</div>
          ) : (
            runtimes.map((r) => (
              <div className="kv" key={r.key}>
                <span>{r.label}</span>
                <span>{r.target}</span>
              </div>
            ))
          )}
          <div className="kv"><span>Deployment</span><span>{caps?.deployment || "—"}</span></div>
          <div className="kv"><span>Collection</span><span>{stats?.collection || "—"}</span></div>
        </div>

        <div className="card">
          <div className="card-top"><div className="card-title">Roles &amp; corpus reach</div></div>
          {(roles || []).length === 0 ? (
            <div className="empty" style={{ padding: "22px 10px" }}>Roles unavailable.</div>
          ) : (
            roles.map((r) => (
              <div className="kv" key={r.role}>
                <span>{r.role.replace(/_/g, " ")}</span>
                <span>{(r.chunks || 0).toLocaleString()}</span>
              </div>
            ))
          )}
        </div>

        <div className="card">
          <div className="card-top"><div className="card-title">Accepted inputs</div></div>
          {fileTypes.length === 0 ? (
            <div className="empty" style={{ padding: "22px 10px" }}>Not reported.</div>
          ) : (
            fileTypes.map((t) => (
              <div className="kv" key={t.kind}>
                <span>{t.label} <span style={{ opacity: 0.55 }}>→ {t.agent}</span></span>
                <span>{t.exts.join(" ")}</span>
              </div>
            ))
          )}
        </div>

        <div className="card">
          <div className="card-top"><div className="card-title">Corpus</div></div>
          {depts.map((d) => (
            <div className="kv" key={d}><span>{d.replace(/_/g, " ")}</span><span /></div>
          ))}
          <div className="kv">
            <span>Total chunks</span>
            <span>{(stats?.chunks_total ?? 0).toLocaleString()}</span>
          </div>
          <div className="kv">
            <span>Readable by {stats?.role || "role"}</span>
            <span>{(stats?.chunks_visible_to_role ?? 0).toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
