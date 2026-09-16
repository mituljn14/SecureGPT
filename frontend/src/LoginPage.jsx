import { useEffect, useState } from "react";
import { getRoles, login } from "./api";
import { Logo } from "./icons";

/**
 * Sign-in page.
 *
 * Authenticates against the backend's own user table (POST /login).
 * Passwords are typed, never bundled. The account catalogue (roles,
 * departments, corpus reach) lives on the Admin page after sign-in.
 */
export default function LoginPage({ onSignedIn }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [offline, setOffline] = useState(null);

  // Backend needs ~90s on a cold boot; poll until it answers so the form
  // can say "starting" instead of failing the first sign-in attempt.
  useEffect(() => {
    let cancelled = false, attempt = 0, timer = null;
    const load = () => {
      attempt += 1;
      getRoles()
        .then(() => { if (!cancelled) setOffline(null); })
        .catch((e) => {
          if (cancelled) return;
          const wait = Math.min(2000 * attempt, 10000);
          setOffline(`Backend starting… (${e.message})`);
          timer = setTimeout(load, wait);
        });
    };
    load();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  async function submit(e) {
    e.preventDefault();
    if (!username.trim() || !password || busy) return;
    setBusy(true); setErr(null);
    try {
      const d = await login(username.trim(), password);
      onSignedIn(d.access_token);
    } catch (ex) {
      setErr(ex.message === "HTTP 422" ? "Enter a username and password." : ex.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login">
      <div className="login-card">
        <div className="login-logo"><Logo size={26} /></div>
        <h1>Welcome back</h1>
        <p className="login-sub">Sign in to SecureGPT. What you can retrieve is decided by your role&rsquo;s access rights, enforced on the server.</p>

        <form onSubmit={submit}>
          <div className="field">
            <input
              autoFocus
              autoComplete="username"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="field">
            <input
              type="password"
              autoComplete="current-password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {err && <div className="login-err">{err}</div>}
          {offline && !err && <div className="login-note">{offline}</div>}
          <button className="login-btn" type="submit" disabled={busy || !username.trim() || !password}>
            {busy ? "Signing in…" : "Continue"}
          </button>
        </form>
      </div>

    </div>
  );
}
