import { useCallback, useEffect, useState } from "react";
import Workbench from "./Workbench";
import LoginPage from "./LoginPage";
import { tokenClaims } from "./api";

// Session token lives in sessionStorage: a refresh keeps you signed in, closing
// the tab signs you out. Expiry is read from the token itself.
const KEY = "securegpt.token";

function stored() {
  try {
    const t = sessionStorage.getItem(KEY);
    const c = t && tokenClaims(t);
    return c && c.exp * 1000 > Date.now() ? t : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [token, setToken] = useState(stored);

  const signIn = useCallback((t) => {
    try { sessionStorage.setItem(KEY, t); } catch { /* private mode */ }
    setToken(t);
  }, []);
  const signOut = useCallback(() => {
    try { sessionStorage.removeItem(KEY); } catch { /* ignore */ }
    setToken(null);
  }, []);

  // Drop the session the moment the token expires rather than on the next
  // failed request.
  useEffect(() => {
    if (!token) return;
    const c = tokenClaims(token);
    if (!c?.exp) return;
    const ms = c.exp * 1000 - Date.now();
    const id = setTimeout(signOut, Math.max(0, ms));
    return () => clearTimeout(id);
  }, [token, signOut]);

  return token
    ? <Workbench key={token} token={token} onSignOut={signOut} />
    : <LoginPage onSignedIn={signIn} />;
}

