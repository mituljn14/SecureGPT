import { API_BASE } from "./config";

/**
 * Thin API layer. No catalogues live here.
 *
 * Roles, models, departments, accepted file extensions and runtime targets
 * are all served by the backend (/roles, /stats, /capabilities), because the
 * backend is what actually enforces them. A second copy in the UI would
 * eventually disagree -- and the visible symptom is a file the picker
 * accepted that no agent can read.
 */

async function jsonOrThrow(res) {
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* no JSON body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const getRoles = () => fetch(`${API_BASE}/roles`).then(jsonOrThrow);
export const getCapabilities = () => fetch(`${API_BASE}/capabilities`).then(jsonOrThrow);

/**
 * Sign in as a role. The backend issues the token from its own user table, so
 * no credential is embedded in this bundle. Gated server-side by
 * PROTOTYPE_AUTH; the normal password flow is untouched.
 */
export function loginAsRole(role) {
  return fetch(`${API_BASE}/login/demo`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role }),
  }).then(jsonOrThrow);
}

/**
 * Real sign-in: username + password against the backend's user table
 * (OAuth2 password form). The returned JWT carries username, role and
 * department, which the backend then enforces on every request.
 */
export function login(username, password) {
  const form = new URLSearchParams({ username, password });
  return fetch(`${API_BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form,
  }).then(jsonOrThrow);
}

/** Decode the JWT payload for display (sub / role / department / exp). Not verification. */
export function tokenClaims(token) {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(escape(atob(b64))));
  } catch {
    return null;
  }
}

export function fetchStats(token) {
  return fetch(`${API_BASE}/stats`, {
    headers: { Authorization: `Bearer ${token}` },
  }).then(jsonOrThrow);
}

export function uploadFile(file) {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${API_BASE}/upload`, { method: "POST", body: form }).then(jsonOrThrow);
}

export function sendQuery(token, { query, uploadedFiles, modelOverride, sessionId, signal }) {
  return fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      query,
      uploaded_files: uploadedFiles,
      model_override: modelOverride,
      session_id: sessionId,
    }),
    // Lets the UI stop waiting on a long run. Note this aborts the REQUEST,
    // not the work: the backend keeps processing to completion, so a
    // cancelled coding query still finishes its sandbox run server-side.
    signal,
  }).then(jsonOrThrow);
}


/* ---------- chat history (server-side, SQLite) ---------- */

const auth = (token) => ({ Authorization: `Bearer ${token}` });

export const listSessions = (token) =>
  fetch(`${API_BASE}/sessions`, { headers: auth(token) }).then(jsonOrThrow);

export const getSession = (token, id) =>
  fetch(`${API_BASE}/sessions/${id}`, { headers: auth(token) }).then(jsonOrThrow);

export const deleteSession = (token, id) =>
  fetch(`${API_BASE}/sessions/${id}`, { method: "DELETE", headers: auth(token) })
    .then(jsonOrThrow);

/** Last `limit` trace events for the signed-in user, across all their chats. */
export const fetchAudit = (token, limit = 100) =>
  fetch(`${API_BASE}/audit?limit=${limit}`, { headers: auth(token) }).then(jsonOrThrow);

export const searchCorpus = (token, q, limit = 8) =>
  fetch(`${API_BASE}/search?q=${encodeURIComponent(q)}&limit=${limit}`,
        { headers: auth(token) }).then(jsonOrThrow);

export const logsUrl = () => `${API_BASE}/logs/download`;

export const downloadUrl = (path) => `${API_BASE}${path}`;

/* ---------- helpers over server-supplied capability data ---------- */

/** Build the file input's accept attribute from what the backend reported. */
export const acceptAttr = (fileTypes = []) =>
  fileTypes.flatMap((t) => t.exts).join(",");

/** Match a filename against the backend's file-type table. */
export function kindOf(filename = "", fileTypes = []) {
  const dot = filename.lastIndexOf(".");
  const ext = dot === -1 ? "" : filename.slice(dot).toLowerCase();
  return fileTypes.find((t) => t.exts.includes(ext)) || null;
}

/** Which agent handled a turn, inferred from its own trajectory. */
export function agentFromTrajectory(trajectory = []) {
  const actions = trajectory.map((t) => (t.action || "").toLowerCase());
  if (actions.some((a) => a.includes("vision"))) return { name: "Vision Agent", icon: "V" };
  if (actions.some((a) => a.includes("meeting"))) return { name: "Meeting Agent", icon: "M" };
  if (actions.some((a) => a.includes("coding") || a.includes("sandbox")))
    return { name: "Coding Agent", icon: "C" };
  if (actions.some((a) => a.includes("enterprise") || a.includes("retrieval")))
    return { name: "Enterprise Agent", icon: "E" };
  return { name: "SecureGPT", icon: "S" };
}

/** True only if every logged step stayed inside the local perimeter. */
export const isFullyLocal = (trajectory = []) =>
  trajectory.every((t) => t.is_local !== false);
