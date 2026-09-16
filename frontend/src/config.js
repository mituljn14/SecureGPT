// Single source of truth for the backend base URL.
//
// NOT port 8000: an unrelated Docker service (ocrapi-api-1) publishes
// 0.0.0.0:8000 on this machine. Windows lets a 127.0.0.1 bind coexist with
// a 0.0.0.0 bind on the same port, so both processes start successfully and
// which one answers a request is a race. Using a dedicated port removes the
// ambiguity entirely.
export const API_BASE = "http://127.0.0.1:8001";
