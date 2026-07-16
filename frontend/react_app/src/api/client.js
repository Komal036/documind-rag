const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, token, isForm = false } = {}) {
  const headers = {};
  if (!isForm) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: isForm ? body : body ? JSON.stringify(body) : undefined,
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    // no JSON body (e.g. some error responses)
  }

  if (!res.ok) {
    const message = data?.detail || data?.error || `Request failed (${res.status})`;
    throw new ApiError(message, res.status);
  }
  return data;
}

export const api = {
  health: () => request("/health"),

  signup: (email, password, fullName) =>
    request("/api/v1/auth/signup", {
      method: "POST",
      body: { email, password, full_name: fullName },
    }),

  login: (email, password) =>
    request("/api/v1/auth/login", { method: "POST", body: { email, password } }),

  me: (token) => request("/api/v1/auth/me", { token }),

  stats: (token) => request("/api/v1/stats", { token }),

  ingest: (token, file) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/v1/ingest", { method: "POST", token, body: form, isForm: true });
  },

  query: (token, question, options = {}) =>
    request("/api/v1/query", {
      method: "POST",
      token,
      body: {
        question,
        use_reranker: options.useReranker ?? true,
        top_k: options.topK,
        top_n: options.topN,
        session_id: options.sessionId,
      },
    }),

  deleteDocument: (token, filename) =>
    request(`/api/v1/documents/${encodeURIComponent(filename)}`, { method: "DELETE", token }),

  createChatSession: (token) =>
    request("/api/v1/chat/sessions", { method: "POST", token }),
};

export { ApiError };
