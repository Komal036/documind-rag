import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const result =
        mode === "login"
          ? await api.login(email, password)
          : await api.signup(email, password, fullName);
      login(result.access_token);
      navigate("/");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Try again.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4 relative overflow-hidden">
      <div className="absolute top-[-10%] right-[-5%] w-[420px] h-[420px] rounded-full bg-accent-200/40 blur-3xl" />
      <div className="absolute bottom-[-15%] left-[-10%] w-[420px] h-[420px] rounded-full bg-accent-100/50 blur-3xl" />

      <div className="w-full max-w-sm relative">
        <div className="flex items-center gap-2.5 justify-center mb-8">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-accent-400 to-accent-600 flex items-center justify-center text-white font-bold text-lg shadow-soft-lg">
            D
          </div>
          <span className="text-xl font-bold text-ink-900 tracking-tight">DocuMind</span>
        </div>

        <div className="bg-white/90 backdrop-blur-xl border border-ink-100/60 rounded-3xl p-7 shadow-soft-lg">
          <h1 className="text-xl font-bold text-ink-900 mb-1 tracking-tight">
            {mode === "login" ? "Log in" : "Create your account"}
          </h1>
          <p className="text-sm text-ink-400 mb-6">
            {mode === "login"
              ? "Access your documents and conversations."
              : "Start indexing documents in seconds."}
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            {mode === "signup" && (
              <div>
                <label className="block text-xs font-semibold text-ink-600 mb-1.5">Full name</label>
                <input
                  type="text"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  placeholder="Komal Sharma"
                  className="w-full px-3.5 py-2.5 rounded-xl border border-ink-100 text-sm bg-white/60 focus:outline-none focus:ring-2 focus:ring-accent-400 focus:bg-white transition"
                />
              </div>
            )}
            <div>
              <label className="block text-xs font-semibold text-ink-600 mb-1.5">Email</label>
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full px-3.5 py-2.5 rounded-xl border border-ink-100 text-sm bg-white/60 focus:outline-none focus:ring-2 focus:ring-accent-400 focus:bg-white transition"
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-ink-600 mb-1.5">Password</label>
              <input
                type="password"
                required
                minLength={8}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="At least 8 characters"
                className="w-full px-3.5 py-2.5 rounded-xl border border-ink-100 text-sm bg-white/60 focus:outline-none focus:ring-2 focus:ring-accent-400 focus:bg-white transition"
              />
            </div>

            {error && (
              <p className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-3.5 py-2.5">
                {error}
              </p>
            )}

            <button
              type="submit"
              disabled={submitting}
              className="btn-lift w-full py-3 rounded-xl bg-gradient-to-br from-ink-800 to-ink-900 text-white text-sm font-semibold shadow-soft-lg disabled:opacity-50 transition"
            >
              {submitting ? "Please wait…" : mode === "login" ? "Log in" : "Sign up"}
            </button>
          </form>

          <p className="text-sm text-ink-400 text-center mt-6">
            {mode === "login" ? "New here?" : "Already have an account?"}{" "}
            <button
              onClick={() => {
                setMode(mode === "login" ? "signup" : "login");
                setError("");
              }}
              className="text-accent-600 font-semibold hover:underline"
            >
              {mode === "login" ? "Create an account" : "Log in"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}
