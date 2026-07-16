import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function Layout() {
  const { token, user, logout } = useAuth();
  const [healthy, setHealthy] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        await api.health();
        if (!cancelled) setHealthy(true);
        const s = await api.stats(token);
        if (!cancelled) setStats(s);
      } catch {
        if (!cancelled) setHealthy(false);
      }
    }

    poll();
    const interval = setInterval(poll, 15000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [token]);

  const navItem = ({ isActive }) =>
    `flex items-center gap-2.5 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
      isActive
        ? "bg-gradient-to-r from-accent-500 to-accent-600 text-white shadow-soft"
        : "text-ink-600 hover:bg-white hover:shadow-soft"
    }`;

  return (
    <div className="min-h-screen flex">
      <aside className="w-64 border-r border-ink-100/60 bg-white/70 backdrop-blur-xl flex flex-col shadow-soft-lg">
        <div className="px-5 py-5 flex items-center gap-2.5 border-b border-ink-100/60">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-accent-400 to-accent-600 flex items-center justify-center text-white font-bold text-sm shadow-soft">
            D
          </div>
          <span className="font-bold text-ink-900 tracking-tight">DocuMind</span>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1.5">
          <NavLink to="/" end className={navItem}>
            Ask a question
          </NavLink>
          <NavLink to="/documents" className={navItem}>
            Documents
          </NavLink>
        </nav>

        <div className="px-4 py-4 border-t border-ink-100/60">
          <div className="flex items-center gap-2 text-xs mb-4">
            <span className="relative flex w-2 h-2">
              {healthy && (
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-accent-400 opacity-60" />
              )}
              <span
                className={`relative inline-flex rounded-full w-2 h-2 ${
                  healthy === null ? "bg-ink-200" : healthy ? "bg-accent-500" : "bg-red-500"
                }`}
              />
            </span>
            <span className="text-ink-400 font-medium">
              {healthy === null ? "Checking…" : healthy ? "API connected" : "API offline"}
            </span>
          </div>

          {stats && (
            <div className="grid grid-cols-2 gap-2 mb-4">
              <div className="bg-gradient-to-br from-white to-ink-50 rounded-xl px-3 py-2.5 shadow-soft border border-ink-100/60">
                <p className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">
                  Chunks
                </p>
                <p className="text-lg font-bold text-ink-900">{stats.total_chunks}</p>
              </div>
              <div className="bg-gradient-to-br from-white to-ink-50 rounded-xl px-3 py-2.5 shadow-soft border border-ink-100/60">
                <p className="text-[10px] text-ink-400 uppercase tracking-wider font-semibold">
                  Docs
                </p>
                <p className="text-lg font-bold text-ink-900">
                  {stats.indexed_files?.length ?? 0}
                </p>
              </div>
            </div>
          )}

          <div className="flex items-center justify-between pt-1">
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-6 h-6 rounded-full bg-gradient-to-br from-ink-700 to-ink-900 flex items-center justify-center text-white text-[10px] font-bold shrink-0">
                {user?.email?.[0]?.toUpperCase()}
              </div>
              <p className="text-xs text-ink-500 truncate font-medium">{user?.email}</p>
            </div>
            <button
              onClick={logout}
              className="text-xs text-ink-400 hover:text-red-600 font-semibold shrink-0 ml-2"
            >
              Log out
            </button>
          </div>
        </div>
      </aside>

      <main className="flex-1 overflow-y-auto">
        <Outlet context={{ stats, refreshStats: () => api.stats(token).then(setStats) }} />
      </main>
    </div>
  );
}
