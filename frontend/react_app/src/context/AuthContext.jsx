import { createContext, useContext, useEffect, useState } from "react";
import { api } from "../api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("documind_token"));
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!token) {
      setLoading(false);
      return;
    }
    api
      .me(token)
      .then((u) => setUser(u))
      .catch(() => {
        // token expired/invalid
        localStorage.removeItem("documind_token");
        setToken(null);
      })
      .finally(() => setLoading(false));
  }, [token]);

  function login(newToken) {
    localStorage.setItem("documind_token", newToken);
    setToken(newToken);
  }

  function logout() {
    localStorage.removeItem("documind_token");
    setToken(null);
    setUser(null);
  }

  return (
    <AuthContext.Provider value={{ token, user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
