import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { useAuth } from "../context/AuthContext";

const EXAMPLES = [
  "What is the remote work policy?",
  "How many days can employees work from home?",
  "What is the expense reimbursement limit?",
  "Who is eligible for remote work?",
];

export default function ChatPage() {
  const { token } = useAuth();
  const [sessionId, setSessionId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [showSettings, setShowSettings] = useState(false);
  const [useReranker, setUseReranker] = useState(true);
  const [topK, setTopK] = useState(20);
  const [topN, setTopN] = useState(5);
  const bottomRef = useRef(null);

  useEffect(() => {
    api.createChatSession(token).then((s) => setSessionId(s.session_id)).catch(() => {});
  }, [token]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function sendQuestion(question) {
    if (!question.trim() || loading) return;
    setError("");
    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setLoading(true);

    const t0 = performance.now();
    try {
      const result = await api.query(token, question, {
        useReranker,
        topK,
        topN,
        sessionId,
      });
      const elapsed = ((performance.now() - t0) / 1000).toFixed(1);
      setMessages((m) => [
        ...m,
        { role: "assistant", content: result.answer, sources: result.sources, meta: { ...result, elapsed } },
      ]);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setMessages((m) => m.slice(0, -1));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 min-h-screen flex flex-col">
      <div className="flex items-start justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-ink-900 tracking-tight">Ask a question</h1>
          <p className="text-sm text-ink-400 mt-1">
            Answers are grounded in your indexed documents, with citations.
          </p>
        </div>
        <button
          onClick={() => setShowSettings((s) => !s)}
          className="btn-lift text-xs font-semibold text-ink-600 bg-white border border-ink-100 rounded-xl px-4 py-2 shadow-soft hover:shadow-soft-lg"
        >
          Settings
        </button>
      </div>

      {showSettings && (
        <div className="bg-white/80 backdrop-blur border border-ink-100/60 rounded-2xl p-5 mb-6 space-y-5 shadow-soft-lg">
          <label className="flex items-center justify-between text-sm">
            <span className="text-ink-600 font-medium">Cross-encoder re-ranking</span>
            <input
              type="checkbox"
              checked={useReranker}
              onChange={(e) => setUseReranker(e.target.checked)}
              className="accent-accent-500 w-4 h-4"
            />
          </label>
          <div>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="text-ink-600 font-medium">Retrieval candidates (top_k)</span>
              <span className="text-ink-900 font-bold">{topK}</span>
            </div>
            <input
              type="range"
              min={5}
              max={50}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-accent-500"
            />
          </div>
          <div>
            <div className="flex justify-between text-sm mb-1.5">
              <span className="text-ink-600 font-medium">Chunks sent to LLM (top_n)</span>
              <span className="text-ink-900 font-bold">{topN}</span>
            </div>
            <input
              type="range"
              min={1}
              max={10}
              value={topN}
              onChange={(e) => setTopN(Number(e.target.value))}
              className="w-full accent-accent-500"
            />
          </div>
        </div>
      )}

      {messages.length === 0 && (
        <div className="mb-8">
          <p className="text-xs font-bold text-ink-400 uppercase tracking-wider mb-3">
            Try asking
          </p>
          <div className="grid grid-cols-2 gap-3">
            {EXAMPLES.map((ex) => (
              <button
                key={ex}
                onClick={() => sendQuestion(ex)}
                className="btn-lift text-left text-sm px-4 py-3.5 rounded-2xl border border-ink-100 bg-white shadow-soft hover:shadow-soft-lg hover:border-accent-200 transition text-ink-600 font-medium"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="flex-1 space-y-6">
        {messages.map((msg, i) =>
          msg.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="bg-gradient-to-br from-ink-800 to-ink-900 text-white text-sm rounded-2xl rounded-br-md px-4 py-3 max-w-[80%] shadow-soft-lg">
                {msg.content}
              </div>
            </div>
          ) : (
            <div key={i} className="space-y-3">
              <div className="bg-white rounded-2xl rounded-bl-md px-4 py-3.5 text-sm text-ink-800 whitespace-pre-wrap leading-relaxed shadow-soft-lg border border-ink-100/60">
                {msg.content}
              </div>

              {msg.meta && (
                <div className="flex items-center gap-2 text-xs text-ink-400 px-1 font-medium">
                  <span>{msg.meta.elapsed}s</span>
                  <span className="text-ink-200">·</span>
                  <span>{msg.meta.chunks_used} chunks</span>
                  <span className="text-ink-200">·</span>
                  <span className="px-2 py-0.5 rounded-full bg-gradient-to-r from-accent-50 to-accent-100 text-accent-700 font-semibold">
                    {msg.meta.model}
                  </span>
                </div>
              )}

              {msg.sources?.length > 0 && (
                <div className="space-y-2 px-1">
                  {msg.sources.map((src) => (
                    <details
                      key={src.chunk_id}
                      className="text-xs bg-white/70 backdrop-blur rounded-xl px-3.5 py-2.5 border border-ink-100/60 shadow-soft"
                    >
                      <summary className="cursor-pointer font-semibold text-ink-600">
                        [{src.citation_number}] {src.filename} · score {src.relevance_score.toFixed(2)}
                      </summary>
                      <p className="mt-2 text-ink-400 leading-relaxed">{src.preview}</p>
                    </details>
                  ))}
                </div>
              )}
            </div>
          )
        )}

        {loading && (
          <div className="flex items-center gap-2 bg-white rounded-2xl rounded-bl-md px-4 py-3.5 text-sm text-ink-400 max-w-[80%] shadow-soft-lg border border-ink-100/60">
            <span className="flex gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1.5 h-1.5 rounded-full bg-accent-400 animate-bounce" />
            </span>
            Retrieving relevant passages and generating an answer…
          </div>
        )}

        {error && (
          <p className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-xl px-4 py-2.5 shadow-soft">
            {error}
          </p>
        )}
        <div ref={bottomRef} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          sendQuestion(input);
        }}
        className="mt-6 flex gap-2.5 sticky bottom-6"
      >
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="e.g. What is the vacation policy?"
          className="flex-1 px-4 py-3 rounded-2xl border border-ink-100 bg-white/90 backdrop-blur text-sm focus:outline-none focus:ring-2 focus:ring-accent-400 shadow-soft-lg"
        />
        <button
          type="submit"
          disabled={loading || !input.trim()}
          className="btn-lift px-6 py-3 rounded-2xl bg-gradient-to-br from-ink-800 to-ink-900 text-white text-sm font-semibold shadow-soft-lg disabled:opacity-40 disabled:hover:translate-y-0 transition"
        >
          Ask
        </button>
      </form>
    </div>
  );
}
