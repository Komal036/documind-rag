import { useCallback, useEffect, useState } from "react";
import { useOutletContext } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../context/AuthContext";

export default function DocumentsPage() {
  const { token } = useAuth();
  const { refreshStats } = useOutletContext();
  const [files, setFiles] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState(null);
  const [deletingFile, setDeletingFile] = useState(null);

  const loadFiles = useCallback(() => {
    api.stats(token).then((s) => setFiles(s.indexed_files || [])).catch(() => {});
  }, [token]);

  useEffect(() => {
    loadFiles();
  }, [loadFiles]);

  async function handleUpload(file) {
    if (!file) return;
    setUploading(true);
    setMessage(null);
    try {
      const result = await api.ingest(token, file);
      setMessage({
        type: "success",
        text: `Indexed ${result.chunk_count} chunks from "${result.filename}".`,
      });
      loadFiles();
      refreshStats?.();
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof ApiError ? err.message : "Upload failed.",
      });
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(filename) {
    setDeletingFile(filename);
    try {
      await api.deleteDocument(token, filename);
      loadFiles();
      refreshStats?.();
    } catch (err) {
      setMessage({
        type: "error",
        text: err instanceof ApiError ? err.message : "Delete failed.",
      });
    } finally {
      setDeletingFile(null);
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-6 py-8">
      <h1 className="text-2xl font-bold text-ink-900 mb-1 tracking-tight">Documents</h1>
      <p className="text-sm text-ink-400 mb-6">
        Upload PDF, TXT, DOCX, or MD files to index them for question answering.
      </p>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          handleUpload(e.dataTransfer.files?.[0]);
        }}
        className={`btn-lift block border-2 border-dashed rounded-3xl px-6 py-12 text-center cursor-pointer transition mb-4 ${
          dragOver
            ? "border-accent-400 bg-accent-50 shadow-soft-lg"
            : "border-ink-200 bg-white/70 backdrop-blur shadow-soft hover:shadow-soft-lg hover:border-accent-300"
        }`}
      >
        <input
          type="file"
          accept=".pdf,.txt,.docx,.md"
          className="hidden"
          onChange={(e) => handleUpload(e.target.files?.[0])}
        />
        <div className="w-12 h-12 mx-auto mb-3 rounded-2xl bg-gradient-to-br from-accent-100 to-accent-200 flex items-center justify-center">
          <span className="text-accent-600 font-bold text-lg">↑</span>
        </div>
        <p className="text-sm font-semibold text-ink-700">
          {uploading ? "Uploading…" : "Drop a file here, or click to browse"}
        </p>
        <p className="text-xs text-ink-400 mt-1">PDF, TXT, DOCX, or MD — max 50MB</p>
      </label>

      {message && (
        <p
          className={`text-sm rounded-xl px-4 py-2.5 mb-6 shadow-soft border ${
            message.type === "success"
              ? "text-accent-700 bg-accent-50 border-accent-100"
              : "text-red-600 bg-red-50 border-red-100"
          }`}
        >
          {message.text}
        </p>
      )}

      <h2 className="text-sm font-bold text-ink-600 mb-3">
        {files.length} document{files.length === 1 ? "" : "s"} indexed
      </h2>

      {files.length === 0 ? (
        <p className="text-sm text-ink-400 bg-white/70 backdrop-blur border border-ink-100/60 rounded-2xl px-4 py-8 text-center shadow-soft">
          No documents indexed yet. Upload one above to get started.
        </p>
      ) : (
        <div className="space-y-2.5">
          {files.map((f) => (
            <div
              key={f}
              className="btn-lift flex items-center justify-between bg-white rounded-xl px-4 py-3 shadow-soft border border-ink-100/60"
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-ink-50 to-ink-100 flex items-center justify-center shrink-0">
                  <span className="text-ink-500 text-xs font-bold">📄</span>
                </div>
                <span className="text-sm text-ink-800 font-mono truncate">{f}</span>
              </div>
              <button
                onClick={() => handleDelete(f)}
                disabled={deletingFile === f}
                className="text-xs font-semibold text-ink-400 hover:text-red-600 disabled:opacity-50 shrink-0 ml-3"
              >
                {deletingFile === f ? "Deleting…" : "Delete"}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
