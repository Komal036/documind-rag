# 🧠 DocuMind — Enterprise RAG System

> Production-quality Retrieval-Augmented Generation system. Upload documents, ask questions, get grounded answers with citations.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.0-orange.svg)](https://www.trychroma.com)

---

## ✨ Features

- **Document Ingestion** — PDF, TXT, DOCX, MD (PyMuPDF + pdfplumber fallback)
- **Smart Chunking** — RecursiveCharacterTextSplitter (512 chars, 50 overlap)
- **Embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store** — ChromaDB (local) → Pinecone (production-ready)
- **Re-Ranking** — Cross-encoder `ms-marco-MiniLM-L-6-v2` for precision
- **Answer Generation** — GPT-4o / Mistral with inline `[N]` citations
- **REST API** — FastAPI with OpenAPI docs
- **Frontend** — Streamlit UI

---

## 🚀 Quick Start

### 1. Setup

```bash
git clone https://github.com/YOUR_USERNAME/documind-rag.git
cd documind-rag

python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 3. Run Backend

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: **http://localhost:8000/docs**

### 4. Run Frontend

```bash
streamlit run frontend/app.py
```

UI: **http://localhost:8501**

---

## 🐳 Docker

```bash
cd deployment/docker
docker-compose up --build
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/api/v1/ingest` | Upload & index document |
| `POST` | `/api/v1/query` | Ask a question |
| `GET` | `/api/v1/stats` | Index statistics |
| `DELETE` | `/api/v1/documents/{filename}` | Remove document |

### Example

```bash
# Ingest
curl -X POST http://localhost:8000/api/v1/ingest -F "file=@data/samples/sample_hr_policy.txt"

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the remote work policy?"}'
```

---

## ⚙️ Key Config (`.env`)

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required** |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `RETRIEVAL_TOP_K` | `20` | Candidates before rerank |
| `RERANKING_TOP_N` | `5` | Chunks sent to LLM |

---

## 🗂 Project Structure

```
src/
├── ingestion/     # Doc loading (PDF, DOCX, TXT)
├── chunking/      # Text splitting
├── embeddings/    # sentence-transformers
├── retrieval/     # Vector store + similarity search
├── reranking/     # Cross-encoder reranker
├── generation/    # LLM answer generation
├── api/           # FastAPI routes + schemas
├── utils/         # config, logger, exceptions
├── pipeline.py    # RAG orchestrator
└── main.py        # FastAPI app entry point
frontend/
└── app.py         # Streamlit UI
```

---

## 🧪 Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

*Built with LangChain · sentence-transformers · ChromaDB · FastAPI · Streamlit*
