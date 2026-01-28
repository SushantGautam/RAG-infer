# Quick Start Guide

## ✅ Prerequisites

- **Python 3.8+** (the project is tested on modern Python versions; the dev container uses Python 3.12)
- `pip` to install dependencies
- **OpenAI API key** (required by default) or an OpenAI-compatible LLM endpoint (see notes below)

> Note: The project uses a local file-based Milvus (via `milvus-lite`) for the vector store, so you don't need a separate Milvus service or Docker unless you prefer one.

## 🚀 Setup Steps

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

The server requires an OpenAI API key by default. You can provide it via an environment variable or a `.env` file:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
cp .env.example .env
# Edit .env and add your API key (OPENAI_API_KEY=...)
```

> Tip: You can pass `--openai-api-key` when starting `rag_server.py`.
> 
> To **secure** this server so only authorized clients can call it, set `API_SECRET` (or `OPENAI_API_SECRET`) in your `.env` or pass `--api-secret` when starting. When `API_SECRET` is set, include it in requests using `Authorization: Bearer $API_SECRET` or `x-api-secret: $API_SECRET`.

### 3. Index your documents

Before starting the server, run the ingestion step to load and embed documents:

```bash
python ingest_documents.py
```

This will:
- Load files from `./documents` (default `**/*.txt`, configurable)
- Split text into chunks
- Create embeddings and store them in a local Milvus file (`./milvus_demo.db` by default)

Example with custom embedding endpoint or API key:

```bash
python ingest_documents.py \
  --documents-path ./documents \
  --milvus-db ./milvus_demo.db \
  --embedding-base-url http://localhost:8002/v1 \
  --embedding-model-name text-embedding-ada-002 \
  --openai-api-key your-api-key
```

Use `--recreate` to rebuild the collection from scratch.

### 4. Start the server

Run directly with Python (entrypoint runs Uvicorn for you):

```bash
python rag_server.py
```

Or run Uvicorn yourself if you prefer:

```bash
uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

Or use Astral `uv` to run the console script directly:

```bash
uv run --refresh --with  git+https://github.com/SushantGautam/vllm-RAG.git rag-server
```

You can set a `.env` file in the current directory and `uv run ` will pick it up so environment variables (like `OPENAI_API_KEY` or `API_SECRET`) are available to the process.

Note about `API_SECRET` and how to provide it:
- If starting the server via `python rag_server.py`, you can pass `--api-secret` on the command line.
- If running `uvicorn rag_server:app` (which imports the module instead of running the script), set `API_SECRET` (or `OPENAI_API_SECRET`) in the environment or in your `.env` before starting the process so the import-time dotenv load picks it up. For example:

```bash
export API_SECRET="your-secret"
uvicorn rag_server:app --host 0.0.0.0 --port 8000
```

When deployed under a process manager (systemd, Docker, etc.), set the environment variable in the service/unit or container environment to keep the secret out of source control.

Default host/port: `0.0.0.0:8000` (configurable via CLI flags). The server loads the pre-built Milvus DB at startup and will fail if the DB does not exist.

### 5. Test the server

- Interactive API docs (Swagger): http://localhost:8000/docs

- Health check:

```bash
curl http://localhost:8000/health
```

- Ask a question (OpenAI-compatible endpoint):

Unauthenticated example (when `API_SECRET` is not set):

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is FastAPI?"}]
  }'
```

Authenticated example (when `API_SECRET` is set):

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_SECRET" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is FastAPI?"}]
  }'
```

- Using the example client:

```bash
# Interactive mode
python client.py

# Single question
python client.py --question "What is FastAPI?"

# Health check
python client.py --health
```

## 🔧 Configuration

All runtime options are available via CLI flags (run `--help` for details):

```bash
python rag_server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --milvus-db ./milvus_demo.db \
  --collection-name rag_collection \
  --openai-api-key your-api-key \
  --openai-base-url http://localhost:8001/v1 \
  --model-name gpt-3.5-turbo
```

See `python rag_server.py --help` and `python ingest_documents.py --help` for full options.

## 📁 Adding Your Own Documents

1. Add `.txt` files (or other supported formats) to the `documents` directory (or point `--documents-path` to another folder).
2. Run `python ingest_documents.py --recreate` to (re)populate the collection.
3. Restart the server to pick up the new DB file.

> Note: The server reads the Milvus DB at startup, so you must recreate/re-ingest and restart to update the in-memory retriever.

## 🛠 Troubleshooting

- OpenAI API Error: verify `OPENAI_API_KEY` or `--openai-api-key` and ensure the key is valid. If using a local OpenAI-compatible endpoint, use `--openai-base-url` and provide any required keys for that service.

- Server won't start: check if port is in use or if `./milvus_demo.db` exists. Try `python rag_server.py --port 8001` or re-run ingestion.

- Database issues: use `python ingest_documents.py --recreate` to rebuild the collection or delete `./milvus_demo.db` if you prefer to start fresh.

## 🧭 Architecture Overview

```
┌──────────────────┐
│  ingest_documents │
│      .py         │
└────────┬─────────┘
         │
         │ 1. Load & split documents
         │ 2. Create embeddings
         │ 3. Store in local Milvus (milvus-lite)
         ▼
   ┌─────────────┐
   │ milvus_demo │
   │    .db      │
   │ (local file)│
   └──────┬──────┘
          │
          │ Read vectors
          │
┌─────────▼───────────────────────────────┐
│         rag_server.py (FastAPI)         │
│  ┌──────────────────────────────────┐  │
│  │   Lifespan (Startup)             │  │
│  │   - Validate API key             │  │
│  │   - Initialize embeddings        │  │
│  │   - Connect to local Milvus DB   │  │
│  │   - Initialize retriever & LLM   │  │
│  └──────────────────────────────────┘  │
│                                         │
│  ┌──────────────────────────────────┐  │
│  │   POST /v1/chat/completions      │  │
│  │   - Receive OpenAI-style request │  │
│  │   - Run retrieval + LLM          │  │
│  │   - Return OpenAI-compatible     │  │
│  └──────────────────────────────────┘  │
└─────────────────┬───────────────────────┘
                  │
                  │ API calls (LLM & embeddings)
                  ▼
            ┌──────────┐
            │  OpenAI  │
            │ or other │
            │ compatible│
            └──────────┘
```

---

## Next Steps

- Add more document formats (PDF, Word, etc.)
- Implement authentication
- Add rate limiting
- Set up monitoring and logging
- Deploy to production
