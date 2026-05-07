"""FastAPI entrypoint for the RAG application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.deps import require_api_key_if_configured
from app.rag_pipeline import RagPipeline
from app.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.production_mode:
        qu = (settings.qdrant_url or "").strip().lower()
        if qu in {"", "memory", ":memory:", "inmem"}:
            raise RuntimeError(
                "PRODUCTION_MODE=1 requires a managed QDRANT_URL (not memory). "
                "Set QDRANT_URL to your Qdrant Cloud cluster URL."
            )

    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.bm25_state_path).parent.mkdir(parents=True, exist_ok=True)
    app.state.rag = RagPipeline()
    yield


def _qdrant_health_headers() -> dict[str, str]:
    h: dict[str, str] = {}
    if settings.qdrant_api_key:
        h["api-key"] = settings.qdrant_api_key
    return h


def create_app() -> FastAPI:
    docs = settings.docs_enabled
    app = FastAPI(
        title="Production RAG",
        lifespan=lifespan,
        docs_url="/docs" if docs else None,
        redoc_url="/redoc" if docs else None,
        openapi_url="/openapi.json" if docs else None,
    )

    cors = [o.strip() for o in (settings.cors_origins or "").split(",") if o.strip()]
    if cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    if STATIC_DIR.is_dir():
        app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="ui")

    def _rag() -> RagPipeline:
        return app.state.rag

    @app.get("/health", response_model=HealthResponse)
    async def health():
        qdrant_ok = "unknown"
        qu = (settings.qdrant_url or "").strip().lower()
        if qu in {"memory", ":memory:", "inmem"}:
            qdrant_ok = "in_memory"
        else:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    base = settings.qdrant_url.rstrip("/")
                    r = await client.get(
                        f"{base}/collections",
                        headers=_qdrant_health_headers(),
                    )
                    qdrant_ok = "ok" if r.status_code < 400 else f"status_{r.status_code}"
            except Exception as e:
                qdrant_ok = f"error:{type(e).__name__}"

        llm_ok = "skipped"
        if settings.groq_api_key:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.get(
                        "https://api.groq.com/openai/v1/models",
                        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                    )
                    llm_ok = "ok" if r.status_code < 400 else f"status_{r.status_code}"
            except Exception as e:
                llm_ok = f"error:{type(e).__name__}"
        embeddings_ok = "loading"
        try:
            _rag().gemini.init()
            embeddings_ok = "ok (local)"
        except Exception as e:
            embeddings_ok = f"error:{type(e).__name__}"

        rer = "disabled"
        if settings.reranker_enabled and settings.reranker_url:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    r = await client.get(settings.reranker_url.rstrip("/") + "/health")
                    rer = "ok" if r.status_code < 400 else f"status_{r.status_code}"
            except Exception as e:
                rer = f"error:{type(e).__name__}"

        return HealthResponse(status="ok", qdrant=qdrant_ok, embeddings=embeddings_ok, llm=llm_ok, reranker=rer)

    @app.post("/v1/ingest", response_model=IngestResponse)
    async def ingest(
        file: UploadFile = File(...),
        _: None = Depends(require_api_key_if_configured),
    ):
        if not file.filename:
            raise HTTPException(400, "filename required")
        raw = await file.read()
        if not raw:
            raise HTTPException(400, "empty file")
        if len(raw) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / (1024 * 1024)
            raise HTTPException(413, f"file too large; max upload size is {limit_mb:g} MB")
        try:
            n = await _rag().ingest_bytes_async(file.filename, raw)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except Exception as e:
            raise HTTPException(500, f"ingest failed: {e}") from e
        return IngestResponse(
            document=Path(file.filename).name,
            chunks_indexed=n,
            message="indexed" if n else "no text extracted",
        )

    @app.delete("/v1/documents/{name:path}")
    def delete_document(
        name: str,
        _: None = Depends(require_api_key_if_configured),
    ):
        _rag().delete_document(name)
        return {"deleted": name}

    @app.post("/v1/query", response_model=QueryResponse)
    async def query(
        body: QueryRequest,
        _: None = Depends(require_api_key_if_configured),
    ):
        if not body.question.strip():
            raise HTTPException(400, "question required")
        try:
            answer, sources = await _rag().answer_async(body.question, body.top_k)
        except Exception as e:
            raise HTTPException(500, f"query failed: {e}") from e
        return QueryResponse(answer=answer, sources=sources)

    @app.get("/")
    async def root():
        if STATIC_DIR.is_dir():
            return RedirectResponse(url="/ui/", status_code=302)
        return {
            "service": "rag-app",
            "ui": "/ui/",
            "techniques": [
                "hybrid_dense_sparse_bm25_rrf",
                "adjacent_chunk_expansion",
                "cross_encoder_rerank_optional",
                "local_hash_embeddings_and_groq_generation",
            ],
            "docs": "/docs" if settings.docs_enabled else None,
        }

    return app


app = create_app()
