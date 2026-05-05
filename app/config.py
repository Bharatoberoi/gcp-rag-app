from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _env_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).lower().strip()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return bool(v)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini API (only used for answer generation — 1 call per query)
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    # Local embeddings (sentence-transformers, no API key needed)
    embedding_dimensions: int = 384
    # Enable agentic features (multiple LLM calls per query — needs higher rate limits)
    agentic_mode: bool = False

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "documents_hybrid"
    dense_vector_weight: float = 0.7

    # Chunking
    max_chunk_tokens: int = 400
    chunk_overlap_tokens: int = 50

    # Retrieval
    adjacent_chunk_count: int = 1
    retrieval_multiplier: int = 2

    # Reranker (built-in uses Gemini; set reranker_url for external cross-encoder)
    reranker_url: str | None = None
    reranker_enabled: bool = True

    # BM25 state
    bm25_state_path: str = "./data/bm25_df.json"
    upload_dir: str = "./data/uploads"

    # Production
    production_mode: bool = False
    docs_enabled: bool = True
    cors_origins: str = ""
    api_keys: str = ""

    @field_validator("qdrant_url", mode="before")
    @classmethod
    def strip_qdrant_url(cls, v: object) -> object:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("production_mode", "docs_enabled", "reranker_enabled", mode="before")
    @classmethod
    def coerce_bool_flags(cls, v: object) -> bool:
        return _env_bool(v)


settings = Settings()
