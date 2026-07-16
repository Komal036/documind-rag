"""
DocuMind Configuration Management
----------------------------------
Centralised settings using Pydantic BaseSettings.
All values are read from environment variables (or .env file).
Never hard-code secrets — this module is the single source of truth.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_temperature: float = Field(default=0.0, alias="OPENAI_TEMPERATURE")
    openai_max_tokens: int = Field(default=1024, alias="OPENAI_MAX_TOKENS")
    mistral_api_key: str = Field(default="", alias="MISTRAL_API_KEY")
    mistral_model: str = Field(default="mistral-small", alias="MISTRAL_MODEL")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    llm_provider: Literal["openai", "mistral", "groq"] = Field(default="openai", alias="LLM_PROVIDER")


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_dimension: int = Field(default=384, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=64, alias="EMBEDDING_BATCH_SIZE")
    embedding_cache_dir: str = Field(
        default=str(PROJECT_ROOT / ".model_cache"), alias="EMBEDDING_CACHE_DIR"
    )


class VectorStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    vector_store_type: Literal["chroma", "pinecone", "pgvector"] = Field(
        default="chroma", alias="VECTOR_STORE_TYPE"
    )
    chroma_persist_dir: str = Field(
        default=str(PROJECT_ROOT / "vector_store" / "chroma_db"),
        alias="CHROMA_PERSIST_DIR",
    )
    chroma_collection_name: str = Field(
        default="documind_collection", alias="CHROMA_COLLECTION_NAME"
    )
    pinecone_api_key: str = Field(default="", alias="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="", alias="PINECONE_ENVIRONMENT")
    pinecone_index_name: str = Field(default="documind-index", alias="PINECONE_INDEX_NAME")


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")
    min_chunk_length: int = Field(default=50, alias="MIN_CHUNK_LENGTH")

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 512)
        if v >= chunk_size:
            raise ValueError(f"chunk_overlap ({v}) must be < chunk_size ({chunk_size})")
        return v


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    retrieval_top_k: int = Field(default=20, alias="RETRIEVAL_TOP_K")
    reranking_top_n: int = Field(default=5, alias="RERANKING_TOP_N")
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANKER_MODEL_NAME"
    )
    similarity_threshold: float = Field(default=0.3, alias="SIMILARITY_THRESHOLD")


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    database_url: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/documind",
        alias="DATABASE_URL",
    )
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_echo: bool = Field(default=False, alias="DB_ECHO")


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_chat_ttl_seconds: int = Field(default=86400, alias="REDIS_CHAT_TTL_SECONDS")
    redis_max_turns: int = Field(default=10, alias="REDIS_MAX_TURNS")


class SelfRAGSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    self_rag_max_retries: int = Field(default=2, alias="SELF_RAG_MAX_RETRIES")
    # Reranker (cross-encoder) raw score threshold above which retrieval is
    # considered "confident enough" to answer without reformulating the query.
    # Empirically, this reranker's scores for genuinely relevant top chunks
    # tend to land above ~1.0; weak/irrelevant matches trend negative.
    self_rag_confidence_threshold: float = Field(default=1.0, alias="SELF_RAG_CONFIDENCE_THRESHOLD")


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    jwt_secret_key: str = Field(
        default="dev-secret-change-in-prod", alias="JWT_SECRET_KEY"
    )
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=60 * 24, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"  # 24h default
    )


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_reload: bool = Field(default=True, alias="API_RELOAD")
    api_key: str = Field(default="dev-api-key-change-in-prod", alias="API_KEY")
    cors_origins: str = Field(default="http://localhost:8501", alias="CORS_ORIGINS")
    max_upload_size_mb: int = Field(default=50, alias="MAX_UPLOAD_SIZE_MB")
    allowed_file_types: str = Field(default="pdf,docx,txt,md", alias="ALLOWED_FILE_TYPES")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    @property
    def allowed_file_types_list(self) -> list[str]:
        return [ft.strip().lower() for ft in self.allowed_file_types.split(",")]


class LoggingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "console"] = Field(default="console", alias="LOG_FORMAT")
    log_file: str = Field(
        default=str(PROJECT_ROOT / "logs" / "documind.log"), alias="LOG_FILE"
    )


class Settings(BaseSettings):
    """
    Master settings object — the single source of truth for all configuration.

    Usage:
        from src.utils.config import get_settings
        cfg = get_settings()
        print(cfg.llm.openai_model)
    """
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    environment: Literal["development", "staging", "production"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"
    samples_dir: Path = PROJECT_ROOT / "data" / "samples"

    @property
    def llm(self) -> LLMSettings:
        return LLMSettings()

    @property
    def embedding(self) -> EmbeddingSettings:
        return EmbeddingSettings()

    @property
    def vector_store(self) -> VectorStoreSettings:
        return VectorStoreSettings()

    @property
    def chunking(self) -> ChunkingSettings:
        return ChunkingSettings()

    @property
    def retrieval(self) -> RetrievalSettings:
        return RetrievalSettings()

    @property
    def api(self) -> APISettings:
        return APISettings()

    @property
    def database(self) -> DatabaseSettings:
        return DatabaseSettings()

    @property
    def auth(self) -> AuthSettings:
        return AuthSettings()

    @property
    def redis(self) -> RedisSettings:
        return RedisSettings()

    @property
    def self_rag(self) -> SelfRAGSettings:
        return SelfRAGSettings()

    @property
    def logging(self) -> LoggingSettings:
        return LoggingSettings()

    def is_production(self) -> bool:
        return self.environment == "production"

    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a process-wide cached Settings instance.
    Call get_settings.cache_clear() in tests to reload.
    """
    return Settings()