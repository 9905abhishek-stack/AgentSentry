"""Environment configuration for the audit service."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "AgentSentry"
    app_env: str = "development"
    debug: bool = False
    port: int = Field(default=8000, ge=1, le=65535)
    host: str = "127.0.0.1"
    cors_origins: list[str] = Field(default_factory=list)

    qdrant_host: str | None = None
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_api_key: str | None = None
    qdrant_collection: str = "audit_evidence"
    hybrid_rrf_k: int = Field(default=60, ge=1)
    corpus_revision: str = "1"

    redis_url: str | None = None
    cache_ttl_seconds: int = Field(default=3600, ge=1)
    cache_max_entries: int = Field(default=128, ge=1)

    vllm_api_base: str | None = None
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_max_tokens: int = Field(default=2048, ge=1)
    max_verification_retries: int = Field(default=2, ge=0, le=3)

    evidence_locked_strict: bool = True
    confidence_threshold: float = Field(default=0.85, ge=0, le=1)

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
