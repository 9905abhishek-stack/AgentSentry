"""
AgentSentry Configuration Module
Uses Pydantic Settings to manage environment variables and system thresholds.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "AgentSentry"
    app_env: str = "development"
    debug: bool = False
    port: int = 8000
    host: str = "0.0.0.0"

    # Qdrant Vector Store (Hybrid Retrieval)
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "audit_evidence"
    dense_vector_dim: int = 1024  # BAAI/bge-m3 dense dimension
    hybrid_rrf_k: int = 60         # Reciprocal Rank Fusion constant

    # Redis Semantic Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    semantic_cache_similarity_threshold: float = 0.94

    # vLLM Serving (OpenAI-compatible inference endpoint)
    vllm_api_base: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"
    vllm_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # Verification & Audit Protocol
    evidence_locked_strict: bool = True
    confidence_threshold: float = 0.85
    max_verification_retries: int = 2

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
