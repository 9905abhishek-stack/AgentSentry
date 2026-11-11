"""TTL cache for completed corpus audits, with a bounded local fallback."""

import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import settings
from app.schemas.agent_state import AuditResponse

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self) -> None:
        self._redis: Redis | None = None
        self._memory: OrderedDict[str, tuple[float, str]] = OrderedDict()

    def init(self) -> None:
        if settings.redis_url and self._redis is None:
            try:
                self._redis = Redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=1.0,
                    socket_timeout=1.0,
                )
            except ValueError:
                logger.warning("Redis configuration is invalid. Using the local cache.")

    @staticmethod
    def key(query: str, strict: bool) -> str:
        components = [
            query,
            strict,
            settings.qdrant_host,
            settings.qdrant_port,
            settings.qdrant_collection,
            settings.corpus_revision,
            settings.hybrid_rrf_k,
            settings.vllm_api_base,
            settings.vllm_model,
            settings.llm_max_tokens,
            settings.max_verification_retries,
            settings.confidence_threshold,
        ]
        digest = hashlib.sha256(json.dumps(components).encode("utf-8")).hexdigest()
        return f"agentsentry:audit:v3:{digest}"

    async def get(self, key: str) -> AuditResponse | None:
        if self._redis is not None:
            try:
                value = await asyncio.wait_for(self._redis.get(key), timeout=1.5)
                if value is not None:
                    return AuditResponse.model_validate_json(value)
            except (RedisError, OSError, TimeoutError, ValidationError):
                logger.warning("Redis cache lookup failed. Checking the local cache.")

        cached = self._memory.get(key)
        if cached is None:
            return None
        expires, value = cached
        if expires <= time.monotonic():
            self._memory.pop(key, None)
            return None
        self._memory.move_to_end(key)
        return AuditResponse.model_validate_json(value)

    async def set(self, key: str, response: AuditResponse) -> None:
        value = response.model_dump_json()
        ttl = settings.cache_ttl_seconds
        if self._redis is not None:
            try:
                await asyncio.wait_for(self._redis.set(key, value, ex=ttl), timeout=1.5)
                self._memory.pop(key, None)
                return
            except (RedisError, OSError, TimeoutError):
                logger.warning("Redis cache write failed. Using the local cache.")

        self._memory[key] = (time.monotonic() + ttl, value)
        self._memory.move_to_end(key)
        while len(self._memory) > settings.cache_max_entries:
            self._memory.popitem(last=False)

    async def health(self) -> str:
        if self._redis is None:
            return "unavailable" if settings.redis_url else "disabled"
        try:
            if await asyncio.wait_for(self._redis.ping(), timeout=1.5):
                return "available"
        except (RedisError, OSError, TimeoutError):
            pass
        return "unavailable"

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None
        self._memory.clear()


cache_service = CacheService()
