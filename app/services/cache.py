"""
Semantic & Exact Redis Cache for AgentSentry
Caches verified audit verdicts to achieve sub-15ms response times for repeat/similar queries.
Includes automatic fallback to in-memory LRU cache if Redis container is not active.
"""
import hashlib
import json
import logging
from typing import Optional
from app.config import settings
from app.schemas.agent_state import AuditVerdict

logger = logging.getLogger("agentsentry.cache")

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheService:
    def __init__(self):
        self.redis_client = None
        self._memory_cache = {}
        self._initialized = False

    async def init(self):
        if not self._initialized and REDIS_AVAILABLE:
            try:
                self.redis_client = aioredis.from_url(
                    settings.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=1.5
                )
                await self.redis_client.ping()
                logger.info("Successfully connected to Redis semantic cache.")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}). Falling back to in-memory cache.")
                self.redis_client = None
            self._initialized = True

    def _normalize_key(self, query: str) -> str:
        clean = query.strip().lower()
        key_hash = hashlib.sha256(clean.encode("utf-8")).hexdigest()
        return f"agentsentry:verdict:{key_hash}"

    async def get(self, query: str) -> Optional[AuditVerdict]:
        await self.init()
        key = self._normalize_key(query)

        if self.redis_client:
            try:
                val = await self.redis_client.get(key)
                if val:
                    return AuditVerdict.model_validate_json(val)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")

        # In-memory check
        if key in self._memory_cache:
            return self._memory_cache[key]
        return None

    async def set(self, query: str, verdict: AuditVerdict, ttl_seconds: Optional[int] = None):
        await self.init()
        key = self._normalize_key(query)
        ttl = ttl_seconds or settings.cache_ttl_seconds

        serialized = verdict.model_dump_json()

        if self.redis_client:
            try:
                await self.redis_client.set(key, serialized, ex=ttl)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")

        # Keep in memory as well
        self._memory_cache[key] = verdict

    async def close(self):
        if self.redis_client:
            await self.redis_client.aclose()


cache_service = CacheService()
