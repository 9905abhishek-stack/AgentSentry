"""Async client for an OpenAI-compatible vLLM chat endpoint."""

import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class InferenceUnavailable(RuntimeError):
    """The model did not return a usable chat completion."""


class VLLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = base_url or settings.vllm_api_base
        self.model = model or settings.vllm_model
        self._client: httpx.AsyncClient | None = None

    def get_client(self) -> httpx.AsyncClient:
        if not self.base_url:
            raise InferenceUnavailable("Inference endpoint is disabled.")
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/") + "/",
                headers={"Authorization": f"Bearer {settings.vllm_api_key}"},
                timeout=httpx.Timeout(30.0, connect=3.0),
            )
        return self._client

    async def generate(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": settings.llm_max_tokens,
            "response_format": {"type": "json_object"},
        }
        try:
            response = await self.get_client().post("chat/completions", json=payload)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            content = choice["message"]["content"]
            if (
                choice.get("finish_reason") not in (None, "stop")
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise InferenceUnavailable("Incomplete or empty model output.")
            return content
        except (
            httpx.HTTPError,
            httpx.InvalidURL,
            ValueError,
            KeyError,
            IndexError,
            TypeError,
        ) as exc:
            logger.warning("Chat completion failed (%s)", type(exc).__name__)
            raise InferenceUnavailable("Inference request failed.") from exc

    async def health(self) -> str:
        if not self.base_url:
            return "disabled"
        try:
            response = await self.get_client().get("models", timeout=3.0)
            response.raise_for_status()
            if any(item["id"] == self.model for item in response.json()["data"]):
                return "available"
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, KeyError, TypeError):
            pass
        return "unavailable"

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


vllm_service = VLLMClient()
