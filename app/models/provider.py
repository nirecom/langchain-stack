"""
LLM provider with direct endpoint fallback.

Probes llama-swap endpoints before chain execution, falling back
to LiteLLM Proxy only when all direct endpoints are unreachable.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx
from langchain_openai import ChatOpenAI
from settings import settings

logger = logging.getLogger(__name__)

_DEAD_TTL = 300  # seconds before a dead endpoint is re-probed


# --- EndpointHealth ---


class EndpointHealth:
    """Track dead endpoints with a TTL-based expiry."""

    def __init__(self, ttl: float = _DEAD_TTL) -> None:
        self._dead: dict[str, float] = {}  # url -> monotonic timestamp
        self._ttl = ttl

    def is_alive(self, url: str) -> bool:
        ts = self._dead.get(url)
        if ts is None:
            return True
        if time.monotonic() - ts > self._ttl:
            del self._dead[url]
            return True
        return False

    def mark_dead(self, url: str) -> None:
        self._dead[url] = time.monotonic()


_endpoint_health = EndpointHealth()


# --- Endpoint list builder ---

_ENDPOINT_DEFS = [
    # (slot, url_attr, model_template, is_cloud)
    ("local", "llama_server_url", "{role}_local_model", False),
    ("portable", "portable_llm_server_url", "{role}_portable_model", False),
    ("cloud", "cloud_api_url", "{role}_cloud_model", True),
]


def _build_endpoints(role: str) -> list[dict[str, Any]]:
    """Build ordered endpoint list for a role from env vars."""
    endpoints: list[dict[str, Any]] = []
    model_cfg = settings.models.get(role, {})
    ep_timeouts = model_cfg.get("endpoints", {})

    for slot, url_attr, model_tmpl, is_cloud in _ENDPOINT_DEFS:
        url = getattr(settings, url_attr, "")
        model_attr = model_tmpl.format(role=role)
        model = getattr(settings, model_attr, "")

        if not url or not model:
            continue

        # Strip openai/ prefix (LiteLLM convention, not needed for direct)
        if model.startswith("openai/"):
            model = model[len("openai/"):]

        timeout = ep_timeouts.get(slot, {}).get(
            "timeout", model_cfg.get("timeout", 120)
        )
        api_key = settings.cloud_api_key if is_cloud else "not-needed"

        endpoints.append({
            "url": url,
            "model": model,
            "timeout": timeout,
            "api_key": api_key,
            "is_cloud": is_cloud,
        })

    return endpoints


# --- Probe ---


async def _probe_single(url: str, timeout: float) -> bool:
    """Probe a single endpoint. Returns True if alive."""
    async with httpx.AsyncClient(verify=False, timeout=timeout) as client:
        resp = await client.get(f"{url}/models")
        resp.raise_for_status()
    return True


async def probe_endpoints() -> EndpointHealth:
    """Probe all non-cloud endpoints and update health state."""
    global _endpoint_health

    # Collect unique non-cloud URLs from all roles
    urls_to_probe: set[str] = set()
    for role in ("reasoner", "judge"):
        for ep in _build_endpoints(role):
            if not ep["is_cloud"]:
                urls_to_probe.add(ep["url"])

    timeout = settings.health_probe_timeout

    async def _probe_one(url: str) -> None:
        try:
            await _probe_single(url, timeout)
            logger.info("Endpoint alive: %s", url)
        except Exception:
            logger.warning("Endpoint dead: %s", url)
            _endpoint_health.mark_dead(url)

    await asyncio.gather(*[_probe_one(u) for u in urls_to_probe])
    return _endpoint_health


# --- LLM selection ---


def _get_llm_for_role(role: str, temperature: float | None = None) -> ChatOpenAI:
    """Select the first alive direct endpoint, or fall back to LiteLLM."""
    model_cfg = settings.models.get(role, {})
    temp = (
        temperature if temperature is not None
        else model_cfg.get("temperature", 0.7)
    )

    endpoints = _build_endpoints(role)
    for ep in endpoints:
        if _endpoint_health.is_alive(ep["url"]):
            kwargs: dict[str, Any] = {
                "base_url": ep["url"],
                "api_key": ep["api_key"],
                "model": ep["model"],
                "temperature": temp,
                "timeout": ep["timeout"],
            }
            # Self-signed certs on local llama-swap
            if not ep["is_cloud"]:
                kwargs["http_async_client"] = httpx.AsyncClient(verify=False)
            return ChatOpenAI(**kwargs)

    # All direct endpoints dead or none configured -> LiteLLM fallback
    logger.warning("All direct endpoints down for %s, using LiteLLM", role)
    return ChatOpenAI(
        base_url=settings.litellm_proxy_url,
        api_key=settings.litellm_api_key,
        model=model_cfg.get("litellm_model_name", role),
        temperature=temp,
        timeout=model_cfg.get("timeout", 120),
    )


def get_reasoner(temperature: float | None = None) -> ChatOpenAI:
    """Get reasoner LLM (direct endpoint with LiteLLM fallback)."""
    return _get_llm_for_role("reasoner", temperature)


def get_judge(temperature: float | None = None) -> ChatOpenAI:
    """Get judge LLM (direct endpoint with LiteLLM fallback)."""
    return _get_llm_for_role("judge", temperature)
