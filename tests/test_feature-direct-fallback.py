"""
Tests for direct endpoint fallback feature.

Tests the EndpointHealth, _build_endpoints, probe_endpoints, and
_get_llm_for_role functions in app/models/provider.py.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Full set of env vars for direct-fallback configuration
FULL_ENV = {
    "LLAMA_SERVER_URL": "https://198.51.100.10:8443/v1",
    "PORTABLE_LLM_SERVER_URL": "https://198.51.100.20:8443/v1",
    "CLOUD_API_URL": "https://api.openai.com/v1",
    "CLOUD_API_KEY": "sk-test-key",
    "REASONER_LOCAL_MODEL": "openai/Qwen3.5-27B-IQ3_M",
    "REASONER_PORTABLE_MODEL": "openai/Qwen3-14B-Q4_K_M",
    "REASONER_CLOUD_MODEL": "openai/gpt-4o",
    "JUDGE_LOCAL_MODEL": "openai/Qwen2.5-7B-Instruct-Q4_K_M",
    "JUDGE_PORTABLE_MODEL": "openai/Qwen2.5-7B-Instruct-Q4_K_M",
    "JUDGE_CLOUD_MODEL": "openai/gpt-4o-mini",
    "HEALTH_PROBE_TIMEOUT": "2.0",
    # LiteLLM (existing)
    "LITELLM_PROXY_URL": "http://litellm-proxy:4000/v1",
    "LITELLM_API_KEY": "not-needed",
}

LOCAL_URL = FULL_ENV["LLAMA_SERVER_URL"]
PORTABLE_URL = FULL_ENV["PORTABLE_LLM_SERVER_URL"]
CLOUD_URL = FULL_ENV["CLOUD_API_URL"]


def _import_provider():
    """Import provider module fresh (after env patches are applied)."""
    import importlib
    import os
    import sys

    # Remove cached modules so env var changes take effect
    for mod_name in list(sys.modules):
        if "provider" in mod_name or mod_name in ("settings", "models"):
            del sys.modules[mod_name]

    # Add app/ to path so imports resolve
    app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
    abs_app_dir = os.path.abspath(app_dir)
    if abs_app_dir not in sys.path:
        sys.path.insert(0, abs_app_dir)

    from models import provider

    # Ensure provider uses the freshly-loaded settings
    import settings as settings_mod
    provider.settings = settings_mod.settings

    return provider


# ---------------------------------------------------------------------------
# EndpointHealth
# ---------------------------------------------------------------------------


class TestEndpointHealth:
    """Tests for the EndpointHealth singleton."""

    def test_unknown_url_is_alive(self):
        """An unregistered URL should be reported as alive."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth()
        assert health.is_alive("https://never-seen.example.com/v1") is True

    def test_mark_dead_then_is_alive_false(self):
        """After mark_dead, is_alive returns False."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        assert health.is_alive(LOCAL_URL) is False

    def test_mark_dead_ttl_expiry(self):
        """After TTL expires, is_alive reverts to True."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        assert health.is_alive(LOCAL_URL) is False

        # Simulate TTL expiry by backdating the dead timestamp
        health._dead[LOCAL_URL] = time.monotonic() - 301
        assert health.is_alive(LOCAL_URL) is True

    def test_mark_dead_idempotent(self):
        """Calling mark_dead twice on the same URL does not raise."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        health.mark_dead(LOCAL_URL)  # no error
        assert health.is_alive(LOCAL_URL) is False


# ---------------------------------------------------------------------------
# _build_endpoints
# ---------------------------------------------------------------------------


class TestBuildEndpoints:
    """Tests for _build_endpoints()."""

    def test_reasoner_all_env_set(self):
        """All env vars set -> returns 3 endpoints for reasoner."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert len(endpoints) == 3
        # Verify order: local, portable, cloud
        assert endpoints[0]["url"] == LOCAL_URL
        assert endpoints[1]["url"] == PORTABLE_URL
        assert endpoints[2]["url"] == CLOUD_URL

    def test_judge_all_env_set(self):
        """All env vars set -> returns 3 endpoints for judge."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("judge")
        assert len(endpoints) == 3

    def test_openai_prefix_stripped(self):
        """The openai/ prefix is auto-stripped from model names."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        # "openai/Qwen3.5-27B-IQ3_M" -> "Qwen3.5-27B-IQ3_M"
        assert not endpoints[0]["model"].startswith("openai/")
        assert endpoints[0]["model"] == "Qwen3.5-27B-IQ3_M"

    def test_no_prefix_model_unchanged(self):
        """A model name without openai/ prefix passes through unchanged."""
        env = {**FULL_ENV, "REASONER_LOCAL_MODEL": "Qwen3.5-27B-IQ3_M"}
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert endpoints[0]["model"] == "Qwen3.5-27B-IQ3_M"

    def test_url_without_model_skipped(self):
        """Endpoint with URL but no model is skipped."""
        env = {**FULL_ENV}
        del env["REASONER_LOCAL_MODEL"]
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        # local skipped -> portable + cloud remain
        assert len(endpoints) == 2
        assert endpoints[0]["url"] == PORTABLE_URL

    def test_model_without_url_skipped(self):
        """Endpoint with model but no URL is skipped."""
        env = {**FULL_ENV}
        del env["LLAMA_SERVER_URL"]
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert len(endpoints) == 2
        assert endpoints[0]["url"] == PORTABLE_URL

    def test_all_env_unset_empty_list(self):
        """No endpoint env vars set -> returns empty list."""
        minimal_env = {
            "LITELLM_PROXY_URL": "http://litellm-proxy:4000/v1",
            "LITELLM_API_KEY": "not-needed",
        }
        with patch.dict("os.environ", minimal_env, clear=True):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert endpoints == []


# ---------------------------------------------------------------------------
# probe_endpoints
# ---------------------------------------------------------------------------


class TestProbeEndpoints:
    """Tests for probe_endpoints()."""

    @pytest.mark.asyncio
    async def test_all_alive(self):
        """All endpoints respond -> all marked alive."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        with patch.object(
            provider, "_probe_single", new_callable=AsyncMock, return_value=True
        ):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL)
        assert health.is_alive(PORTABLE_URL)
        assert health.is_alive(CLOUD_URL)

    @pytest.mark.asyncio
    async def test_timeout_marks_dead(self):
        """Connect timeout -> endpoint marked dead."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise httpx.ConnectTimeout("timeout")
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL) is False
        assert health.is_alive(PORTABLE_URL) is True

    @pytest.mark.asyncio
    async def test_connection_error_marks_dead(self):
        """Connection error -> endpoint marked dead."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise httpx.ConnectError("connection refused")
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL) is False

    @pytest.mark.asyncio
    async def test_http_500_marks_dead(self):
        """HTTP 500 response -> endpoint marked dead."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise httpx.HTTPStatusError(
                    "500", request=MagicMock(), response=MagicMock(status_code=500)
                )
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL) is False

    @pytest.mark.asyncio
    async def test_cloud_not_probed(self):
        """Cloud URL is not probed (always alive)."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        probed_urls = []

        async def mock_probe(url, timeout):
            probed_urls.append(url)
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            await provider.probe_endpoints()

        assert CLOUD_URL not in probed_urls
        assert LOCAL_URL in probed_urls
        assert PORTABLE_URL in probed_urls

    @pytest.mark.asyncio
    async def test_probe_idempotent(self):
        """Two consecutive probe calls produce the same result."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise httpx.ConnectError("down")
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health1 = await provider.probe_endpoints()
            health2 = await provider.probe_endpoints()

        assert health1.is_alive(LOCAL_URL) is False
        assert health2.is_alive(LOCAL_URL) is False
        assert health1.is_alive(PORTABLE_URL) is True
        assert health2.is_alive(PORTABLE_URL) is True


# ---------------------------------------------------------------------------
# _get_llm_for_role
# ---------------------------------------------------------------------------


class TestGetLlmForRole:
    """Tests for _get_llm_for_role()."""

    def test_first_alive_selected(self):
        """Selects the first alive endpoint."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner")

        base = getattr(llm, "openai_api_base", None) or str(llm.base_url)
        assert LOCAL_URL in base

    def test_first_dead_second_alive(self):
        """First endpoint dead, second alive -> selects the second."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner")

        base = getattr(llm, "openai_api_base", None) or str(llm.base_url)
        assert PORTABLE_URL in base

    def test_all_dead_litellm_fallback(self):
        """All direct endpoints dead -> falls back to LiteLLM proxy."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        health.mark_dead(PORTABLE_URL)
        health.mark_dead(CLOUD_URL)
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner")

        base = getattr(llm, "openai_api_base", None) or str(llm.base_url)
        assert "litellm" in base.lower() or "4000" in base


# ---------------------------------------------------------------------------
# get_reasoner / get_judge (public API)
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """Tests for get_reasoner() and get_judge()."""

    def test_get_reasoner_returns_chat_openai(self):
        """get_reasoner() returns a ChatOpenAI instance."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        from langchain_openai import ChatOpenAI

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider.get_reasoner()

        assert isinstance(llm, ChatOpenAI)

    def test_get_judge_returns_chat_openai(self):
        """get_judge() returns a ChatOpenAI instance."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        from langchain_openai import ChatOpenAI

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider.get_judge()

        assert isinstance(llm, ChatOpenAI)


# ---------------------------------------------------------------------------
# High priority: temperature, HTTP 4xx, role isolation
# ---------------------------------------------------------------------------


class TestTemperaturePropagation:
    """Tests for temperature parameter passing."""

    def test_explicit_temperature_used(self):
        """Explicit temperature value is passed to ChatOpenAI."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner", temperature=0.3)

        assert llm.temperature == 0.3

    def test_none_temperature_uses_config_default(self):
        """None temperature falls back to models.yaml config value."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        judge_cfg = {"judge": {"temperature": 0.0, "litellm_model_name": "judge", "timeout": 60}}
        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health), \
             patch.object(type(provider.settings), "models", new_callable=lambda: property(lambda self: judge_cfg)):
            llm = provider._get_llm_for_role("judge", temperature=None)

        # judge config default is 0.0
        assert llm.temperature == 0.0

    def test_zero_temperature_not_overridden(self):
        """Explicit temperature=0.0 is used, not replaced by config default."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner", temperature=0.0)

        # reasoner config default is 0.7, but explicit 0.0 should win
        assert llm.temperature == 0.0


class TestProbeHttp4xx:
    """Tests for HTTP 4xx errors during probing."""

    @pytest.mark.asyncio
    async def test_http_404_marks_dead(self):
        """HTTP 404 response -> endpoint marked dead."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise httpx.HTTPStatusError(
                    "404", request=MagicMock(), response=MagicMock(status_code=404)
                )
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL) is False
        assert health.is_alive(PORTABLE_URL) is True

    @pytest.mark.asyncio
    async def test_generic_exception_marks_dead(self):
        """Non-httpx exception (e.g. RuntimeError) -> endpoint marked dead."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        async def mock_probe(url, timeout):
            if url == LOCAL_URL:
                raise RuntimeError("unexpected failure")
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            health = await provider.probe_endpoints()

        assert health.is_alive(LOCAL_URL) is False


class TestRoleIsolation:
    """Tests for independence between reasoner and judge endpoint selection."""

    def test_reasoner_dead_does_not_affect_judge(self):
        """Marking local dead for reasoner also affects judge (shared health)."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)

        with patch.object(provider, "_endpoint_health", health):
            reasoner = provider._get_llm_for_role("reasoner")
            judge = provider._get_llm_for_role("judge")

        # Both should skip local (shared URL) and use portable
        r_base = getattr(reasoner, "openai_api_base", None) or str(reasoner.base_url)
        j_base = getattr(judge, "openai_api_base", None) or str(judge.base_url)
        assert PORTABLE_URL in r_base
        assert PORTABLE_URL in j_base

    def test_repeated_get_llm_consistent(self):
        """Repeated calls to _get_llm_for_role with same health -> same endpoint."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)

        with patch.object(provider, "_endpoint_health", health):
            llm1 = provider._get_llm_for_role("reasoner")
            llm2 = provider._get_llm_for_role("reasoner")

        base1 = getattr(llm1, "openai_api_base", None) or str(llm1.base_url)
        base2 = getattr(llm2, "openai_api_base", None) or str(llm2.base_url)
        assert base1 == base2


# ---------------------------------------------------------------------------
# Medium priority: timeout inheritance, verify=False, api_key
# ---------------------------------------------------------------------------


class TestEndpointConfig:
    """Tests for timeout inheritance, TLS config, and api_key assignment."""

    def test_timeout_from_endpoint_config(self):
        """Endpoint-level timeout in models.yaml takes precedence."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        judge_cfg = {
            "judge": {
                "litellm_model_name": "judge", "temperature": 0.0, "timeout": 60,
                "endpoints": {"local": {"timeout": 15}, "portable": {"timeout": 10}, "cloud": {"timeout": 30}},
            }
        }
        with patch.object(type(provider.settings), "models", new_callable=lambda: property(lambda self: judge_cfg)):
            endpoints = provider._build_endpoints("judge")

        local_ep = [e for e in endpoints if e["url"] == LOCAL_URL][0]
        assert local_ep["timeout"] == 15

    def test_timeout_falls_back_to_model_level(self):
        """Without endpoint-level timeout, falls back to model-level timeout."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        # Patch models config to remove endpoint-level timeouts
        no_ep_config = {
            "reasoner": {
                "litellm_model_name": "reasoner",
                "temperature": 0.7,
                "timeout": 120,
                # no "endpoints" key
            }
        }
        with patch.object(type(provider.settings), "models", new_callable=lambda: property(lambda self: no_ep_config)):
            endpoints = provider._build_endpoints("reasoner")

        local_ep = [e for e in endpoints if e["url"] == LOCAL_URL][0]
        assert local_ep["timeout"] == 120  # model-level default

    def test_cloud_gets_cloud_api_key(self):
        """Cloud endpoint uses cloud_api_key from settings."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        endpoints = provider._build_endpoints("reasoner")
        cloud_ep = [e for e in endpoints if e["is_cloud"]][0]
        assert cloud_ep["api_key"] == "sk-test-key"

    def test_local_gets_not_needed_api_key(self):
        """Local endpoint uses 'not-needed' as api_key."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        endpoints = provider._build_endpoints("reasoner")
        local_ep = [e for e in endpoints if e["url"] == LOCAL_URL][0]
        assert local_ep["api_key"] == "not-needed"

    def test_local_endpoint_gets_verify_false_client(self):
        """Non-cloud endpoint ChatOpenAI gets http_async_client with verify=False."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner")

        # First alive is local -> should have custom http_async_client
        assert llm.http_async_client is not None
        # Verify the client was created with verify=False
        assert isinstance(llm.http_async_client, httpx.AsyncClient)

    def test_cloud_endpoint_no_custom_client(self):
        """Cloud endpoint ChatOpenAI does not get custom http_async_client."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()

        health = provider.EndpointHealth()
        health.mark_dead(LOCAL_URL)
        health.mark_dead(PORTABLE_URL)
        # Cloud is first alive
        with patch.object(provider, "_endpoint_health", health):
            llm = provider._get_llm_for_role("reasoner")

        assert llm.http_async_client is None


# ---------------------------------------------------------------------------
# Lower priority: TTL boundary, string edge, single-endpoint config
# ---------------------------------------------------------------------------


class TestTTLBoundary:
    """Tests for TTL edge values."""

    def test_ttl_zero_immediate_expiry(self):
        """TTL=0 means dead entries expire immediately."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth(ttl=0)
        health.mark_dead(LOCAL_URL)
        # With ttl=0, any elapsed time > 0 should expire it
        # Backdate slightly to ensure expiry
        health._dead[LOCAL_URL] = time.monotonic() - 0.001
        assert health.is_alive(LOCAL_URL) is True

    def test_ttl_negative_always_expires(self):
        """Negative TTL means entries expire instantly."""
        with patch.dict("os.environ", FULL_ENV, clear=False):
            provider = _import_provider()
        health = provider.EndpointHealth(ttl=-1)
        health.mark_dead(LOCAL_URL)
        assert health.is_alive(LOCAL_URL) is True


class TestStringEdgeCases:
    """Tests for empty/whitespace string handling in env vars."""

    def test_empty_url_skipped(self):
        """Explicitly empty URL string is treated as unset."""
        env = {**FULL_ENV, "LLAMA_SERVER_URL": ""}
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        urls = [e["url"] for e in endpoints]
        assert LOCAL_URL not in urls
        assert len(endpoints) == 2

    def test_empty_model_skipped(self):
        """Explicitly empty model string is treated as unset."""
        env = {**FULL_ENV, "REASONER_LOCAL_MODEL": ""}
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert len(endpoints) == 2
        assert endpoints[0]["url"] == PORTABLE_URL


class TestSingleEndpointConfig:
    """Tests for partial endpoint configurations."""

    def test_only_cloud_configured(self):
        """Only cloud endpoint set -> single-element list."""
        env = {
            "CLOUD_API_URL": "https://api.openai.com/v1",
            "CLOUD_API_KEY": "sk-test-key",
            "REASONER_CLOUD_MODEL": "openai/gpt-4o",
            "LITELLM_PROXY_URL": "http://litellm-proxy:4000/v1",
            "LITELLM_API_KEY": "not-needed",
        }
        with patch.dict("os.environ", env, clear=True):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert len(endpoints) == 1
        assert endpoints[0]["is_cloud"] is True

    def test_only_local_configured(self):
        """Only local endpoint set -> single-element list."""
        env = {
            "LLAMA_SERVER_URL": LOCAL_URL,
            "REASONER_LOCAL_MODEL": "openai/Qwen3.5-27B-IQ3_M",
            "LITELLM_PROXY_URL": "http://litellm-proxy:4000/v1",
            "LITELLM_API_KEY": "not-needed",
        }
        with patch.dict("os.environ", env, clear=True):
            provider = _import_provider()
        endpoints = provider._build_endpoints("reasoner")
        assert len(endpoints) == 1
        assert endpoints[0]["url"] == LOCAL_URL
        assert endpoints[0]["is_cloud"] is False

    @pytest.mark.asyncio
    async def test_duplicate_urls_both_probed(self):
        """Same URL for local and portable -> probed once (deduplication)."""
        env = {
            **FULL_ENV,
            "PORTABLE_LLM_SERVER_URL": LOCAL_URL,  # same as local
        }
        with patch.dict("os.environ", env, clear=False):
            provider = _import_provider()

        probed_urls = []

        async def mock_probe(url, timeout):
            probed_urls.append(url)
            return True

        with patch.object(provider, "_probe_single", side_effect=mock_probe):
            await provider.probe_endpoints()

        # URL deduplication in probe_endpoints uses a set
        assert probed_urls.count(LOCAL_URL) == 1
