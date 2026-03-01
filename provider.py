"""
LLM provider — all calls go through LiteLLM Proxy.
No direct endpoint URLs or API keys in this file.
"""
from langchain_openai import ChatOpenAI
from settings import settings


def get_reasoner(temperature: float | None = None) -> ChatOpenAI:
    """Get reasoner LLM via LiteLLM Proxy."""
    model_config = settings.models.get("reasoner", {})
    return ChatOpenAI(
        base_url=settings.litellm_proxy_url,
        api_key=settings.litellm_api_key,
        model=model_config.get("litellm_model_name", "reasoner"),
        temperature=temperature if temperature is not None
        else model_config.get("temperature", 0.7),
        timeout=model_config.get("timeout", 120),
    )


def get_judge(temperature: float | None = None) -> ChatOpenAI:
    """Get judge LLM via LiteLLM Proxy (used for feedback generation only)."""
    model_config = settings.models.get("judge", {})
    return ChatOpenAI(
        base_url=settings.litellm_proxy_url,
        api_key=settings.litellm_api_key,
        model=model_config.get("litellm_model_name", "judge"),
        temperature=temperature if temperature is not None
        else model_config.get("temperature", 0.0),
        timeout=model_config.get("timeout", 60),
    )
