"""
Application settings.
All configuration is loaded from environment variables and YAML files.
No hardcoded hosts, IPs, ports, or model names.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml
from pathlib import Path


def _load_yaml(path: str) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


class Settings(BaseSettings):
    # LiteLLM Proxy (the ONLY LLM endpoint this app talks to)
    litellm_proxy_url: str = Field(default="http://litellm-proxy:4000/v1")
    litellm_api_key: str = Field(default="not-needed")

    # ChromaDB
    chroma_host: str = Field(default="chromadb")
    chroma_port: int = Field(default=8000)

    # Judge settings
    max_judge_retries: int = Field(default=2)

    # RAGAS embedding model (used for Response Relevancy cosine similarity)
    ragas_embedding_model: str = Field(
        default="intfloat/multilingual-e5-small"
    )

    # Config file paths
    models_config_path: str = Field(default="/config/models.yaml")
    judge_rules_path: str = Field(default="/config/judge_rules.yaml")

    @property
    def models(self) -> dict:
        return _load_yaml(self.models_config_path)

    @property
    def judge_rules(self) -> dict:
        return _load_yaml(self.judge_rules_path)

    def get_judge_profile(self, profile: str = "default") -> dict:
        """Load a specific judge evaluation profile."""
        rules = self.judge_rules
        return rules.get(profile, rules.get("default", {}))

    def get_feedback_criteria(self, profile: str = "default") -> str:
        """Get feedback criteria as formatted string for Judge LLM prompt."""
        p = self.get_judge_profile(profile)
        criteria = p.get("feedback_criteria", [])
        return "\n".join(f"- {c}" for c in criteria)

    def get_pass_threshold(self, profile: str = "default") -> float:
        """Get Response Relevancy pass threshold."""
        p = self.get_judge_profile(profile)
        metrics = p.get("metrics", {})
        rr = metrics.get("response_relevancy", {})
        return rr.get("pass_threshold", 0.7)

    def get_strictness(self, profile: str = "default") -> int:
        """Get RAGAS strictness (number of questions to generate)."""
        p = self.get_judge_profile(profile)
        metrics = p.get("metrics", {})
        rr = metrics.get("response_relevancy", {})
        return rr.get("strictness", 3)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
