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

    # RAGAS evaluation
    ragas_response_relevancy_threshold: float = Field(default=0.7)

    # Embedding model
    embedding_model_name: str = Field(default="cl-nagoya/ruri-v3-310m")

    # Ingestion
    ingest_chunk_size: int = Field(default=1000)
    ingest_chunk_overlap: int = Field(default=200)

    # Config file paths
    models_config_path: str = Field(default="/config/models.yaml")
    judge_rules_path: str = Field(default="/config/judge_rules.yaml")

    @property
    def models(self) -> dict:
        return _load_yaml(self.models_config_path)

    @property
    def judge_criteria(self) -> str:
        rules = _load_yaml(self.judge_rules_path)
        criteria = rules.get("default", {}).get("criteria", [])
        return "\n".join(f"- {c}" for c in criteria)

    @property
    def rag_judge_criteria(self) -> str:
        rules = _load_yaml(self.judge_rules_path)
        criteria = rules.get("rag", {}).get("criteria", [])
        return "\n".join(f"- {c}" for c in criteria)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
