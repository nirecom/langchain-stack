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
    # LiteLLM Proxy (fallback when all direct endpoints are down)
    litellm_proxy_url: str = Field(default="http://litellm-proxy:4000/v1")
    litellm_api_key: str = Field(default="not-needed")

    # Direct endpoint URLs
    llama_server_url: str = Field(default="")
    portable_llm_server_url: str = Field(default="")
    cloud_api_url: str = Field(default="")
    cloud_api_key: str = Field(default="")

    # Direct endpoint model names (per role)
    reasoner_local_model: str = Field(default="")
    reasoner_portable_model: str = Field(default="")
    reasoner_cloud_model: str = Field(default="")
    judge_local_model: str = Field(default="")
    judge_portable_model: str = Field(default="")
    judge_cloud_model: str = Field(default="")

    # Health probe
    health_probe_timeout: float = Field(default=2.0)

    # ChromaDB
    chroma_host: str = Field(default="chromadb")
    chroma_port: int = Field(default=8000)

    # OpenSearch (Phase 4F+)
    opensearch_url: str = Field(default="http://opensearch:9200")
    os_index_prefix: str = Field(default="ls_")
    vector_backend: str = Field(default="opensearch")
    hybrid_pipeline_name: str = Field(default="hybrid-bm25-knn")
    search_mode: str = Field(default="hybrid+header")

    # Judge settings
    max_judge_retries: int = Field(default=2)

    # RAGAS evaluation
    ragas_response_relevancy_threshold: float = Field(default=0.7)
    ragas_faithfulness_threshold: float = Field(default=0.7)
    ragas_context_precision_threshold: float = Field(default=0.7)

    # Embedding model
    embedding_model_name: str = Field(default="BAAI/bge-m3")
    ingest_device: str = Field(default="cpu")
    embedding_ab_suffix: str = Field(default="")

    # Ingestion
    ingest_chunk_size: int = Field(default=1000)
    ingest_chunk_overlap: int = Field(default=200)

    # RAG retrieval
    rag_top_k: int = Field(default=3)

    # Config file paths
    models_config_path: str = Field(default="/config/models.yaml")
    judge_rules_path: str = Field(default="/config/judge_rules.yaml")
    access_control_path: str = Field(default="/config/access_control.yaml")

    # API keys (empty = auth disabled)
    ingest_api_key: str = Field(default="")

    # Audit log
    audit_log_path: str = Field(default="/data/audit/ingest.jsonl")

    @property
    def models(self) -> dict:
        return _load_yaml(self.models_config_path)

    @property
    def access_control(self) -> dict:
        return _load_yaml(self.access_control_path)

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
