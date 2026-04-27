from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(_REPO_ROOT / ".env"),
            str(_BACKEND_ROOT / ".env"),
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_temperature: float = 0.7

    rag_enabled: bool = True
    rag_mode: str = "bm25"  # bm25 / vector / hybrid
    rag_top_k: int = 4
    chroma_host: str = "localhost"
    chroma_port: int = 8100
    chroma_collection: str = "daihatsu_regulations"
    bm25_index_path: str = str(_REPO_ROOT / "daihatsu_rag" / "bm25_index.pkl")
    bm25_docs_path: str = str(_REPO_ROOT / "daihatsu_rag" / "bm25_docs.pkl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
