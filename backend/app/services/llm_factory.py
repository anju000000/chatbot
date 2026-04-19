from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama

from app.core.config import Settings


def get_chat_model(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return ChatOllama(
            base_url=settings.ollama_base_url.rstrip("/"),
            model=settings.ollama_model,
            temperature=settings.ollama_temperature,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER={settings.llm_provider!r}. "
        "P0 supports 'ollama' only; use later bolts for bedrock / openai_compatible."
    )
