from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.services.rag_service import retrieve

_ROLE_MAP: dict[str, type[BaseMessage]] = {
    "user": HumanMessage,
    "assistant": AIMessage,
    "system": SystemMessage,
}


def messages_from_payload(messages: list[dict[str, str]]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for m in messages:
        role = m.get("role", "").lower().strip()
        content = m.get("content", "")
        if role not in _ROLE_MAP:
            raise ValueError(f"Unknown message role: {m.get('role')!r}")
        out.append(_ROLE_MAP[role](content=content))
    return out


def run_chat(llm: BaseChatModel, messages: list[dict[str, str]]) -> str:
    lc_messages = messages_from_payload(messages)
    result = llm.invoke(lc_messages)
    if isinstance(result, AIMessage):
        text = result.content
    else:
        text = getattr(result, "content", str(result))
    if isinstance(text, list):
        return "".join(str(part) for part in text)
    return str(text)


def _build_rag_system_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    context_parts = []
    for c in chunks:
        header = f"【{c['pdf_name']} {c['article_number']} {c['article_title']}】".strip()
        context_parts.append(f"{header}\n{c['text']}")
    context = "\n\n".join(context_parts)
    return (
        "あなたは就業規則・社内規程の専門アシスタントです。\n"
        "以下の条文を参考にして質問に答えてください。\n"
        "条文に記載がない場合は「規則に記載がありません」と答えてください。\n\n"
        f"--- 参考条文 ---\n{context}\n--- ここまで ---"
    )


def run_chat_with_rag(llm: BaseChatModel, messages: list[dict[str, str]], settings: Settings) -> str:
    # 最後のユーザーメッセージをクエリとして検索
    user_messages = [m for m in messages if m.get("role") == "user"]
    query = user_messages[-1]["content"] if user_messages else ""

    chunks: list[dict] = []
    if query and settings.rag_enabled:
        try:
            chunks = retrieve(query, settings)
        except Exception:
            pass  # 検索失敗時はRAGなしで続行

    system_prompt = _build_rag_system_prompt(chunks)

    # 既存のsystemメッセージを除いた上でRAGシステムプロンプトを先頭に挿入
    filtered = [m for m in messages if m.get("role") != "system"]
    if system_prompt:
        augmented = [{"role": "system", "content": system_prompt}] + filtered
    else:
        augmented = filtered

    return run_chat(llm, augmented)
