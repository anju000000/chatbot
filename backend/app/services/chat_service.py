import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from app.core.config import Settings
from app.services.rag_service import retrieve

logger = logging.getLogger(__name__)

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
        "あなたは就業規則・社内規程の専門アシスタントです。\n\n"
        f"--- 参考条文 ---\n{context}\n--- ここまで ---\n\n"
        "【回答ルール】\n"
        "1. 上記の参考条文を根拠に、必ず具体的な内容（数値・日数・金額・手続き・条件など）を含めて答えること。\n"
        "   条文番号だけを返してはいけない。必ず文章で説明すること。\n"
        "2. 「〜できますか」「〜いけませんか」「〜ありますか」などYes/No型の質問には、\n"
        "   回答の冒頭で必ず「はい」または「いいえ」を明示してから説明すること。\n"
        "3. 回答の最後に必ず出典を書くこと。形式: 「（規程名 条項番号）」\n"
        "   例: （育児・介護休業等に関する規程 第13条）\n"
        "4. 参考条文に直接的な記述が見当たらない場合でも、関連する条文から合理的に推測して答えること。\n"
        "   推測の場合は「規程上の明記はありませんが、〜と考えられます」と前置きすること。\n"
        "   それでも全く手がかりがない場合のみ「規程に記載がありません。担当部署にご確認ください。」と答えること。\n"
        "5. 挨拶や雑談には条文を使わず自然に返答し、出典は書かないこと。"
    )


def run_chat_with_rag(llm: BaseChatModel, messages: list[dict[str, str]], settings: Settings) -> str:
    # 最後のユーザーメッセージをクエリとして検索
    user_messages = [m for m in messages if m.get("role") == "user"]
    query = user_messages[-1]["content"] if user_messages else ""

    chunks: list[dict] = []
    if query and settings.rag_enabled:
        try:
            chunks = retrieve(query, settings)
            logger.info("RAG retrieved %d chunks for query: %s", len(chunks), query[:50])
        except Exception as e:
            logger.error("RAG retrieval failed, falling back to no-context: %s", e)

    system_prompt = _build_rag_system_prompt(chunks)

    # 既存のsystemメッセージを除いた上でRAGシステムプロンプトを先頭に挿入
    filtered = [m for m in messages if m.get("role") != "system"]
    if system_prompt:
        augmented = [{"role": "system", "content": system_prompt}] + filtered
    else:
        augmented = filtered

    return run_chat(llm, augmented)
