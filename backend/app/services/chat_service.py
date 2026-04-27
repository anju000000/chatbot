import logging
import re

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
        "あなたは就業規則・社内規程の専門アシスタントです。必ず日本語のみで回答してください。\n\n"
        f"--- 参考条文 ---\n{context}\n--- ここまで ---\n\n"
        "【回答ルール】\n"
        "0. 回答は必ず日本語のみで記述すること。他の言語を一切混入させないこと。\n"
        "1. 上記の参考条文を根拠に、必ず具体的な内容（数値・日数・金額・手続き・条件など）を含めて答えること。\n"
        "   条文番号だけを返してはいけない。必ず文章で説明すること。\n"
        "   金額・日数・期間・対象者の条件・例外（試用期間中は除外など）は省略せずすべて明記すること。\n"
        "   列挙が必要な内容（ハラスメントの類型・懲戒の種類・安全衛生の措置など）はすべての項目を漏らさず列挙すること。\n"
        "2. 「〜できますか」「〜いけませんか」「〜ありますか」などYes/No型の質問には、\n"
        "   回答の冒頭で必ず「はい」または「いいえ」を明示してから説明すること。\n"
        "3. 参考条文の内容を使って回答した場合のみ、回答の末尾に出典をまとめて1回だけ書くこと。\n"
        "   形式: 「（規程名 条項番号）」 例: （育児・介護休業等に関する規程 第13条）\n"
        "   複数の条文を引用した場合も、同じ出典は重複させず、末尾にまとめて一度だけ列挙すること。\n"
        "   参考条文を根拠に使っていない場合（手がかりがない・挨拶・雑談など）は出典を書かないこと。\n"
        "4. 参考条文に直接的な記述が見当たらない場合でも、関連する条文から合理的に推測して答えること。\n"
        "   推測の場合は「規程上の明記はありませんが、〜と考えられます」と前置きすること。\n"
        "   全く手がかりがない場合のみ「規程に記載がありません。担当部署にご確認ください。」と答えること。\n"
        "   この場合は出典を一切書かないこと。質問と無関係な条文を出典として添付してはいけない。\n"
        "   一般的な社会常識・法律の一般論・インターネット情報を混入させないこと。あくまで参考条文のみを根拠にすること。\n"
        "5. 挨拶や雑談には条文を使わず自然に返答し、出典は書かないこと。\n"
        "6. 自分自身（AI・アシスタント）の能力・機能・制限について一切言及しないこと。\n"
        "   「私はテキスト対応しかできません」「私の機能では〜できません」などの自己説明は禁止。\n"
        "   規程に記載がない質問には「規程に記載がありません。担当部署にご確認ください。」とだけ答えること。"
    )


_CITATION_PATTERN = re.compile(r'（[^）]*第\d+条[^）]*）')

_NO_CITATION_SIGNALS = [
    "規程に記載がありません",
    "担当部署にご確認ください",
    "担当部署に直接お問い合わせ",
    "所属長または",
    "人事部にお問い合わせ",
    "総務部門にお問い合わせ",
    "システム担当部門",
    "ご確認いただくのが最善",
    "直接ご確認ください",
    "直接お問い合わせください",
    "確認されることをお勧め",
]

_GREETING_PATTERN = re.compile(
    r'^[\s　]*('
    r'こんにちは|こんばんは|おはよう|はじめまして|よろしく|お疲れ|ありがとう|'
    r'どうぞよろしく|お願いします|失礼します|さようなら|またね|bye|hello|hi'
    r')',
    re.IGNORECASE,
)


def _postprocess_citations(text: str, query: str) -> str:
    citations = _CITATION_PATTERN.findall(text)
    if not citations:
        return text

    # 出典を除いた本文
    body = _CITATION_PATTERN.sub('', text).strip()

    # 挨拶クエリ または 「情報なし」系の返答なら出典を除去して終了
    if _GREETING_PATTERN.match(query) or any(sig in body for sig in _NO_CITATION_SIGNALS):
        return body

    # 重複を除きつつ順序を保持
    seen: set[str] = set()
    unique: list[str] = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return body + "\n" + "　".join(unique)


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

    response = run_chat(llm, augmented)
    return _postprocess_citations(response, query)
