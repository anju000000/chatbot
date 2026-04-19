from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

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
    # LangChain により content がブロックの list になる場合がある
    if isinstance(text, list):
        return "".join(str(part) for part in text)
    return str(text)
