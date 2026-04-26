from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.services.chat_service import run_chat_with_rag
from app.services.llm_factory import get_chat_model

router = APIRouter(prefix="/api/v1", tags=["chat"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)


class ChatResponse(BaseModel):
    content: str


@router.post("/chat", response_model=ChatResponse)
def post_chat(
    body: ChatRequest,
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    try:
        llm = get_chat_model(settings)
    except ValueError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    payload = [m.model_dump() for m in body.messages]
    try:
        text = run_chat_with_rag(llm, payload, settings)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ChatResponse(content=text)
