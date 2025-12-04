from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class PromptRequest(BaseModel):
    messages: List[ChatMessage]
    chat_id: Optional[UUID] = None

    # {
    #   "messages": [
    #     { "role": "user", "content": "Why did Ted build you?" },
    #     { "role": "assistant", "content": "So I can fetch his resume faster than his dumbbell progress." },
    #     { "role": "user", "content": "Nice." }
    #   ]
    # }


class ContentChunk(BaseModel):
    slug: str
    title: str
    content: str
    metadata: Optional[Dict[str, Any]] = None


class UpdateContentRequest(BaseModel):
    timestamp: str
    content: List[ContentChunk]


class UpdateContentResponse(BaseModel):
    processed: int
    updated: int
    skipped: int
    errors: List[Dict[str, str]]
