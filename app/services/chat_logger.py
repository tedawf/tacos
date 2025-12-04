import uuid
from typing import List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage


class ChatLogger:
    """Minimal helper to persist chat turns."""

    def __init__(self, db: Session, chat_model=ChatMessage):
        self.db = db
        self.chat_model = chat_model

    def ensure_chat_id(self, chat_id: Optional[uuid.UUID]) -> uuid.UUID:
        return chat_id or uuid.uuid4()

    def next_sequence(self, chat_id: uuid.UUID) -> int:
        last_seq = (
            self.db.query(func.max(self.chat_model.seq))
            .filter(self.chat_model.chat_id == chat_id)
            .scalar()
        )
        return (last_seq or 0) + 1

    def log_message(
        self,
        chat_id: uuid.UUID,
        role: str,
        seq: int,
        content: str,
        context_slugs: Optional[List[str]] = None,
    ) -> None:
        message = self.chat_model(
            chat_id=chat_id,
            seq=seq,
            role=role,
            content=content,
            context_slugs=context_slugs,
        )
        self.db.add(message)
