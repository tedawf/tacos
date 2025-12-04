import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Integer,
    Text,
    UniqueConstraint,
    UUID,
    func,
)

from app.db.postgres.base import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    chat_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    seq = Column(Integer, nullable=False)
    role = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    context_slugs = Column(JSON, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )

    __table_args__ = (
        CheckConstraint(
            "role in ('user','assistant')", name="ck_chat_messages_role_valid"
        ),
        UniqueConstraint("chat_id", "seq", name="uq_chat_messages_chat_seq_unique"),
    )
