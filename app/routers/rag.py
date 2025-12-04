import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.db.couchdb import get_couch
from app.db.postgres.base import get_db
from app.schemas.doc import DocResult
from app.schemas.rag import PromptRequest, UpdateContentRequest, UpdateContentResponse
from app.services.chat_logger import ChatLogger
from app.services.docs_ingester import ingest_all
from app.services.rag_service import RAGService
from app.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    return RAGService(db, api_key=settings.OPENAI_API_KEY)


def get_chat_logger(db: Session = Depends(get_db)) -> ChatLogger:
    return ChatLogger(db)


def get_ingest_all():
    """Provide ingest_all for dependency override in tests."""
    return ingest_all


@router.post("/prompt")
async def prompt_rag(
    request: PromptRequest,
    limit: int = Query(15, ge=1, le=50, description="Number of documents to retrieve"),
    threshold: float = Query(0.25, ge=0.0, le=1.0, description="Similarity threshold"),
    rag_service: RAGService = Depends(get_rag_service),
    chat_logger: ChatLogger = Depends(get_chat_logger),
    chat_id_header: Optional[str] = Header(
        default=None, alias="X-Chat-Id", convert_underscores=False
    ),
):
    """
    Prompt endpoint for RAG chatbot.
    Accepts a list of messages and streams back the response.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    try:
        logger.debug(f"Received prompt request with {len(request.messages)} messages.")

        # Determine session and context (prefer body, then header, else create)
        try:
            provided_chat = request.chat_id or (
                uuid.UUID(chat_id_header) if chat_id_header else None
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid X-Chat-Id header")
        chat_id = chat_logger.ensure_chat_id(provided_chat)
        latest_user_message = request.messages[-1].content

        relevant_docs = rag_service.get_relevant_documents_with_navigation(
            query=latest_user_message, limit=limit, threshold=threshold
        )
        context_slugs = [doc.slug for doc in relevant_docs if doc.slug]

        # Persist the user turn
        next_seq = chat_logger.next_sequence(chat_id)
        chat_logger.log_message(
            chat_id=chat_id,
            role="user",
            seq=next_seq,
            content=latest_user_message,
            context_slugs=context_slugs,
        )
        chat_logger.db.commit()

        # Stream assistant reply while buffering to save after stream completes
        streamer = rag_service.stream_chat_response(
            messages=request.messages,
            limit=limit,
            threshold=threshold,
            relevant_docs=relevant_docs,
        )

        async def streaming_wrapper():
            assistant_chunks = []
            try:
                async for chunk in streamer:
                    if chunk is not None:
                        assistant_chunks.append(chunk)
                        yield chunk
            finally:
                assistant_message = "".join(assistant_chunks).strip()
                if assistant_message:
                    try:
                        chat_logger.log_message(
                            chat_id=chat_id,
                            role="assistant",
                            seq=next_seq + 1,
                            content=assistant_message,
                            context_slugs=context_slugs,
                        )
                        chat_logger.db.commit()
                    except Exception as log_error:
                        chat_logger.db.rollback()
                        logger.error(
                            f"Failed to log assistant message: {log_error}",
                            exc_info=True,
                        )

        response = StreamingResponse(streaming_wrapper(), media_type="text/plain")
        response.headers["X-Chat-Id"] = str(chat_id)
        return response

    except HTTPException as http_exc:
        try:
            chat_logger.db.rollback()
        except Exception:
            logger.error("Rollback failed after prompt error", exc_info=True)
        # Preserve original status/detail for client errors
        raise http_exc
    except Exception as e:
        try:
            chat_logger.db.rollback()
        except Exception:
            # If rollback fails, just log; we still want to surface the error.
            logger.error("Rollback failed after prompt error", exc_info=True)
        logger.error(f"Error in /prompt endpoint: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@router.post("/reingest")
def reingest(
    db: Session = Depends(get_db),
    couch=Depends(get_couch),
    ingest_all_fn=Depends(get_ingest_all),
):
    """
    Full ingestion/reset endpoint.
    Deletes old docs and re-ingests all CouchDB content.
    """
    try:
        couch_db, parser = couch
        ingest_all_fn(db, parser=parser)
        return {"status": "success", "message": "ingestion completed."}
    except Exception as e:
        logger.error(f"Reset ingest failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="ingestion failed")


@router.get("/query", response_model=List[DocResult])
def query_docs(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    threshold: float = Query(0.25, ge=0.0, le=1.0),
    debug: bool = Query(False),
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Perform semantic search with cosine similarity.
    Returns ranked document chunks above the similarity threshold.
    """
    try:
        results = rag_service.get_relevant_documents(
            query=q, limit=limit, threshold=threshold
        )

        if debug:
            # For debug, we return a list of dicts with truncated content
            return [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "content": truncate(doc.content, 100),
                    "similarity": doc.similarity,
                }
                for doc in results
            ]

        # normal API output (validated)
        return results

    except Exception as e:
        logger.error(f"Error in /query endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def truncate(text: str | None, length: int) -> str | None:
    if not text:
        return None
    return text if len(text) <= length else text[:length] + "..."


@router.post("/update", response_model=UpdateContentResponse)
def update_portfolio_content(
    request: UpdateContentRequest,
    rag_service: RAGService = Depends(get_rag_service),
):
    """
    Update portfolio content with complete replacement strategy.
    Accepts portfolio content and manages embeddings for semantic search.
    """
    try:
        logger.info(
            f"Processing portfolio content update with {len(request.content)} chunks"
        )

        # Update portfolio content
        stats = rag_service.update_portfolio_content(request.content)

        return UpdateContentResponse(
            processed=stats["processed"],
            updated=stats["updated"],
            skipped=stats["skipped"],
            errors=stats["errors"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /update endpoint: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to process portfolio content update"
        )
