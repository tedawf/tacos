import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.models.doc import Doc
from app.schemas.doc import DocResult
from app.schemas.rag import ChatMessage, ContentChunk
from app.services.content_enhancer import ContentEnhancer
from app.services.query_expander import QueryExpander
from app.services.text_embedder import embed_text
from app.settings import settings

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(
        self,
        db: Session,
        ai_client: AsyncOpenAI | None = None,
        api_key: str | None = None,
        query_expander_service: QueryExpander | None = None,
        embed_text_fn=embed_text,
        content_enhancer_service: ContentEnhancer | None = None,
        doc_model=Doc,
    ):
        self.db = db
        if ai_client is None:
            key = api_key or settings.OPENAI_API_KEY
            if not key:
                raise ValueError("OPENAI_API_KEY must be set to use RAGService")
            ai_client = AsyncOpenAI(api_key=key)
        self.ai_client = ai_client
        self.query_expander = query_expander_service or QueryExpander()
        self.embed_text_fn = embed_text_fn
        self.content_enhancer = content_enhancer_service or ContentEnhancer()
        self.doc_model = doc_model

    def get_relevant_documents(
        self, query: str, limit: int, threshold: float
    ) -> List[DocResult]:
        """
        Embeds a query and performs semantic search with cosine similarity.
        Retrieves the content of documents with similarity above the threshold.
        """

        try:
            # Expand query to improve semantic matching
            expanded_query = self.query_expander.expand_query(query)
            embedding = self.embed_text_fn(expanded_query)
            if not embedding or len(embedding) != 1536:
                raise HTTPException(400, "Invalid embedding for query")

            distance = self.doc_model.embedding.cosine_distance(embedding)
            similarity = (1 - distance).label("similarity")

            results = (
                self.db.query(self.doc_model, similarity)
                .filter(self.doc_model.embedding.isnot(None))
                .filter(similarity >= threshold)
                .order_by(similarity.desc())
                .limit(limit)
                .all()
            )

            return [
                DocResult(
                    id=doc.id,
                    slug=doc.slug,
                    title=doc.title,
                    content=doc.content,
                    doc_metadata=doc.doc_metadata,
                    similarity=round(sim, 4),
                )
                for doc, sim in results
            ]

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            # Re-raise as a standard exception for the router to handle
            raise Exception("Internal error during search") from e

    def get_relevant_documents_with_navigation(
        self, query: str, limit: int, threshold: float
    ) -> List[DocResult]:
        """
        Enhanced version that always includes navigation content at the top.
        Ensures navigation is available for any chatbot questions.
        """
        try:
            # Get regular results (reserve 1 spot for navigation)
            regular_results = self.get_relevant_documents(query, limit - 1, threshold)

            # Force include navigation content with high priority
            navigation_docs = (
                self.db.query(self.doc_model)
                .filter(
                    self.doc_model.doc_metadata.op("->>")("contentType") == "navigation"
                )
                .all()
            )

            # Convert navigation docs to DocResult with high similarity
            navigation_results = [
                DocResult(
                    id=doc.id,
                    slug=doc.slug,
                    title=doc.title,
                    content=doc.content,
                    doc_metadata=doc.doc_metadata,
                    similarity=0.95,  # High fixed similarity to ensure top ranking
                )
                for doc in navigation_docs
            ]

            # Combine and ensure navigation is at top
            combined = navigation_results + regular_results
            combined.sort(key=lambda x: x.similarity, reverse=True)
            return combined[:limit]

        except Exception as e:
            logger.error(f"Error during navigation-enhanced search: {e}")
            # Fall back to regular search if navigation enhancement fails
            return self.get_relevant_documents(query, limit, threshold)

    async def stream_chat_response(
        self,
        messages: List[ChatMessage],
        limit: int,
        threshold: float,
        relevant_docs: Optional[List[DocResult]] = None,
    ):
        """
        Streams a chat response, using the retrieval logic to build context.
        Accepts pre-fetched relevant_docs to avoid duplicate queries.
        """

        # The last message is always the user's question
        latest_user_question = messages[-1].content

        # 1. Retrieve relevant docs using navigation-enhanced search
        relevant_docs = relevant_docs or self.get_relevant_documents_with_navigation(
            query=latest_user_question, limit=limit, threshold=threshold
        )

        # 2. Build enriched context
        context_parts = []
        for doc in relevant_docs:
            md = doc.doc_metadata or {}

            # Generate URL based on document source
            if doc.slug.startswith(settings.BLOG_PREFIX):
                url = f"{settings.BASE_BLOG_URL}/{doc.slug}"

            # Navigation content describes all routes, so use the base URL
            elif doc.slug == "navigation:routes":
                url = f"{settings.BASE_BLOG_URL}"
            # Project
            elif doc.slug.startswith("projects"):
                url = f"{settings.BASE_BLOG_URL}/projects"
            # Homepage
            elif doc.slug.startswith("/"):
                url = f"{settings.BASE_BLOG_URL}/"
            else:
                url = "N/A"

            tags = ", ".join(md.get("tags", [])) or "N/A"
            content_snippet = (
                (doc.content[:500] + "...") if len(doc.content) > 500 else doc.content
            )
            context_parts.append(
                f"Title: {doc.title}\n"
                f"Summary: {md.get('summary', 'N/A')}\n"
                f"Tags: {tags}\n"
                f"Source: {md.get('source', 'N/A')}\n"
                f"Created: {md.get('created_at', 'N/A')}\n"
                f"Updated: {md.get('updated_at', 'N/A')}\n"
                f"URL: {url}\n"
                f"Content: {content_snippet}\n"
                "-----"
            )

        context_text = "\n".join(context_parts) or "No relevant context available."
        logger.debug(f"Context for RAG:\n{context_text}")

        # 3. Prepare system prompt
        system_prompt = (
            f"You are Ted Support, a helpful AI chatbot for Ted's developer portfolio.\n"
            f"The current year is {datetime.now().year}.\n"
            "Use the following context from Ted's portfolio content, blog posts and personal knowledge base to answer the user's question:\n\n"
            f"{context_text}\n\n"
            "CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:\n"
            "- Answer the user's question directly and specifically using ONLY information from the provided context.\n"
            "- Keep responses concise (2-4 sentences) and actionable.\n"
            "- ALWAYS include the exact URL from the 'URL' field when referencing any page - NEVER use relative paths.\n"
            "- When discussing projects, highlight relevant technologies and specific accomplishments mentioned in the context.\n"
            "- STRICTLY NO HALLUCINATION: If information isn't in the context, say 'I don't have enough information about that in my knowledge base.'\n"
            "- Navigation content is always available - use it to guide users to relevant site sections and provide clear URLs in markdown format."
        )

        # 4. Build messages with existing chat history
        prompt_messages = [{"role": "system", "content": system_prompt}]
        for message in messages:
            prompt_messages.append({"role": message.role, "content": message.content})

        # 5. Stream response
        try:
            stream = await self.ai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=prompt_messages,
                stream=True,
            )
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content is not None:
                    yield content
        except Exception as e:
            logger.error(f"Error streaming OpenAI response: {e}")
            yield f"\n[Error: Could not get response from the model]"

    def _generate_content_hash(self, chunk: ContentChunk) -> str:
        """Generate SHA-256 hash for content change detection."""
        content_to_hash = f"{chunk.slug}{chunk.title}{chunk.content}{json.dumps(chunk.metadata or {}, sort_keys=True)}"
        return hashlib.sha256(content_to_hash.encode()).hexdigest()

    def update_portfolio_content(
        self, content_chunks: List[ContentChunk]
    ) -> Dict[str, Any]:
        """
        Update portfolio content with hash-based change detection.
        Only updates documents that have changed and removes deleted ones.
        """
        stats = {"processed": 0, "updated": 0, "skipped": 0, "errors": []}

        try:
            # Get existing portfolio documents for comparison
            existing_docs = (
                self.db.query(self.doc_model)
                .filter(
                    self.doc_model.document_id.like(f"{settings.PORTFOLIO_PREFIX}%")
                )
                .all()
            )

            existing_docs_map = {doc.document_id: doc for doc in existing_docs}
            processed_doc_ids = set()

            # Process each content chunk
            for chunk in content_chunks:
                stats["processed"] += 1

                try:
                    # Generate content hash for change detection
                    content_hash = self._generate_content_hash(chunk)

                    # Create document ID with portfolio prefix (now using unique slugs)
                    document_id = f"{settings.PORTFOLIO_PREFIX}{chunk.slug}"
                    processed_doc_ids.add(document_id)

                    # Check if document exists and has same content hash
                    existing_doc = existing_docs_map.get(document_id)
                    if existing_doc:
                        existing_hash = existing_doc.doc_metadata.get("content_hash")
                        if existing_hash == content_hash:
                            # Skip unchanged content
                            stats["skipped"] += 1
                            logger.debug(f"Skipping unchanged content: {document_id}")
                            continue
                        else:
                            logger.debug(
                                f"Content changed for {document_id}: {existing_hash} != {content_hash}"
                            )
                    else:
                        logger.debug(f"New content: {document_id}")

                    # Enhance content with metadata before embedding
                    enhanced_content = self.content_enhancer.enhance_content(
                        content=chunk.content,
                        title=chunk.title,
                        metadata=chunk.metadata or {},
                    )

                    # Generate embedding for the enhanced content (only for new/changed content)
                    embedding = self.embed_text_fn(enhanced_content)

                    if existing_doc:
                        # Update existing document
                        existing_doc.title = chunk.title
                        existing_doc.content = chunk.content
                        existing_doc.doc_metadata = {
                            "content_hash": content_hash,
                            "source": "portfolio",
                            "updated_at": datetime.now().isoformat(),
                            **(chunk.metadata or {}),
                        }
                        existing_doc.embedding = embedding
                    else:
                        # Create new document
                        doc = self.doc_model(
                            document_id=document_id,
                            slug=chunk.slug,
                            title=chunk.title,
                            content=chunk.content,
                            doc_metadata={
                                "content_hash": content_hash,
                                "source": "portfolio",
                                "updated_at": datetime.now().isoformat(),
                                **(chunk.metadata or {}),
                            },
                            embedding=embedding,
                        )
                        self.db.add(doc)

                    stats["updated"] += 1

                except Exception as e:
                    error_msg = f"Failed to process chunk {chunk.slug}: {str(e)}"
                    logger.error(error_msg)
                    stats["errors"].append({"slug": chunk.slug, "error": str(e)})
                    continue

            # Remove documents that are no longer in the current content
            doc_ids_to_remove = set(existing_docs_map.keys()) - processed_doc_ids
            if doc_ids_to_remove:
                deleted_count = (
                    self.db.query(self.doc_model)
                    .filter(self.doc_model.document_id.in_(doc_ids_to_remove))
                    .delete(synchronize_session=False)
                )
                logger.info(f"Deleted {deleted_count} removed portfolio documents")

            # Commit all changes
            self.db.commit()
            logger.info(f"Portfolio content update completed: {stats}")

        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to update portfolio content: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to update portfolio content: {str(e)}"
            )

        return stats
