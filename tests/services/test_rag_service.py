import uuid
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.schemas.doc import DocResult
from app.schemas.rag import ChatMessage, ContentChunk
from app.services.rag_service import RAGService
from app.settings import settings
from tests.conftest import FakeSession


class FakeSimilarity:
    def __init__(self, value):
        self.value = value
        self.label_name = None
        self.ge_other = None

    def label(self, name):
        self.label_name = name
        return self

    def __ge__(self, other):
        self.ge_other = other
        return ("ge", self.value, other)

    def desc(self):
        return f"desc({self.value})"


class FakeDistance:
    def __init__(self, value):
        self.value = value

    def __rsub__(self, other):
        return FakeSimilarity(other - self.value)


class FakeEmbedding:
    def __init__(self, distance_value=0.1):
        self.distance_value = distance_value
        self.used_embedding = None
        self.isnot_arg = None

    def cosine_distance(self, embedding):
        self.used_embedding = embedding
        return FakeDistance(self.distance_value)

    def isnot(self, value):
        self.isnot_arg = value
        return ("isnot", value)


class FakeOp:
    def __init__(self, name, arg):
        self.name = name
        self.arg = arg

    def __eq__(self, other):
        return (self.name, self.arg, other)


class FakeColumn:
    def __init__(self, name):
        self.name = name

    def like(self, pattern):
        return (self.name, "like", pattern)

    def op(self, operator):
        return lambda arg: FakeOp(operator, arg)

    def in_(self, values):
        return (self.name, "in", values)


class FakeDocModel:
    embedding = FakeEmbedding()
    doc_metadata = FakeColumn("metadata")
    document_id = FakeColumn("document_id")

    def __init__(
        self,
        document_id,
        slug,
        title,
        content,
        doc_metadata,
        embedding,
        id=None,
    ):
        self.id = id or uuid.uuid4()
        self.document_id = document_id
        self.slug = slug
        self.title = title
        self.content = content
        self.doc_metadata = doc_metadata
        self.embedding = embedding


class PortfolioFakeQuery:
    def __init__(self, rows, delete_return=0):
        self.rows = rows
        self.filtered = None
        self.delete_return = delete_return
        self.delete_kwargs = None

    def filter(self, expr):
        self.filtered = expr
        return self

    def all(self):
        return self.rows

    def delete(self, synchronize_session=None):
        self.delete_kwargs = {"synchronize_session": synchronize_session}
        return self.delete_return


class PortfolioFakeSession:
    def __init__(self, rows=None, delete_return=0):
        self.rows = rows or []
        self.delete_return = delete_return
        self.added = []
        self.committed = False
        self.rolled_back = False
        self.queries = []

    def query(self, *_cols):
        query = PortfolioFakeQuery(self.rows, delete_return=self.delete_return)
        self.queries.append(query)
        return query

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class FakeStreamChunk:
    def __init__(self, content):
        delta = SimpleNamespace(content=content)
        self.choices = [SimpleNamespace(delta=delta)]


class FakeStream:
    def __init__(self, contents):
        self._iter = iter(FakeStreamChunk(c) for c in contents)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


class FakeCompletions:
    def __init__(self, stream):
        self.stream = stream
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.stream


class FakeAIClient:
    def __init__(self, stream):
        self.chat = SimpleNamespace(completions=FakeCompletions(stream))


def make_doc(slug: str, content: str = "body", metadata=None, similarity: float = 0.9):
    metadata = metadata or {}
    doc = SimpleNamespace(
        id=uuid.uuid4(),
        slug=slug,
        title=f"Title for {slug}",
        content=content,
        doc_metadata=metadata,
    )
    return doc, similarity


def make_content_chunk(slug: str, content: str, title: str, metadata=None):
    return ContentChunk(slug=slug, title=title, content=content, metadata=metadata)


def test_get_relevant_documents_returns_results():
    rows = [
        make_doc("blog/alpha", "Alpha content", {"summary": "S"}, 0.91234),
        make_doc("blog/bravo", "Bravo content", {"summary": "S"}, 0.5),
    ]
    session = FakeSession(rows=rows)

    calls = {}
    fake_embedding = FakeEmbedding()
    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        query_expander_service=SimpleNamespace(
            expand_query=lambda q: calls.setdefault("query", q) or q
        ),
        embed_text_fn=lambda _q: [0.0] * 1536,
        doc_model=SimpleNamespace(embedding=fake_embedding),
    )
    results = service.get_relevant_documents("hello world", limit=1, threshold=0.2)

    assert len(results) == 1
    assert results[0].slug == "blog/alpha"
    assert results[0].similarity == round(rows[0][1], 4)
    assert calls["query"] == "hello world"
    assert fake_embedding.used_embedding == [0.0] * 1536
    assert session.last_query.limit_value == 1


def test_get_relevant_documents_invalid_embedding_raises():
    session = FakeSession([])
    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        embed_text_fn=lambda _q: [1.0, 2.0],
        doc_model=SimpleNamespace(embedding=FakeEmbedding()),
    )

    with pytest.raises(HTTPException) as excinfo:
        service.get_relevant_documents("short", limit=2, threshold=0.1)

    assert excinfo.value.status_code == 400


def test_get_relevant_documents_with_navigation_prioritizes_nav():
    regular = [
        DocResult(
            id=uuid.uuid4(),
            slug="blog/regular",
            title="Regular",
            content="regular content",
            doc_metadata={},
            similarity=0.4,
        )
    ]
    nav_doc = FakeDocModel(
        document_id="portfolio/navigation",
        slug="navigation:routes",
        title="Nav",
        content="nav content",
        doc_metadata={"contentType": "navigation"},
        embedding=None,
    )
    session = PortfolioFakeSession(rows=[nav_doc])

    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        doc_model=FakeDocModel,
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        embed_text_fn=lambda _q: [0.0] * 1536,
    )
    service.get_relevant_documents = lambda query, limit, threshold: regular

    results = service.get_relevant_documents_with_navigation(
        "where am i", limit=2, threshold=0.1
    )

    assert results[0].slug == "navigation:routes"
    assert results[0].similarity == 0.95
    assert len(results) == 2
    assert session.queries[0].filtered is not None


def test_generate_content_hash_is_stable():
    session = PortfolioFakeSession()
    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        embed_text_fn=lambda q: [0.0] * 1536,
        doc_model=FakeDocModel,
    )

    chunk1 = make_content_chunk("slug", "body", "Title", {"a": 1, "b": 2})
    chunk2 = make_content_chunk("slug", "body", "Title", {"b": 2, "a": 1})

    assert service._generate_content_hash(chunk1) == service._generate_content_hash(
        chunk2
    )


def test_update_portfolio_content_adds_new_doc():
    session = PortfolioFakeSession()
    enhance_calls = []
    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        content_enhancer_service=SimpleNamespace(
            enhance_content=lambda **kwargs: enhance_calls.append(kwargs)
            or "enhanced body"
        ),
        embed_text_fn=lambda _text: [1.0, 2.0, 3.0],
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        doc_model=FakeDocModel,
    )

    chunk = make_content_chunk(
        "project-1", "hello world", "Project 1", {"tags": ["py"]}
    )

    stats = service.update_portfolio_content([chunk])

    assert stats == {"processed": 1, "updated": 1, "skipped": 0, "errors": []}
    assert len(session.added) == 1
    stored = session.added[0]
    assert stored.document_id == f"{settings.PORTFOLIO_PREFIX}project-1"
    assert stored.doc_metadata["content_hash"] == service._generate_content_hash(chunk)
    assert stored.doc_metadata["source"] == "portfolio"
    assert stored.embedding == [1.0, 2.0, 3.0]
    assert session.committed is True
    assert enhance_calls[0]["title"] == "Project 1"
    assert enhance_calls[0]["metadata"]["tags"] == ["py"]


def test_update_portfolio_content_skips_unchanged_and_deletes_removed():
    service = RAGService(
        db=PortfolioFakeSession(delete_return=1),
        ai_client=SimpleNamespace(),
        content_enhancer_service=SimpleNamespace(
            enhance_content=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("should skip")
            )
        ),
        embed_text_fn=lambda _text: (_ for _ in ()).throw(RuntimeError("should skip")),
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        doc_model=FakeDocModel,
    )

    chunk = make_content_chunk("keep", "same body", "Keep", {})
    unchanged_hash = service._generate_content_hash(chunk)

    keep_doc = FakeDocModel(
        document_id=f"{settings.PORTFOLIO_PREFIX}keep",
        slug="keep",
        title="Keep old",
        content="same body",
        doc_metadata={"content_hash": unchanged_hash},
        embedding=[0.1],
    )
    remove_doc = FakeDocModel(
        document_id=f"{settings.PORTFOLIO_PREFIX}remove",
        slug="remove",
        title="Remove me",
        content="gone",
        doc_metadata={},
        embedding=[0.2],
    )
    service.db.rows = [keep_doc, remove_doc]

    stats = service.update_portfolio_content([chunk])

    assert stats == {"processed": 1, "updated": 0, "skipped": 1, "errors": []}
    assert service.db.committed is True
    assert len(service.db.queries) == 2  # one for fetch, one for delete
    assert service.db.queries[1].delete_kwargs == {"synchronize_session": False}
    assert remove_doc in service.db.rows


def test_update_portfolio_content_updates_changed_doc():
    session = PortfolioFakeSession()
    existing = FakeDocModel(
        document_id=f"{settings.PORTFOLIO_PREFIX}project-2",
        slug="project-2",
        title="Old Title",
        content="old body",
        doc_metadata={"content_hash": "oldhash", "source": "portfolio"},
        embedding=[0.0],
    )
    session.rows = [existing]

    service = RAGService(
        db=session,
        ai_client=SimpleNamespace(),
        content_enhancer_service=SimpleNamespace(
            enhance_content=lambda **kwargs: f"enhanced:{kwargs['content']}"
        ),
        embed_text_fn=lambda _text: ["vec"],
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        doc_model=FakeDocModel,
    )

    chunk = make_content_chunk(
        "project-2", "new body", "New Title", {"summary": "updated summary"}
    )

    stats = service.update_portfolio_content([chunk])

    assert stats == {"processed": 1, "updated": 1, "skipped": 0, "errors": []}
    assert existing.title == "New Title"
    assert existing.content == "new body"
    assert existing.embedding == ["vec"]
    assert existing.doc_metadata["content_hash"] == service._generate_content_hash(
        chunk
    )
    assert existing.doc_metadata["source"] == "portfolio"
    assert "updated_at" in existing.doc_metadata
    assert session.committed is True


@pytest.mark.asyncio
async def test_stream_chat_response_yields_chunks_and_builds_prompt():
    stream = FakeStream(["Hi", " there"])
    ai_client = FakeAIClient(stream)
    service = RAGService(
        db=SimpleNamespace(),
        ai_client=ai_client,
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        embed_text_fn=lambda q: [0.0] * 1536,
        doc_model=FakeDocModel,
        content_enhancer_service=SimpleNamespace(
            enhance_content=lambda **kwargs: kwargs["content"]
        ),
    )

    doc = DocResult(
        id=uuid.uuid4(),
        slug=f"{settings.BLOG_PREFIX}post-1",
        title="Post 1",
        content="Post content",
        doc_metadata={
            "summary": "A summary",
            "tags": ["tech"],
            "source": "blog",
            "created_at": "2024-01-01",
            "updated_at": "2024-02-01",
        },
        similarity=0.8,
    )
    service.get_relevant_documents_with_navigation = lambda query, limit, threshold: [
        doc
    ]

    messages = [ChatMessage(role="user", content="What's new?")]

    tokens = []
    async for token in service.stream_chat_response(messages, limit=2, threshold=0.2):
        tokens.append(token)

    assert tokens == ["Hi", " there"]

    assert ai_client.chat.completions.kwargs["model"] == settings.RAG_CHAT_MODEL
    sent = ai_client.chat.completions.kwargs["messages"]
    assert sent[0]["role"] == "system"
    assert "Title: Post 1" in sent[0]["content"]
    assert sent[-1]["content"] == "What's new?"


@pytest.mark.asyncio
async def test_stream_chat_response_uses_override_model():
    stream = FakeStream(["ok"])
    ai_client = FakeAIClient(stream)
    service = RAGService(
        db=SimpleNamespace(),
        ai_client=ai_client,
        chat_model="gpt-test-model",
        query_expander_service=SimpleNamespace(expand_query=lambda q: q),
        embed_text_fn=lambda q: [0.0] * 1536,
        doc_model=FakeDocModel,
    )
    service.get_relevant_documents_with_navigation = lambda query, limit, threshold: []

    messages = [ChatMessage(role="user", content="ping")]
    async for _token in service.stream_chat_response(messages, limit=2, threshold=0.2):
        pass

    assert ai_client.chat.completions.kwargs["model"] == "gpt-test-model"
