import pytest

from app.services.couchdb_listener import process_change
from app.settings import Settings
from tests.conftest import FakeSession


def test_process_change_skips_when_no_doc(caplog):
    session = FakeSession()
    process_change({"id": "1"}, session, parser=object())
    assert session.committed is False
    assert session.last_query is None


def test_process_change_deletes_when_marked_deleted():
    session = FakeSession()
    change = {"doc": {"_id": "doc-1", "deleted": True}}

    process_change(change, session, parser=object())

    assert session.last_query.filtered is not None
    assert session.committed is True


def test_process_change_skips_non_plain_type():
    session = FakeSession()
    ingest_calls = []

    process_change(
        {"doc": {"_id": "doc-2", "type": "binary"}},
        session,
        parser=object(),
        ingest_fn=lambda *args, **kwargs: ingest_calls.append(True),
    )

    assert ingest_calls == []
    assert session.committed is False


def test_process_change_skips_outside_paths():
    session = FakeSession()
    ingest_calls = []
    custom_settings = Settings(BLOG_PREFIX="blog/", KB_PREFIX="kb/")

    process_change(
        {"doc": {"_id": "doc-3", "type": "plain", "path": "other/path.md"}},
        session,
        parser=object(),
        ingest_fn=lambda *args, **kwargs: ingest_calls.append(True),
        settings_obj=custom_settings,
    )

    assert ingest_calls == []
    assert session.committed is False


def test_process_change_ingests_plain_doc():
    session = FakeSession()
    ingest_calls = []

    process_change(
        {
            "doc": {
                "_id": "blog/keep.md",
                "type": "plain",
                "path": "blog/keep.md",
            }
        },
        session,
        parser="parser",
        ingest_fn=lambda db, doc, parser: ingest_calls.append((db, doc, parser)),
    )

    assert ingest_calls == [
        (
            session,
            {"_id": "blog/keep.md", "type": "plain", "path": "blog/keep.md"},
            "parser",
        )
    ]
    assert session.committed is False


def test_process_change_triggers_revalidation_for_blog_docs():
    session = FakeSession()
    revalidated = []
    custom_settings = Settings(BLOG_PREFIX="blog/", KB_PREFIX="kb/")

    process_change(
        {
            "doc": {
                "_id": "blog/keep.md",
                "type": "plain",
                "path": "blog/keep.md",
            }
        },
        session,
        parser="parser",
        ingest_fn=lambda *_args, **_kwargs: None,
        revalidate_posts_fn=lambda slug: revalidated.append(slug),
        settings_obj=custom_settings,
    )

    assert revalidated == ["keep"]


def test_process_change_does_not_revalidate_for_kb_docs():
    session = FakeSession()
    revalidated = []
    custom_settings = Settings(BLOG_PREFIX="blog/", KB_PREFIX="kb/")

    process_change(
        {
            "doc": {
                "_id": "kb/keep.md",
                "type": "plain",
                "path": "kb/keep.md",
            }
        },
        session,
        parser="parser",
        ingest_fn=lambda *_args, **_kwargs: None,
        revalidate_posts_fn=lambda slug: revalidated.append(slug),
        settings_obj=custom_settings,
    )

    assert revalidated == []


def test_process_change_revalidates_on_blog_deletes():
    session = FakeSession()
    revalidated = []
    custom_settings = Settings(BLOG_PREFIX="blog/", KB_PREFIX="kb/")

    process_change(
        {"doc": {"_id": "blog/gone.md", "deleted": True}},
        session,
        parser="parser",
        revalidate_posts_fn=lambda slug: revalidated.append(slug),
        settings_obj=custom_settings,
    )

    assert revalidated == ["gone"]
