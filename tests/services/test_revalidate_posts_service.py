from app.services.revalidate_posts import RevalidatePostsService


class FakeResponse:
    def __init__(self, *, raise_exc: Exception | None = None):
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc


class FakeClient:
    def __init__(
        self, *, response: FakeResponse | None = None, post_exc: Exception | None = None
    ):
        self.response = response or FakeResponse()
        self.post_exc = post_exc
        self.calls = []

    def post(self, url: str, *, headers=None, json=None):
        if self.post_exc is not None:
            raise self.post_exc
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.response


def test_revalidate_posts_noops_when_secret_missing():
    client = FakeClient()
    service = RevalidatePostsService(url="http://example.com", secret="", client=client)

    assert service.revalidate_posts("hello") is False
    assert client.calls == []


def test_revalidate_posts_posts_without_body_when_slug_missing():
    client = FakeClient()
    service = RevalidatePostsService(
        url="http://example.com/revalidate", secret="s", client=client
    )

    assert service.revalidate_posts(None) is True
    assert client.calls == [
        {
            "url": "http://example.com/revalidate",
            "headers": {"x-revalidate-secret": "s"},
            "json": None,
        }
    ]


def test_revalidate_posts_posts_slug_body_when_provided():
    client = FakeClient()
    service = RevalidatePostsService(
        url="http://example.com/revalidate", secret="s", client=client
    )

    assert service.revalidate_posts("my-slug") is True
    assert client.calls[0]["json"] == {"slug": "my-slug"}


def test_revalidate_posts_returns_false_when_request_fails():
    client = FakeClient(post_exc=RuntimeError("boom"))
    service = RevalidatePostsService(
        url="http://example.com/revalidate", secret="s", client=client
    )

    assert service.revalidate_posts("my-slug") is False
