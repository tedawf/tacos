import logging

import httpx

from app.settings import Settings

logger = logging.getLogger(__name__)


class RevalidatePostsService:
    def __init__(
        self,
        *,
        url: str,
        secret: str,
        timeout_seconds: float = 5.0,
        client: httpx.Client | None = None,
    ):
        self.url = url
        self.secret = secret
        self.timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_settings(cls, settings_obj: Settings) -> "RevalidatePostsService":
        return cls(
            url=settings_obj.REVALIDATE_POSTS_URL,
            secret=settings_obj.REVALIDATE_SECRET,
        )

    def revalidate_posts(self, slug: str | None = None) -> bool:
        """
        Trigger on-demand revalidation of the blog index (and optionally a single post).

        Returns True on success. When not configured (no secret) or on failure, returns False.
        """
        if not self.secret:
            return False

        headers = {"x-revalidate-secret": self.secret}
        payload = {"slug": slug} if slug else None

        if self._client is not None:
            return self._post(self._client, headers=headers, payload=payload)

        with httpx.Client(timeout=self.timeout_seconds) as client:
            return self._post(client, headers=headers, payload=payload)

    def _post(
        self,
        client: httpx.Client,
        *,
        headers: dict[str, str],
        payload: dict[str, str] | None,
    ) -> bool:
        try:
            logger.info(
                "Calling revalidation endpoint url=%s slug=%s",
                self.url,
                payload["slug"] if payload else None,
            )
            if payload is None:
                response = client.post(self.url, headers=headers)
            else:
                response = client.post(self.url, headers=headers, json=payload)
            response.raise_for_status()
            logger.info(
                "Revalidation succeeded url=%s slug=%s status=%s",
                self.url,
                payload["slug"] if payload else None,
                getattr(response, "status_code", None),
            )
            return True
        except Exception:
            logger.warning(
                "Revalidation request failed",
                extra={
                    "url": self.url,
                    "has_slug": bool(payload and payload.get("slug")),
                },
                exc_info=True,
            )
            return False
