"""Pure-ASGI middleware: streamed request-body limit with read timeout, and per-client rate limiting."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fleetops.security.ratelimit import TokenBucketLimiter

Metric = Callable[[str], None]


async def _respond(send: Send, status: int, code: str, message: str, headers: dict[str, str] | None = None) -> None:
    body = json.dumps({"error": {"code": code, "message": message, "details": None}}).encode()
    raw = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    raw += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": raw})
    await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    """Reject bodies over ``max_bytes`` *as they stream* (not just by Content-Length) and slow senders.

    The body (<= max_bytes) is buffered and replayed to the application, so the app never sees an oversized or
    half-delivered request. A client that stalls longer than ``read_timeout`` gets 408 (slowloris protection).
    """

    def __init__(
        self, app: ASGIApp, max_bytes: int, read_timeout: float = 10.0, on_reject: Metric | None = None
    ) -> None:
        self.app, self.max_bytes, self.read_timeout, self.on_reject = app, max_bytes, read_timeout, on_reject

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD", "OPTIONS"):
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            self._reject("too_large")
            await _respond(send, 413, "payload_too_large", "request body too large")
            return
        chunks: list[bytes] = []
        total = 0
        try:
            async with asyncio.timeout(self.read_timeout):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    total += len(chunk)
                    if total > self.max_bytes:
                        self._reject("too_large")
                        await _respond(send, 413, "payload_too_large", "request body too large")
                        return
                    chunks.append(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError:
            self._reject("slow_body")
            await _respond(send, 408, "request_timeout", "request body not received in time")
            return
        body = b"".join(chunks)
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()  # afterwards: disconnect notifications

        await self.app(scope, replay, send)

    def _reject(self, reason: str) -> None:
        if self.on_reject:
            self.on_reject(reason)


class RateLimitMiddleware:
    """Per-client-address limit for ``/api/`` requests. Behind a reverse proxy the address is the proxy's: configure
    the proxy's own limits (X-Forwarded-For is deliberately not trusted)."""

    def __init__(
        self, app: ASGIApp, get_limiter: Callable[[Scope], TokenBucketLimiter | None], on_limited: Metric | None = None
    ) -> None:
        self.app, self.get_limiter, self.on_limited = app, get_limiter, on_limited

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/api/"):
            limiter = self.get_limiter(scope)
            client = (scope.get("client") or ("unknown", 0))[0]
            if limiter is not None and not limiter.allow(client):
                if self.on_limited:
                    self.on_limited("api")
                wait = max(1, int(limiter.retry_after(client)) + 1)
                await _respond(send, 429, "rate_limited", "too many requests", {"Retry-After": str(wait)})
                return
        await self.app(scope, receive, send)
