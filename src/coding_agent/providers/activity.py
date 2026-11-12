"""Per-request HTTP activity observation for SDK-backed providers."""

from __future__ import annotations

import asyncio
from importlib import import_module
from contextvars import ContextVar
from typing import Any, AsyncIterator


class RequestActivity:
    """Track model and HTTP activity without retaining request or response payloads."""

    def __init__(self) -> None:
        self.http_request_attempts = 0
        self.last_activity_type: str | None = None
        self.last_activity_source: str | None = None
        self.last_activity_at: float | None = None
        self.last_http_status: int | None = None
        self._provider_http_retry_counts: list[int] = []
        self._provider_http_observation: list[bool] = []

    @property
    def http_retry_count(self) -> int:
        return sum(self._provider_http_retry_counts)

    @property
    def http_observation_supported(self) -> bool:
        return bool(self._provider_http_observation) and all(self._provider_http_observation)

    def begin_provider_attempt(self, *, http_observed: bool) -> None:
        self._provider_http_retry_counts.append(0)
        self._provider_http_observation.append(http_observed)

    def mark_model_event(self, event_type: str) -> None:
        self._mark(event_type, "model")

    def mark_http_request(self, retry_count: int | None) -> None:
        self.http_request_attempts += 1
        if retry_count is not None and self._provider_http_retry_counts:
            self._provider_http_retry_counts[-1] = max(
                self._provider_http_retry_counts[-1], retry_count,
            )
        self._mark("http_request_started", "http")

    def mark_http_response(self, status_code: int) -> None:
        self.last_http_status = status_code
        self._mark("http_response", "http")

    def mark_http_response_data(self) -> None:
        self._mark("http_response_data", "http")

    def mark_http_error(self) -> None:
        self._mark("http_error", "http")

    def _mark(self, event_type: str, source: str) -> None:
        self.last_activity_type = event_type
        self.last_activity_source = source
        self.last_activity_at = asyncio.get_running_loop().time()


_current_request_activity: ContextVar[RequestActivity | None] = ContextVar(
    "coding_agent_request_activity", default=None,
)


def _observed_stream_type(stream_type: type) -> type:
    class ObservedByteStream(stream_type):
        def __init__(self, stream: Any, activity: RequestActivity) -> None:
            self._stream = stream
            self._activity = activity

        async def __aiter__(self) -> AsyncIterator[bytes]:
            try:
                async for chunk in self._stream:
                    self._activity.mark_http_response_data()
                    yield chunk
            except Exception:
                self._activity.mark_http_error()
                raise

        async def aclose(self) -> None:
            await self._stream.aclose()

    return ObservedByteStream


class _ObservedTransport:
    def __init__(self, transport: Any, stream_type: type) -> None:
        self._transport = transport
        self._stream_type = _observed_stream_type(stream_type)

    async def handle_async_request(self, request: Any) -> Any:
        activity = _current_request_activity.get()
        if activity is None:
            return await self._transport.handle_async_request(request)

        retry_header = request.headers.get("x-stainless-retry-count")
        try:
            retry_count = int(retry_header) if retry_header is not None else None
        except ValueError:
            retry_count = None
        activity.mark_http_request(retry_count)
        try:
            response = await self._transport.handle_async_request(request)
        except Exception:
            activity.mark_http_error()
            raise

        activity.mark_http_response(response.status_code)
        response.stream = self._stream_type(response.stream, activity)
        return response

    async def aclose(self) -> None:
        await self._transport.aclose()


def _sdk_http_types(sdk_client: Any) -> tuple[type, type]:
    sdk_module = type(sdk_client).__module__.split(".", 1)[0]
    base_client = import_module(f"{sdk_module}._base_client")
    http_module = getattr(base_client, "httpx2", None) or getattr(base_client, "httpx")
    return http_module.AsyncByteStream, http_module.AsyncBaseTransport


def _wrap_transports(client: Any, transport_type: type, stream_type: type) -> bool:
    wrapped = True
    transport = client._transport
    if isinstance(transport, _ObservedTransport):
        pass
    elif isinstance(transport, transport_type):
        client._transport = _ObservedTransport(transport, stream_type)
    else:
        wrapped = False

    mounts = {}
    for pattern, mount in client._mounts.items():
        if mount is None or isinstance(mount, _ObservedTransport):
            mounts[pattern] = mount
        elif isinstance(mount, transport_type):
            mounts[pattern] = _ObservedTransport(mount, stream_type)
        else:
            mounts[pattern] = mount
            wrapped = False
    client._mounts = mounts
    return wrapped


def observe_http_client(sdk_client: Any) -> bool:
    """Wrap SDK-owned transports without changing client timeout or retry defaults."""
    stream_type, transport_type = _sdk_http_types(sdk_client)
    client = sdk_client._client
    return _wrap_transports(client, transport_type, stream_type)


def current_request_activity() -> RequestActivity | None:
    return _current_request_activity.get()

