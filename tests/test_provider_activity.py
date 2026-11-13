"""HTTP activity and retry observation through real SDK clients."""

import asyncio
import json
from importlib import import_module

import pytest

from coding_agent.loop import AgentLoop
from coding_agent.providers import AnthropicProvider, OpenAICompatProvider
from coding_agent.providers.activity import (
    RequestActivity,
    _current_request_activity,
    observe_http_client,
)
from coding_agent.providers.base import ProviderResponse, StreamEvent


def _sdk_http(sdk_name: str):
    base_client = import_module(f"{sdk_name}._base_client")
    return getattr(base_client, "httpx2", None) or base_client.httpx


@pytest.mark.asyncio
async def test_openai_sdk_retry_is_counted_from_real_http_sends():
    http = _sdk_http("openai")
    sends = 0

    async def respond(request):
        nonlocal sends
        sends += 1
        if sends == 1:
            return http.Response(
                503,
                headers={"retry-after": "0"},
                json={"error": {"message": "retry", "type": "server_error", "param": None, "code": None}},
                request=request,
            )
        return http.Response(
            200,
            json={
                "id": "chatcmpl-probe",
                "object": "chat.completion",
                "created": 1,
                "model": "probe",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=request,
        )

    provider = OpenAICompatProvider(api_key="test-key")
    original_timeout = provider._client.timeout
    original_max_retries = provider._client.max_retries
    assert original_timeout.connect == 20.0
    assert original_timeout.read == 121.0
    assert original_max_retries == 0
    await provider._client.close()
    provider._client._client = http.AsyncClient(transport=http.MockTransport(respond), timeout=600)
    provider._client.max_retries = 1
    observe_http_client(provider._client)
    assert provider._client.timeout is original_timeout
    assert provider._client._client.timeout.read == 600
    assert provider._client.max_retries == 1
    assert original_timeout.read == 121.0
    assert original_max_retries == 0
    activity = RequestActivity()
    activity.begin_provider_attempt(http_observed=True)
    token = _current_request_activity.set(activity)
    try:
        response = await provider.create_message([{"role": "user", "content": "probe"}])
    finally:
        _current_request_activity.reset(token)
        await provider.aclose()

    assert response.text == "ok"
    assert sends == 2
    assert activity.http_request_attempts == 2
    assert activity.http_retry_count == 1
    assert activity.http_observation_supported is True
    assert activity.last_http_status == 200
    assert activity.last_activity_source == "http"


@pytest.mark.asyncio
async def test_anthropic_sdk_uses_http_observer():
    http = _sdk_http("anthropic")
    sends = 0

    async def respond(request):
        nonlocal sends
        sends += 1
        return http.Response(
            200,
            json={
                "id": "msg_probe", "type": "message", "role": "assistant", "model": "probe",
                "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            request=request,
        )

    provider = AnthropicProvider(api_key="test-key")
    assert provider._client.max_retries == 0
    assert provider._client.timeout.connect == 20.0
    assert provider._client.timeout.read == 121.0
    await provider.aclose()
    provider._client._client = http.AsyncClient(transport=http.MockTransport(respond), timeout=600)
    observe_http_client(provider._client)
    activity = RequestActivity()
    activity.begin_provider_attempt(http_observed=True)
    token = _current_request_activity.set(activity)
    try:
        response = await provider.create_message([{"role": "user", "content": "probe"}])
    finally:
        _current_request_activity.reset(token)
        await provider.aclose()

    assert response.text == "ok"
    assert sends == 1
    assert activity.http_request_attempts == 1
    assert activity.last_http_status == 200
    assert activity.last_activity_source == "http"


@pytest.mark.asyncio
async def test_openai_connection_retries_are_counted_and_error_propagates():
    import openai

    http = _sdk_http("openai")
    sends = 0

    async def fail(request):
        nonlocal sends
        sends += 1
        raise http.ConnectError("offline", request=request)

    provider = OpenAICompatProvider(api_key="test-key")
    await provider.aclose()
    provider._client._client = http.AsyncClient(transport=http.MockTransport(fail), timeout=600)
    provider._client.max_retries = 2

    async def no_delay(**kwargs):
        return None

    provider._client._sleep_for_retry = no_delay
    observe_http_client(provider._client)
    activity = RequestActivity()
    activity.begin_provider_attempt(http_observed=True)
    token = _current_request_activity.set(activity)
    try:
        with pytest.raises(openai.APIConnectionError):
            await provider.create_message([{"role": "user", "content": "probe"}])
    finally:
        _current_request_activity.reset(token)
        await provider.aclose()

    assert sends == 3
    assert activity.http_request_attempts == 3
    assert activity.http_retry_count == 2
    assert activity.last_activity_type == "http_error"


@pytest.mark.asyncio
async def test_openai_request_cancellation_propagates_without_becoming_http_error():
    http = _sdk_http("openai")
    entered = asyncio.Event()
    release = asyncio.Event()

    async def wait_for_release(request):
        entered.set()
        await release.wait()
        return http.Response(200, json={}, request=request)

    provider = OpenAICompatProvider(api_key="test-key")
    await provider.aclose()
    provider._client._client = http.AsyncClient(
        transport=http.MockTransport(wait_for_release), timeout=600,
    )
    observe_http_client(provider._client)
    activity = RequestActivity()
    activity.begin_provider_attempt(http_observed=True)
    token = _current_request_activity.set(activity)
    request = asyncio.create_task(
        provider.create_message([{"role": "user", "content": "probe"}]),
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
    finally:
        release.set()
        _current_request_activity.reset(token)
        await provider.aclose()

    assert activity.http_request_attempts == 1
    assert activity.last_activity_type == "http_request_started"
    assert activity.http_retry_count == 0


@pytest.mark.asyncio
async def test_stream_bytes_are_observed_before_model_events():
    http = _sdk_http("openai")

    class DelayedSSE(http.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.02)
            yield b": keep-alive\n\n"
            await asyncio.sleep(0.02)
            first = {
                "id": "chatcmpl-probe", "object": "chat.completion.chunk", "created": 1,
                "model": "probe", "choices": [{"index": 0, "delta": {"content": "ok"},
                                                   "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n".encode()
            final = {
                "id": "chatcmpl-probe", "object": "chat.completion.chunk", "created": 1,
                "model": "probe", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            pass

    async def respond(request):
        return http.Response(200, headers={"content-type": "text/event-stream"},
                             stream=DelayedSSE(), request=request)

    provider = OpenAICompatProvider(api_key="test-key")
    await provider.aclose()
    provider._client._client = http.AsyncClient(transport=http.MockTransport(respond), timeout=600)
    observe_http_client(provider._client)
    activity = RequestActivity()
    activity.begin_provider_attempt(http_observed=True)
    token = _current_request_activity.set(activity)
    try:
        events = [event async for event in provider.stream_message([{"role": "user", "content": "probe"}])]
    finally:
        _current_request_activity.reset(token)
        await provider.aclose()

    assert [event.type for event in events] == ["text_delta", "done"]
    assert events[-1].response.text == "ok"
    assert activity.http_request_attempts == 1
    assert activity.last_activity_type == "http_response_data"
    assert activity.last_activity_source == "http"


@pytest.mark.asyncio
async def test_wait_warning_reports_http_activity_and_retry_counts():
    release = asyncio.Event()
    response = ProviderResponse(content=[], stop_reason="end_turn")

    class WaitingProvider:
        model_id = "probe"
        http_activity_supported = True

        async def stream_message(self, **kwargs):
            activity = _current_request_activity.get()
            assert activity is not None
            activity.mark_http_request(0)
            activity.mark_http_response(200)
            activity.mark_http_response_data()
            await release.wait()
            yield StreamEvent(type="done", response=response)

    events = AgentLoop(WaitingProvider(), [], progress_interval=0.01).run(
        [{"role": "user", "content": "probe"}],
    )
    try:
        warning = await asyncio.wait_for(anext(events), timeout=1)
        assert warning.type == "warning"
        assert warning.data["last_activity_source"] == "http"
        assert warning.data["last_activity_type"] == "http_response_data"
        assert warning.data["last_protocol_event"] is None
        assert warning.data["http_request_attempts"] == 1
        assert warning.data["http_retry_count"] == 0
        assert warning.data["http_observation_supported"] is True
        assert warning.data["last_http_status"] == 200
        assert "HTTP 响应数据" in warning.text
        assert "SDK HTTP 重试0次" in warning.text
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
async def test_wait_warning_distinguishes_request_sent_from_response_received():
    release = asyncio.Event()

    class WaitingProvider:
        model_id = "probe"
        http_activity_supported = True

        async def stream_message(self, **kwargs):
            activity = _current_request_activity.get()
            assert activity is not None
            activity.mark_http_request(0)
            await release.wait()
            yield StreamEvent(type="done", response=ProviderResponse([], "end_turn"))

    events = AgentLoop(WaitingProvider(), [], progress_interval=0.01).run(
        [{"role": "user", "content": "probe"}],
    )
    try:
        warning = await asyncio.wait_for(anext(events), timeout=1)
        assert warning.data["last_activity_type"] == "http_request_started"
        assert warning.data["last_activity_source"] == "http"
        assert warning.data["last_http_status"] is None
        assert "HTTP 请求已发出，尚未收到响应" in warning.text
        assert "HTTP 请求1次（SDK HTTP 重试0次）" in warning.text
    finally:
        release.set()
        await events.aclose()


@pytest.mark.asyncio
async def test_custom_provider_wait_does_not_claim_http_counts():
    release = asyncio.Event()

    class WaitingProvider:
        model_id = "custom"

        async def stream_message(self, **kwargs):
            await release.wait()
            yield StreamEvent(type="done", response=ProviderResponse([], "end_turn"))

    events = AgentLoop(WaitingProvider(), [], progress_interval=0.01).run(
        [{"role": "user", "content": "probe"}],
    )
    try:
        warning = await asyncio.wait_for(anext(events), timeout=1)
        assert warning.data["http_observation_supported"] is False
        assert "HTTP 网络计数不可用" in warning.text
    finally:
        release.set()
        await events.aclose()
