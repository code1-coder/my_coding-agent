"""验证真实 SDK 事件结构的适配和流式连接清理。"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from coding_agent.providers import AnthropicProvider, OpenAICompatProvider


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event


@pytest.mark.asyncio
async def test_anthropic_stream_uses_raw_sdk_deltas_and_awaits_final_message():
    from anthropic.types import RawContentBlockStartEvent, RawContentBlockDeltaEvent

    stream = FakeStream([
        RawContentBlockStartEvent(type="content_block_start", index=0,
                                  content_block={"type": "text", "text": ""}),
        RawContentBlockDeltaEvent(type="content_block_delta", index=0,
                                  delta={"type": "text_delta", "text": "你好"}),
        RawContentBlockStartEvent(type="content_block_start", index=1,
                                  content_block={"type": "tool_use", "id": "tool-1", "name": "read_file", "input": {}}),
        RawContentBlockDeltaEvent(type="content_block_delta", index=1,
                                  delta={"type": "input_json_delta", "partial_json": '{"path":"a.txt"}'}),
    ])
    stream.get_final_message = AsyncMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text="你好")], stop_reason="end_turn",
        model="mock", usage=SimpleNamespace(input_tokens=2, output_tokens=3),
    ))
    provider = AnthropicProvider(api_key="test-key")
    await provider.aclose()
    provider._client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kwargs: stream))
    events = [event async for event in provider.stream_message([{"role": "user", "content": "你好"}])]
    assert events[0].type == "text_delta"
    assert events[0].text == "你好"
    assert events[2].tool_id == "tool-1"
    assert events[2].text == '{"path":"a.txt"}'
    assert events[-1].response.usage == {"input_tokens": 2, "output_tokens": 3}
    stream.get_final_message.assert_awaited_once()
    assert stream.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [True, False])
async def test_openai_stream_is_closed_on_success_and_failure(fail):
    from openai.types.chat import ChatCompletionChunk

    chunk = ChatCompletionChunk(id="reply", created=0, object="chat.completion.chunk", model="mock",
                                choices=[{"index": 0, "delta": {"content": "你好"}, "finish_reason": "stop"}])
    stream = FakeStream([chunk, RuntimeError("中途断开")] if fail else [chunk])
    provider = OpenAICompatProvider(api_key="test-key")
    await provider.aclose()
    create = AsyncMock(return_value=stream)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    async def consume():
        return [event async for event in provider.stream_message([{"role": "user", "content": "你好"}])]

    if fail:
        with pytest.raises(RuntimeError, match="中途断开"):
            await consume()
    else:
        events = await consume()
        assert events[-1].response.text == "你好"
    assert stream.closed
