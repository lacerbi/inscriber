"""VLM truncation detection (finish_reason) + ChatClient request shape.

No ``max_tokens`` is sent on VLM calls — generation is bounded by ``ctx_size``
alone, and hitting the window yields ``finish_reason: "length"``.
"""

from __future__ import annotations

from inscriber.llama.client import ChatClient
from inscriber.vlm.gemma import GemmaVlmBackend


class _Resp:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def json(self):
        return self.payload


def test_chat_client_records_usage_and_finish_reason(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, json, timeout):
        bodies.append(json)
        return _Resp(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 123},
            }
        )

    monkeypatch.setattr("inscriber.llama.client.httpx.post", fake_post)
    client = ChatClient("http://local")
    assert client.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert client.last_completion_tokens == 123
    assert client.last_finish_reason == "length"
    # No max_tokens knob: the field is omitted so ctx_size alone bounds generation.
    assert "max_tokens" not in bodies[0]


def test_chat_client_sends_max_tokens_when_given(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, json, timeout):
        bodies.append(json)
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("inscriber.llama.client.httpx.post", fake_post)
    ChatClient("http://local").chat([{"role": "user", "content": "hi"}], max_tokens=8192)
    assert bodies[0]["max_tokens"] == 8192  # OCR's anti-loop guard still passes one


def test_chat_client_sends_chat_template_kwargs(monkeypatch):
    bodies: list[dict] = []

    def fake_post(url, *, json, timeout):
        bodies.append(json)
        return _Resp({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("inscriber.llama.client.httpx.post", fake_post)
    ChatClient("http://local").chat(
        [{"role": "user", "content": "hi"}],
        chat_template_kwargs={"enable_thinking": True},
    )
    assert bodies[0]["chat_template_kwargs"] == {"enable_thinking": True}


class _FakeClient:
    def __init__(self, finish_reason: str | None) -> None:
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = 10

    def chat_image(self, **kwargs):
        return "<img_desc>A long partial description</img_desc>"


def test_gemma_appends_truncated_marker_on_length():
    backend = GemmaVlmBackend(client=_FakeClient("length"))
    assert backend.describe(b"png", None) == "A long partial description [...]"


def test_gemma_does_not_mark_on_stop():
    backend = GemmaVlmBackend(client=_FakeClient("stop"))
    assert backend.describe(b"png", None) == "A long partial description"


def test_gemma_does_not_mark_when_finish_reason_unknown():
    backend = GemmaVlmBackend(client=_FakeClient(None))
    assert backend.describe(b"png", None) == "A long partial description"
