"""VLM truncation marker behavior."""

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


def test_chat_client_records_completion_tokens(monkeypatch):
    def fake_post(*args, **kwargs):
        return _Resp(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 123},
            }
        )

    monkeypatch.setattr("inscriber.llama.client.httpx.post", fake_post)
    client = ChatClient("http://local")
    assert client.chat([{"role": "user", "content": "hi"}], max_tokens=4096) == "ok"
    assert client.last_completion_tokens == 123


class _FakeClient:
    def __init__(self, completion_tokens: int | None) -> None:
        self.last_completion_tokens = completion_tokens

    def chat_image(self, **kwargs):
        return "<img_desc>A long partial description</img_desc>"


def test_gemma_appends_truncated_marker_at_token_cap():
    backend = GemmaVlmBackend(client=_FakeClient(4096), max_tokens=4096)
    assert backend.describe(b"png", None) == "A long partial description [...]"


def test_gemma_does_not_mark_below_token_cap():
    backend = GemmaVlmBackend(client=_FakeClient(4095), max_tokens=4096)
    assert backend.describe(b"png", None) == "A long partial description"
