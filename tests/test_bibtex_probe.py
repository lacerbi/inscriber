"""BibTeX citability/metadata probe (DESIGN §12 auto mode, PLAN-bibtex-auto B1).

Prompt assembly + JSON parsing units, the Gemma text-only call (thinking kwarg,
truncation→None), probe cache-key disjointness, and mocked pipeline integration
(probe inside the session, caching, never-cache-failure, no-VLM degrade).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from inscriber import pipeline
from inscriber.bibtex.probe import (
    PROBE_PAGE_CHARS,
    ProbeResult,
    format_probe_prompt,
    parse_probe_response,
)
from inscriber.cache import make_bibtex_probe_key, make_table_key, make_vlm_key
from inscriber.errors import InferenceError
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import RunConfig
from inscriber.vlm.gemma import GemmaVlmBackend

FIXTURES = Path(__file__).parent / "fixtures"

FULL_JSON = (
    '{"citable": true, "title": "Attention Is All You Need", '
    '"authors": ["Ada Lovelace", "Charles Babbage"], "year": "2017", "venue": "NeurIPS"}'
)


# --------------------------------------------------------------------------- #
# prompt assembly
# --------------------------------------------------------------------------- #


def test_probe_prompt_contains_discriminator_and_page_text():
    prompt = format_probe_prompt("My Paper Title\nJane Doe\nAbstract...")
    # The pinned mock-dispatch discriminator (AGENTS.md) — must survive tuning.
    assert "bibliographic metadata" in prompt
    assert "<page_text>\nMy Paper Title\nJane Doe\nAbstract...\n</page_text>" in prompt
    assert '"citable": false' in prompt  # the abstain instruction is present
    # disjoint from the other discriminators:
    assert "reconstructing ONE table" not in prompt
    assert "Convert the document to markdown" not in prompt


def test_probe_prompt_truncates_long_page_text():
    prompt = format_probe_prompt("x" * (PROBE_PAGE_CHARS * 2))
    body = prompt.split("<page_text>\n")[1].split("\n</page_text>")[0]
    assert len(body) == PROBE_PAGE_CHARS
    assert body.endswith("...")


def test_probe_prompt_is_deterministic():
    assert format_probe_prompt("same text") == format_probe_prompt("same text")


# --------------------------------------------------------------------------- #
# response parsing
# --------------------------------------------------------------------------- #


def test_parse_full_metadata_roundtrip():
    r = parse_probe_response(FULL_JSON)
    assert r is not None
    assert r.citable is True
    assert r.title == "Attention Is All You Need"
    assert r.authors == ["Ada Lovelace", "Charles Babbage"]
    assert r.year == "2017"
    assert r.venue == "NeurIPS"
    assert r.raw == FULL_JSON


def test_parse_tolerates_code_fence():
    r = parse_probe_response(f"```json\n{FULL_JSON}\n```")
    assert r is not None and r.citable is True
    assert r.raw == FULL_JSON  # fence stripped from the cached raw


def test_parse_minimal_and_partial():
    r = parse_probe_response('{"citable": false}')
    assert r is not None
    assert r.citable is False
    assert r.title is None and r.authors == [] and r.year is None and r.venue is None
    r = parse_probe_response('{"citable": true, "title": "T", "year": 2023}')
    assert r is not None and r.year == "2023"  # bare-number year tolerated


def test_parse_strips_and_drops_empty_fields():
    r = parse_probe_response('{"citable": true, "title": "  ", "authors": [" A ", ""]}')
    assert r is not None
    assert r.title is None
    assert r.authors == ["A"]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "not json at all",
        '"just a string"',
        "[1, 2]",
        '{"title": "no citable field"}',
        '{"citable": "yes"}',  # non-bool citable
        '{"citable": true, "title": 42}',  # wrong-typed title
        '{"citable": true, "authors": "Jane"}',  # authors not a list
        '{"citable": true, "authors": [1]}',  # non-str author
        f"Sure! Here you go:\n{FULL_JSON}",  # commentary
    ],
)
def test_parse_rejects_malformed(raw):
    assert parse_probe_response(raw) is None


# --------------------------------------------------------------------------- #
# Gemma text-only call
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, response: str, finish_reason: str = "stop") -> None:
        self.response = response
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = 10
        self.calls: list[dict] = []

    def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.response


def test_probe_metadata_sends_text_only_with_thinking_kwarg():
    client = _FakeClient(FULL_JSON)
    backend = GemmaVlmBackend(client=client)
    prompt = backend.build_bibtex_probe_prompt("page text")
    assert backend.probe_metadata(prompt) == FULL_JSON
    call = client.calls[0]
    assert call["messages"] == [{"role": "user", "content": prompt}]  # no image part
    assert call["chat_template_kwargs"] == {"enable_thinking": True}
    assert call["sampling"]["temperature"] == 0
    assert "max_tokens" not in call  # ctx_size is the single size knob


def test_probe_metadata_truncated_returns_none():
    client = _FakeClient(FULL_JSON, finish_reason="length")
    backend = GemmaVlmBackend(client=client)
    assert backend.probe_metadata(backend.build_bibtex_probe_prompt("t")) is None


def test_probe_metadata_without_client_raises():
    backend = GemmaVlmBackend()
    with pytest.raises(InferenceError):
        backend.probe_metadata("prompt")


# --------------------------------------------------------------------------- #
# cache key
# --------------------------------------------------------------------------- #


def test_probe_key_never_collides_with_vlm_or_table_keys():
    shared = dict(
        vlm_backend_name="gemma",
        vlm_model_identity="m",
        vlm_mmproj_identity="p",
        server_identity="version: 9587 (test)",
        full_assembled_prompt="prompt",
        sampling={"temperature": 0},
        chat_template_kwargs={"enable_thinking": True},
    )
    probe_key = make_bibtex_probe_key(**shared)
    assert probe_key != make_vlm_key(figure_crop_hash="h", **shared)
    assert probe_key != make_table_key(page_image_hash="h", **shared)


def test_probe_key_varies_with_prompt():
    base = dict(
        vlm_backend_name="gemma",
        vlm_model_identity="m",
        vlm_mmproj_identity="p",
        server_identity="version: 9587 (test)",
        sampling={"temperature": 0},
        chat_template_kwargs={"enable_thinking": True},
    )
    a = make_bibtex_probe_key(full_assembled_prompt="page one text", **base)
    b = make_bibtex_probe_key(full_assembled_prompt="different text", **base)
    assert a != b


# --------------------------------------------------------------------------- #
# pipeline integration (mocked at the chat boundary)
# --------------------------------------------------------------------------- #


# hermetic_cache comes from tests/conftest.py (shared; review E1).


def _dummy_models(tmp_path) -> dict:
    paths = {}
    for name in ("ocr", "ocr_mmproj", "vlm", "vlm_mmproj"):
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(name.encode() + b"-bytes")
        paths[name] = str(p)
    return paths


def _auto_cfg(tmp_path, out):
    models = _dummy_models(tmp_path)
    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.vlm.model = models["vlm"]
    cfg.vlm.mmproj = models["vlm_mmproj"]
    cfg.bibtex.mode = "auto"
    cfg.net.offline = True  # probe + best-effort only: no network in tests
    return cfg


def _mock_inference(monkeypatch, *, probe_response=FULL_JSON):
    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        if "<|grounding|>" in prompt:
            return raw
        return "<img_desc>A chart.</img_desc>"

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)

    probe_calls: list[str] = []

    def fake_chat(self, messages, *, max_tokens=None, sampling=None,
                  timeout_s=None, chat_template_kwargs=None):
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "bibliographic metadata" in text:
            probe_calls.append(text)
            return probe_response
        raise AssertionError(f"unexpected text-only chat call: {text[:80]!r}")

    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    return probe_calls


def test_probe_runs_inside_session_and_caches(tmp_path, monkeypatch, hermetic_cache):
    probe_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    pipeline.run(cfg)
    assert len(probe_calls) == 1  # the probe was dispatched on its discriminator
    assert "## Abstract" in probe_calls[0]  # page-1 text embedded in the prompt
    pipeline.run(cfg)
    assert len(probe_calls) == 1  # second run served from the probe cache


def test_probe_failure_is_not_cached(tmp_path, monkeypatch, hermetic_cache):
    probe_calls = _mock_inference(monkeypatch, probe_response="I cannot answer that.")
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    pipeline.run(cfg)
    pipeline.run(cfg)
    # Unparseable output → treated as unknown and NOT cached → probed again.
    assert len(probe_calls) == 2


def test_probe_skipped_without_vlm(tmp_path, monkeypatch, hermetic_cache):
    probe_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    cfg.vlm.model = ""
    cfg.vlm.mmproj = ""
    cfg.figure.detect = "none"  # no VLM configured at all
    pipeline.run(cfg)  # must not raise
    assert probe_calls == []


def test_probe_inert_outside_auto_mode(tmp_path, monkeypatch, hermetic_cache):
    probe_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    cfg.bibtex.mode = "off"
    pipeline.run(cfg)
    assert probe_calls == []


def test_probe_result_shape():
    # ProbeResult is a plain dataclass: defaults are inert.
    r = ProbeResult(citable=False)
    assert r.title is None and r.authors == [] and r.raw == ""
