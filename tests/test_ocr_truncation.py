"""OCR loop/truncation detection (DESIGN §2.2, §8.6).

A page whose generation stops at the token cap (``finish_reason != "stop"``) is
the repetition-loop signature: page text after the loop point is missing. The
page is flagged ``truncated``, kept best-effort, cached WITH the flag (the OCR
key contains every output-determining knob, so a recompute would reproduce the
same loop — unlike the table pass, whose key excludes ``ctx_size``), and loudly
re-warned on every cache hit. The bundle manifest records the flag per page.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest

from inscriber import cache as cache_mod
from inscriber import pipeline
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import OcrPageResult, PageImage, ResolutionMode, RunConfig
from inscriber.ocr.base import HttpInferencer, inference_truncated
from inscriber.ocr.deepseek import DeepSeekOcrBackend
from inscriber.serialize import ocr_page_result_from_dict, ocr_page_result_to_dict

FIXTURES = Path(__file__).parent / "fixtures"


def _page() -> PageImage:
    return PageImage(page_number=1, png_bytes=b"png-bytes", width_px=100, height_px=140)


# --------------------------------------------------------------------------- #
# Unit: the finish_reason mirror and the truncation predicate
# --------------------------------------------------------------------------- #


def test_http_inferencer_mirrors_finish_reason_and_tokens(monkeypatch):
    def fake_chat_image(self, **kwargs):
        self.last_finish_reason = "length"
        self.last_completion_tokens = 8192
        return "looped output"

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)
    inf = HttpInferencer("http://fake:1")
    assert inf.last_finish_reason is None  # unknown until the first call
    out = inf.infer(
        _page(), "p", sampling={}, chat_template=None, max_tokens=8192, timeout_s=1
    )
    assert out == "looped output"
    assert inf.last_finish_reason == "length"
    assert inf.last_completion_tokens == 8192


class _FakeInf:
    """Inferencer test double; ``None`` finish_reason = unknown (mtmd-cli, mocks)."""

    def __init__(self, finish_reason: str | None) -> None:
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = 8192 if finish_reason == "length" else 10
        self.last_raw = ""

    def infer(self, image, prompt, *, sampling, chat_template, max_tokens, timeout_s):
        self.last_raw = "text[[10, 10, 900, 200]]\nThe page text before the loop."
        return self.last_raw


@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [("stop", False), ("length", True), (None, False)],
)
def test_inference_truncated_predicate(finish_reason, truncated):
    assert inference_truncated(_FakeInf(finish_reason)) is truncated


def test_inference_truncated_tolerates_missing_attribute():
    class Bare:
        pass

    assert inference_truncated(Bare()) is False


@pytest.mark.parametrize(
    ("finish_reason", "truncated"),
    [("stop", False), ("length", True), (None, False)],
)
def test_deepseek_ocr_page_sets_truncated_flag(finish_reason, truncated):
    backend = DeepSeekOcrBackend()
    res = backend.ocr_page(_FakeInf(finish_reason), _page(), ResolutionMode.GUNDAM)
    assert res.truncated is truncated
    assert "The page text before the loop." in res.markdown  # best-effort parse kept


# --------------------------------------------------------------------------- #
# Unit: serialization is additive (clean pages keep the original shape)
# --------------------------------------------------------------------------- #


def test_truncated_serialized_only_when_true():
    clean = OcrPageResult(page_number=1, markdown="ok")
    assert "truncated" not in ocr_page_result_to_dict(clean)
    bad = OcrPageResult(page_number=2, markdown="cut", truncated=True)
    d = ocr_page_result_to_dict(bad)
    assert d["truncated"] is True
    assert ocr_page_result_from_dict(d).truncated is True


def test_old_entries_read_as_not_truncated():
    # Cache entries / bundles written before detection existed lack the field.
    old = {"page_number": 1, "markdown": "old", "regions": []}
    assert ocr_page_result_from_dict(old).truncated is False


# --------------------------------------------------------------------------- #
# Pipeline: warn on compute, cache flagged, re-warn on every hit
# --------------------------------------------------------------------------- #


@pytest.fixture
def hermetic_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "default_cache_dir", lambda: tmp_path / "ocrcache")
    monkeypatch.setattr(cache_mod, "default_vlm_cache_dir", lambda: tmp_path / "vlmcache")
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "version: 9587 (test)")


def _mock_ocr(monkeypatch, *, finish_reason: str):
    """Serve + chat_image fakes; every OCR page ends with ``finish_reason``.
    Returns the list of prompts (to count real OCR calls vs cache hits)."""

    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    calls: list[str] = []

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        calls.append(prompt)
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = 8192 if finish_reason == "length" else 10
        return "text[[10, 10, 900, 200]]\nThe page text before the loop."

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)
    return calls


def _ocr_only_cfg(tmp_path, out) -> RunConfig:
    # Text-only offline run: no VLM configured (table refine / probe skip with a
    # warning), so the OCR call is the only inference in play.
    models = {}
    for name in ("ocr", "ocr_mmproj"):
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(name.encode() + b"-bytes")
        models[name] = str(p)
    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.figure.detect = "none"
    cfg.net.offline = True
    return cfg


def _ocr_cache_entries(tmp_path) -> list[Path]:
    return [p for p in (tmp_path / "ocrcache").glob("*.json") if p.name != "hashes.json"]


def test_truncated_page_warns_and_is_cached_flagged(
    tmp_path, monkeypatch, hermetic_cache, caplog
):
    _mock_ocr(monkeypatch, finish_reason="length")
    out = tmp_path / "out"
    cfg = _ocr_only_cfg(tmp_path, out)

    logging.getLogger("inscriber").propagate = True  # let caplog see records
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        pipeline.run(cfg)

    assert "likely a repetition loop" in caplog.text
    assert "after 8192 tokens" in caplog.text
    # The best-effort parse still reaches the output…
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "The page text before the loop." in text
    # …and the page IS cached, marked truncated (DESIGN §8.6).
    entries = _ocr_cache_entries(tmp_path)
    assert entries
    for entry in entries:
        data = json.loads(entry.read_text(encoding="utf-8"))
        assert data["result"]["truncated"] is True


def test_truncated_cache_hit_rewarns_without_recompute(
    tmp_path, monkeypatch, hermetic_cache, caplog
):
    calls = _mock_ocr(monkeypatch, finish_reason="length")
    out = tmp_path / "out"
    cfg = _ocr_only_cfg(tmp_path, out)
    pipeline.run(cfg)
    n_first = len(calls)

    logging.getLogger("inscriber").propagate = True
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        pipeline.run(cfg)

    assert len(calls) == n_first  # served from cache — no OCR recompute
    assert "truncated when OCR'd" in caplog.text
    assert "likely a repetition loop" in caplog.text
    # The flagged cached page still reaches the document output.
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "The page text before the loop." in text


def test_clean_page_neither_warns_nor_flags(tmp_path, monkeypatch, hermetic_cache, caplog):
    _mock_ocr(monkeypatch, finish_reason="stop")
    out = tmp_path / "out"
    cfg = _ocr_only_cfg(tmp_path, out)

    logging.getLogger("inscriber").propagate = True
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        pipeline.run(cfg)

    assert "repetition loop" not in caplog.text
    for entry in _ocr_cache_entries(tmp_path):
        data = json.loads(entry.read_text(encoding="utf-8"))
        assert "truncated" not in data["result"]  # additive: clean shape unchanged


# --------------------------------------------------------------------------- #
# Bundle: the manifest records the flag per page (DESIGN §8.5)
# --------------------------------------------------------------------------- #


def test_bundle_manifest_records_truncated(tmp_path, monkeypatch, hermetic_cache):
    _mock_ocr(monkeypatch, finish_reason="length")
    out = tmp_path / "out"
    cfg = _ocr_only_cfg(tmp_path, out)
    cfg.command = "ocr"
    pipeline.run_ocr(cfg)

    manifest = json.loads(
        (out / "sample_paper.inscriber-ocr" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pages"]
    assert all(p["truncated"] is True for p in manifest["pages"])


def test_bundle_manifest_clean_pages_have_no_flag(tmp_path, monkeypatch, hermetic_cache):
    _mock_ocr(monkeypatch, finish_reason="stop")
    out = tmp_path / "out"
    cfg = _ocr_only_cfg(tmp_path, out)
    cfg.command = "ocr"
    pipeline.run_ocr(cfg)

    manifest = json.loads(
        (out / "sample_paper.inscriber-ocr" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["pages"]
    assert all("truncated" not in p for p in manifest["pages"])
