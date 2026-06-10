"""M5: full `run` pipeline end-to-end with OCR+VLM mocked at the chat boundary.

No real servers: ``LlamaServerManager.serve`` yields a fake URL and
``ChatClient.chat_image`` returns canned responses (grounding raw for OCR,
``<img_desc>`` for the VLM). Asserts the full output-file set + figure injection.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from inscriber import cache as cache_mod
from inscriber import pipeline
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import RunConfig

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def hermetic_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "default_cache_dir", lambda: tmp_path / "ocrcache")
    monkeypatch.setattr(cache_mod, "default_vlm_cache_dir", lambda: tmp_path / "vlmcache")
    # Cache keys probe the llama.cpp build identity; no real binary in tests.
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "version: 9587 (test)")


def _dummy_models(tmp_path) -> dict:
    paths = {}
    for name in ("ocr", "ocr_mmproj", "vlm", "vlm_mmproj"):
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(name.encode() + b"-bytes")
        paths[name] = str(p)
    return paths


def _mock_inference(monkeypatch, *, probe_response='{"citable": false}'):
    """Mock serve + both chat surfaces. The text-only ``chat`` fake serves the
    BibTeX probe ("bibliographic metadata" discriminator); its default answer is
    non-citable so auto-mode runs stay network-free unless a test opts in.
    Returns the list of probe prompts (for cache-hit assertions)."""

    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        # Discriminate on the OCR grounding TOKEN (the page text itself mentions the
        # word "grounding", so a substring check on "grounding" would misfire).
        if "<|grounding|>" in prompt:  # OCR grounding call
            return raw
        return "<img_desc>A line chart trending upward.</img_desc>"  # VLM call

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)

    probe_calls: list[str] = []

    def fake_chat(self, messages, *, max_tokens=None, sampling=None,
                  timeout_s=None, chat_template_kwargs=None):
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        text = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for m in messages
            for part in (m["content"] if isinstance(m["content"], list) else [m["content"]])
        )
        if "bibliographic metadata" in text:  # BibTeX probe (text-only)
            probe_calls.append(text)
            return probe_response
        raise AssertionError(f"unexpected text-only chat call: {text[:80]!r}")

    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    return probe_calls


def test_full_run_mocked(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    out = tmp_path / "out"

    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.vlm.model = models["vlm"]
    cfg.vlm.mmproj = models["vlm_mmproj"]

    written = pipeline.run(cfg)

    assert (out / "sample_paper.md").is_file()
    assert (out / "sample_paper.main.md").is_file()
    assert str(out / "sample_paper.md") in written

    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "> **Image description.** A line chart trending upward." in text
    assert "⟦INSCRIBER_FIG" not in text  # placeholder consumed
    assert "## Abstract" in text  # OCR text carried through
    assert text.rstrip().endswith(
        "*Transcribed with OCR and VLMs; text, equations, and figure descriptions "
        "may contain mistakes.*"
    )


def _base_cfg(tmp_path, models, out):
    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.vlm.model = models["vlm"]
    cfg.vlm.mmproj = models["vlm_mmproj"]
    return cfg


def test_concurrent_mode_runs(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.inference.mode = "concurrent"  # pre-launches the VLM server alongside OCR
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "> **Image description.** A line chart trending upward." in text


def test_describe_and_keep_copies_figures(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.figure.mode = "describe-and-keep"
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "![" in text and "figures/fig_p1_1.png" in text  # image ref kept
    assert (out / "figures" / "fig_p1_1.png").is_file()  # crop copied to output


def test_page_numbers_survive_into_document(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.output.page_numbers = True
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "#### Page 1" in text


def test_notice_can_be_disabled(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.output.notice = False
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "Transcribed with OCR" not in text


def test_no_clobber_errors_on_rerun(tmp_path, monkeypatch, hermetic_cache):
    from inscriber.output import OutputError

    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    cfg.output.clobber = False
    with pytest.raises(OutputError):
        pipeline.run(cfg)


def test_no_cache_writes_nothing_to_cache_dir(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.cache.enabled = False
    pipeline.run(cfg)
    # --no-cache: neither OCR/VLM entries nor the hash sidecar are written.
    assert not (tmp_path / "ocrcache").exists()
    assert not (tmp_path / "vlmcache").exists()


def test_old_llama_build_is_refused(tmp_path, monkeypatch, hermetic_cache):
    # DeepSeek-OCR pins min_server_build = 9587 (the grounding coordinate frame
    # changed upstream, dev/docs/build-9587-verification.md): an older server
    # must be refused up front, not silently mis-crop every figure.
    from inscriber.config import ConfigError

    _mock_inference(monkeypatch)
    monkeypatch.setattr(
        pipeline, "llama_build_identity", lambda *a, **k: "version: 9028 (d6e7b033a)"
    )
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), tmp_path / "out")
    with pytest.raises(ConfigError, match="too old"):
        pipeline.run(cfg)


def test_unknown_build_warns_but_runs(tmp_path, monkeypatch, hermetic_cache):
    # An endpoint without /props build_info yields "unknown" — the gate warns
    # (the user manages that server) but must not block the run.
    _mock_inference(monkeypatch)
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "unknown")
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    assert (out / "sample_paper.md").is_file()


def test_split_files_get_correct_sections(tmp_path):
    # Regression: prepare_formatted_sections returns (main, backmatter, appendix);
    # the pipeline must unpack in that order so .appendix.md / .backmatter.md aren't swapped.
    out = tmp_path / "out"
    cfg = RunConfig(command="run", input="x")
    cfg.output.dir = str(out)
    full_md = (
        "# My Paper\n\nMain body here.\n\n"
        "## Acknowledgments\n\nWe thank the reviewers.\n\n"
        "## Appendix\n\nExtra derivations.\n"
    )
    pipeline._write_documents(cfg, "p", full_md, out)
    appendix = (out / "p.appendix.md").read_text(encoding="utf-8")
    backmatter = (out / "p.backmatter.md").read_text(encoding="utf-8")
    assert appendix.startswith("# My Paper - Appendix")
    assert "Extra derivations" in appendix and "thank the reviewers" not in appendix
    assert backmatter.startswith("# My Paper - Backmatter")
    assert "thank the reviewers" in backmatter and "Extra derivations" not in backmatter


def test_failed_ocr_page_is_not_cached(tmp_path, monkeypatch, hermetic_cache):
    from inscriber.llama.client import ChatError

    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    def boom(self, **kwargs):
        raise ChatError("transient server error")

    monkeypatch.setattr(ChatClient, "chat_image", boom)

    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.figure.detect = "none"  # OCR-only; the OCR call fails for every page
    pipeline.run(cfg)  # resilient: empty page, run still completes

    # The failed page must NOT have been written to the OCR cache (only the hash
    # sidecar may exist), so a later retry re-attempts instead of serving empty.
    ocr_cache = tmp_path / "ocrcache"
    page_entries = [p for p in ocr_cache.glob("*.json") if p.name != "hashes.json"]
    assert page_entries == []


def test_default_auto_bibtex_not_citable_writes_no_bib(tmp_path, monkeypatch, hermetic_cache):
    # bibtex.mode defaults to "auto"; the harness probe answers {"citable": false}
    # → abstain: no .bib, no network (PLAN-bibtex-auto B4 default flip).
    probe_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    written = pipeline.run(cfg)
    assert len(probe_calls) == 1
    assert not (out / "sample_paper.bib").exists()
    assert all(not w.endswith(".bib") for w in written)


def test_default_auto_bibtex_offline_citable_best_efforts(tmp_path, monkeypatch, hermetic_cache):
    # Default auto + --offline + citable probe → marked best-effort entry,
    # assembled fully locally (the chain makes no network call offline).
    probe_calls = _mock_inference(
        monkeypatch,
        probe_response='{"citable": true, "title": "A Sample Paper", "authors": ["Ada B"]}',
    )
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.net.offline = True
    written = pipeline.run(cfg)
    assert len(probe_calls) == 1
    bib = out / "sample_paper.bib"
    assert str(bib) in written
    text = bib.read_text(encoding="utf-8")
    assert text.startswith("% NOTE: Best-effort entry")
    assert "title={A Sample Paper}" in text


def test_run_no_figures_offline_smoke(tmp_path, monkeypatch, hermetic_cache):
    # The README/CI smoke: --no-figures --offline never launches a VLM and
    # produces a clean text-only document.
    _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    out = tmp_path / "out"

    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.figure.detect = "none"
    cfg.net.offline = True

    pipeline.run(cfg)
    assert (out / "sample_paper.md").is_file()
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "⟦INSCRIBER_FIG" not in text
    assert "Image description" not in text  # no figures described
    assert text.rstrip().endswith("*Transcribed with OCR; text may contain mistakes.*")
