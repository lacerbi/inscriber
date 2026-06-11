"""M5: full `run` pipeline end-to-end with OCR+VLM mocked at the chat boundary.

No real servers: ``LlamaServerManager.serve`` yields a fake URL and
``ChatClient.chat_image`` returns canned responses (grounding raw for OCR,
``<img_desc>`` for the VLM). Asserts the full output-file set + figure injection.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from inscriber import pipeline
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import RunConfig

FIXTURES = Path(__file__).parent / "fixtures"

# hermetic_cache comes from tests/conftest.py (shared; review E1).


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

    assert (out / "sample_paper_full.md").is_file()
    assert (out / "sample_paper_main.md").is_file()
    assert str(out / "sample_paper_full.md") in written

    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
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
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "> **Image description.** A line chart trending upward." in text


def test_describe_and_keep_copies_figures(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.figure.mode = "describe-and-keep"
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "![" in text and "figures/fig_p1_1.png" in text  # image ref kept
    assert (out / "figures" / "fig_p1_1.png").is_file()  # crop copied to output


def test_page_numbers_survive_into_document(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.output.page_numbers = True
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "#### Page 1" in text


def test_notice_can_be_disabled(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.output.notice = False
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
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
    # changed upstream, dev/notes/2026-06-10-build-9587-verification.md): an older server
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
    assert (out / "sample_paper_full.md").is_file()


def test_split_files_get_correct_sections(tmp_path):
    # Regression: prepare_formatted_sections returns (main, backmatter, appendix);
    # the pipeline must unpack in that order so _appendix.md / _backmatter.md aren't swapped.
    out = tmp_path / "out"
    cfg = RunConfig(command="run", input="x")
    cfg.output.dir = str(out)
    full_md = (
        "# My Paper\n\nMain body here.\n\n"
        "## Acknowledgments\n\nWe thank the reviewers.\n\n"
        "## Appendix\n\nExtra derivations.\n"
    )
    pipeline._write_documents(cfg, "p", full_md, out)
    appendix = (out / "p_appendix.md").read_text(encoding="utf-8")
    backmatter = (out / "p_backmatter.md").read_text(encoding="utf-8")
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
    # No year in the probe metadata → the citation key would embed the current
    # year (clock-dependent name) — pin the source-derived name here.
    cfg.output.name_from_bibtex = False
    written = pipeline.run(cfg)
    assert len(probe_calls) == 1
    bib = out / "sample_paper.bib"
    assert str(bib) in written
    text = bib.read_text(encoding="utf-8")
    assert text.startswith("% NOTE: Best-effort entry")
    assert "title={A Sample Paper}" in text


def test_explicit_name_overrides_everything(tmp_path, monkeypatch, hermetic_cache):
    # --name wins over both the bibtex key and the source name; it is sanitized.
    probe_calls = _mock_inference(
        monkeypatch,
        probe_response='{"citable": true, "title": "A Sample Paper", '
                       '"authors": ["Ada B"], "year": "2026"}',
    )
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.net.offline = True
    cfg.name = "My Paper (v2)"
    written = pipeline.run(cfg)
    assert len(probe_calls) == 1
    assert (out / "My_Paper_v2_full.md").is_file()
    assert (out / "My_Paper_v2_main.md").is_file()
    assert str(out / "My_Paper_v2.bib") in written  # entry produced, name pinned


def test_no_full_suffix_writes_bare_base_md(tmp_path, monkeypatch, hermetic_cache):
    # full_suffix=False: the full document is {base}.md (library-style); the
    # split files keep their _part suffixes.
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.output.full_suffix = False
    written = pipeline.run(cfg)
    assert (out / "sample_paper.md").is_file()
    assert not (out / "sample_paper_full.md").exists()
    assert (out / "sample_paper_main.md").is_file()
    assert str(out / "sample_paper.md") in written


def test_mock_bibtex_entry_never_names_outputs(tmp_path, monkeypatch, hermetic_cache):
    # on-mode's fallback mock (key unknownYear) is not a usable name source.
    from inscriber.bibtex.semantic_scholar import mock_bibtex

    _mock_inference(monkeypatch)
    monkeypatch.setattr(
        pipeline, "generate_bibtex", lambda title, **k: mock_bibtex(title)
    )
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.bibtex.mode = "on"
    written = pipeline.run(cfg)
    assert (out / "sample_paper_full.md").is_file()  # source-derived, not unknownYear
    assert str(out / "sample_paper.bib") in written  # the mock is still written


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
    assert (out / "sample_paper_full.md").is_file()
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "⟦INSCRIBER_FIG" not in text
    assert "Image description" not in text  # no figures described
    assert text.rstrip().endswith("*Transcribed with OCR; text may contain mistakes.*")


# --------------------------------------------------------------------------- #
# Figure-description cache key: (raster, bbox, padding) — review C2+C3
# --------------------------------------------------------------------------- #


def _counting_figure_mock(monkeypatch) -> list[str]:
    """Re-patch ``chat_image`` (after ``_mock_inference``) to count figure calls."""
    fig_calls: list[str] = []
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")

    def counting_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                            timeout_s=None, image_first=True, chat_template_kwargs=None):
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        if "<|grounding|>" in prompt:  # OCR grounding call
            return raw
        fig_calls.append(prompt)
        return "<img_desc>A line chart trending upward.</img_desc>"

    monkeypatch.setattr(ChatClient, "chat_image", counting_chat_image)
    return fig_calls


def _describe_cfg(models, out, bundle_dir):
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    return dcfg


def test_run_then_describe_share_figure_cache(tmp_path, monkeypatch, hermetic_cache):
    # Review C2: the figure key is (raster hash, bbox, padding) — a describe
    # after a run is a pure cache hit, EVEN when the bundle's crop PNG was
    # re-encoded by a different PNG writer (same pixels, different bytes).
    import io

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    _mock_inference(monkeypatch)
    fig_calls = _counting_figure_mock(monkeypatch)
    models = _dummy_models(tmp_path)
    out = tmp_path / "out"

    pipeline.run(_base_cfg(tmp_path, models, out))
    assert len(fig_calls) == 1

    ocr_cfg = _base_cfg(tmp_path, models, out)
    ocr_cfg.command = "ocr"
    bundle_dir = Path(pipeline.run_ocr(ocr_cfg)[0])

    # Simulate PNG-encoder (Pillow) churn: identical pixels, different bytes
    # (an added tEXt chunk guarantees the byte difference).
    crop = bundle_dir / "figures" / "fig_p1_1.png"
    original = crop.read_bytes()
    info = PngInfo()
    info.add_text("Software", "a different png encoder")
    buf = io.BytesIO()
    Image.open(io.BytesIO(original)).save(buf, "PNG", pnginfo=info)
    assert buf.getvalue() != original
    crop.write_bytes(buf.getvalue())

    pipeline.describe(_describe_cfg(models, out, bundle_dir))
    assert len(fig_calls) == 1  # no second figure call: key is byte-independent


def test_describe_old_bundle_falls_back_to_crop_hash(tmp_path, monkeypatch, hermetic_cache):
    # An old bundle (manifest predating raster_sha256 / figure_crop_padding)
    # still describes: the key falls back to hashing the stored crop bytes —
    # a recompute, never a crash (DESIGN §9.6).
    import json

    _mock_inference(monkeypatch)
    fig_calls = _counting_figure_mock(monkeypatch)
    models = _dummy_models(tmp_path)
    out = tmp_path / "out"

    ocr_cfg = _base_cfg(tmp_path, models, out)
    ocr_cfg.command = "ocr"
    bundle_dir = Path(pipeline.run_ocr(ocr_cfg)[0])

    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("figure_crop_padding", None)
    for p in manifest["pages"]:
        p.pop("raster_sha256", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pipeline.describe(_describe_cfg(models, out, bundle_dir))
    assert len(fig_calls) == 1  # described via the fallback key
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "A line chart trending upward." in text

    # Same old bundle again: the fallback key itself caches normally.
    pipeline.describe(_describe_cfg(models, out, bundle_dir))
    assert len(fig_calls) == 1
