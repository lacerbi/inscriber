"""Table restructuring pass (dev/docs/table-reconstruction-findings.md).

Unit tests for the tables module (blob detection, guards, prompt, sanitation,
splicing), the Gemma table call (thinking kwarg, truncation→None), the table
cache key, and mocked pipeline/bundle integration (run + ocr→describe).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from inscriber import cache as cache_mod
from inscriber import pipeline
from inscriber.bundle import BundleError, read_bundle
from inscriber.cache import make_table_key, make_vlm_key
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import ResolutionMode, RunConfig
from inscriber.ocr.deepseek import DeepSeekOcrBackend
from inscriber.pdf.rasterize import rasterize
from inscriber.postprocess.tables import (
    blob_is_refinable,
    find_table_blobs,
    format_table_prompt,
    locator_text,
    sanitize_table_output,
    splice_tables,
    table_page_context,
)
from inscriber.vlm.gemma import GemmaVlmBackend

FIXTURES = Path(__file__).parent / "fixtures"

BLOB = "<table><td>Dep. Variable:CC</td><td>SR-squared:0.616</td><br><td colspan=\"2\">Model:OLS</td></table>"
PIPE_TABLE = "| Dep. Variable | CC |\n| --- | --- |\n| R-squared | 0.616 |"


# --------------------------------------------------------------------------- #
# tables module units
# --------------------------------------------------------------------------- #


def test_find_table_blobs_keeps_adjacent_tables_separate():
    md = f"intro\n\n{BLOB}\n\nmiddle\n\n<table><td>x1</td></table>\n\nend"
    blobs = find_table_blobs(md)
    assert len(blobs) == 2
    assert blobs[0][2] == BLOB
    assert blobs[1][2] == "<table><td>x1</td></table>"


def test_find_table_blobs_ignores_unclosed_table():
    assert find_table_blobs("text <table><td>a</td> never closed") == []


def test_blob_with_figure_placeholder_is_not_refinable():
    assert not blob_is_refinable("<table><td>a</td>⟦INSCRIBER_FIG:fig_p1_1⟧</table>")


def test_empty_blob_is_not_refinable():
    assert not blob_is_refinable("<table></table>")
    assert not blob_is_refinable("<table> <br> </table>")
    assert blob_is_refinable(BLOB)


def test_nested_table_blob_is_not_refinable():
    md = "before\n<table><td>a</td><table><td>b</td></table></table>\nafter"
    ((_, _, blob),) = find_table_blobs(md)
    assert blob.endswith("<td>b</td></table>")  # non-greedy match stops at the INNER close
    assert not blob_is_refinable(blob)  # splicing would orphan the outer </table> tail


def test_table_page_context_strips_blobs_and_placeholders():
    md = f"Prose before.\n\n{BLOB}\n\n⟦INSCRIBER_FIG:fig_p1_1⟧\n\nProse after."
    ctx = table_page_context(md)
    assert "Prose before." in ctx and "Prose after." in ctx
    assert "<table" not in ctx and "INSCRIBER_FIG" not in ctx


def test_locator_single_and_multiple():
    assert locator_text(1, 1) == "This page contains a single table; reconstruct it."
    multi = locator_text(2, 3)
    assert multi.startswith("This page contains 3 tables; reconstruct the 2nd table")
    assert "values match the OCR text below" in multi


@pytest.mark.parametrize(
    "n,expected", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (21, "21st")]
)
def test_locator_ordinals(n, expected):
    assert expected in locator_text(n, 99)


def test_format_table_prompt_contains_all_parts():
    prompt = format_table_prompt(BLOB, "Page prose here.", table_index=1, table_count=1)
    assert prompt.startswith(
        "You are reconstructing ONE table from a scientific paper as clean GitHub-flavored Markdown."
    )
    assert "This page contains a single table; reconstruct it." in prompt
    assert "<page_text>\nPage prose here.\n</page_text>" in prompt
    assert prompt.rstrip().endswith(BLOB)
    assert "Output ONLY the markdown table. No commentary." in prompt


def test_sanitize_accepts_clean_pipe_table():
    assert sanitize_table_output(PIPE_TABLE) == PIPE_TABLE


def test_sanitize_unwraps_code_fence():
    assert sanitize_table_output(f"```markdown\n{PIPE_TABLE}\n```") == PIPE_TABLE
    assert sanitize_table_output(f"```\n{PIPE_TABLE}\n```\n") == PIPE_TABLE


def test_sanitize_rejects_commentary_and_non_tables():
    assert sanitize_table_output(f"Here is the table:\n{PIPE_TABLE}") is None
    assert sanitize_table_output("Just prose, no table.") is None
    assert sanitize_table_output("| single row |") is None
    assert sanitize_table_output("") is None
    assert sanitize_table_output(None) is None


def test_splice_tables_replaces_blob_with_spacing():
    md = f"before\n{BLOB}\nafter"
    (start, end, _),  = find_table_blobs(md)
    out = splice_tables(md, [(start, end, PIPE_TABLE)])
    assert out == f"before\n\n{PIPE_TABLE}\n\nafter"


def test_splice_tables_multiple_reverse_order_safe():
    blob2 = "<table><td>x1</td></table>"
    md = f"a\n\n{BLOB}\n\nb\n\n{blob2}\n\nc"
    spans = find_table_blobs(md)
    out = splice_tables(
        md, [(spans[0][0], spans[0][1], "| t1 |\n| -- |"), (spans[1][0], spans[1][1], "| t2 |\n| -- |")]
    )
    assert out == "a\n\n| t1 |\n| -- |\n\nb\n\n| t2 |\n| -- |\n\nc"
    assert "<table" not in out


# --------------------------------------------------------------------------- #
# Gemma table call: thinking kwarg + truncation
# --------------------------------------------------------------------------- #


class _FakeClient:
    def __init__(self, response: str, finish_reason: str = "stop") -> None:
        self.response = response
        self.last_finish_reason = finish_reason
        self.last_completion_tokens = 10
        self.calls: list[dict] = []

    def chat_image(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_restructure_table_returns_raw_and_sends_thinking_kwarg():
    client = _FakeClient(PIPE_TABLE)
    backend = GemmaVlmBackend(client=client)
    prompt = backend.build_table_prompt(BLOB, "page text", table_index=1, table_count=1)
    out = backend.restructure_table(b"png", prompt)
    assert out == PIPE_TABLE
    call = client.calls[0]
    assert call["prompt"] == prompt  # the cache-key prompt is what gets sent
    assert call["chat_template_kwargs"] == {"enable_thinking": True}
    assert "max_tokens" not in call  # ctx_size is the single size knob
    assert call["image_first"] is True
    assert call["sampling"]["temperature"] == 0


def test_restructure_table_truncated_returns_none():
    client = _FakeClient(PIPE_TABLE, finish_reason="length")
    backend = GemmaVlmBackend(client=client)
    prompt = backend.build_table_prompt(BLOB, "page text", table_index=1, table_count=1)
    assert backend.restructure_table(b"png", prompt) is None


def test_describe_sends_thinking_kwarg():
    client = _FakeClient("<img_desc>A chart.</img_desc>")
    backend = GemmaVlmBackend(client=client)
    assert backend.describe(b"png", backend.build_prompt(None)) == "A chart."
    assert client.calls[0]["chat_template_kwargs"] == {"enable_thinking": True}


# --------------------------------------------------------------------------- #
# cache keys
# --------------------------------------------------------------------------- #


def test_table_key_never_collides_with_vlm_key():
    shared = dict(
        vlm_backend_name="gemma",
        vlm_model_identity="m",
        vlm_mmproj_identity="p",
        server_identity="version: 9028 (abc1234)",
        full_assembled_prompt="prompt",
        sampling={"temperature": 0},
        chat_template_kwargs={"enable_thinking": True},
    )
    assert make_table_key(page_image_hash="h", **shared) != make_vlm_key(
        figure_crop_hash="h", **shared
    )


def test_thinking_kwarg_is_key_material():
    base = dict(
        page_image_hash="h",
        vlm_backend_name="gemma",
        vlm_model_identity="m",
        vlm_mmproj_identity="p",
        server_identity="version: 9028 (abc1234)",
        full_assembled_prompt="prompt",
        sampling={"temperature": 0},
    )
    on = make_table_key(**base, chat_template_kwargs={"enable_thinking": True})
    off = make_table_key(**base, chat_template_kwargs=None)
    assert on != off


# --------------------------------------------------------------------------- #
# pipeline integration (mocked at the chat boundary)
# --------------------------------------------------------------------------- #

RAW_TABLE_BLOCK = f"\n\ntable[[120, 870, 880, 960]]\n{BLOB}\n"


@pytest.fixture
def hermetic_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "default_cache_dir", lambda: tmp_path / "ocrcache")
    monkeypatch.setattr(cache_mod, "default_vlm_cache_dir", lambda: tmp_path / "vlmcache")
    # Cache keys probe the llama.cpp build identity; no real binary in tests.
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "version: 0 (test)")


def _dummy_models(tmp_path) -> dict:
    paths = {}
    for name in ("ocr", "ocr_mmproj", "vlm", "vlm_mmproj"):
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(name.encode() + b"-bytes")
        paths[name] = str(p)
    return paths


def _base_cfg(tmp_path, models, out, command="run"):
    cfg = RunConfig(command=command, input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.vlm.model = models["vlm"]
    cfg.vlm.mmproj = models["vlm_mmproj"]
    return cfg


def _mock_inference(monkeypatch, *, table_response=PIPE_TABLE, table_finish="stop"):
    """Mock serve + chat. OCR returns the fixture raw plus a degenerate <table>
    block; table prompts return ``table_response``; figure prompts an img_desc.
    Returns the list of table-prompt calls (for cache-hit assertions)."""

    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    raw += RAW_TABLE_BLOCK
    table_calls: list[str] = []

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        self.last_completion_tokens = 10
        if "reconstructing ONE table" in prompt:  # table restructure call
            table_calls.append(prompt)
            self.last_finish_reason = table_finish
            return table_response
        self.last_finish_reason = "stop"
        if "Convert the document to markdown" in prompt:  # OCR (grounded or plain)
            return raw
        return "<img_desc>A line chart trending upward.</img_desc>"  # figure call

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)
    return table_calls


def test_run_refines_table(tmp_path, monkeypatch, hermetic_cache):
    table_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)

    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text  # blob replaced by the restructured table
    assert "<table" not in text
    assert "> **Image description.** A line chart trending upward." in text
    assert len(table_calls) == 1
    # The locator + context + blob all reached the prompt.
    assert "This page contains a single table; reconstruct it." in table_calls[0]
    assert "## Abstract" in table_calls[0]  # page text context
    assert BLOB in table_calls[0]
    assert text.rstrip().endswith(
        "*Transcribed with OCR and VLMs; text, equations, tables, and figure "
        "descriptions may contain mistakes.*"
    )


def test_run_table_cache_hit_on_rerun(tmp_path, monkeypatch, hermetic_cache):
    table_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    pipeline.run(cfg)
    assert len(table_calls) == 1  # second run served from the table cache


def test_no_table_refine_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    table_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.table.refine = False
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert BLOB in text
    assert table_calls == []


def test_truncated_table_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch, table_finish="length")
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert BLOB in text  # truncated output discarded, original kept
    assert PIPE_TABLE not in text


def test_commentary_table_output_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch, table_response=f"Sure! Here it is:\n{PIPE_TABLE}")
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert BLOB in text


def test_tables_refined_even_without_figures(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.figure.detect = "none"  # --no-figures must not disable table refinement
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text
    assert "Image description" not in text
    # VLM credited for tables even with no figure descriptions:
    assert text.rstrip().endswith(
        "*Transcribed with OCR and VLMs; text, equations, and tables "
        "may contain mistakes.*"
    )


def test_concurrent_mode_refines_tables(tmp_path, monkeypatch, hermetic_cache):
    # Concurrent mode pre-launches the VLM server OUTSIDE the lazy session; the
    # table pass must reach it via the endpoint override (no second server).
    table_calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.inference.mode = "concurrent"
    pipeline.run(cfg)
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text
    assert len(table_calls) == 1


def test_multiple_tables_on_page_get_ordinal_locators(tmp_path, monkeypatch, hermetic_cache):
    blob2 = "<table><td>Alpha:1.0</td><td>Beta:2.0</td></table>"
    extra = f"{RAW_TABLE_BLOCK}\ntext[[100, 965, 500, 980]]\nMore prose.\n\ntable[[120, 982, 880, 995]]\n{blob2}\n"
    table_calls = _mock_inference(monkeypatch)
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8") + extra

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        if "reconstructing ONE table" in prompt:
            table_calls.append(prompt)
            return PIPE_TABLE
        if "Convert the document to markdown" in prompt:
            return raw
        return "<img_desc>A chart.</img_desc>"

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)

    assert len(table_calls) == 2
    assert "This page contains 2 tables; reconstruct the 1st table" in table_calls[0]
    assert "This page contains 2 tables; reconstruct the 2nd table" in table_calls[1]
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert "<table" not in text
    assert text.count("| Dep. Variable | CC |") == 2  # both blobs replaced


def test_tables_skipped_gracefully_without_vlm_config(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    models = _dummy_models(tmp_path)
    cfg = RunConfig(command="run", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.figure.detect = "none"  # text-only user: no VLM configured at all
    pipeline.run(cfg)  # must not raise
    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert BLOB in text  # blob kept, run completed
    assert text.rstrip().endswith(
        "*Transcribed with OCR; text, equations, and tables may contain mistakes.*"
    )


# --------------------------------------------------------------------------- #
# bundle two-step: rasters for table pages, describe-time refinement
# --------------------------------------------------------------------------- #


@pytest.fixture
def fixture_pages_results_with_table():
    pdf = (FIXTURES / "sample_paper.pdf").read_bytes()
    pages = rasterize(pdf, ResolutionMode.LARGE)
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    raw += RAW_TABLE_BLOCK
    results = [DeepSeekOcrBackend().parse(raw, pages[0])]
    return pages, results


def _ocr_cfg(tmp_path, out):
    model = tmp_path / "ocr.gguf"
    model.write_bytes(b"ocr-model-bytes")
    mmproj = tmp_path / "mmproj.gguf"
    mmproj.write_bytes(b"ocr-mmproj-bytes")
    cfg = RunConfig(command="ocr", input=str(FIXTURES / "sample_paper.pdf"))
    cfg.output.dir = str(out)
    cfg.ocr.model = str(model)
    cfg.ocr.mmproj = str(mmproj)
    return cfg


def test_bundle_gets_verbatim_raster_for_table_page(
    tmp_path, monkeypatch, hermetic_cache, fixture_pages_results_with_table
):
    pages, results = fixture_pages_results_with_table
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, tmp_path / "out"))[0])

    bundle = read_bundle(bundle_dir)
    assert bundle.pages[0].raster_path == "pages/page_0001.png"
    # Verbatim bytes — run and a later describe share table cache keys.
    assert (bundle_dir / "pages" / "page_0001.png").read_bytes() == pages[0].png_bytes


def test_bundle_without_tables_gets_no_raster(tmp_path, monkeypatch, hermetic_cache):
    pdf = (FIXTURES / "sample_paper.pdf").read_bytes()
    pages = rasterize(pdf, ResolutionMode.LARGE)
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    results = [DeepSeekOcrBackend().parse(raw, pages[0])]
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, tmp_path / "out"))[0])
    bundle = read_bundle(bundle_dir)
    assert bundle.pages[0].raster_path is None
    assert not (bundle_dir / "pages").exists()


def test_describe_refines_tables_from_bundle(
    tmp_path, monkeypatch, hermetic_cache, fixture_pages_results_with_table
):
    pages, results = fixture_pages_results_with_table
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, out))[0])

    table_calls = _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    pipeline.describe(dcfg)

    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text
    assert "<table" not in text
    assert len(table_calls) == 1


def test_describe_old_bundle_without_raster_keeps_blob(
    tmp_path, monkeypatch, hermetic_cache, fixture_pages_results_with_table
):
    pages, results = fixture_pages_results_with_table
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, out))[0])

    # Simulate a pre-table-pass bundle: drop raster_path + the pages/ dir.
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][0].pop("raster_path", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    table_calls = _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    pipeline.describe(dcfg)  # degrades gracefully (warning), never fails

    text = (out / "sample_paper.md").read_text(encoding="utf-8")
    assert BLOB in text
    assert table_calls == []


def test_bundle_missing_referenced_raster_rejected(tmp_path):
    bundle_dir = tmp_path / "b.inscriber-ocr"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps({
            "bundle_schema": 1,
            "source": {"name": "x"},
            "pages": [{
                "page_number": 1,
                "markdown": "<table><td>a1</td></table>",
                "regions": [],
                "figures": [],
                "raster_path": "pages/page_0001.png",
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(BundleError, match="missing referenced page raster"):
        read_bundle(bundle_dir)
