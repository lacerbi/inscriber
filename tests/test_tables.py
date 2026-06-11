"""Table restructuring pass (dev/notes/2026-06-10-table-reconstruction-findings.md).

Unit tests for the tables module (blob detection, guards, prompt, sanitation,
splicing), the Gemma table call (thinking kwarg, truncation→None), the table
cache key, and mocked pipeline/bundle integration (run + ocr→describe).
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
from contextlib import contextmanager
from pathlib import Path

import pytest
from PIL import Image

from inscriber import pipeline
from inscriber.bundle import BundleError, read_bundle
from inscriber.cache import make_table_key, make_vlm_key
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import Region, ResolutionMode, RunConfig
from inscriber.ocr.deepseek import DeepSeekOcrBackend
from inscriber.pdf.crop import crop_region_bytes
from inscriber.pdf.rasterize import rasterize
from inscriber.postprocess.tables import (
    TABLE_PROMPT_TEMPLATE,
    TABLE_PROMPT_TEMPLATE_CROPPED,
    blob_is_refinable,
    digit_coverage_ok,
    find_table_blobs,
    format_table_prompt,
    locator_text,
    match_table_regions,
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


def test_format_cropped_table_prompt():
    prompt = format_table_prompt(
        BLOB, "Page prose here.", table_index=1, table_count=3, cropped=True
    )
    # Same pinned mock discriminator as the whole-page variant.
    assert prompt.startswith(
        "You are reconstructing ONE table from a scientific paper as clean GitHub-flavored Markdown."
    )
    assert "cropped view of the table" in prompt
    assert "This page contains" not in prompt  # no locator on the cropped path
    assert "<page_text>\nPage prose here.\n</page_text>" in prompt
    assert prompt.rstrip().endswith(BLOB)


def test_cropped_template_shares_validated_tail():
    # Everything from the OCR caveat onward (guidelines, context, blob slots)
    # must stay byte-identical to the validated whole-page template — only the
    # locator/crop preamble and the "you are given …" image wording differ.
    marker = "The OCR is generally accurate but NOT perfect"
    assert (
        TABLE_PROMPT_TEMPLATE.split(marker, 1)[1]
        == TABLE_PROMPT_TEMPLATE_CROPPED.split(marker, 1)[1]
    )


# --------------------------------------------------------------------------- #
# blob ↔ grounded-table-region matching (cropped input path)
# --------------------------------------------------------------------------- #

BLOB2 = "<table><td>Alpha:1.0</td><td>Beta:2.0</td></table>"


def _table_region(text: str, bbox=(0.1, 0.5, 0.9, 0.7), label: str = "table") -> Region:
    return Region(label=label, bbox_norm=bbox, text=text)


def test_match_table_regions_by_content():
    r1 = _table_region(BLOB, bbox=(0.1, 0.2, 0.9, 0.4))
    r2 = _table_region(BLOB2, bbox=(0.1, 0.6, 0.9, 0.8))
    # Region order ≠ blob order: content decides, not position in the list.
    assert match_table_regions([BLOB2, BLOB], [r1, r2]) == [r2, r1]


def test_match_table_regions_ignores_non_table_labels():
    # A text-block bbox is not a table bbox even when its text holds the blob.
    assert match_table_regions([BLOB], [_table_region(BLOB, label="text")]) == [None]


def test_match_table_regions_unmatched_blob_is_none():
    # Hand-edited markdown / stale region: content no longer matches.
    assert match_table_regions([BLOB], [_table_region(BLOB2)]) == [None]


def test_match_table_regions_duplicate_blobs_match_in_order():
    r1 = _table_region(BLOB, bbox=(0.1, 0.1, 0.9, 0.3))
    r2 = _table_region(BLOB, bbox=(0.1, 0.6, 0.9, 0.8))
    assert match_table_regions([BLOB, BLOB], [r1, r2]) == [r1, r2]


def test_match_table_regions_prefers_exact_over_containment():
    # An earlier region whose text merely CONTAINS the blob must not steal it
    # from the later region whose text IS the blob.
    aggregate = _table_region(f"prose around {BLOB} more prose", bbox=(0.1, 0.1, 0.9, 0.3))
    exact = _table_region(BLOB, bbox=(0.1, 0.6, 0.9, 0.8))
    assert match_table_regions([BLOB], [aggregate, exact]) == [exact]


def test_match_table_regions_skips_degenerate_bbox():
    sliver = _table_region(BLOB, bbox=(0.5, 0.2, 0.505, 0.8))  # x-span < MIN_TABLE_REGION_SPAN
    assert match_table_regions([BLOB], [sliver]) == [None]


def test_match_table_regions_anchorless_region_is_skipped():
    # Textless table region with no following region: nothing to anchor on.
    assert match_table_regions([BLOB], [_table_region("")]) == [None]
    # ...or a following region that doesn't carry the blob either.
    follower = Region(label="text", bbox_norm=(0.1, 0.8, 0.9, 0.9), text="prose only")
    assert match_table_regions([BLOB], [_table_region(""), follower]) == [None]


def test_match_table_regions_caption_carried_blob():
    # The REAL build-9587 shape (deepseek_paper_table_p27_raw.txt): table[[bbox]]
    # is an empty block — like image — and the following table_caption block
    # carries the caption AND the <table> HTML. The TABLE region's bbox wins.
    table = _table_region(None, bbox=(0.3, 0.1, 0.7, 0.25))
    caption = Region(
        label="table_caption",
        bbox_norm=(0.3, 0.05, 0.7, 0.09),
        text=f"Table A1: Characteristics.\n\n{BLOB}",
    )
    assert match_table_regions([BLOB], [table, caption]) == [table]


def test_match_table_regions_caption_carried_multiple_tables():
    def pair(blob: str, y: float) -> list[Region]:
        return [
            _table_region(None, bbox=(0.2, y, 0.8, y + 0.1)),
            Region(
                label="table_caption", bbox_norm=(0.2, y - 0.04, 0.8, y - 0.01),
                text=f"Table: caption.\n\n{blob}",
            ),
        ]

    regions = pair(BLOB, 0.1) + pair(BLOB2, 0.5)
    matched = match_table_regions([BLOB, BLOB2], regions)
    assert matched == [regions[0], regions[2]]


def test_parse_and_match_real_table_page_fixture():
    """Golden end-to-end pin on REAL captured output (build 9587, PriorGuide
    p27 at gundam/2048): the parser passes table/table_caption through and the
    matcher anchors the blob via the caption block to the table region's bbox."""
    raw = (FIXTURES / "deepseek_paper_table_p27_raw.txt").read_text(encoding="utf-8")
    from inscriber.models import PageImage

    page = PageImage(page_number=27, png_bytes=b"", width_px=1583, height_px=2048)
    res = DeepSeekOcrBackend().parse(raw, page)

    tables = [r for r in res.regions if r.label == "table"]
    assert len(tables) == 1
    assert tables[0].text is None  # the empty block (anchor lives in the caption)
    captions = [r for r in res.regions if r.label == "table_caption"]
    assert len(captions) == 1 and "<table>" in captions[0].text

    blobs = [b for _, _, b in find_table_blobs(res.markdown)]
    assert len(blobs) == 1 and "Two-Moons" in blobs[0]
    matched = match_table_regions(blobs, res.regions)
    assert matched == [tables[0]]
    # Raw emits table[[333, 128, 663, 226]] — per-axis grid/999.
    assert matched[0].bbox_norm == pytest.approx(
        (333 / 999, 128 / 999, 663 / 999, 226 / 999), abs=1e-6
    )


# --------------------------------------------------------------------------- #
# region cropping (in-memory)
# --------------------------------------------------------------------------- #


def _png(width: int, height: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (250, 250, 250)).save(buf, format="PNG")
    return buf.getvalue()


def test_crop_region_bytes_dimensions():
    out = crop_region_bytes(_png(1000, 2000), (0.25, 0.25, 0.75, 0.5), padding=0.02)
    img = Image.open(io.BytesIO(out))
    # x: (0.23..0.77)×1000 → 540 px; y: (0.23..0.52)×2000 → 580 px.
    assert img.size == (540, 580)


def test_crop_region_bytes_clamps_at_page_edges():
    out = crop_region_bytes(_png(100, 100), (0.0, 0.0, 1.0, 1.0), padding=0.05)
    assert Image.open(io.BytesIO(out)).size == (100, 100)


def test_crop_region_bytes_degenerate_returns_none():
    assert crop_region_bytes(_png(100, 100), (0.5, 0.5, 0.5, 0.5), padding=0.0) is None


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


# --------------------------------------------------------------------------- #
# digit-coverage guard (silent-data-loss; DESIGN §9.7)
# --------------------------------------------------------------------------- #


def test_digit_coverage_accepts_correct_resegmentation():
    # The digit stream is invariant under re-splitting fused values — the
    # validated fusion fix (159.99346.68300.4 → 159.9 | 9346.6 | 8300.4).
    blob = '<table><td rowspan="2">Turin159.99346.68300.41037.48.8</td></table>'
    out = "| Turin | 159.9 | 9346.6 | 8300.4 | 1037.4 | 8.8 |"
    assert digit_coverage_ok(blob, out)


def test_digit_coverage_rejects_dropped_rows():
    blob = "<table>A0.11B0.22C0.33D0.44E0.55F0.66</table>"
    out = "| A | 0.11 |\n| B | 0.22 |"  # 4 of 12 digits: rows silently dropped
    assert not digit_coverage_ok(blob, out)


def test_digit_coverage_ignores_tag_and_entity_digits():
    # colspan="4" / rowspan="3" / &#x27; must not inflate the blob's count.
    blob = '<table><td colspan="4" rowspan="3">x&#x27;s value 7</td></table>'
    assert digit_coverage_ok(blob, "| x's value | 7 |")


def test_digit_coverage_trivially_ok_without_digits():
    assert digit_coverage_ok("<table><td>alpha</td></table>", "| alpha | beta |")


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


def test_build_table_prompt_cropped_variant():
    backend = GemmaVlmBackend()
    prompt = backend.build_table_prompt(
        BLOB, "page text", table_index=1, table_count=2, cropped=True
    )
    assert "cropped view of the table" in prompt
    assert "This page contains" not in prompt


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


_KEY_KWARGS = dict(
    page_image_hash="h",
    vlm_backend_name="gemma",
    vlm_model_identity="m",
    vlm_mmproj_identity="p",
    server_identity="version: 9587 (abc1234)",
    full_assembled_prompt="prompt",
    sampling={"temperature": 0},
    chat_template_kwargs={"enable_thinking": True},
)


def test_table_key_page_path_unchanged_by_crop_feature():
    # The crop fields are added to the payload CONDITIONALLY so whole-page-path
    # keys stay byte-identical to the pre-crop scheme (warm caches preserved).
    # This pins the legacy payload shape.
    legacy_payload = json.dumps(
        {
            "kind": "table-restructure",
            "page_image": "h",
            "backend": "gemma",
            "model": "m",
            "mmproj": "p",
            "server": "version: 9587 (abc1234)",
            "prompt": "prompt",
            "sampling": {"temperature": 0},
            "chat_template_kwargs": {"enable_thinking": True},
        },
        sort_keys=True,
    )
    expected = hashlib.sha256(legacy_payload.encode("utf-8")).hexdigest()
    assert make_table_key(**_KEY_KWARGS) == expected


def test_table_key_crop_fields_are_key_material():
    k_page = make_table_key(**_KEY_KWARGS)
    k_crop = make_table_key(**_KEY_KWARGS, crop_bbox=(0.1, 0.2, 0.9, 0.5), crop_padding=0.02)
    k_other_bbox = make_table_key(
        **_KEY_KWARGS, crop_bbox=(0.1, 0.2, 0.9, 0.6), crop_padding=0.02
    )
    k_other_pad = make_table_key(
        **_KEY_KWARGS, crop_bbox=(0.1, 0.2, 0.9, 0.5), crop_padding=0.05
    )
    assert len({k_page, k_crop, k_other_bbox, k_other_pad}) == 4


# --------------------------------------------------------------------------- #
# pipeline integration (mocked at the chat boundary)
# --------------------------------------------------------------------------- #

# The REAL build-9587 table shape (deepseek_paper_table_p27_raw.txt): an empty
# table[[bbox]] block; the following table_caption block carries caption + HTML.
RAW_TABLE_BLOCK = (
    f"\n\ntable[[120, 870, 880, 960]]\ntable_caption[[120, 845, 600, 865]]\n"
    f"Table 1: Mock results.\n\n{BLOB}\n"
)


# hermetic_cache comes from tests/conftest.py (shared; review E1).


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


class _RecordedCalls:
    """What the mocked chat boundary saw: table prompts + the image each table
    call carried (crop vs whole page), and the page image the OCR call sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.images: list[bytes] = []
        self.ocr_images: list[bytes] = []


def _mock_inference(monkeypatch, *, table_response=PIPE_TABLE, table_finish="stop",
                    raw_extra=RAW_TABLE_BLOCK):
    """Mock serve + chat. OCR returns the fixture raw plus ``raw_extra`` (a
    grounded degenerate <table> block by default); table prompts return
    ``table_response``; figure prompts an img_desc. Returns a
    :class:`_RecordedCalls` (for cache-hit / crop-path assertions)."""

    @contextmanager
    def fake_serve(self, spec):
        yield "http://fake:1"

    monkeypatch.setattr(LlamaServerManager, "serve", fake_serve)
    monkeypatch.setattr(pipeline, "find_binary", lambda *a, **k: Path("llama-server"))

    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
    raw += raw_extra
    calls = _RecordedCalls()

    def fake_chat_image(self, *, image_png, prompt, max_tokens=None, sampling=None,
                        timeout_s=None, image_first=True, chat_template_kwargs=None):
        self.last_completion_tokens = 10
        if "reconstructing ONE table" in prompt:  # table restructure call
            calls.prompts.append(prompt)
            calls.images.append(image_png)
            self.last_finish_reason = table_finish
            return table_response
        self.last_finish_reason = "stop"
        if "Convert the document to markdown" in prompt:  # OCR (grounded or plain)
            calls.ocr_images.append(image_png)
            return raw
        return "<img_desc>A line chart trending upward.</img_desc>"  # figure call

    monkeypatch.setattr(ChatClient, "chat_image", fake_chat_image)

    def fake_chat(self, messages, *, max_tokens=None, sampling=None,
                  timeout_s=None, chat_template_kwargs=None):
        # Text-only surface: only the BibTeX probe lands here; non-citable by
        # default so auto-mode runs stay inert/network-free in these tests.
        self.last_finish_reason = "stop"
        self.last_completion_tokens = 10
        text = " ".join(str(m.get("content", "")) for m in messages)
        if "bibliographic metadata" in text:
            return '{"citable": false}'
        raise AssertionError(f"unexpected text-only chat call: {text[:80]!r}")

    monkeypatch.setattr(ChatClient, "chat", fake_chat)
    return calls


def test_run_refines_table_with_cropped_input(tmp_path, monkeypatch, hermetic_cache):
    calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)

    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text  # blob replaced by the restructured table
    assert "<table" not in text
    assert "> **Image description.** A line chart trending upward." in text
    assert len(calls.prompts) == 1
    # The grounded table region matched → the cropped prompt variant, no locator;
    # context + blob still reach the prompt.
    assert "cropped view of the table" in calls.prompts[0]
    assert "This page contains" not in calls.prompts[0]
    assert "## Abstract" in calls.prompts[0]  # page text context
    assert BLOB in calls.prompts[0]
    # …and the image sent is the crop, not the page raster the OCR call saw.
    page_img = Image.open(io.BytesIO(calls.ocr_images[0]))
    crop_img = Image.open(io.BytesIO(calls.images[0]))
    assert crop_img.width < page_img.width
    assert crop_img.height < page_img.height
    assert text.rstrip().endswith(
        "*Transcribed with OCR and VLMs; text, equations, tables, and figure "
        "descriptions may contain mistakes.*"
    )


def test_run_table_cache_hit_on_rerun(tmp_path, monkeypatch, hermetic_cache):
    calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    pipeline.run(cfg)
    assert len(calls.prompts) == 1  # second run served from the table cache


def test_no_table_refine_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.table.refine = False
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert BLOB in text
    assert calls.prompts == []


def test_truncated_table_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch, table_finish="length")
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert BLOB in text  # truncated output discarded, original kept
    assert PIPE_TABLE not in text


def test_commentary_table_output_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch, table_response=f"Sure! Here it is:\n{PIPE_TABLE}")
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert BLOB in text


def test_value_dropping_table_output_keeps_blob(tmp_path, monkeypatch, hermetic_cache):
    # A syntactically clean pipe table that silently lost the blob's number
    # (the worst observed failure class) must be rejected by the
    # digit-coverage guard — the raw blob still holds every value.
    dropping = "| Dep. Variable | CC |\n| --- | --- |\n| Model | OLS |"
    _mock_inference(monkeypatch, table_response=dropping)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert BLOB in text
    assert "| Model | OLS |" not in text


def test_tables_refined_even_without_figures(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.figure.detect = "none"  # --no-figures must not disable table refinement
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
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
    calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    cfg.inference.mode = "concurrent"
    pipeline.run(cfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text
    assert len(calls.prompts) == 1


def test_multiple_grounded_tables_get_distinct_crops(tmp_path, monkeypatch, hermetic_cache):
    extra = (
        f"{RAW_TABLE_BLOCK}\ntext[[100, 950, 500, 962]]\nMore prose.\n\n"
        f"table[[120, 965, 880, 995]]\n{BLOB2}\n"
    )
    calls = _mock_inference(monkeypatch, raw_extra=extra)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)

    assert len(calls.prompts) == 2
    for prompt in calls.prompts:  # both grounded → both cropped, no locators
        assert "cropped view of the table" in prompt
        assert "This page contains" not in prompt
    assert calls.images[0] != calls.images[1]  # different bboxes → different crops
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "<table" not in text
    assert text.count("| Dep. Variable | CC |") == 2  # both blobs replaced


def test_ungrounded_tables_fall_back_to_page_with_locators(
    tmp_path, monkeypatch, hermetic_cache, caplog
):
    # Blobs inside text-labeled blocks: no table region to crop to → the
    # validated whole-page path, count-aware locators, and an INFO line.
    extra = (
        f"\n\ntext[[100, 850, 880, 905]]\n{BLOB}\n\n"
        f"text[[100, 910, 880, 960]]\n{BLOB2}\n"
    )
    calls = _mock_inference(monkeypatch, raw_extra=extra)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    logging.getLogger("inscriber").propagate = True  # let caplog see records
    with caplog.at_level(logging.INFO, logger="inscriber"):
        pipeline.run(cfg)

    assert len(calls.prompts) == 2
    assert "This page contains 2 tables; reconstruct the 1st table" in calls.prompts[0]
    assert "This page contains 2 tables; reconstruct the 2nd table" in calls.prompts[1]
    assert calls.images[0] == calls.ocr_images[0]  # whole page raster sent
    assert calls.images[1] == calls.ocr_images[0]
    assert "no grounded table region matched" in caplog.text
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "<table" not in text


def test_mixed_page_crops_matched_table_and_falls_back_for_other(
    tmp_path, monkeypatch, hermetic_cache
):
    # First blob grounded (table region), second not (text block): the first
    # gets the crop, the second the whole page + a correct 2-of-2 locator.
    extra = f"{RAW_TABLE_BLOCK}\ntext[[100, 965, 880, 990]]\n{BLOB2}\n"
    calls = _mock_inference(monkeypatch, raw_extra=extra)
    out = tmp_path / "out"
    cfg = _base_cfg(tmp_path, _dummy_models(tmp_path), out)
    pipeline.run(cfg)

    assert len(calls.prompts) == 2
    assert "cropped view of the table" in calls.prompts[0]
    assert "This page contains 2 tables; reconstruct the 2nd table" in calls.prompts[1]
    assert calls.images[0] != calls.ocr_images[0]  # crop
    assert calls.images[1] == calls.ocr_images[0]  # whole page


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
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
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

    calls = _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    pipeline.describe(dcfg)

    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert PIPE_TABLE in text
    assert "<table" not in text
    assert len(calls.prompts) == 1
    # Regions ride the bundle manifest, so describe takes the cropped path too.
    assert "cropped view of the table" in calls.prompts[0]
    crop_img = Image.open(io.BytesIO(calls.images[0]))
    page_img = Image.open(io.BytesIO(pages[0].png_bytes))
    assert crop_img.width < page_img.width and crop_img.height < page_img.height


def test_run_then_describe_share_table_cache(tmp_path, monkeypatch, hermetic_cache):
    # The cropped-path key is (raster hash + bbox + padding); the bundle stores
    # the raster VERBATIM, so a describe after a run is a pure cache hit.
    calls = _mock_inference(monkeypatch)
    out = tmp_path / "out"
    models = _dummy_models(tmp_path)
    pipeline.run(_base_cfg(tmp_path, models, out))
    assert len(calls.prompts) == 1

    bundle_dir = Path(pipeline.run_ocr(_base_cfg(tmp_path, models, out, command="ocr"))[0])
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    pipeline.describe(dcfg)
    assert len(calls.prompts) == 1  # no second VLM table call


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

    calls = _mock_inference(monkeypatch)
    models = _dummy_models(tmp_path)
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.llama.bin_dir = "/fake/bin"
    dcfg.vlm.model = models["vlm"]
    dcfg.vlm.mmproj = models["vlm_mmproj"]
    pipeline.describe(dcfg)  # degrades gracefully (warning), never fails

    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert BLOB in text
    assert calls.prompts == []


def test_no_raster_warning_gated_on_refinable_blobs(caplog):
    # Review A6: a page whose only <table> blobs are non-refinable (here: empty)
    # and which has no raster must NOT warn — nothing refinable is lost. With no
    # refinable work the session is never touched (session=None is safe).
    cfg = RunConfig(command="describe", input="x")
    logging.getLogger("inscriber").propagate = True  # let caplog see records
    empty_pg = pipeline._Page(
        page_number=1, markdown="prose\n\n<table></table>", page_text="prose"
    )
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        assert pipeline._refine_tables(cfg, [empty_pg], session=None) == 0
    assert "no page raster" not in caplog.text

    # ...while a refinable blob without a raster still warns (and keeps the blob).
    refinable_pg = pipeline._Page(page_number=2, markdown=BLOB, page_text="")
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        assert pipeline._refine_tables(cfg, [refinable_pg], session=None) == 0
    assert "no page raster" in caplog.text


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
