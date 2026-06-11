"""M2: ocr→describe two-step roundtrip (DESIGN §3.1, §8.5).

Mocks the OCR pass and the VLM describe pass (no servers): `ocr` writes a bundle
from real parsed fixture output; `describe` loads it and produces output consistent
with the bundle; a hand-edited page survives; a too-new bundle_schema is rejected.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inscriber import pipeline
from inscriber.bundle import BundleError, read_bundle
from inscriber.models import ResolutionMode, RunConfig
from inscriber.ocr.deepseek import DeepSeekOcrBackend
from inscriber.pdf.rasterize import rasterize

FIXTURES = Path(__file__).parent / "fixtures"


# hermetic_cache comes from tests/conftest.py (shared; review E1).


@pytest.fixture
def fixture_pages_results():
    pdf = (FIXTURES / "sample_paper.pdf").read_bytes()
    pages = rasterize(pdf, ResolutionMode.LARGE)
    raw = (FIXTURES / "deepseek_paper_p1_raw.txt").read_text(encoding="utf-8")
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


def test_ocr_then_describe_roundtrip(tmp_path, monkeypatch, hermetic_cache, fixture_pages_results):
    pages, results = fixture_pages_results
    out = tmp_path / "out"

    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    cfg = _ocr_cfg(tmp_path, out)
    written = pipeline.run_ocr(cfg)
    bundle_dir = Path(written[0])

    # Bundle structure (DESIGN §8.5).
    assert bundle_dir.name == "sample_paper.inscriber-ocr"
    assert (bundle_dir / "manifest.json").is_file()
    assert (bundle_dir / "figures" / "fig_p1_1.png").is_file()

    bundle = read_bundle(bundle_dir)
    assert bundle.source_name == "sample_paper"
    assert any("⟦INSCRIBER_FIG:fig_p1_1⟧" in p.markdown for p in bundle.pages)
    assert bundle.pages[0].figures[0].id == "fig_p1_1"

    # New-bundle figure cache-key material rides the manifest (DESIGN §9.6).
    assert bundle.pages[0].raster_sha256 is not None
    assert bundle.figure_crop_padding == 0.02

    # describe with a mocked VLM pass.
    monkeypatch.setattr(
        pipeline, "_vlm_describe",
        lambda cfg, pages, crop_base, session, **kw: {
            "fig_p1_1": "A line chart trending upward."
        },
    )
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    written2 = pipeline.describe(dcfg)
    paper_md = out / "sample_paper_full.md"
    assert paper_md in [Path(p) for p in written2]
    text = paper_md.read_text(encoding="utf-8")
    assert "> **Image description.** A line chart trending upward." in text
    assert "⟦INSCRIBER_FIG" not in text  # placeholder consumed
    assert "## Abstract" in text  # OCR text carried through


def test_ocr_no_clobber_protects_existing_bundle(
    tmp_path, monkeypatch, hermetic_cache, fixture_pages_results
):
    # Review batch 5: a re-run overwrites the bundle — including hand-edited
    # page markdown (an advertised workflow, DESIGN §8.5) — so `ocr` honors
    # output.clobber and fails fast, before any model work.
    from inscriber.output import OutputError

    pages, results = fixture_pages_results
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    out = tmp_path / "out"
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, out))[0])
    manifest = bundle_dir / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace("## Abstract", "## Edited"),
        encoding="utf-8",
    )

    cfg = _ocr_cfg(tmp_path, out)
    cfg.output.clobber = False
    with pytest.raises(OutputError, match="--no-clobber"):
        pipeline.run_ocr(cfg)
    assert "## Edited" in manifest.read_text(encoding="utf-8")  # hand-edit intact

    # Default clobber=True still overwrites (behavior unchanged for re-runs).
    pipeline.run_ocr(_ocr_cfg(tmp_path, out))
    assert "## Edited" not in manifest.read_text(encoding="utf-8")


def test_write_bundle_unwritable_manifest_raises_bundle_error(tmp_path):
    # Review batch 5: an unwritable manifest after a full OCR pass must be an
    # actionable BundleError, not a raw OSError traceback.
    from inscriber.bundle import write_bundle

    bdir = tmp_path / "b.inscriber-ocr"
    (bdir / "manifest.json").mkdir(parents=True)  # a dir blocks the text write
    with pytest.raises(BundleError, match="could not write bundle manifest"):
        write_bundle(bdir, base_name="x", source={}, ocr_meta={},
                     figure_detect="none", page_results=[], page_figures={})


def test_hand_edited_page_survives(tmp_path, monkeypatch, hermetic_cache, fixture_pages_results):
    pages, results = fixture_pages_results
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, out))[0])

    # Hand-edit the bundle's page markdown (keeping the placeholder).
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["pages"][0]["markdown"] = manifest["pages"][0]["markdown"].replace(
        "## Abstract", "## Edited Abstract"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(pipeline, "_vlm_describe", lambda *a, **k: {"fig_p1_1": "desc"})
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    pipeline.describe(dcfg)
    text = (out / "sample_paper_full.md").read_text(encoding="utf-8")
    assert "## Edited Abstract" in text


def test_describe_with_bibtex_injection(tmp_path, monkeypatch, hermetic_cache, fixture_pages_results):
    pages, results = fixture_pages_results
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    bundle_dir = Path(pipeline.run_ocr(_ocr_cfg(tmp_path, out))[0])

    monkeypatch.setattr(pipeline, "_vlm_describe", lambda *a, **k: {"fig_p1_1": "desc"})
    monkeypatch.setattr(
        pipeline, "generate_bibtex",
        lambda title, **k: "@article{key2026,\n  title={" + title + "}\n}",
    )
    dcfg = RunConfig(command="describe", input=str(bundle_dir))
    dcfg.output.dir = str(out)
    dcfg.bibtex.mode = "on"
    dcfg.bibtex.append_to_document = True
    written = pipeline.describe(dcfg)

    # name_from_bibtex (default): the entry's citation key names every output.
    bib = out / "key2026.bib"
    assert bib in [Path(p) for p in written]
    assert "@article{key2026," in bib.read_text(encoding="utf-8")
    # injected (prepended, fenced, --- separator) into the full doc and main split:
    full = (out / "key2026_full.md").read_text(encoding="utf-8")
    assert full.startswith("```\n@article{key2026,")
    assert "\n```\n\n---\n\n" in full
    main = (out / "key2026_main.md").read_text(encoding="utf-8")
    assert main.startswith("```\n@article{key2026,")


def test_ocr_explicit_name_names_the_bundle(
    tmp_path, monkeypatch, hermetic_cache, fixture_pages_results
):
    # --name applies to the ocr bundle; bibtex-derived naming never can (no
    # BibTeX exists at ocr time, DESIGN §14).
    pages, results = fixture_pages_results
    out = tmp_path / "out"
    monkeypatch.setattr(pipeline, "run_ocr_pass", lambda cfg, resolved, work: (pages, results))
    cfg = _ocr_cfg(tmp_path, out)
    cfg.name = "custom name"
    bundle_dir = Path(pipeline.run_ocr(cfg)[0])
    assert bundle_dir.name == "custom_name.inscriber-ocr"
    assert read_bundle(bundle_dir).source_name == "custom_name"


def test_higher_bundle_schema_rejected(tmp_path):
    bundle_dir = tmp_path / "b.inscriber-ocr"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps({"bundle_schema": 999, "source": {"name": "x"}, "pages": []}),
        encoding="utf-8",
    )
    with pytest.raises(BundleError, match="newer than supported"):
        read_bundle(bundle_dir)


def test_missing_crop_rejected(tmp_path):
    bundle_dir = tmp_path / "b.inscriber-ocr"
    bundle_dir.mkdir()
    (bundle_dir / "manifest.json").write_text(
        json.dumps({
            "bundle_schema": 1,
            "source": {"name": "x"},
            "pages": [{
                "page_number": 1,
                "markdown": "⟦INSCRIBER_FIG:fig_p1_1⟧",
                "regions": [],
                "figures": [{"id": "fig_p1_1", "page": 1, "bbox_norm": [0, 0, 1, 1],
                             "crop_path": "figures/fig_p1_1.png", "caption": None}],
            }],
        }),
        encoding="utf-8",
    )
    with pytest.raises(BundleError, match="missing referenced crop"):
        read_bundle(bundle_dir)
