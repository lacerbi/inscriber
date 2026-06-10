"""BibTeX auto-mode source chain (DESIGN §12; PLAN-bibtex-auto B3).

httpx is mocked; every chain order / fall-through / degrade path is exercised,
plus pipeline-level provenance behavior (the probe always runs in auto mode —
even with a repository URL — so best-effort survives online failures; describe
reads provenance from the bundle; run and describe share the probe cache).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from inscriber import cache as cache_mod
from inscriber import pipeline
from inscriber.bibtex.arxiv import arxiv_bibtex, arxiv_id_from_url
from inscriber.bibtex.chain import citable_provenance, generate_bibtex_auto
from inscriber.bibtex.probe import ProbeResult
from inscriber.bibtex.semantic_scholar import lookup_arxiv, strip_arxiv_version
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager
from inscriber.models import ResolvedInput, RunConfig

FIXTURES = Path(__file__).parent / "fixtures"

ARXIV_URL = "https://arxiv.org/abs/2510.18234v2"
PROBE = ProbeResult(
    citable=True,
    title="Attention Is All You Need",
    authors=["Ada Lovelace"],
    year="2017",
    venue="arXiv",
)

ATOM_OK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2510.18234v2</id>
    <published>2025-10-21T00:00:00Z</published>
    <title>DeepSeek-OCR: Contexts
  Optical Compression</title>
    <author><name>Haoran Wei</name></author>
    <author><name>Yaofeng Sun</name></author>
    <arxiv:primary_category term="cs.CV"/>
  </entry>
</feed>
"""

ATOM_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
  </entry>
</feed>
"""

S2_PUBLISHED = {
    "title": "Attention Is All You Need",
    "authors": [{"name": "Ashish Vaswani"}],
    "year": 2017,
    "venue": "Advances in Neural Information Processing Systems",
    "externalIds": {"DOI": "10.5555/nips2017"},
    "url": "https://www.semanticscholar.org/paper/x",
}
S2_PREPRINT = {
    "title": "A Fresh Preprint",
    "authors": [{"name": "Jane Smith"}],
    "year": 2026,
    "venue": "",
}

S2_BY_ID = "https://api.semanticscholar.org/graph/v1/paper/arXiv:"
S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_API = "https://export.arxiv.org/api/query"


class _Resp:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


@pytest.fixture
def fake_http(monkeypatch):
    """Route-table httpx.get mock shared by both bibtex modules. Tests fill
    ``routes`` (URL prefix → _Resp | Exception); ``calls`` logs every URL."""
    calls: list[str] = []
    routes: dict[str, object] = {}

    def fake_get(url, *, params=None, timeout=None, headers=None):
        calls.append(url)
        for prefix, resp in routes.items():
            if url.startswith(prefix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        raise AssertionError(f"unrouted httpx.get: {url}")

    monkeypatch.setattr(httpx, "get", fake_get)
    return routes, calls


# --------------------------------------------------------------------------- #
# units: ID extraction, provenance, by-ID lookups
# --------------------------------------------------------------------------- #


def test_arxiv_id_from_url_shapes():
    assert arxiv_id_from_url("https://arxiv.org/abs/2510.18234") == "2510.18234"
    assert arxiv_id_from_url("https://arxiv.org/abs/2510.18234v2") == "2510.18234v2"
    assert arxiv_id_from_url("https://arxiv.org/pdf/2510.18234v2.pdf") == "2510.18234v2"
    assert arxiv_id_from_url("https://arxiv.org/html/2510.18234") == "2510.18234"
    assert arxiv_id_from_url("https://arxiv.org/abs/cs.AI/0301001") == "cs.AI/0301001"
    assert arxiv_id_from_url("https://example.org/abs/2510.18234") is None
    assert arxiv_id_from_url("https://arxiv.org/list/cs.AI/recent") is None
    assert arxiv_id_from_url(None) is None


def test_strip_arxiv_version():
    assert strip_arxiv_version("2510.18234v2") == "2510.18234"
    assert strip_arxiv_version("2510.18234") == "2510.18234"
    assert strip_arxiv_version("cs.AI/0301001") == "cs.AI/0301001"


def test_citable_provenance_recognizes_all_repositories():
    assert citable_provenance("https://arxiv.org/abs/2510.18234")
    assert citable_provenance("https://www.biorxiv.org/content/10.1101/2024.01.01.573000v1")
    assert citable_provenance("https://papers.nips.cc/paper/2017/hash/abc123-Abstract.html")
    assert citable_provenance("https://openreview.net/forum?id=xyz")
    assert citable_provenance("https://proceedings.mlr.press/v202/smith23a.html")
    assert not citable_provenance("https://example.com/whatever.pdf")
    assert not citable_provenance(None)


def test_lookup_arxiv_strips_version_and_degrades(fake_http):
    routes, calls = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PUBLISHED)
    assert lookup_arxiv("2510.18234v2") == S2_PUBLISHED
    assert calls[0].startswith(S2_BY_ID + "2510.18234")
    assert "v2" not in calls[0]

    routes[S2_BY_ID] = _Resp(status_code=404)
    assert lookup_arxiv("2510.18234") is None
    routes[S2_BY_ID] = _Resp(status_code=429)
    assert lookup_arxiv("2510.18234") is None
    routes[S2_BY_ID] = _Resp(json_data={"no": "title"})
    assert lookup_arxiv("2510.18234") is None
    routes[S2_BY_ID] = httpx.HTTPError("network down")
    assert lookup_arxiv("2510.18234") is None


def test_arxiv_bibtex_formats_atom_entry(fake_http):
    routes, _ = fake_http
    routes[ARXIV_API] = _Resp(text=ATOM_OK)
    entry = arxiv_bibtex("2510.18234v2")
    assert entry is not None
    assert entry.startswith("@misc{wei2025deepseekocr,")
    # whitespace in the Atom title collapsed:
    assert "title={DeepSeek-OCR: Contexts Optical Compression}" in entry
    assert "author={Haoran Wei and Yaofeng Sun}" in entry
    assert "year={2025}" in entry
    assert "eprint={2510.18234v2}" in entry
    assert "archivePrefix={arXiv}" in entry
    assert "primaryClass={cs.CV}" in entry
    assert "url={https://arxiv.org/abs/2510.18234v2}" in entry


def test_arxiv_bibtex_degrades(fake_http):
    routes, _ = fake_http
    routes[ARXIV_API] = _Resp(text=ATOM_ERROR)  # bad-ID error entry
    assert arxiv_bibtex("nope") is None
    routes[ARXIV_API] = _Resp(text="<not-xml")
    assert arxiv_bibtex("2510.18234") is None
    routes[ARXIV_API] = _Resp(status_code=500)
    assert arxiv_bibtex("2510.18234") is None
    routes[ARXIV_API] = httpx.HTTPError("network down")
    assert arxiv_bibtex("2510.18234") is None


# --------------------------------------------------------------------------- #
# the chain: order and every fall-through
# --------------------------------------------------------------------------- #


def _auto(probe, url=ARXIV_URL, online=True, fallback="Fallback Title"):
    return generate_bibtex_auto(
        probe, original_url=url, online_allowed=online, fallback_title=fallback
    )


def test_chain_s2_by_id_published_version_wins(fake_http):
    routes, calls = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PUBLISHED)
    bibtex, source = _auto(PROBE)
    assert source == "s2-arxiv-id"
    # the PUBLISHED entry (decision 8), not the arXiv @misc:
    assert bibtex.startswith("@article{vaswani2017attention,")
    assert "journal={Advances in Neural Information Processing Systems}" in bibtex
    assert "% WARNING" not in bibtex  # by-ID match: no title validation
    assert all(not c.startswith(ARXIV_API) for c in calls)  # export API not consulted


def test_chain_s2_by_id_preprint_gets_misc_shape(fake_http):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PREPRINT)
    bibtex, source = _auto(PROBE)
    assert source == "s2-arxiv-id"
    assert bibtex.startswith("@misc{smith2026fresh,")
    assert "eprint={2510.18234v2}" in bibtex  # version preserved in the eprint
    assert "archivePrefix={arXiv}" in bibtex
    assert "journal=" not in bibtex


def test_chain_s2_429_falls_to_arxiv_export(fake_http):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(status_code=429)
    routes[ARXIV_API] = _Resp(text=ATOM_OK)
    bibtex, source = _auto(PROBE)
    assert source == "arxiv-export"
    assert "eprint={2510.18234v2}" in bibtex


def test_chain_both_by_id_sources_fail_falls_to_title_search(fake_http):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(status_code=500)
    routes[ARXIV_API] = _Resp(status_code=500)
    routes[S2_SEARCH] = _Resp(json_data={"data": [S2_PUBLISHED]})
    bibtex, source = _auto(PROBE)
    assert source == "s2-title"
    assert bibtex.startswith("@article{vaswani2017attention,")


def test_chain_no_arxiv_id_uses_title_search_with_probe_title(fake_http):
    routes, calls = fake_http
    routes[S2_SEARCH] = _Resp(json_data={"data": [S2_PUBLISHED]})
    bibtex, source = _auto(PROBE, url=None)
    assert source == "s2-title"
    assert "% WARNING" not in bibtex  # probe title == S2 title → validation passes
    assert all(not c.startswith(S2_BY_ID) for c in calls)


def test_chain_title_validation_compares_against_query(fake_http):
    routes, _ = fake_http
    routes[S2_SEARCH] = _Resp(json_data={"data": [S2_PUBLISHED]})
    probe = ProbeResult(citable=True, title="A Completely Different Title About Bees")
    bibtex, source = _auto(probe, url=None)
    assert source == "s2-title"
    assert bibtex.startswith("% WARNING: The retrieved citation title may not match")
    assert '% Paper title: "A Completely Different Title About Bees"' in bibtex


def test_chain_s2_search_empty_falls_to_best_effort(fake_http):
    routes, _ = fake_http
    routes[S2_SEARCH] = _Resp(json_data={"data": []})
    bibtex, source = _auto(PROBE, url=None)
    assert source == "best-effort"
    assert bibtex.startswith("% NOTE: Best-effort entry")
    assert "title={Attention Is All You Need}" in bibtex


def test_chain_network_unreachable_degrades_to_best_effort(fake_http):
    routes, _ = fake_http
    err = httpx.HTTPError("network unreachable")
    routes[S2_BY_ID] = err
    routes[ARXIV_API] = err
    routes[S2_SEARCH] = err
    bibtex, source = _auto(PROBE)  # never raises
    assert source == "best-effort"
    assert "@misc{" in bibtex


def test_chain_offline_makes_no_http_call(fake_http):
    _, calls = fake_http
    bibtex, source = _auto(PROBE, online=False)
    assert source == "best-effort"
    assert calls == []


def test_chain_provenance_wins_over_probe_says_no(fake_http):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PUBLISHED)
    not_citable = ProbeResult(citable=False)
    bibtex, source = _auto(not_citable)  # arXiv URL: provenance settles it
    assert source == "s2-arxiv-id"
    assert bibtex.startswith("@article{")


def test_chain_offline_provenance_probe_no_still_best_efforts(fake_http):
    _, calls = fake_http
    probe = ProbeResult(citable=False, title="Withdrawn But Real Paper")
    bibtex, source = _auto(probe, online=False)
    assert source == "best-effort"
    assert "title={Withdrawn But Real Paper}" in bibtex
    assert calls == []


def test_chain_abstains_without_provenance():
    assert _auto(ProbeResult(citable=False), url=None) == (None, "not-citable")
    assert _auto(None, url=None) == (None, "unknown")


def test_chain_provenance_without_probe_offline_skips(fake_http):
    _, calls = fake_http
    bibtex, source = _auto(None, online=False)  # probe skipped/failed, offline
    assert bibtex is None
    assert source == "no usable metadata"
    assert calls == []


def test_chain_non_arxiv_provenance_uses_title_search(fake_http):
    routes, calls = fake_http
    routes[S2_SEARCH] = _Resp(json_data={"data": [S2_PUBLISHED]})
    biorxiv = "https://www.biorxiv.org/content/10.1101/2024.01.01.573000v1"
    bibtex, source = _auto(None, url=biorxiv, fallback="Attention Is All You Need")
    assert source == "s2-title"  # provenance counts; query = fallback title
    assert all(not c.startswith(S2_BY_ID) for c in calls)


# --------------------------------------------------------------------------- #
# pipeline integration (mocked chat + serve + httpx)
# --------------------------------------------------------------------------- #


@pytest.fixture
def hermetic_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "default_cache_dir", lambda: tmp_path / "ocrcache")
    monkeypatch.setattr(cache_mod, "default_vlm_cache_dir", lambda: tmp_path / "vlmcache")
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "version: 9587 (test)")


def _dummy_models(tmp_path) -> dict:
    paths = {}
    for name in ("ocr", "ocr_mmproj", "vlm", "vlm_mmproj"):
        p = tmp_path / f"{name}.gguf"
        p.write_bytes(name.encode() + b"-bytes")
        paths[name] = str(p)
    return paths


def _auto_cfg(tmp_path, out, command="run", input_arg=None):
    models = _dummy_models(tmp_path)
    cfg = RunConfig(
        command=command, input=input_arg or str(FIXTURES / "sample_paper.pdf")
    )
    cfg.output.dir = str(out)
    cfg.llama.bin_dir = "/fake/bin"
    cfg.ocr.model = models["ocr"]
    cfg.ocr.mmproj = models["ocr_mmproj"]
    cfg.vlm.model = models["vlm"]
    cfg.vlm.mmproj = models["vlm_mmproj"]
    cfg.bibtex.mode = "auto"
    return cfg


def _mock_inference(monkeypatch, *, probe_response):
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


def test_run_with_arxiv_url_probes_but_prefers_s2_by_id(
    tmp_path, monkeypatch, hermetic_cache, fake_http
):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PUBLISHED)
    probe_calls = _mock_inference(monkeypatch, probe_response='{"citable": true}')

    pdf_bytes = (FIXTURES / "sample_paper.pdf").read_bytes()
    monkeypatch.setattr(
        pipeline, "resolve_input",
        lambda *a, **k: ResolvedInput(
            pdf_bytes=pdf_bytes, source="url", original_url=ARXIV_URL,
            suggested_name="arxiv-2510.18234v2",
        ),
    )
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out, input_arg=ARXIV_URL)
    written = pipeline.run(cfg)

    # The probe always runs (its metadata backs best-effort if online sources
    # fail at lookup time), but the by-ID source still wins the chain.
    assert len(probe_calls) == 1
    bib = out / "arxiv-2510_18234v2.bib"
    assert str(bib) in written
    assert bib.read_text(encoding="utf-8").startswith("@article{vaswani2017attention,")


def test_run_provenance_with_rate_limited_s2_falls_back_to_best_effort(
    tmp_path, monkeypatch, hermetic_cache, fake_http
):
    """The regression behind always-probing: a repository URL with no arXiv ID
    (OpenReview) + a rate-limited Semantic Scholar exhausts every online
    source; the probe's metadata — collected while the VLM was still up — is
    all that's left for best-effort."""
    routes, _ = fake_http
    routes[S2_SEARCH] = _Resp(status_code=429)
    full_json = (
        '{"citable": true, "title": "A Sample Paper", '
        '"authors": ["Ada B"], "year": "2026"}'
    )
    probe_calls = _mock_inference(monkeypatch, probe_response=full_json)

    pdf_bytes = (FIXTURES / "sample_paper.pdf").read_bytes()
    openreview_url = "https://openreview.net/pdf?id=G4I23g5Ugh"
    monkeypatch.setattr(
        pipeline, "resolve_input",
        lambda *a, **k: ResolvedInput(
            pdf_bytes=pdf_bytes, source="url", original_url=openreview_url,
            suggested_name="openreview-G4I23g5Ugh",
        ),
    )
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out, input_arg=openreview_url)
    written = pipeline.run(cfg)

    assert len(probe_calls) == 1  # probed despite recognized provenance
    bib = out / "openreview-G4I23g5Ugh.bib"
    assert str(bib) in written
    content = bib.read_text(encoding="utf-8")
    assert content.startswith("% NOTE: Best-effort entry")
    assert "title={A Sample Paper}" in content


def test_describe_reads_provenance_from_bundle(tmp_path, monkeypatch, hermetic_cache, fake_http):
    routes, _ = fake_http
    routes[S2_BY_ID] = _Resp(json_data=S2_PUBLISHED)
    probe_calls = _mock_inference(monkeypatch, probe_response='{"citable": true}')

    pdf_bytes = (FIXTURES / "sample_paper.pdf").read_bytes()
    monkeypatch.setattr(
        pipeline, "resolve_input",
        lambda *a, **k: ResolvedInput(
            pdf_bytes=pdf_bytes, source="url", original_url=ARXIV_URL,
            suggested_name="arxiv-paper",
        ),
    )
    out = tmp_path / "out"
    ocr_cfg = _auto_cfg(tmp_path, out, command="ocr", input_arg=ARXIV_URL)
    bundle_dir = Path(pipeline.run_ocr(ocr_cfg)[0])

    dcfg = _auto_cfg(tmp_path, out, command="describe", input_arg=str(bundle_dir))
    written = pipeline.describe(dcfg)

    # One probe call (describe always probes in auto mode); the @article below
    # proves the chain got the arXiv URL from the bundle manifest's provenance.
    assert len(probe_calls) == 1
    bib = out / "arxiv-paper.bib"
    assert str(bib) in written
    assert "@article{vaswani2017attention," in bib.read_text(encoding="utf-8")


def test_run_and_describe_share_probe_cache_and_bib(tmp_path, monkeypatch, hermetic_cache):
    full_json = (
        '{"citable": true, "title": "Attention Is All You Need", '
        '"authors": ["Ada Lovelace"], "year": "2017"}'
    )
    probe_calls = _mock_inference(monkeypatch, probe_response=full_json)

    out_run = tmp_path / "out-run"
    cfg = _auto_cfg(tmp_path, out_run)
    cfg.net.offline = True  # local file + offline → probe + best-effort only
    pipeline.run(cfg)
    assert len(probe_calls) == 1
    bib_run = (out_run / "sample_paper.bib").read_text(encoding="utf-8")
    assert bib_run.startswith("% NOTE: Best-effort entry")

    out_desc = tmp_path / "out-desc"
    ocr_cfg = _auto_cfg(tmp_path, out_desc, command="ocr")
    bundle_dir = Path(pipeline.run_ocr(ocr_cfg)[0])
    dcfg = _auto_cfg(tmp_path, out_desc, command="describe", input_arg=str(bundle_dir))
    dcfg.net.offline = True
    pipeline.describe(dcfg)

    assert len(probe_calls) == 1  # describe served from the shared probe cache
    bib_desc = (out_desc / "sample_paper.bib").read_text(encoding="utf-8")
    assert bib_desc == bib_run


def test_run_not_citable_skips_bib(tmp_path, monkeypatch, hermetic_cache):
    _mock_inference(monkeypatch, probe_response='{"citable": false}')
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    cfg.net.offline = True
    written = pipeline.run(cfg)
    assert not (out / "sample_paper.bib").exists()
    assert all(not w.endswith(".bib") for w in written)


def test_auto_append_to_document_prepends_fenced_entry(tmp_path, monkeypatch, hermetic_cache):
    full_json = '{"citable": true, "title": "A Sample Paper", "authors": ["Ada B"]}'
    _mock_inference(monkeypatch, probe_response=full_json)
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    cfg.net.offline = True
    cfg.bibtex.append_to_document = True
    pipeline.run(cfg)
    full = (out / "sample_paper.md").read_text(encoding="utf-8")
    # auto entries get the same prepended, fenced, ----separated injection as on-mode:
    assert full.startswith("```\n% NOTE: Best-effort entry")
    assert "\n```\n\n---\n\n" in full


def test_bibtex_failure_never_fails_the_run(tmp_path, monkeypatch, hermetic_cache):
    # DESIGN §16: even an unexpected exception inside the chain (e.g. a
    # malformed-but-HTTP-200 API body) degrades to a logged skip.
    _mock_inference(monkeypatch, probe_response='{"citable": true, "title": "T"}')
    monkeypatch.setattr(
        pipeline, "generate_bibtex_auto",
        lambda *a, **k: (_ for _ in ()).throw(TypeError("malformed API body")),
    )
    out = tmp_path / "out"
    cfg = _auto_cfg(tmp_path, out)
    cfg.net.offline = True
    written = pipeline.run(cfg)  # must not raise
    assert (out / "sample_paper.md").is_file()
    assert all(not w.endswith(".bib") for w in written)
