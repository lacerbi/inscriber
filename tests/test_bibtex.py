"""M4: BibTeX generation (DESIGN §12) + the auto-mode best-effort entry."""

from __future__ import annotations

from pathlib import Path

from inscriber.bibtex import semantic_scholar as ss
from inscriber.bibtex.local import best_effort_bibtex
from inscriber.bibtex.probe import ProbeResult

FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# Citation key (DESIGN §12 / Correction #4)
# --------------------------------------------------------------------------- #


def test_citation_key_basic():
    assert ss.generate_citation_key("Deep Learning for Vision", ["Jane Smith"], "2023") == (
        "smith2023deep"
    )


def test_citation_key_skips_short_and_stopwords():
    # "A" (stopword) and "of" skipped; first >2-char non-skip word is "study".
    assert ss.generate_citation_key("A Study of Things", ["Bob Lee"], "2021") == "lee2021study"


def test_citation_key_no_authors():
    key = ss.generate_citation_key("Networks", [], "2020")
    assert key.startswith("Unknown2020")


# --------------------------------------------------------------------------- #
# Title validation (DESIGN §12 / Correction #3)
# --------------------------------------------------------------------------- #


def test_normalize_title():
    assert ss.normalize_title("The Great Paper!  (v2)") == "the great paper v"


def test_short_titles_require_exact():
    assert ss.titles_match("Deep Net", "Deep Net") is True
    assert ss.titles_match("Deep Net", "Deep Nets") is False  # < 10 normalized chars


def test_long_titles_word_overlap():
    assert ss.titles_match(
        "deep learning for image classification tasks",
        "deep learning for image classification",
    ) is True
    assert ss.titles_match(
        "deep learning for image classification tasks",
        "a totally unrelated paper about gardening",
    ) is False


# --------------------------------------------------------------------------- #
# Mock fallback (canonical fixture, review Fix 6)
# --------------------------------------------------------------------------- #


def test_mock_matches_canonical_fixture():
    expected = (FIXTURES / "bibtex_mock.txt").read_text(encoding="utf-8")
    got = ss.mock_bibtex("Test Paper Title", date="2026-06-09", year="2026")
    assert got == expected


def test_generate_falls_back_to_mock_on_no_results(monkeypatch):
    monkeypatch.setattr(ss, "search_semantic_scholar", lambda *a, **k: [])
    out = ss.generate_bibtex("Some Paper", date="2026-06-09")
    assert "fallback mock citation" in out
    assert "@article{unknownYear," in out


# --------------------------------------------------------------------------- #
# Real-match + mismatch paths (mocked API)
# --------------------------------------------------------------------------- #


def test_generate_formats_matching_entry(monkeypatch):
    paper = {
        "title": "Deep Learning for Image Classification Tasks",
        "authors": [{"name": "Jane Smith"}, {"name": "Bob Lee"}],
        "year": 2023,
        "venue": "Journal of ML",
        "externalIds": {"DOI": "10.1/xyz"},
        "url": "https://example.org/p",
    }
    monkeypatch.setattr(ss, "search_semantic_scholar", lambda *a, **k: [paper])
    out = ss.generate_bibtex("Deep Learning for Image Classification Tasks")
    assert out.startswith("@article{smith2023deep,")
    assert "author={Jane Smith and Bob Lee}" in out
    assert "journal={Journal of ML}" in out
    assert "doi={10.1/xyz}" in out
    assert "WARNING" not in out  # titles match → no warning


def test_generate_prepends_mismatch_warning(monkeypatch):
    paper = {
        "title": "An Entirely Different Paper About Gardening",
        "authors": [{"name": "X Y"}],
        "year": 2022,
    }
    monkeypatch.setattr(ss, "search_semantic_scholar", lambda *a, **k: [paper])
    out = ss.generate_bibtex("Deep Learning for Image Classification Tasks")
    assert out.startswith("% WARNING: The retrieved citation title may not match")
    assert '% Paper title: "Deep Learning for Image Classification Tasks"' in out
    assert "@article{" in out


def test_sanitize_escapes_specials():
    assert ss.sanitize_bibtex_text("A & B_C") == "A \\& B\\_C"


# --------------------------------------------------------------------------- #
# Best-effort @misc entry (auto mode, PLAN-bibtex-auto B2)
# --------------------------------------------------------------------------- #


def test_best_effort_full_metadata_matches_canonical_fixture():
    probe = ProbeResult(
        citable=True,
        title="Attention Is All You Need",
        authors=["Ada Lovelace", "Charles Babbage"],
        year="2017",
        venue="NeurIPS",
    )
    expected = (FIXTURES / "bibtex_best_effort.txt").read_text(encoding="utf-8")
    assert best_effort_bibtex(probe) == expected.rstrip("\n")


def test_best_effort_partial_metadata_omits_fields():
    # Transcription, not recall (decision 5): absent fields are absent — no
    # "Unknown Author" / "Unknown Journal" filler; venue goes in note, never journal.
    out = best_effort_bibtex(ProbeResult(citable=True, title="Sparse Title Only"))
    assert out is not None
    assert out.startswith("% NOTE: Best-effort entry")
    assert "@misc{" in out
    assert "title={Sparse Title Only}" in out
    assert "author=" not in out
    assert "year=" not in out
    assert "journal=" not in out
    assert "Unknown" not in out.split("@misc{", 1)[1].split(",", 1)[1]  # fields only


def test_best_effort_requires_title():
    assert best_effort_bibtex(None) is None
    assert best_effort_bibtex(ProbeResult(citable=True, authors=["A B"])) is None


def test_best_effort_sanitizes_fields():
    probe = ProbeResult(
        citable=True,
        title="Results & Methods — “quoted”",
        authors=["Mary O_Connor"],
        venue="Conf & Expo",
    )
    out = best_effort_bibtex(probe)
    assert out is not None
    assert "title={Results \\& Methods --- ``quoted``}" in out
    assert "author={Mary O\\_Connor}" in out
    assert "note={Conf \\& Expo}" in out
