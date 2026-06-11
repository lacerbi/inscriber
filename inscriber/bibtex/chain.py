"""BibTeX auto-mode orchestration: citability → ordered source chain (DESIGN §12).

Citability: a URL matching **any of the seven recognized paper repositories**
settles it — the probe never vetoes provenance (a disagreement is logged,
nothing more). The probe governs only provenance-less documents, and it is
abstain-biased: with a default-on feature, a false positive (an unwanted
``.bib``) is worse than a false negative.

Sources, in order (preprint provenance ≠ preprint citation — many preprints are
later published at a venue): Semantic Scholar **by arXiv ID** (exact match;
prefers the published version when one exists) → arXiv export API (``@misc``
availability fallback) → Semantic Scholar title search → local best-effort.
Any failure falls through; this module never raises (DESIGN §16: BibTeX never
fails the run).
"""

from __future__ import annotations

from inscriber.bibtex.arxiv import arxiv_bibtex, arxiv_id_from_url, format_arxiv_misc
from inscriber.bibtex.local import best_effort_bibtex
from inscriber.bibtex.probe import ProbeResult
from inscriber.bibtex.semantic_scholar import (
    _format_entry,
    _mismatch_warning,
    lookup_arxiv,
    search_semantic_scholar,
)
from inscriber.input.domain_handlers import find_handler
from inscriber.logging import get_logger

logger = get_logger()


def citable_provenance(url: str | None) -> bool:
    """Whether ``url`` matches any of the seven recognized paper repositories
    (the domain-handler configs, DESIGN §6) — citable by construction."""
    if not url:
        return False
    return find_handler(url) is not None


def _is_preprint_venue(venue) -> bool:
    return not venue or str(venue).strip().lower().startswith("arxiv")


def _s2_arxiv_entry(paper: dict, arxiv_id: str) -> str:
    """Format an S2 by-ID record: a real publication venue → the published
    ``@article`` shape (shared with the title-search path); no venue (or an
    "arXiv.org"-style one) → the preprint ``@misc`` + ``eprint`` shape. No
    title validation on this path — the ID match is exact."""
    if not _is_preprint_venue(paper.get("venue")):
        bibtex, _ = _format_entry(paper, paper.get("title", "") or "")
        return bibtex
    authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
    year = str(paper["year"]) if paper.get("year") else None
    return format_arxiv_misc(paper.get("title", "") or "", authors, year, arxiv_id)


def generate_bibtex_auto(
    probe: ProbeResult | None,
    *,
    original_url: str | None,
    online_allowed: bool,
    fallback_title: str,
    timeout: float = 30.0,
) -> tuple[str | None, str]:
    """Walk the auto-mode chain; returns ``(bibtex | None, source_label)``.

    ``source_label`` ∈ {``s2-arxiv-id``, ``arxiv-export``, ``s2-title``,
    ``best-effort``} on success; on a skip it is the reason
    (``not-citable`` / ``unknown`` / ``no usable metadata``).
    """
    provenance = citable_provenance(original_url)
    if provenance and probe is not None and not probe.citable:
        logger.info(
            "BibTeX (auto): probe judged the document not citable, but the source "
            "URL is a recognized paper repository; provenance wins"
        )
    if not provenance and (probe is None or not probe.citable):
        return None, "not-citable" if probe is not None else "unknown"

    if online_allowed:
        aid = arxiv_id_from_url(original_url)
        if aid:
            paper = lookup_arxiv(aid, timeout=timeout)
            if paper:
                return _s2_arxiv_entry(paper, aid), "s2-arxiv-id"
            entry = arxiv_bibtex(aid, timeout=timeout)
            if entry:
                return entry, "arxiv-export"
        # Title search; validation compares against the same string used as the
        # query (avoids a spurious % WARNING from a mangled OCR `# Title`).
        title = (probe.title if probe and probe.title else None) or fallback_title
        results = search_semantic_scholar(title, timeout=timeout)
        if results:
            bibtex, matches = _format_entry(results[0], title)
            if not matches:
                bibtex = _mismatch_warning(title, results[0].get("title", "")) + bibtex
            return bibtex, "s2-title"

    entry = best_effort_bibtex(probe)
    if entry:
        return entry, "best-effort"
    return None, "no usable metadata"
