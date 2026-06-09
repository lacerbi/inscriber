"""BibTeX generation via Semantic Scholar (DESIGN §12) — optional & online.

Ported from paper2llm ``core/utils/bibtex-generator.ts`` + ``content-utils.ts``:
title→entry lookup, citation key, title validation, and the mock fallback. The
inscriber **standardizes on the single 4-line mismatch warning** (DESIGN §12) and
**adds a clean 429 / network degrade path** (the source had none).
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import httpx

from inscriber.logging import get_logger

logger = get_logger()

API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,venue,year,abstract,externalIds,url"
USER_AGENT = "inscriber/0.1 (+https://github.com/lacerbi/inscriber)"

_SKIP_WORDS = {"a", "an", "the", "on", "in", "of", "for", "and", "or"}


def _current_year() -> str:
    return str(datetime.now(timezone.utc).year)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def sanitize_bibtex_text(text: str) -> str:
    """Escape BibTeX special characters (port of ``sanitizeBibTeXText``)."""
    if not text:
        return ""
    text = re.sub(r"[&%$#_{}~^\\\s]", lambda m: m.group() if m.group() == " " else "\\" + m.group(), text)
    text = text.replace("“", "``").replace("”", "``")  # curly double quotes
    text = text.replace("‘", "''").replace("’", "''")  # curly single quotes
    text = text.replace("—", "---").replace("–", "--")  # em / en dash
    return text


def generate_citation_key(title: str, authors: list[str], year: str | None) -> str:
    """``{firstAuthorLastName}{year}{firstSubstantiveTitleWord}`` (DESIGN §12)."""
    author_part = "Unknown"
    if authors:
        author_part = authors[0].split(" ")[-1].lower()

    title_part = ""
    title_words = title.split(" ")
    for word in title_words:
        clean = re.sub(r"[^a-z0-9]", "", word.lower())
        if len(clean) > 2 and clean not in _SKIP_WORDS:
            title_part = clean
            break
    if not title_part and title_words:
        title_part = re.sub(r"[^a-z0-9]", "", title_words[0].lower())

    year_part = year or _current_year()
    return f"{author_part}{year_part}{title_part}"


def normalize_title(title: str) -> str:
    """lower → strip all but ``[a-z\\s]`` → collapse whitespace → trim (DESIGN §12).

    Whitespace is preserved through the strip and collapsed afterward (matching the
    TS ``normalizeTitleForComparison``), so an embedded tab/newline keeps words apart
    rather than fusing them.
    """
    if not title:
        return ""
    s = re.sub(r"[^a-z\s]", "", title.lower())
    return re.sub(r"\s+", " ", s).strip()


def titles_match(original: str, bibtex_title: str) -> bool:
    """Validate a retrieved title vs the paper title (DESIGN §12)."""
    no, nb = normalize_title(original), normalize_title(bibtex_title)
    if len(no) < 10 or len(nb) < 10:
        return no == nb  # short titles require exact normalized match
    orig_words = no.split(" ")
    bib_words = set(nb.split(" "))
    common = sum(1 for w in orig_words if w in bib_words)
    similarity = common / max(len(orig_words), len(bib_words))
    return similarity > 0.75


def search_semantic_scholar(title: str, *, limit: int = 3, timeout: float = 30.0) -> list[dict]:
    """Query Semantic Scholar; return ``data`` (or ``[]`` on any error/429)."""
    try:
        resp = httpx.get(
            API_URL,
            params={"query": title, "limit": limit, "fields": FIELDS},
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as e:
        logger.warning("Semantic Scholar request failed: %s; using fallback citation", e)
        return []
    if resp.status_code == 429:
        logger.warning("Semantic Scholar rate-limited (HTTP 429); using fallback citation")
        return []
    if resp.status_code != 200:
        logger.warning("Semantic Scholar returned HTTP %d; using fallback citation", resp.status_code)
        return []
    try:
        return resp.json().get("data", []) or []
    except ValueError:
        logger.warning("Semantic Scholar returned non-JSON; using fallback citation")
        return []


def _format_entry(paper: dict, original_title: str) -> tuple[str, bool]:
    """Format a Semantic Scholar paper as BibTeX; return ``(bibtex, title_matches)``."""
    authors = [a.get("name", "") for a in paper.get("authors", [])]
    year = str(paper["year"]) if paper.get("year") else None
    bib_title = paper.get("title", "") or ""
    key = generate_citation_key(bib_title, authors, year)

    fields = [f"  title={{{sanitize_bibtex_text(bib_title)}}}"]
    if authors:
        fields.append("  author={" + " and ".join(sanitize_bibtex_text(a) for a in authors) + "}")
    else:
        fields.append("  author={Unknown}")
    if year:
        fields.append(f"  year={{{year}}}")
    if paper.get("venue"):
        fields.append(f"  journal={{{sanitize_bibtex_text(paper['venue'])}}}")
    doi = (paper.get("externalIds") or {}).get("DOI")
    if doi:
        fields.append(f"  doi={{{doi}}}")
    if paper.get("url"):
        fields.append(f"  url={{{paper['url']}}}")

    bibtex = f"@article{{{key},\n" + ",\n".join(fields) + "\n}"
    return bibtex, titles_match(original_title, bib_title)


def _mismatch_warning(original: str, bibtex_title: str) -> str:
    # The standardized 4-line form (DESIGN §12); note the trailing "% " line.
    return (
        "% WARNING: The retrieved citation title may not match the paper title.\n"
        f'% Paper title: "{original}"\n'
        f'% Citation title: "{bibtex_title}"\n'
        "% \n"
    )


def mock_bibtex(title: str, *, date: str | None = None, year: str | None = None) -> str:
    """The canonical fallback mock entry (DESIGN §12; review Fix 6)."""
    return (
        "% WARNING: This is a fallback mock citation.\n"
        "% BibTeX generation failed to find this paper in academic databases.\n"
        "% Please replace with the correct citation if available.\n"
        "%\n"
        f"% Generated: {date or _today()}\n"
        "@article{unknownYear,\n"
        f"  title={{{title}}},\n"
        "  author={Unknown Author},\n"
        "  journal={Unknown Journal},\n"
        f"  year={{{year or _current_year()}}},\n"
        "  note={This is an automatically generated fallback citation}\n"
        "}"
    )


def generate_bibtex(title: str, *, timeout: float = 30.0, date: str | None = None) -> str:
    """Generate a BibTeX string for ``title`` (DESIGN §12).

    On a confident match → the formatted entry. On a title mismatch → the entry
    prefixed with the 4-line warning. On no result / API error / 429 → the mock
    fallback (never raises — BibTeX never fails the whole run).
    """
    results = search_semantic_scholar(title, timeout=timeout)
    if results:
        bibtex, matches = _format_entry(results[0], title)
        if not matches:
            bibtex = _mismatch_warning(title, results[0].get("title", "")) + bibtex
        return bibtex
    return mock_bibtex(title, date=date)
