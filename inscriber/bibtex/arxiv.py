"""arXiv sources for BibTeX auto mode (DESIGN §12).

``arxiv_id_from_url`` extracts the arXiv ID from a source URL (provenance).
``arxiv_bibtex`` is the export-API **availability fallback**: Semantic Scholar
(which knows the *published* version of a preprint) is consulted first; the
export API is authoritative for identification but can never know about later
venue publication. ``format_arxiv_misc`` is the standard arXiv ``@misc`` +
``eprint`` shape, shared with the chain's S2-preprint path.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx

from inscriber.bibtex.semantic_scholar import (
    USER_AGENT,
    generate_citation_key,
    sanitize_bibtex_text,
)
from inscriber.logging import get_logger

logger = get_logger()

API_URL = "https://export.arxiv.org/api/query"
_ATOM = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

# The domain handler's FILENAME-RULE pattern shape (input/domain_handlers.py
# ``_arxiv()``): it preserves ``v2``-style version suffixes and old-style
# ``cs.AI/0301001`` IDs. (The handler's ``url_patterns`` detection regex is NOT
# reusable here — its ``\d+\.\d+`` stops before the version suffix.)
_ARXIV_ID_RE = re.compile(r"/(?:abs|pdf|html)/([\w.-]+/?\d+|\d+\.\d+)")


def arxiv_id_from_url(url: str | None) -> str | None:
    """The arXiv ID (version suffix preserved) from an arxiv.org URL, else None."""
    if not url:
        return None
    parsed = urlparse(url)
    if "arxiv.org" not in (parsed.hostname or ""):
        return None
    m = _ARXIV_ID_RE.search(parsed.path)
    return m.group(1) if m else None


def format_arxiv_misc(
    title: str,
    authors: list[str],
    year: str | None,
    arxiv_id: str,
    *,
    primary_class: str | None = None,
) -> str:
    """The standard arXiv ``@misc`` + ``eprint`` shape (humble entry types)."""
    key = generate_citation_key(title, authors, year)
    fields = [f"  title={{{sanitize_bibtex_text(title)}}}"]
    if authors:
        fields.append(
            "  author={" + " and ".join(sanitize_bibtex_text(a) for a in authors) + "}"
        )
    if year:
        fields.append(f"  year={{{year}}}")
    fields.append(f"  eprint={{{arxiv_id}}}")
    fields.append("  archivePrefix={arXiv}")
    if primary_class:
        fields.append(f"  primaryClass={{{primary_class}}}")
    fields.append(f"  url={{https://arxiv.org/abs/{arxiv_id}}}")
    return f"@misc{{{key},\n" + ",\n".join(fields) + "\n}"


def arxiv_bibtex(arxiv_id: str, *, timeout: float = 30.0) -> str | None:
    """Fetch + format the arXiv entry by ID via the export API (Atom, stdlib
    ``xml.etree``); ``None`` on any HTTP/parse failure — log + fall through,
    mirroring ``search_semantic_scholar``'s degrade style."""
    try:
        resp = httpx.get(
            API_URL,
            params={"id_list": arxiv_id},
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
        )
    except httpx.HTTPError as e:
        logger.warning("arXiv API request failed: %s", e)
        return None
    if resp.status_code != 200:
        logger.warning("arXiv API returned HTTP %d", resp.status_code)
        return None
    try:
        root = ElementTree.fromstring(resp.text)
    except ElementTree.ParseError as e:
        logger.warning("arXiv API returned unparseable XML: %s", e)
        return None
    entry = root.find("atom:entry", _ATOM)
    if entry is None:
        return None
    entry_id = entry.findtext("atom:id", default="", namespaces=_ATOM)
    if "api/errors" in entry_id:  # arXiv reports a bad ID as an <entry> with an error id
        logger.warning("arXiv API has no record for id %s", arxiv_id)
        return None
    raw_title = entry.findtext("atom:title", default="", namespaces=_ATOM)
    title = re.sub(r"\s+", " ", raw_title).strip()
    if not title:
        return None
    authors = []
    for a in entry.findall("atom:author", _ATOM):
        name = (a.findtext("atom:name", default="", namespaces=_ATOM) or "").strip()
        if name:
            authors.append(name)
    published = entry.findtext("atom:published", default="", namespaces=_ATOM)
    year = published[:4] if published[:4].isdigit() else None
    primary = entry.find("arxiv:primary_category", _ATOM)
    primary_class = primary.get("term") if primary is not None else None
    return format_arxiv_misc(title, authors, year, arxiv_id, primary_class=primary_class)
