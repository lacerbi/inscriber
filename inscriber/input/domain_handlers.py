"""Config-driven domain handlers for academic repositories (DESIGN §6).

Ported from paper2llm ``core/domain-handlers/generic-handler.ts``: a single
``GenericDomainHandler`` instantiated per repository from a regex config
(host match + PDF-URL transform + filename rule). The reusable asset is the
**7 regex configs**, ported verbatim and pinned by fixtures. URLs matching no
config are **not handled** (no catch-all).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse, urlunparse

# A transform rule replaces the first match of ``pattern`` in the path with the
# string returned by ``repl`` (which receives the match) — mirrors TS
# ``pathname.replace(pattern, replacement)``.
TransformRepl = Callable[[re.Match], str]
# A filename rule turns a path match (+ parsed URL) into a filename.
FilenameTemplate = Callable[[re.Match, object], str]


@dataclass
class RepositoryConfig:
    domain: str
    host_patterns: list[str]
    url_patterns: list[re.Pattern]
    pdf_transform_rules: list[tuple[re.Pattern, TransformRepl]]
    filename_rules: list[tuple[re.Pattern, FilenameTemplate]]


def _ensure_pdf_extension(url: str) -> str:
    return url if url.lower().endswith(".pdf") else f"{url}.pdf"


def host_matches(host: str, pattern: str) -> bool:
    """Suffix host matching: ``host`` IS ``pattern`` or a subdomain of it.

    Deliberate, documented parity break from the TS source's substring
    ``hostname.includes(...)`` (DESIGN §6): a lookalike host that merely
    *contains* a repository domain (``arxiv.org.evil.com``, ``evilarxiv.org``)
    must not match — it would route the download (and, via provenance, the
    BibTeX chain) to an attacker host. Real subdomains (``www.biorxiv.org``,
    ``export.arxiv.org``) still match.
    """
    return host == pattern or host.endswith("." + pattern)


class GenericDomainHandler:
    def __init__(self, config: RepositoryConfig) -> None:
        self.config = config

    def can_handle(self, url: str) -> bool:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if not any(host_matches(host, p) for p in self.config.host_patterns):
            return False
        return any(p.search(parsed.path) for p in self.config.url_patterns)

    def normalize_pdf_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path
        if path.lower().endswith(".pdf"):
            return url

        # OpenReview special case (host-level branch BEFORE the generic rules,
        # DESIGN §6): require ?id=, set path /pdf, PRESERVE the query.
        if host_matches(parsed.hostname or "", "openreview.net"):
            id_ = parse_qs(parsed.query).get("id", [None])[0]
            if not id_:
                return url  # unchanged if no ID
            return urlunparse(parsed._replace(path="/pdf"))

        for pattern, repl in self.config.pdf_transform_rules:
            m = pattern.search(path)
            if m:
                # Apply ensure-.pdf to the PATH (not the full URL) so a query string
                # like "?id=x" doesn't become "?id=x.pdf".
                new_path = _ensure_pdf_extension(path[: m.start()] + repl(m) + path[m.end():])
                return urlunparse(parsed._replace(path=new_path))
        return urlunparse(parsed._replace(path=_ensure_pdf_extension(path)))

    def file_name(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path
        for pattern, template in self.config.filename_rules:
            m = pattern.search(path)
            if m:
                return template(m, parsed)
        return f"{self.config.domain}-paper.pdf"


# --------------------------------------------------------------------------- #
# The 7 repository configs (ported verbatim from generic-handler.ts)
# --------------------------------------------------------------------------- #


def _arxiv() -> RepositoryConfig:
    return RepositoryConfig(
        domain="arxiv",
        host_patterns=["arxiv.org"],
        url_patterns=[re.compile(r"/(abs|pdf|html)/(\d+\.\d+|[\w-]+/\d+)")],
        pdf_transform_rules=[(re.compile(r"/(abs|html)/"), lambda m: "/pdf/")],
        filename_rules=[
            (re.compile(r"/(abs|pdf|html)/([\w.-]+/?\d+|\d+\.\d+)"),
             lambda m, _u: f"arxiv-{m.group(2)}.pdf"),
        ],
    )


def _openreview() -> RepositoryConfig:
    def _fname(_m: re.Match, parsed) -> str:
        id_ = parse_qs(parsed.query).get("id", [None])[0]
        return f"openreview-{id_}.pdf" if id_ else "openreview-paper.pdf"

    return RepositoryConfig(
        domain="openreview",
        host_patterns=["openreview.net"],
        url_patterns=[re.compile(r"/(forum|pdf|attachment)")],
        # normalize_pdf_url handles OpenReview via its host-level branch.
        pdf_transform_rules=[(re.compile(r"/(forum|attachment)"), lambda m: "/pdf")],
        filename_rules=[(re.compile(r".*"), _fname)],
    )


def _acl() -> RepositoryConfig:
    return RepositoryConfig(
        domain="acl",
        host_patterns=["aclanthology.org"],
        url_patterns=[re.compile(r"/\d{4}\.\w+-\w+\.\d+"), re.compile(r"/[A-Z]\d{2}-\d{4}")],
        pdf_transform_rules=[(re.compile(r"/([^/]+)$"), lambda m: f"/{m.group(1)}.pdf")],
        filename_rules=[
            (re.compile(r"/([^/]+?)(?:\.pdf)?$"), lambda m, _u: f"acl-{m.group(1)}.pdf"),
        ],
    )


def _biorxiv_like(domain: str, host: str) -> RepositoryConfig:
    return RepositoryConfig(
        domain=domain,
        host_patterns=[host],
        url_patterns=[re.compile(r"/content/10\.1101/")],
        pdf_transform_rules=[
            (re.compile(r"/content/(10\.1101/[\d.]+)(v\d+)?(?:\.full\.pdf|\.full|$)"),
             lambda m: f"/content/{m.group(1)}{m.group(2) or ''}.full.pdf"),
        ],
        filename_rules=[
            (re.compile(r"10\.1101/([\d.]+)"), lambda m, _u: f"{domain}-{m.group(1)}.pdf"),
        ],
    )


def _neurips() -> RepositoryConfig:
    return RepositoryConfig(
        domain="neurips",
        host_patterns=["papers.nips.cc", "papers.neurips.cc"],
        url_patterns=[re.compile(r"/paper/"), re.compile(r"/paper_files/paper/")],
        pdf_transform_rules=[
            (re.compile(r"(/paper(?:_files/paper)?/\d{4})/hash/([^/]+)-Abstract\.html"),
             lambda m: f"{m.group(1)}/file/{m.group(2)}-Paper.pdf"),
        ],
        filename_rules=[
            (re.compile(r"/paper(?:_files/paper)?/(\d{4})/(?:hash|file)/([^/-]+)"),
             lambda m, _u: f"neurips-{m.group(1)}-{m.group(2)}.pdf"),
            (re.compile(r"/(?:hash|file)/([^/-]+)"), lambda m, _u: f"neurips-{m.group(1)}.pdf"),
        ],
    )


def _mlrp() -> RepositoryConfig:
    return RepositoryConfig(
        domain="mlrp",
        host_patterns=["proceedings.mlr.press"],
        url_patterns=[re.compile(r"/v\d+/[a-z0-9]+")],
        pdf_transform_rules=[
            (re.compile(r"/(v\d+)/([a-z0-9]+)(?:\.html)?$"),
             lambda m: f"/{m.group(1)}/{m.group(2)}/{m.group(2)}.pdf"),
        ],
        filename_rules=[
            (re.compile(r"/v(\d+)/([a-z0-9]+)"), lambda m, _u: f"mlrp-v{m.group(1)}-{m.group(2)}.pdf"),
        ],
    )


def all_handlers() -> list[GenericDomainHandler]:
    return [
        GenericDomainHandler(_arxiv()),
        GenericDomainHandler(_openreview()),
        GenericDomainHandler(_acl()),
        GenericDomainHandler(_biorxiv_like("biorxiv", "biorxiv.org")),
        GenericDomainHandler(_biorxiv_like("medrxiv", "medrxiv.org")),
        GenericDomainHandler(_neurips()),
        GenericDomainHandler(_mlrp()),
    ]


def find_handler(url: str) -> GenericDomainHandler | None:
    """First handler whose ``can_handle`` matches (DESIGN §6; no catch-all)."""
    for handler in all_handlers():
        if handler.can_handle(url):
            return handler
    return None
