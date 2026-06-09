"""Input resolution (DESIGN §6).

v1 milestone split (review Fix 1): **local-path validation lands in M1a** (here)
so M1a/M1b/M2 don't grow ad-hoc path glue. **URL download + the 7 domain handlers
arrive in M4** and extend this module (:func:`resolve_input` will branch on URL).
"""

from __future__ import annotations

from pathlib import Path

import httpx

from inscriber.errors import InscriberError
from inscriber.input.domain_handlers import find_handler
from inscriber.logging import get_logger
from inscriber.models import ResolvedInput

logger = get_logger()

PDF_MAGIC = b"%PDF"
USER_AGENT = "inscriber/0.1 (+https://github.com/lacerbi/inscriber)"
DOWNLOAD_TIMEOUT = 60.0


class InputError(InscriberError):
    """Raised on a missing/unreadable/non-PDF input."""


def is_url(value: str) -> bool:
    v = value.strip().lower()
    return v.startswith("http://") or v.startswith("https://")


def resolve_local_pdf(path_str: str) -> ResolvedInput:
    """Validate a local PDF path (exists / readable / ``%PDF`` magic) → bytes."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise InputError(f"input PDF not found: {path}")
    if not path.is_file():
        raise InputError(f"input path is not a file: {path}")
    try:
        data = path.read_bytes()
    except OSError as e:
        raise InputError(f"could not read PDF {path}: {e}") from e
    if not data.startswith(PDF_MAGIC):
        raise InputError(f"file does not look like a PDF (no %PDF header): {path}")
    return ResolvedInput(
        pdf_bytes=data,
        source="file",
        original_url=None,
        suggested_name=path.stem,
    )


def _download_pdf(pdf_url: str) -> bytes:
    """Download with redirects, a timeout, and a descriptive User-Agent (DESIGN §6)."""
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = client.get(pdf_url)
    except httpx.HTTPError as e:
        raise InputError(f"failed to download {pdf_url}: {e}") from e
    if resp.status_code != 200:
        raise InputError(f"download of {pdf_url} returned HTTP {resp.status_code}")
    data = resp.content
    if not data.startswith(PDF_MAGIC):
        raise InputError(
            f"downloaded content is not a PDF (no %PDF header): {pdf_url}"
        )
    return data


def resolve_url(url: str) -> ResolvedInput:
    """Resolve an http(s) URL via the domain handlers (DESIGN §6)."""
    handler = find_handler(url)
    if handler is None:
        raise InputError(
            "unsupported URL (no matching repository handler): "
            f"{url}\nSupported: arXiv, OpenReview, ACL, bioRxiv, medRxiv, NeurIPS, MLR Press. "
            "Download the PDF and pass a local path instead."
        )
    pdf_url = handler.normalize_pdf_url(url)
    suggested = Path(handler.file_name(url)).stem
    logger.info("resolving %s → %s", url, pdf_url)
    data = _download_pdf(pdf_url)
    return ResolvedInput(
        pdf_bytes=data, source="url", original_url=url, suggested_name=suggested
    )


def resolve_input(value: str, *, offline: bool = False) -> ResolvedInput:
    """Resolve a PDF path or URL to local bytes (DESIGN §6).

    A URL requires the network; ``--offline`` hard-disables it (errors early).
    """
    if is_url(value):
        if offline:
            raise InputError("--offline is set: URL input is disabled (use a local PDF path)")
        return resolve_url(value)
    return resolve_local_pdf(value)
