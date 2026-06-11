"""Input resolution (DESIGN §6).

v1 milestone split (review Fix 1): **local-path validation lands in M1a** (here)
so M1a/M1b/M2 don't grow ad-hoc path glue. **URL download + the 7 domain handlers
arrive in M4** and extend this module (:func:`resolve_input` will branch on URL).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse, urlunparse

import httpx

from inscriber.errors import InscriberError
from inscriber.input.domain_handlers import find_handler
from inscriber.logging import get_logger
from inscriber.models import ResolvedInput

logger = get_logger()

PDF_MAGIC = b"%PDF"
USER_AGENT = "inscriber/0.1 (+https://github.com/lacerbi/inscriber)"
DOWNLOAD_TIMEOUT = 60.0
# Hard cap on a downloaded "PDF" (DESIGN §6). The body is buffered in memory
# (PyMuPDF consumes bytes), so an unbounded body on a hostile or misconfigured
# URL must not be able to exhaust RAM. Generously above any real academic PDF.
MAX_DOWNLOAD_BYTES = 512 * 1024 * 1024


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


def _too_large_error(pdf_url: str, size_note: str) -> InputError:
    limit_mib = MAX_DOWNLOAD_BYTES // (1024 * 1024)
    return InputError(
        f"download of {pdf_url} {size_note} the {limit_mib} MiB limit; "
        "if it is a real PDF, download it manually and pass the local path"
    )


def _download_pdf(pdf_url: str, *, transport: httpx.BaseTransport | None = None) -> bytes:
    """Download with redirects, a timeout, and a descriptive User-Agent (DESIGN §6).

    The body is **streamed** with a hard size cap (:data:`MAX_DOWNLOAD_BYTES`)
    and the ``%PDF`` magic is checked on the first bytes, so a non-PDF or
    oversized body aborts early instead of being buffered whole into memory.
    ``transport`` is the test seam (cf. ``setup.py``).
    """
    buf = bytearray()
    magic_checked = False
    try:
        with httpx.Client(
            follow_redirects=True,
            timeout=DOWNLOAD_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
            transport=transport,
        ) as client:
            with client.stream("GET", pdf_url) as resp:
                if resp.url.scheme != "https":
                    # Input URLs are upgraded in resolve_url, so plaintext here
                    # means a downgrade redirect — the body is MITM-tamperable.
                    logger.warning(
                        "download is being served over plain HTTP (%s) — the "
                        "content could be tampered with in transit", resp.url,
                    )
                if resp.status_code != 200:
                    raise InputError(
                        f"download of {pdf_url} returned HTTP {resp.status_code}"
                    )
                declared = resp.headers.get("Content-Length", "")
                if declared.isdigit() and int(declared) > MAX_DOWNLOAD_BYTES:
                    raise _too_large_error(pdf_url, f"declares {declared} bytes — over")
                for chunk in resp.iter_bytes():
                    buf += chunk
                    if not magic_checked and len(buf) >= len(PDF_MAGIC):
                        if not buf.startswith(PDF_MAGIC):
                            raise InputError(
                                f"downloaded content is not a PDF (no %PDF header): {pdf_url}"
                            )
                        magic_checked = True
                    if len(buf) > MAX_DOWNLOAD_BYTES:
                        raise _too_large_error(pdf_url, "exceeded")
    except httpx.HTTPError as e:
        raise InputError(f"failed to download {pdf_url}: {e}") from e
    if not buf.startswith(PDF_MAGIC):  # covers bodies shorter than the magic
        raise InputError(
            f"downloaded content is not a PDF (no %PDF header): {pdf_url}"
        )
    return bytes(buf)


def resolve_url(url: str) -> ResolvedInput:
    """Resolve an http(s) URL via the domain handlers (DESIGN §6).

    A plain ``http://`` input is upgraded to ``https://`` before any request —
    every supported repository serves HTTPS, and a plaintext fetch would let a
    MITM feed attacker bytes to PyMuPDF. (Unknown hosts never get fetched at
    all — there is no catch-all handler.)
    """
    handler = find_handler(url)
    if handler is None:
        raise InputError(
            "unsupported URL (no matching repository handler): "
            f"{url}\nSupported: arXiv, OpenReview, ACL, bioRxiv, medRxiv, NeurIPS, MLR Press. "
            "Download the PDF and pass a local path instead."
        )
    parsed = urlparse(url)
    if parsed.scheme == "http":
        url = urlunparse(parsed._replace(scheme="https"))
        logger.info("upgraded plain http:// input URL to https:// (%s)", url)
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
