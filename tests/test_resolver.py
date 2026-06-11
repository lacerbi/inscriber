"""URL download path (DESIGN §6): streamed, size-capped, magic-checked early.

HTTP is mocked at the httpx transport boundary (the ``transport`` test seam,
same pattern as ``test_setup.py``) — no test touches the network.
"""

from __future__ import annotations

import httpx
import pytest

from inscriber.input import resolver
from inscriber.input.resolver import InputError, _download_pdf

URL = "https://arxiv.org/pdf/2301.12345.pdf"


def _transport_returning(response: httpx.Response) -> httpx.MockTransport:
    return httpx.MockTransport(lambda request: response)


def test_download_ok():
    body = b"%PDF-1.7 fake body"
    t = _transport_returning(httpx.Response(200, content=body))
    assert _download_pdf(URL, transport=t) == body


def test_download_http_error_status():
    t = _transport_returning(httpx.Response(404))
    with pytest.raises(InputError, match="returned HTTP 404"):
        _download_pdf(URL, transport=t)


def test_download_network_error():
    def handler(request):
        raise httpx.ConnectError("boom", request=request)

    with pytest.raises(InputError, match="failed to download"):
        _download_pdf(URL, transport=httpx.MockTransport(handler))


def test_download_non_pdf_aborts_on_first_chunk():
    # A non-PDF body must abort as soon as the magic fails — before the rest
    # of the (possibly huge) stream is consumed.
    consumed = []

    def chunks():
        consumed.append(1)
        yield b"<html>not a pdf"
        for _ in range(1000):
            consumed.append(1)
            yield b"x" * 1024

    t = _transport_returning(httpx.Response(200, content=chunks()))
    with pytest.raises(InputError, match="not a PDF"):
        _download_pdf(URL, transport=t)
    assert len(consumed) <= 2  # first chunk (+ at most one read-ahead), not the tail


def test_download_body_shorter_than_magic_is_not_pdf():
    t = _transport_returning(httpx.Response(200, content=b"%P"))
    with pytest.raises(InputError, match="not a PDF"):
        _download_pdf(URL, transport=t)


def test_download_declared_oversize_aborts_before_body(monkeypatch):
    monkeypatch.setattr(resolver, "MAX_DOWNLOAD_BYTES", 1024)
    t = _transport_returning(
        httpx.Response(200, headers={"Content-Length": "2048"}, content=b"%PDF" + b"x" * 2044)
    )
    with pytest.raises(InputError, match="MiB limit"):
        _download_pdf(URL, transport=t)


def test_download_streamed_oversize_aborts_mid_body(monkeypatch):
    # No (honest) Content-Length: the cap must still trip while streaming.
    monkeypatch.setattr(resolver, "MAX_DOWNLOAD_BYTES", 1024)

    def chunks():
        yield b"%PDF"
        while True:  # endless body — the cap is the only way out
            yield b"x" * 512

    t = _transport_returning(httpx.Response(200, content=chunks()))
    with pytest.raises(InputError, match="MiB limit"):
        _download_pdf(URL, transport=t)


def test_http_input_upgraded_to_https(monkeypatch):
    # Review D2: every supported repository serves HTTPS — a plain http:// input
    # is upgraded before any request is made (and recorded as the provenance URL).
    seen = {}

    def fake_download(pdf_url, **kwargs):
        seen["url"] = pdf_url
        return b"%PDF-1.7 fake"

    monkeypatch.setattr(resolver, "_download_pdf", fake_download)
    resolved = resolver.resolve_url("http://arxiv.org/abs/2301.12345")
    assert seen["url"] == "https://arxiv.org/pdf/2301.12345.pdf"
    assert resolved.original_url == "https://arxiv.org/abs/2301.12345"


def test_plain_http_download_warns(caplog):
    # Review D2: a plaintext fetch that slips past the upgrade (downgrade
    # redirect) must warn loudly — the body is MITM-tamperable.
    import logging

    logging.getLogger("inscriber").propagate = True  # let caplog see records
    body = b"%PDF-1.7 fake body"
    t = _transport_returning(httpx.Response(200, content=body))
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        assert _download_pdf("http://insecure.example/p.pdf", transport=t) == body
    assert "plain HTTP" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        _download_pdf(URL, transport=_transport_returning(httpx.Response(200, content=body)))
    assert "plain HTTP" not in caplog.text  # https stays quiet
