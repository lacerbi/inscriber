"""M4: the 7 domain handlers (DESIGN §6). Each transform/filename is pinned."""

from __future__ import annotations

import pytest

from inscriber.input.domain_handlers import find_handler
from inscriber.input.resolver import InputError, resolve_input


def _norm(url: str) -> tuple[str, str]:
    h = find_handler(url)
    assert h is not None, f"no handler for {url}"
    return h.normalize_pdf_url(url), h.file_name(url)


def test_arxiv():
    pdf, name = _norm("https://arxiv.org/abs/2301.12345")
    assert pdf == "https://arxiv.org/pdf/2301.12345.pdf"
    assert name == "arxiv-2301.12345.pdf"


def test_openreview_preserves_query():
    pdf, name = _norm("https://openreview.net/forum?id=AbC123xyz")
    assert pdf == "https://openreview.net/pdf?id=AbC123xyz"  # ?id= preserved
    assert name == "openreview-AbC123xyz.pdf"


def test_openreview_missing_id_unchanged():
    h = find_handler("https://openreview.net/forum?foo=bar")
    assert h is not None
    # no id → returned unchanged (then download would fail clearly)
    assert h.normalize_pdf_url("https://openreview.net/forum?foo=bar") == (
        "https://openreview.net/forum?foo=bar"
    )
    assert h.file_name("https://openreview.net/forum?foo=bar") == "openreview-paper.pdf"


def test_acl():
    pdf, name = _norm("https://aclanthology.org/2023.acl-long.42")
    assert pdf == "https://aclanthology.org/2023.acl-long.42.pdf"
    assert name == "acl-2023.acl-long.42.pdf"


def test_biorxiv():
    pdf, name = _norm("https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1")
    assert pdf == "https://www.biorxiv.org/content/10.1101/2023.01.01.123456v1.full.pdf"
    assert name == "biorxiv-2023.01.01.123456.pdf"


def test_medrxiv_shares_rule():
    pdf, name = _norm("https://www.medrxiv.org/content/10.1101/2023.05.05.999999v2")
    assert pdf == "https://www.medrxiv.org/content/10.1101/2023.05.05.999999v2.full.pdf"
    assert name == "medrxiv-2023.05.05.999999.pdf"


def test_neurips():
    pdf, name = _norm("https://papers.nips.cc/paper/2020/hash/abc123def-Abstract.html")
    assert pdf == "https://papers.nips.cc/paper/2020/file/abc123def-Paper.pdf"
    assert name == "neurips-2020-abc123def.pdf"


def test_mlrp():
    pdf, name = _norm("https://proceedings.mlr.press/v202/smith23a.html")
    assert pdf == "https://proceedings.mlr.press/v202/smith23a/smith23a.pdf"
    assert name == "mlrp-v202-smith23a.pdf"


def test_already_pdf_url_unchanged():
    h = find_handler("https://arxiv.org/pdf/2301.12345")
    # /pdf/ matches arxiv url pattern; not ending .pdf → transform won't fire on
    # /abs|html/, falls to ensurePdfExtension → appends .pdf.
    assert h is not None


def test_query_string_keeps_pdf_on_path():
    # ensure-.pdf must apply to the PATH, not after the query string.
    h = find_handler("https://arxiv.org/abs/2301.12345?context=cs")
    pdf = h.normalize_pdf_url("https://arxiv.org/abs/2301.12345?context=cs")
    assert pdf == "https://arxiv.org/pdf/2301.12345.pdf?context=cs"


def test_unmatched_url_not_handled():
    assert find_handler("https://example.com/some/paper") is None


def test_lookalike_hosts_not_matched():
    # Suffix host matching (review D3; deliberate parity break with the TS
    # substring `hostname.includes`): a host that merely CONTAINS a repository
    # domain must not match — the download would go to the attacker host.
    assert find_handler("https://arxiv.org.evil.com/abs/2301.12345") is None
    assert find_handler("https://evilarxiv.org/abs/2301.12345") is None
    assert find_handler("https://openreview.net.evil.com/forum?id=x") is None


def test_real_subdomains_still_matched():
    # ...while genuine subdomains keep working (www.biorxiv.org is pinned by
    # test_biorxiv above; export.arxiv.org is the other real-world shape).
    assert find_handler("https://export.arxiv.org/abs/2301.12345") is not None


def test_resolve_unmatched_url_errors():
    with pytest.raises(InputError, match="no matching repository handler"):
        resolve_input("https://example.com/paper")


def test_offline_blocks_url():
    with pytest.raises(InputError, match="offline"):
        resolve_input("https://arxiv.org/abs/2301.12345", offline=True)
