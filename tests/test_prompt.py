"""M2: figure-description prompt, context builder, tag extraction (DESIGN §9.3–9.5)."""

from __future__ import annotations

from inscriber.postprocess.prompt import (
    IMAGE_PROMPT_TEMPLATE,
    build_page_context,
    extract_description_from_tags,
    format_image_prompt,
)


def test_prompt_without_context_removes_placeholder():
    p = format_image_prompt(None)
    assert "{contextText}" not in p
    assert "# Context" not in p
    assert "<img_desc>" in p  # the format instruction remains


def test_prompt_with_context_injects_block():
    p = format_image_prompt("Page about training pipelines.")
    assert "# Context" in p
    assert "<context>\nPage about training pipelines.\n</context>" in p
    assert "{contextText}" not in p


def test_template_is_verbatim_marker():
    # Spot-check a couple of exact lines to guard against accidental drift.
    assert IMAGE_PROMPT_TEMPLATE.startswith("# Task")
    assert "wrap your entire description inside <img_desc> and </img_desc>" in IMAGE_PROMPT_TEMPLATE


def test_build_page_context_uses_real_page_number():
    ctx = build_page_context(7, "Some page text.", context_chars=2000)
    assert ctx.startswith("This image appears on page 7.")
    assert "Some page text." in ctx
    # the paper2llm .split('-')[0] bug must NOT appear:
    assert "page img" not in ctx


def test_build_page_context_truncates():
    long = "x" * 5000
    ctx = build_page_context(1, long, context_chars=2000)
    body = ctx.split("\n\n", 1)[1]
    assert len(body) == 2000  # 1997 chars + "..."
    assert body.endswith("...")


def test_extract_with_both_tags():
    assert extract_description_from_tags("<img_desc>Hello world</img_desc>") == "Hello world"


def test_extract_strips_whitespace():
    assert extract_description_from_tags("  <img_desc>  Hi  </img_desc>  ") == "Hi"


def test_extract_open_only_takes_rest():
    assert extract_description_from_tags("noise <img_desc>truncated desc") == "truncated desc"


def test_extract_no_opening_returns_whole(caplog):
    # DESIGN §9.4 divergence (review Fix 5): whole trimmed response + warning.
    out = extract_description_from_tags("  just prose, no tags  ")
    assert out == "just prose, no tags"


def test_extract_empty():
    assert extract_description_from_tags("") == ""
