"""Figure injection: replace ``⟦INSCRIBER_FIG:{id}⟧`` with the description block.

Ports the **format** from paper2llm's ``enhanceImageReferences`` (markdown-processor.ts
:298,:329) — a Markdown blockquote with a bold header — NOT the ``![]()``-matching
loop (DeepSeek-OCR grounding emits no inline image syntax; §8.3 splices placeholders
instead). DESIGN §10.2.

Exact header strings matter (downstream regexes depend on them):
* real description → ``> **Image description.**``
* placeholder      → ``> **Image.** [not displayed]``
"""

from __future__ import annotations

import re

from inscriber.models import Figure, fig_placeholder

PLACEHOLDER_RE = re.compile(r"⟦INSCRIBER_FIG:(?P<id>[^⟧]+)⟧")


def ensure_placeholders(markdown: str, figures: list[Figure]) -> str:
    """Guarantee each figure has a placeholder in ``markdown``.

    For grounding output the placeholders are already inline (no-op). For the
    experimental ``pdf-embedded`` path the markdown has none, so each figure's
    placeholder is appended after the page text, in figure order (DESIGN §8.4).
    """
    md = markdown
    for fig in figures:
        token = fig_placeholder(fig.id)
        if token not in md:
            md = (md.rstrip() + "\n\n" + token) if md.strip() else token
    return md

DESC_HEADER = "Image description."
PLACEHOLDER_BLOCK = "> **Image.** [not displayed]\n"
UNAVAILABLE = "[figure description unavailable]"


def format_blockquote(description: str, header: str = DESC_HEADER) -> str:
    """Render description text as a blockquote (DESIGN §10.2).

    Every line is prefixed with ``> ``; the first line carries the bold header;
    blank lines become a bare ``>`` so the blockquote does not break across
    paragraphs/lists in the description.
    """
    lines = description.strip().split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        if line.strip() == "":
            out.append(">")
        elif i == 0:
            out.append(f"> **{header}** {line}")
        else:
            out.append(f"> {line}")
    return "\n".join(out)


def inject_descriptions(
    markdown: str,
    *,
    descriptions: dict[str, str],
    figures: dict[str, Figure],
    mode: str = "describe-only",
) -> str:
    """Replace each figure placeholder with its formatted block (DESIGN §10.2).

    ``mode`` ∈ {``describe-only`` (default), ``describe-and-keep``, ``placeholder``}.
    A figure with no description gets ``[figure description unavailable]`` (DESIGN §16),
    except in ``placeholder`` mode which always emits the not-displayed block. Each
    block ends with a single ``\\n`` to match paper2llm's spacing behavior.
    """

    def repl(m: re.Match) -> str:
        fig_id = m.group("id")
        if mode == "placeholder":
            return PLACEHOLDER_BLOCK
        desc = descriptions.get(fig_id) or UNAVAILABLE
        block = format_blockquote(desc)
        if mode == "describe-and-keep":
            fig = figures.get(fig_id)
            if fig is not None and fig.crop_path:
                # Escape brackets so a caption containing ] (or [) cannot break
                # the image link (crop_path itself is controlled, no escaping).
                alt = (fig.caption or fig_id).replace("[", "\\[").replace("]", "\\]")
                return f"![{alt}]({fig.crop_path})\n\n{block}\n"
        return f"{block}\n"

    return PLACEHOLDER_RE.sub(repl, markdown)
