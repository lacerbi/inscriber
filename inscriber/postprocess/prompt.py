"""Figure-description prompt, context builder, and tag extraction (DESIGN §9.3–9.5).

Ported from paper2llm:
* prompt template + ``{contextText}`` formatter — ``core/templates/image-prompt-template.ts``
  (verbatim, model-agnostic, well-tuned).
* whole-page context — ``core/markdown-processor.ts`` ``extractImageContext`` (with the
  page-number bug fixed: use the real page N, not ``image.id.split("-")[0]``).
* ``<img_desc>`` extraction — ``extractDescriptionFromTags`` (with the §9.4 divergence,
  review Fix 5: a missing opening tag yields the whole trimmed response + a warning,
  rather than the source's ``null``/throw).
"""

from __future__ import annotations

from inscriber.logging import get_logger

logger = get_logger()

# Ported verbatim from core/templates/image-prompt-template.ts:12-33.
IMAGE_PROMPT_TEMPLATE = """# Task

Please describe the visual content of this image in detail, focusing on all visible elements, text, and relevant information.

- Focus primarily on visual elements directly observable in the image: shapes, colors, objects, arrangements, and any visible text. When appropriate, include reasonable interpretation of what these elements represent based on their visual context.
- For academic or technical visuals: Identify the specific type (bar chart, line graph, flow diagram, etc.). Describe axes, labels, data points, and visual patterns exactly as they appear in the image.
- For any text visible in the image: Provide an accurate transcription, maintaining the original layout where meaningful.
- For images with multiple panels: Describe each panel separately based on its visual appearance. Note any panel labels if present. If the composition is unusual or the panels interact in a non-standard way, explain their relationship.
{contextText}

# Format

- Begin with a concise overview sentence identifying the type of image (e.g., "A line graph showing...", "A diagram illustrating...", "A photograph of...").
- Then provide specific details in a well-structured format. Use multiple paragraphs if necessary to organize different aspects of complex images.
- For complex visuals, you may use bullet points or numbered lists to clearly separate distinct elements.
- Adjust the length of your description based on the complexity of the image - simple images may need only a paragraph, while complex diagrams might require more detailed explanations.

IMPORTANT: You must wrap your entire description inside <img_desc> and </img_desc> XML tags like this:

<img_desc>Your detailed description goes here.</img_desc>

Do not include anything else outside these tags."""

# Context block injected for {contextText} (verbatim from image-prompt-template.ts:48).
_CONTEXT_BLOCK = (
    "# Context\n\nContext for reference:\n\n<context>\n{context}\n</context>\n\n"
    "Use this to correctly identify technical terms and provide reasonable "
    "interpretations of what you can see in the image.\n"
    "Your image description should still focus primarily on the visual aspects of the "
    "figure and not be a mere repetition of the image caption or provided context.\n"
)

_OPEN_TAG = "<img_desc>"
_CLOSE_TAG = "</img_desc>"


def format_image_prompt(context_text: str | None) -> str:
    """Format the prompt with optional context (paper2llm ``formatImagePrompt``)."""
    if not context_text:
        return IMAGE_PROMPT_TEMPLATE.replace("{contextText}", "")
    formatted = _CONTEXT_BLOCK.format(context=context_text)
    return IMAGE_PROMPT_TEMPLATE.replace("{contextText}", formatted)


def build_page_context(page_number: int, page_text: str, context_chars: int = 2000) -> str:
    """Whole-page context for a figure (DESIGN §9.5).

    Reproduces paper2llm's behavior: the entire page's text, prefixed with a short
    preamble, truncated to ``context_chars`` (paper2llm uses 2000;
    ``substring(0, 1997) + "..."`` only when the page exceeds the cap). The page
    number is the **real** page N (paper2llm's ``.split("-")[0]`` bug is fixed here).
    """
    summary = (
        f"This image appears on page {page_number}. "
        "The surrounding page content follows."
    )
    text = page_text
    if context_chars > 0 and len(text) > context_chars:
        text = text[: max(0, context_chars - 3)] + "..."
    return f"{summary}\n\n{text}"


def extract_description_from_tags(response: str) -> str:
    """Extract the description from ``<img_desc>…</img_desc>`` (DESIGN §9.4).

    * opening + closing tag → the substring between them;
    * opening but no closing (truncated output) → everything after the opening tag
      (a faithful port of ``extractDescriptionFromTags``);
    * **no opening tag** → the whole trimmed response, with a warning (inscriber
      divergence per DESIGN §9.4 / review Fix 5 — the source returns ``null``).
    """
    if not response:
        return ""
    trimmed = response.strip()
    open_idx = trimmed.find(_OPEN_TAG)
    if open_idx == -1:
        logger.warning(
            "VLM response missing <img_desc> opening tag; using whole response"
        )
        return trimmed
    content_start = open_idx + len(_OPEN_TAG)
    close_idx = trimmed.find(_CLOSE_TAG, content_start)
    content = (
        trimmed[content_start:close_idx]
        if close_idx != -1
        else trimmed[content_start:]
    )
    return content.strip()
