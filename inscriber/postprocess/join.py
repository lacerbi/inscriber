"""Rejoin split files into a full document (DESIGN §11 — the allparts form).

``inscriber join`` reads ``{base}_main.md`` (+ ``_appendix.md`` /
``_backmatter.md`` when present) — typically after the user hand-edited them —
and regenerates ``{base}_full.md``. The standalone split files already carry
the allparts framing (``# {title} - Appendix`` + ``---``, §11), so joining is:
strip each file's per-file extras (the transcription notice footer; main's
optional prepended BibTeX block), concatenate in the deliberate
**main → appendix → backmatter** order, then re-apply the BibTeX block and one
regenerated notice to the joined document.

Note the joined document is the *allparts* form: appendix precedes backmatter
under derived headings, which differs from ``run``'s original ``{base}_full.md``
(source order, original headings). Rejoining from splits cannot recover the
source order — this is the paper2llm-faithful combined shape, not a bug.
"""

from __future__ import annotations

import re
from pathlib import Path

from inscriber.errors import InscriberError
from inscriber.postprocess.notice import append_transcription_notice

MAIN_SUFFIX = "_main.md"

# The exact shapes the pipeline writes (pipeline._write_outputs):
#   notice  = rstrip() + "\n\n---\n\n" + "*Transcribed with …*" + "\n"
#   bibtex  = "```\n{entry}\n```\n\n---\n\n" prepended to full/main
# Tolerant of hand-edited trailing whitespace, strict about the notice text.
_NOTICE_RE = re.compile(
    r"\n\s*---\s*\n\s*\*Transcribed with (OCR|OCR and VLMs);[^\n]*"
    r"may contain mistakes\.\*\s*\Z"
)
_BIBTEX_BLOCK_RE = re.compile(r"\A```\n[\s\S]*?\n```\n\n---\n\n")


class JoinError(InscriberError):
    """Raised when the split files for ``join`` can't be located or read."""


def resolve_join_input(path: Path) -> Path:
    """Resolve the ``join`` BASE argument to the ``{base}_main.md`` file.

    Accepts the main split file itself, a base path (``out/paper`` →
    ``out/paper_main.md``), or a directory containing exactly one ``*_main.md``.
    """
    if path.is_dir():
        candidates = sorted(path.glob(f"*{MAIN_SUFFIX}"))
        if not candidates:
            raise JoinError(f"no *{MAIN_SUFFIX} file found in directory: {path}")
        if len(candidates) > 1:
            names = ", ".join(c.name for c in candidates)
            raise JoinError(
                f"multiple *{MAIN_SUFFIX} files in {path} ({names}); "
                f"pass the base path or the _main.md file itself"
            )
        return candidates[0]
    if path.name.endswith(MAIN_SUFFIX):
        if not path.is_file():
            raise JoinError(f"main split file not found: {path}")
        return path
    candidate = path.with_name(path.name + MAIN_SUFFIX)
    if candidate.is_file():
        return candidate
    raise JoinError(f"main split file not found: {candidate}")


def _read_split(path: Path) -> tuple[str, bool, bool]:
    """Read one split file → ``(body, had_notice, notice_said_vlm)``.

    Normalizes CRLF (hand-edits on Windows) and strips the per-file
    transcription notice so the join doesn't accumulate one footer per part.
    """
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = _NOTICE_RE.search(text)
    if not m:
        return text.strip(), False, False
    return text[: m.start()].strip(), True, m.group(1) == "OCR and VLMs"


def join_split_files(main_path: Path) -> str:
    """Join ``{base}_main/_appendix/_backmatter.md`` into the full document."""
    base = main_path.name[: -len(MAIN_SUFFIX)]
    main_body, had_notice, vlm = _read_split(main_path)

    bibtex_block = ""
    m = _BIBTEX_BLOCK_RE.match(main_body)
    if m:
        bibtex_block = m.group(0)
        main_body = main_body[m.end() :].strip()

    # Deliberate allparts order (§11): main → appendix → backmatter.
    parts = [main_body] if main_body else []
    for section in ("appendix", "backmatter"):
        section_path = main_path.with_name(f"{base}_{section}.md")
        if section_path.is_file():
            body, notice, section_vlm = _read_split(section_path)
            had_notice = had_notice or notice
            vlm = vlm or section_vlm
            if body:
                parts.append(body)

    joined = "\n\n".join(parts)
    if had_notice:
        # "OCR and VLMs" in any stripped notice implies VLM involvement; passing
        # vlm_tables=True is a no-op when figure descriptions already imply it.
        joined = append_transcription_notice(joined, vlm_tables=vlm)
    else:
        joined = joined.rstrip() + "\n"
    return bibtex_block + joined
