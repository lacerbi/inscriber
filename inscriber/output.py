"""Output writer (DESIGN §14).

Writes the full document (always) and — once splitting lands (M3) — the
main/appendix/backmatter files. All text is UTF-8 with ``\\n`` newlines (never let
Windows inject ``\\r\\n``). ``clobber`` (default true) overwrites; ``--no-clobber``
makes a pre-existing target a hard error.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from inscriber.errors import InscriberError
from inscriber.logging import get_logger
from inscriber.models import Figure

logger = get_logger()


class OutputError(InscriberError):
    """Raised on a clobber conflict or unwritable output."""


def sanitize_base_name(name: str) -> str:
    """Sanitize a base name for use in output filenames (DESIGN §14) —
    dots/spaces/etc → ``_``. Every document output carries a ``_part`` suffix
    (``_full``/``_main``/…), which is what keeps pathological source names
    (e.g. a PDF literally named ``paper_main.pdf``) from colliding with
    another document's outputs.
    """
    cleaned = re.sub(r"[^\w\-]+", "_", name).strip("_")
    return cleaned or "paper"


def write_text_file(path: Path, content: str, *, clobber: bool) -> Path:
    """Write UTF-8 text with ``\\n`` newlines, honoring the clobber policy."""
    if path.exists() and not clobber:
        raise OutputError(f"output exists and --no-clobber set: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    logger.info("wrote %s", path)
    return path


def copy_figures(
    figures: list[Figure], *, src_base: Path, out_dir: Path, clobber: bool = True
) -> list[Path]:
    """Copy referenced crop PNGs into ``out_dir/figures/`` (for describe-and-keep).

    Honors the ``clobber`` policy: a pre-existing destination figure with
    ``--no-clobber`` is a hard error (DESIGN §14).
    """
    written: list[Path] = []
    dest_dir = out_dir / "figures"
    for fig in figures:
        if not fig.crop_path:
            continue
        src = src_base / fig.crop_path
        if not src.is_file():
            logger.warning("crop not found, skipping copy: %s", src)
            continue
        dest = out_dir / fig.crop_path  # crop_path is "figures/<id>.png"
        if src.resolve() == dest.resolve():
            written.append(dest)  # already in place (out_dir == bundle dir)
            continue
        if dest.exists() and not clobber:
            raise OutputError(f"output figure exists and --no-clobber set: {dest}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        written.append(dest)
    return written


def write_full_document(
    out_dir: Path, base_name: str, markdown: str, *, clobber: bool
) -> Path:
    """Write ``{base}_full.md`` — the full stitched document (DESIGN §14)."""
    out_dir = Path(out_dir)
    return write_text_file(out_dir / f"{base_name}_full.md", markdown, clobber=clobber)


def write_split_documents(
    out_dir: Path,
    base_name: str,
    *,
    main: str,
    appendix: str | None,
    backmatter: str | None,
    clobber: bool,
) -> list[Path]:
    """Write ``{base}_main.md`` and, when present, ``_appendix.md`` / ``_backmatter.md``."""
    out_dir = Path(out_dir)
    written = [write_text_file(out_dir / f"{base_name}_main.md", main, clobber=clobber)]
    if appendix is not None:
        written.append(
            write_text_file(out_dir / f"{base_name}_appendix.md", appendix, clobber=clobber)
        )
    if backmatter is not None:
        written.append(
            write_text_file(out_dir / f"{base_name}_backmatter.md", backmatter, clobber=clobber)
        )
    return written
