"""M5 gap coverage: output writer (DESIGN §14)."""

from __future__ import annotations

import logging

import pytest

from inscriber.models import Figure
from inscriber.output import (
    OutputError,
    copy_figures,
    sanitize_base_name,
    write_full_document,
    write_text_file,
)


def test_sanitize_base_name_avoids_collision():
    # Dots/spaces become underscores; the _full/_main/... output suffixes do the
    # rest of the collision avoidance (DESIGN §14).
    assert sanitize_base_name("paper.main") == "paper_main"
    assert sanitize_base_name("My Paper (v2)") == "My_Paper_v2"
    assert sanitize_base_name("") == "paper"
    assert sanitize_base_name("...") == "paper"


def test_sanitize_base_name_windows_reserved_stems():
    # Review B3: a bare CON.md / CON.bib is an unwritable device name on Windows
    # (the extension does not unreserve the stem) — append "_" to stay writable.
    assert sanitize_base_name("CON") == "CON_"
    assert sanitize_base_name("con") == "con_"
    assert sanitize_base_name("COM1") == "COM1_"
    assert sanitize_base_name("lpt9") == "lpt9_"
    assert sanitize_base_name("CONSOLE") == "CONSOLE"  # only exact stems
    assert sanitize_base_name("COM10") == "COM10"  # COM1-9 / LPT1-9 only


def test_no_full_suffix_part_suffix_base_warns(tmp_path, caplog):
    # Review B3: with --no-full-suffix, a base that itself ends in a part suffix
    # writes e.g. x_main.md — which can collide with ANOTHER paper's split file.
    logging.getLogger("inscriber").propagate = True  # let caplog see records
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        write_full_document(tmp_path, "x_main", "body", clobber=True, full_suffix=False)
    assert "could collide" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="inscriber"):
        write_full_document(tmp_path, "x", "body", clobber=True, full_suffix=False)
        write_full_document(tmp_path, "y_main_v2", "body", clobber=True, full_suffix=True)
    assert "could collide" not in caplog.text  # innocuous names stay quiet


def test_write_text_file_clobbers_by_default(tmp_path):
    p = tmp_path / "a.md"
    write_text_file(p, "one", clobber=True)
    write_text_file(p, "two", clobber=True)
    assert p.read_text(encoding="utf-8") == "two"


def test_write_text_file_no_clobber_errors(tmp_path):
    p = tmp_path / "a.md"
    write_text_file(p, "one", clobber=True)
    with pytest.raises(OutputError, match="--no-clobber"):
        write_text_file(p, "two", clobber=False)


def test_write_text_file_uses_lf_newlines(tmp_path):
    p = tmp_path / "a.md"
    write_text_file(p, "line1\nline2\n", clobber=True)
    # Read raw bytes: no \r\n even on Windows.
    assert b"\r\n" not in p.read_bytes()


def test_write_text_file_unwritable_raises_output_error(tmp_path):
    # Review batch 5: a locked/unwritable target after a long run must surface
    # as an actionable OutputError, not a raw OSError traceback.
    target = tmp_path / "a.md"
    target.mkdir()  # writing text over a directory path fails on every OS
    with pytest.raises(OutputError, match="could not write"):
        write_text_file(target, "x", clobber=True)


def test_copy_figures(tmp_path):
    src = tmp_path / "bundle"
    (src / "figures").mkdir(parents=True)
    (src / "figures" / "fig_p1_1.png").write_bytes(b"PNG")
    out = tmp_path / "out"
    figs = [Figure(id="fig_p1_1", page=1, bbox_norm=(0, 0, 1, 1),
                   crop_path="figures/fig_p1_1.png", caption=None)]
    written = copy_figures(figs, src_base=src, out_dir=out)
    assert (out / "figures" / "fig_p1_1.png").read_bytes() == b"PNG"
    assert out / "figures" / "fig_p1_1.png" in written


def test_copy_figures_no_clobber_errors(tmp_path):
    src = tmp_path / "bundle"
    (src / "figures").mkdir(parents=True)
    (src / "figures" / "fig_p1_1.png").write_bytes(b"NEW")
    out = tmp_path / "out"
    (out / "figures").mkdir(parents=True)
    (out / "figures" / "fig_p1_1.png").write_bytes(b"OLD")  # pre-existing
    figs = [Figure(id="fig_p1_1", page=1, bbox_norm=(0, 0, 1, 1),
                   crop_path="figures/fig_p1_1.png", caption=None)]
    with pytest.raises(OutputError, match="--no-clobber"):
        copy_figures(figs, src_base=src, out_dir=out, clobber=False)
    # the existing file is untouched:
    assert (out / "figures" / "fig_p1_1.png").read_bytes() == b"OLD"


def test_copy_figures_self_copy_guard(tmp_path):
    # When out_dir == bundle dir, the source IS the destination — must not error.
    base = tmp_path / "b"
    (base / "figures").mkdir(parents=True)
    (base / "figures" / "fig_p1_1.png").write_bytes(b"X")
    figs = [Figure(id="fig_p1_1", page=1, bbox_norm=(0, 0, 1, 1),
                   crop_path="figures/fig_p1_1.png", caption=None)]
    written = copy_figures(figs, src_base=base, out_dir=base)
    assert (base / "figures" / "fig_p1_1.png").read_bytes() == b"X"
    assert len(written) == 1
