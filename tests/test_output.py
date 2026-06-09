"""M5 gap coverage: output writer (DESIGN §14)."""

from __future__ import annotations

import pytest

from inscriber.models import Figure
from inscriber.output import (
    OutputError,
    copy_figures,
    sanitize_base_name,
    write_text_file,
)


def test_sanitize_base_name_avoids_collision():
    # "paper.main" must not be able to collide with the "paper.main.md" split output.
    assert sanitize_base_name("paper.main") == "paper_main"
    assert sanitize_base_name("My Paper (v2)") == "My_Paper_v2"
    assert sanitize_base_name("") == "paper"
    assert sanitize_base_name("...") == "paper"


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
