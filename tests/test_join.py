"""The ``join`` subcommand: rejoin split files into ``{base}.md`` (DESIGN §11).

The splits are written exactly the way ``pipeline._write_outputs`` writes them
(framed sections + per-file notice + optional BibTeX block on main), so these
tests pin the round-trip contract: join(strip extras, allparts order, re-frame)
equals the allparts assembly of the same sections.
"""

from __future__ import annotations

import pytest

from inscriber.cli import main as cli_main
from inscriber.output import write_split_documents, write_text_file
from inscriber.postprocess.join import (
    JoinError,
    join_split_files,
    resolve_join_input,
)
from inscriber.postprocess.notice import append_transcription_notice
from inscriber.postprocess.splitter import (
    prepare_formatted_sections,
    split_markdown_content,
)

FULL = """# A Study of Things

Intro text with math \\(x_i\\).

## 2. Results

| metric | value |
| --- | --- |
| acc | 0.91 |

## Acknowledgments

We thank everyone.

## References

- Someone 2020.

# Appendix

## A. Extra proofs

More \\(y\\) details.
"""

BIBTEX_BLOCK = "```\n@article{someone2020study, title={A Study of Things}}\n```\n\n---\n\n"


def _formatted_sections():
    sections = split_markdown_content(FULL)
    main, backmatter, appendix = prepare_formatted_sections(sections)
    assert backmatter is not None and appendix is not None  # fixture sanity
    return main, backmatter, appendix


def _write_splits(out_dir, base="paper", *, notice=True, vlm_tables=False, bibtex_block=None):
    """Write splits exactly like ``pipeline._write_outputs`` does."""
    main, backmatter, appendix = _formatted_sections()
    if bibtex_block:
        main = bibtex_block + main
    if notice:
        main = append_transcription_notice(main, vlm_tables=vlm_tables)
        appendix = append_transcription_notice(appendix, vlm_tables=vlm_tables)
        backmatter = append_transcription_notice(backmatter, vlm_tables=vlm_tables)
    write_split_documents(
        out_dir, base, main=main, appendix=appendix, backmatter=backmatter, clobber=True
    )
    return out_dir / f"{base}_main.md"


def _expected_core():
    """The §11 allparts assembly: main → appendix → backmatter."""
    main, backmatter, appendix = _formatted_sections()
    return "\n\n".join([main, appendix, backmatter])


# --------------------------------------------------------------------------- #
# join_split_files
# --------------------------------------------------------------------------- #


def test_round_trip_allparts_with_notice(tmp_path):
    main_path = _write_splits(tmp_path)
    joined = join_split_files(main_path)
    assert joined == append_transcription_notice(_expected_core())
    # one notice, not one per part:
    assert joined.count("Transcribed with") == 1


def test_vlm_table_notice_inferred(tmp_path):
    # vlm_tables with no figure descriptions: only the stripped notices reveal
    # VLM involvement — join must carry it into the regenerated notice.
    main_path = _write_splits(tmp_path, vlm_tables=True)
    joined = join_split_files(main_path)
    assert joined == append_transcription_notice(_expected_core(), vlm_tables=True)
    assert "OCR and VLMs" in joined


def test_bibtex_block_reprepended(tmp_path):
    main_path = _write_splits(tmp_path, bibtex_block=BIBTEX_BLOCK)
    joined = join_split_files(main_path)
    assert joined == BIBTEX_BLOCK + append_transcription_notice(_expected_core())
    assert joined.count("@article{") == 1


def test_no_notice_splits(tmp_path):
    main_path = _write_splits(tmp_path, notice=False)
    joined = join_split_files(main_path)
    assert joined == _expected_core() + "\n"
    assert "Transcribed with" not in joined


def test_main_only(tmp_path):
    main, _, _ = _formatted_sections()
    main_path = write_text_file(
        tmp_path / "paper_main.md", append_transcription_notice(main), clobber=True
    )
    assert join_split_files(main_path) == append_transcription_notice(main)


def test_crlf_split_tolerated(tmp_path):
    # A hand-edit on Windows may save CRLF; the notice must still be stripped.
    main, _, _ = _formatted_sections()
    crlf = append_transcription_notice(main).replace("\n", "\r\n")
    main_path = tmp_path / "paper_main.md"
    main_path.write_bytes(crlf.encode("utf-8"))
    assert join_split_files(main_path) == append_transcription_notice(main)


def test_hand_edit_survives(tmp_path):
    main_path = _write_splits(tmp_path)
    edited = main_path.read_text(encoding="utf-8").replace("acc | 0.91", "acc | 0.92")
    main_path.write_text(edited, encoding="utf-8", newline="\n")
    assert "acc | 0.92" in join_split_files(main_path)


# --------------------------------------------------------------------------- #
# resolve_join_input
# --------------------------------------------------------------------------- #


def test_resolve_accepts_base_main_file_and_dir(tmp_path):
    main_path = _write_splits(tmp_path)
    assert resolve_join_input(tmp_path / "paper") == main_path
    assert resolve_join_input(main_path) == main_path
    assert resolve_join_input(tmp_path) == main_path


def test_resolve_errors(tmp_path):
    with pytest.raises(JoinError, match="not found"):
        resolve_join_input(tmp_path / "missing")
    with pytest.raises(JoinError, match="no \\*_main.md"):
        resolve_join_input(tmp_path)
    _write_splits(tmp_path, "a")
    _write_splits(tmp_path, "b")
    with pytest.raises(JoinError, match="multiple"):
        resolve_join_input(tmp_path)


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #


def _empty_config(tmp_path):
    p = tmp_path / "empty-config.toml"
    p.write_text("", encoding="utf-8")
    return str(p)


def test_cli_join_writes_full_document(tmp_path, capsys):
    _write_splits(tmp_path)
    rc = cli_main(["join", str(tmp_path / "paper"), "-c", _empty_config(tmp_path)])
    assert rc == 0
    out_file = tmp_path / "paper_full.md"
    assert out_file.is_file()
    assert out_file.read_text(encoding="utf-8") == append_transcription_notice(_expected_core())
    assert str(out_file) in capsys.readouterr().out


def test_cli_join_no_clobber(tmp_path):
    _write_splits(tmp_path)
    write_text_file(tmp_path / "paper_full.md", "existing\n", clobber=True)
    rc = cli_main([
        "join", str(tmp_path / "paper"), "--no-clobber", "-c", _empty_config(tmp_path)
    ])
    assert rc == 1
    assert (tmp_path / "paper_full.md").read_text(encoding="utf-8") == "existing\n"


def test_cli_join_missing_input(tmp_path):
    rc = cli_main(["join", str(tmp_path / "nope"), "-c", _empty_config(tmp_path)])
    assert rc == 1


def test_cli_join_no_full_suffix(tmp_path):
    _write_splits(tmp_path)
    rc = cli_main([
        "join", str(tmp_path / "paper"), "--no-full-suffix", "-c", _empty_config(tmp_path)
    ])
    assert rc == 0
    assert (tmp_path / "paper.md").is_file()
    assert not (tmp_path / "paper_full.md").exists()
