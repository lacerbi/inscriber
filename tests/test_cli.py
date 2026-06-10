"""M0: CLI surface behaviors — default-subcommand injection, --version, help."""

from __future__ import annotations

import pytest

from inscriber import __version__
from inscriber.cli import _normalize_argv, main


def test_normalizer_injects_run():
    assert _normalize_argv(["paper.pdf"]) == ["run", "paper.pdf"]
    assert _normalize_argv(["-v", "paper.pdf"]) == ["run", "-v", "paper.pdf"]


def test_normalizer_leaves_subcommands():
    assert _normalize_argv(["ocr", "paper.pdf"]) == ["ocr", "paper.pdf"]
    assert _normalize_argv(["describe", "b"]) == ["describe", "b"]
    assert _normalize_argv(["--version"]) == ["--version"]
    assert _normalize_argv([]) == []


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert __version__ in out


def test_empty_argv_prints_help():
    assert main([]) == 2


def test_missing_input_returns_1(tmp_path):
    # A nonexistent input PDF is an InputError -> CLI maps it to a clean exit 1.
    rc = main(["run", str(tmp_path / "x.pdf")])
    assert rc == 1


def test_bibtex_mode_flag_wiring():
    # --bibtex-mode is the full knob; --bibtex stays a back-compat alias for "on"
    # and loses to an explicit --bibtex-mode (PLAN-bibtex-auto B0).
    from inscriber.cli import build_parser, collect_cli_sections

    args = build_parser().parse_args(["run", "p.pdf", "--bibtex-mode", "auto"])
    assert collect_cli_sections(args)["bibtex"]["mode"] == "auto"
    args = build_parser().parse_args(["run", "p.pdf", "--bibtex"])
    assert collect_cli_sections(args)["bibtex"]["mode"] == "on"
    args = build_parser().parse_args(["run", "p.pdf", "--bibtex", "--bibtex-mode", "off"])
    assert collect_cli_sections(args)["bibtex"]["mode"] == "off"
    # not passed at all -> no override emitted (config/default applies)
    args = build_parser().parse_args(["run", "p.pdf"])
    assert "mode" not in collect_cli_sections(args).get("bibtex", {})
