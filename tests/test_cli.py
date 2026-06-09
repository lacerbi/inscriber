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
