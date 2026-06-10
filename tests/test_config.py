"""M0: config load, CLI-override precedence, and validation (DESIGN §13, §16).

Asserts the §1.2 "every field overridable" contract literally by walking the
§13.3 config<->CLI mapping table.
"""

from __future__ import annotations

import pytest

from inscriber.config import (
    ConfigError,
    load_config_file,
    resolve_config,
    validate_structural,
)
from inscriber.models import RunConfig


def cfg_from(argv, *, validate=True):
    """Resolve a RunConfig from argv with NO config file (isolate CLI/defaults)."""
    parser_argv = list(argv)
    # build_run_config reads the platform config file; for hermetic tests we pass
    # an explicit empty merge via resolve_config instead. Use the lower-level path.
    from inscriber.cli import build_parser, collect_cli_sections

    args = build_parser().parse_args(parser_argv)
    cli_sections = collect_cli_sections(args)
    rc = resolve_config(
        command=args.command,
        input_arg=args.input,
        config_path=None,
        file_dict={},
        cli_sections=cli_sections,
        pages=getattr(args, "pages", None),
        verbose=args.verbose,
        quiet=args.quiet,
    )
    if validate:
        validate_structural(rc)
    return rc


# --------------------------------------------------------------------------- #
# Defaults (DESIGN §13.1)
# --------------------------------------------------------------------------- #


def test_defaults_match_design():
    rc = cfg_from(["run", "paper.pdf"])
    assert rc.command == "run"
    assert rc.input == "paper.pdf"
    assert rc.llama.host == "127.0.0.1"
    assert rc.llama.port == 0
    assert rc.llama.server_start_timeout == 120
    assert rc.llama.ctx_size == 16384  # headroom for the table pass (8k budget)
    assert rc.inference.mode == "sequential"
    assert rc.ocr.backend == "deepseek-ocr"
    assert rc.ocr.resolution == "large"
    assert rc.ocr.n_gpu_layers == "auto"  # GPU offload by default (DESIGN §13.1)
    assert rc.vlm.n_gpu_layers == "auto"
    assert rc.vlm.backend == "gemma"
    assert rc.figure.detect == "auto"
    assert rc.figure.mode == "describe-only"
    assert rc.figure.crop_padding == pytest.approx(0.02)
    assert rc.figure.context_chars == 2000
    assert rc.table.refine is True
    assert rc.output.split is True
    assert rc.output.page_numbers is False
    assert rc.output.page_separators is False
    assert rc.output.normalize_line_breaks is True
    assert rc.output.clean is True
    assert rc.output.clobber is True
    assert rc.output.notice is True
    assert rc.cache.enabled is True
    assert rc.cache.refresh is False
    assert rc.workdir.path == ""
    assert rc.workdir.keep_intermediates is False
    assert rc.bibtex.enabled is False
    assert rc.bibtex.append_to_document is False
    assert rc.net.offline is False


def test_bare_input_is_run():
    # build_run_config injects nothing; the cli.main normalizer does. Here we test
    # the explicit "run" path; the normalizer is covered in test_cli_normalizer.
    rc = cfg_from(["run", "paper.pdf"])
    assert rc.command == "run"


# --------------------------------------------------------------------------- #
# CLI > file > default precedence
# --------------------------------------------------------------------------- #


def test_cli_overrides_file_and_default():
    file_dict = {
        "ocr": {"resolution": "base", "model": "/from/file.gguf"},
        "output": {"split": True},
    }
    from inscriber.cli import build_parser, collect_cli_sections

    args = build_parser().parse_args(
        ["run", "paper.pdf", "--ocr-resolution", "gundam", "--no-split"]
    )
    rc = resolve_config(
        command=args.command,
        input_arg=args.input,
        config_path=None,
        file_dict=file_dict,
        cli_sections=collect_cli_sections(args),
        pages=None,
        verbose=0,
        quiet=False,
    )
    # CLI wins over file:
    assert rc.ocr.resolution == "gundam"
    assert rc.output.split is False
    # file wins over default where no CLI override:
    assert rc.ocr.model == "/from/file.gguf"
    # default holds where neither set:
    assert rc.ocr.backend == "deepseek-ocr"


# --------------------------------------------------------------------------- #
# Every field overridable (§13.3 mapping) — exercises each CLI flag.
# --------------------------------------------------------------------------- #


def test_every_field_overridable():
    argv = [
        "run", "paper.pdf",
        "--config", "ignored.toml",  # parsed but file load bypassed here
        "-o", "OUT",
        "--pages", "1-10",
        "--llama-bin-dir", "/bin",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--ctx", "4096",
        "--server-timeout", "30",
        "--mode", "concurrent",
        "--ocr-backend", "deepseek-ocr",
        "--ocr-model", "/m/ocr.gguf",
        "--ocr-mmproj", "/m/ocr-mmproj.gguf",
        "--ocr-resolution", "base",
        "--ocr-ngl", "20",
        "--ocr-endpoint", "http://localhost:1",
        "--figure-detect", "grounding",
        "--crop-padding", "0.05",
        "--vlm-backend", "gemma",
        "--vlm-model", "/m/vlm.gguf",
        "--vlm-mmproj", "/m/vlm-mmproj.gguf",
        "--vlm-ngl", "10",
        "--vlm-endpoint", "http://localhost:2",
        "--figure-mode", "describe-and-keep",
        "--context-chars", "1500",
        "--no-table-refine",
        "--no-split",
        "--page-numbers",
        "--page-separators",
        "--no-clean",
        "--no-normalize-breaks",
        "--no-clobber",
        "--no-notice",
        "--bibtex",
        "--bibtex-in-doc",
        "--offline",
        "--no-cache",
        "--refresh",
        "--workdir", "/tmp/work",
        "--keep-intermediates",
    ]
    # validate=False: this exercises the override MAPPING for every flag, including
    # the otherwise-conflicting --mode concurrent + --port 9000 combination.
    rc = cfg_from(argv, validate=False)

    assert rc.output.dir == "OUT"
    assert rc.pages == "1-10"
    assert rc.llama.bin_dir == "/bin"
    assert rc.llama.host == "0.0.0.0"
    assert rc.llama.port == 9000
    assert rc.llama.ctx_size == 4096
    assert rc.llama.server_start_timeout == 30
    assert rc.inference.mode == "concurrent"
    assert rc.ocr.model == "/m/ocr.gguf"
    assert rc.ocr.mmproj == "/m/ocr-mmproj.gguf"
    assert rc.ocr.resolution == "base"
    assert rc.ocr.n_gpu_layers == 20
    assert rc.ocr.endpoint == "http://localhost:1"
    assert rc.figure.detect == "grounding"
    assert rc.figure.crop_padding == pytest.approx(0.05)
    assert rc.vlm.model == "/m/vlm.gguf"
    assert rc.vlm.mmproj == "/m/vlm-mmproj.gguf"
    assert rc.vlm.n_gpu_layers == 10
    assert rc.vlm.endpoint == "http://localhost:2"
    assert rc.figure.mode == "describe-and-keep"
    assert rc.figure.context_chars == 1500
    assert rc.table.refine is False
    assert rc.output.split is False
    assert rc.output.page_numbers is True
    assert rc.output.page_separators is True
    assert rc.output.normalize_line_breaks is False
    assert rc.output.clean is False
    assert rc.output.clobber is False
    assert rc.output.notice is False
    assert rc.bibtex.enabled is True
    assert rc.bibtex.append_to_document is True
    assert rc.net.offline is True
    assert rc.cache.enabled is False
    assert rc.cache.refresh is True
    assert rc.workdir.path == "/tmp/work"
    assert rc.workdir.keep_intermediates is True


def test_ngl_accepts_auto_all_and_int():
    assert cfg_from(["run", "p.pdf", "--ocr-ngl", "auto"]).ocr.n_gpu_layers == "auto"
    assert cfg_from(["run", "p.pdf", "--ocr-ngl", "all"]).ocr.n_gpu_layers == "all"
    assert cfg_from(["run", "p.pdf", "--vlm-ngl", "33"]).vlm.n_gpu_layers == 33


def test_ngl_rejects_bad_value():
    from inscriber.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "p.pdf", "--ocr-ngl", "lots"])


def test_no_figures_aliases_detect_none():
    rc = cfg_from(["run", "paper.pdf", "--no-figures"])
    assert rc.figure.detect == "none"


def test_no_figures_wins_over_figure_detect():
    rc = cfg_from(["run", "paper.pdf", "--figure-detect", "grounding", "--no-figures"])
    assert rc.figure.detect == "none"


# --------------------------------------------------------------------------- #
# Validation (layer a: structural)
# --------------------------------------------------------------------------- #


def test_invalid_resolution_rejected():
    from inscriber.cli import build_parser

    # argparse choices reject bad resolution at parse time:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "p.pdf", "--ocr-resolution", "huge"])


def test_structural_validation_catches_bad_values():
    rc = RunConfig(command="run", input="p.pdf")
    rc.figure.crop_padding = 2.0  # out of [0,1]
    rc.llama.ctx_size = 0  # must be > 0
    with pytest.raises(ConfigError) as e:
        validate_structural(rc)
    msg = str(e.value)
    assert "crop_padding" in msg
    assert "ctx_size" in msg


def test_unknown_backend_rejected():
    rc = RunConfig(command="run", input="p.pdf")
    rc.ocr.backend = "made-up-ocr"
    with pytest.raises(ConfigError):
        validate_structural(rc)


def test_bad_numeric_type_is_clean_config_error():
    # A malformed TOML value (wrong type) must yield a ConfigError, not a TypeError.
    rc = RunConfig(command="run", input="p.pdf")
    rc.llama.port = "x"  # e.g. port = "x" in config.toml
    with pytest.raises(ConfigError, match="port must be an integer"):
        validate_structural(rc)
    rc2 = RunConfig(command="run", input="p.pdf")
    rc2.figure.crop_padding = "big"
    with pytest.raises(ConfigError, match="crop_padding must be a number"):
        validate_structural(rc2)


def test_concurrent_requires_auto_port():
    rc = RunConfig(command="run", input="p.pdf")
    rc.inference.mode = "concurrent"
    rc.llama.port = 9000
    with pytest.raises(ConfigError, match="concurrent mode requires an auto port"):
        validate_structural(rc)
    # concurrent + auto port (0) is fine:
    rc.llama.port = 0
    validate_structural(rc)


# --------------------------------------------------------------------------- #
# Config file loading
# --------------------------------------------------------------------------- #


def test_explicit_missing_config_errors(tmp_path):
    with pytest.raises(ConfigError):
        load_config_file(str(tmp_path / "nope.toml"))


def test_load_toml(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text(
        '[ocr]\nresolution = "tiny"\nmodel = "/x/y.gguf"\n[net]\noffline = true\n',
        encoding="utf-8",
    )
    data, used = load_config_file(str(p))
    assert used == p
    assert data["ocr"]["resolution"] == "tiny"
    assert data["net"]["offline"] is True


def test_implicit_local_config_wins_over_platform_config(tmp_path, monkeypatch):
    from inscriber import config as config_module

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    platform_dir = tmp_path / "platform"
    platform_dir.mkdir()
    local_config = project_dir / "config.toml"
    platform_config = platform_dir / "config.toml"
    local_config.write_text('[ocr]\nresolution = "tiny"\n', encoding="utf-8")
    platform_config.write_text('[ocr]\nresolution = "base"\n', encoding="utf-8")

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        config_module.platformdirs,
        "user_config_dir",
        lambda _appname: str(platform_dir),
    )

    data, used = load_config_file(None)
    assert used == local_config
    assert data["ocr"]["resolution"] == "tiny"


def test_implicit_platform_config_used_when_no_local_config(tmp_path, monkeypatch):
    from inscriber import config as config_module

    project_dir = tmp_path / "project"
    project_dir.mkdir()
    platform_dir = tmp_path / "platform"
    platform_dir.mkdir()
    platform_config = platform_dir / "config.toml"
    platform_config.write_text('[ocr]\nresolution = "base"\n', encoding="utf-8")

    monkeypatch.chdir(project_dir)
    monkeypatch.setattr(
        config_module.platformdirs,
        "user_config_dir",
        lambda _appname: str(platform_dir),
    )

    data, used = load_config_file(None)
    assert used == platform_config
    assert data["ocr"]["resolution"] == "base"


def test_invalid_toml_errors(tmp_path):
    p = tmp_path / "config.toml"
    p.write_text("this is = = not valid toml", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config_file(str(p))
