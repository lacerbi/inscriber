"""Command-line interface (DESIGN §13.2).

Three subcommands — ``run`` (default), ``ocr``, ``describe`` — sharing flag
groups. Bare ``inscriber INPUT`` is shorthand for ``inscriber run INPUT``.

Every config field is overridable by a flag (the §1.2 "every field overridable"
contract); unset flags default to ``None`` here so :func:`config.resolve_config`
can tell "not passed" from an explicit value and apply CLI > file > default.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from inscriber import __version__
from inscriber.config import (
    ConfigError,
    load_config_file,
    resolve_config,
    validate_structural,
)
from inscriber.errors import InscriberError
from inscriber.logging import get_logger, setup_logging

SUBCOMMANDS = ("run", "ocr", "describe")


def _ngl(value: str):
    """argparse type for ``-ngl``: ``auto`` | ``all`` | a non-negative integer."""
    v = value.strip().lower()
    if v in ("auto", "all"):
        return v
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"must be an integer, 'auto', or 'all' (got {value!r})"
        ) from e
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0 (got {n})")
    return n


# --------------------------------------------------------------------------- #
# Argument-group helpers (added per subcommand)
# --------------------------------------------------------------------------- #


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("-c", "--config", dest="config", default=None, metavar="PATH",
                   help="config file (default: ./config.toml, then platform config dir)")
    p.add_argument("-o", "--output-dir", dest="output_dir", default=None, metavar="DIR",
                   help="output directory (default: cwd)")
    p.add_argument("-v", "--verbose", action="count", default=0,
                   help="verbose logging (DEBUG)")
    p.add_argument("-q", "--quiet", action="store_true", default=False,
                   help="quiet logging (WARNING)")


def _add_inference(p: argparse.ArgumentParser, *, include_mode: bool) -> None:
    p.add_argument("--llama-bin-dir", dest="llama_bin_dir", default=None, metavar="DIR")
    p.add_argument("--host", dest="host", default=None, metavar="HOST",
                   help="llama-server bind host (default 127.0.0.1)")
    p.add_argument("--port", dest="port", type=int, default=None, metavar="N",
                   help="fixed port (default 0 = auto)")
    p.add_argument("--ctx", dest="ctx", type=int, default=None, metavar="N",
                   help="context size")
    p.add_argument("--server-timeout", dest="server_timeout", type=int, default=None,
                   metavar="SEC", help="seconds to wait for /health")
    if include_mode:
        p.add_argument("--mode", dest="mode", choices=("sequential", "concurrent"),
                       default=None, help="inference mode (run only)")


def _add_ocr_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ocr-backend", dest="ocr_backend", default=None, metavar="NAME",
                   help="v1: deepseek-ocr")
    p.add_argument("--ocr-model", dest="ocr_model", default=None, metavar="PATH")
    p.add_argument("--ocr-mmproj", dest="ocr_mmproj", default=None, metavar="PATH")
    p.add_argument("--ocr-resolution", dest="ocr_resolution", default=None,
                   choices=("tiny", "small", "base", "large", "gundam"), metavar="MODE")
    p.add_argument("--ocr-ngl", dest="ocr_ngl", type=_ngl, default=None, metavar="N",
                   help="OCR GPU layers: auto (default) | all | integer (0 = CPU)")
    p.add_argument("--ocr-endpoint", dest="ocr_endpoint", default=None, metavar="URL",
                   help="use running server; don't spawn")
    p.add_argument("--figure-detect", dest="figure_detect", default=None,
                   choices=("auto", "grounding", "none", "pdf-embedded"), metavar="MODE")
    p.add_argument("--no-figures", dest="no_figures", action="store_const", const=True,
                   default=None, help="alias for --figure-detect none")
    p.add_argument("--crop-padding", dest="crop_padding", type=float, default=None,
                   metavar="FRAC", help="figure crop margin (fraction of page dims)")


def _add_vlm_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--vlm-backend", dest="vlm_backend", default=None, metavar="NAME")
    p.add_argument("--vlm-model", dest="vlm_model", default=None, metavar="PATH")
    p.add_argument("--vlm-mmproj", dest="vlm_mmproj", default=None, metavar="PATH")
    p.add_argument("--vlm-ngl", dest="vlm_ngl", type=_ngl, default=None, metavar="N",
                   help="VLM GPU layers: auto (default) | all | integer (0 = CPU)")
    p.add_argument("--vlm-endpoint", dest="vlm_endpoint", default=None, metavar="URL")
    p.add_argument("--figure-mode", dest="figure_mode", default=None,
                   choices=("describe-only", "describe-and-keep", "placeholder"))
    p.add_argument("--context-chars", dest="context_chars", type=int, default=None,
                   metavar="N", help="whole-page context truncation cap")


def _add_output_stage(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-split", dest="split", action="store_const", const=False,
                   default=None, help="write only the full document")
    p.add_argument("--page-numbers", dest="page_numbers", action="store_const",
                   const=True, default=None, help='insert "#### Page N" before each page')
    p.add_argument("--page-separators", dest="page_separators", action="store_const",
                   const=True, default=None, help='insert "---" between pages')
    p.add_argument("--no-clean", dest="clean", action="store_const", const=False,
                   default=None, help="skip header/footer + de-hyphenation cleanup")
    p.add_argument("--no-normalize-breaks", dest="normalize_line_breaks",
                   action="store_const", const=False, default=None,
                   help="skip blank-line collapsing")
    p.add_argument("--no-clobber", dest="clobber", action="store_const", const=False,
                   default=None, help="error instead of overwriting existing outputs")
    p.add_argument("--bibtex", dest="bibtex", action="store_const", const=True,
                   default=None, help="fetch BibTeX (requires network)")
    p.add_argument("--bibtex-in-doc", dest="bibtex_in_doc", action="store_const",
                   const=True, default=None, help="also inject the BibTeX entry into the document")
    p.add_argument("--offline", dest="offline", action="store_const", const=True,
                   default=None, help="disable ALL network use (URL input + bibtex)")


def _add_caching(p: argparse.ArgumentParser) -> None:
    p.add_argument("--no-cache", dest="cache_enabled", action="store_const", const=False,
                   default=None, help="neither read nor write caches")
    p.add_argument("--refresh", dest="refresh", action="store_const", const=True,
                   default=None, help="ignore + recompute + overwrite caches")
    p.add_argument("--workdir", dest="workdir", default=None, metavar="DIR",
                   help="where intermediate page/crop images go")
    p.add_argument("--keep-intermediates", dest="keep_intermediates",
                   action="store_const", const=True, default=None,
                   help="don't delete the work dir on success")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inscriber",
        description="Convert academic PDFs into LLM-friendly text-only Markdown, locally.",
    )
    parser.add_argument("--version", action="version", version=f"inscriber {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{run,ocr,describe}")

    # run — end-to-end (default)
    p_run = sub.add_parser("run", help="end-to-end OCR → describe → write (default)")
    p_run.set_defaults(command="run")
    _add_common(p_run)
    p_run.add_argument("input", metavar="INPUT", help="PDF file path or http(s) URL")
    p_run.add_argument("--pages", dest="pages", default=None, metavar="RANGE",
                       help='1-indexed inclusive, e.g. "1-10","3","5-","-12","all"')
    _add_inference(p_run, include_mode=True)
    _add_ocr_stage(p_run)
    _add_vlm_stage(p_run)
    _add_output_stage(p_run)
    _add_caching(p_run)

    # ocr — OCR + crop → write bundle, stop
    p_ocr = sub.add_parser("ocr", help="OCR + figure crop → write OCR bundle, stop")
    p_ocr.set_defaults(command="ocr")
    _add_common(p_ocr)
    p_ocr.add_argument("input", metavar="INPUT", help="PDF file path or http(s) URL")
    p_ocr.add_argument("--pages", dest="pages", default=None, metavar="RANGE",
                       help='1-indexed inclusive, e.g. "1-10","3","5-","-12","all"')
    _add_inference(p_ocr, include_mode=False)
    _add_ocr_stage(p_ocr)
    _add_caching(p_ocr)

    # describe — OCR bundle → VLM + assemble + write
    p_desc = sub.add_parser("describe", help="OCR bundle → VLM describe + assemble + write")
    p_desc.set_defaults(command="describe")
    _add_common(p_desc)
    p_desc.add_argument("input", metavar="BUNDLE", help="path to a *.inscriber-ocr dir")
    _add_inference(p_desc, include_mode=False)
    _add_vlm_stage(p_desc)
    _add_output_stage(p_desc)
    _add_caching(p_desc)

    return parser


# --------------------------------------------------------------------------- #
# Args -> nested CLI-override sections (only explicitly-set values)
# --------------------------------------------------------------------------- #


def collect_cli_sections(args: argparse.Namespace) -> dict[str, dict]:
    """Translate parsed args into ``{section: {key: value}}`` for the merge.

    Only non-``None`` values are emitted (None == flag not passed). Handles the
    special transforms: ``--no-figures`` ⇒ ``figure.detect=none``, the set-false
    flags (``--no-split`` etc.), and the dest-renamed numeric flags.
    """
    sections: dict[str, dict] = defaultdict(dict)

    def setv(section: str, key: str, value) -> None:
        if value is not None:
            sections[section][key] = value

    g = lambda name: getattr(args, name, None)  # noqa: E731 - terse local accessor

    # llama
    setv("llama", "bin_dir", g("llama_bin_dir"))
    setv("llama", "host", g("host"))
    setv("llama", "port", g("port"))
    setv("llama", "ctx_size", g("ctx"))
    setv("llama", "server_start_timeout", g("server_timeout"))
    # inference
    setv("inference", "mode", g("mode"))
    # ocr
    setv("ocr", "backend", g("ocr_backend"))
    setv("ocr", "model", g("ocr_model"))
    setv("ocr", "mmproj", g("ocr_mmproj"))
    setv("ocr", "resolution", g("ocr_resolution"))
    setv("ocr", "n_gpu_layers", g("ocr_ngl"))
    setv("ocr", "endpoint", g("ocr_endpoint"))
    # figure.detect: --no-figures wins over --figure-detect
    figure_detect = "none" if g("no_figures") else g("figure_detect")
    setv("figure", "detect", figure_detect)
    setv("figure", "crop_padding", g("crop_padding"))
    setv("figure", "mode", g("figure_mode"))
    setv("figure", "context_chars", g("context_chars"))
    # vlm
    setv("vlm", "backend", g("vlm_backend"))
    setv("vlm", "model", g("vlm_model"))
    setv("vlm", "mmproj", g("vlm_mmproj"))
    setv("vlm", "n_gpu_layers", g("vlm_ngl"))
    setv("vlm", "endpoint", g("vlm_endpoint"))
    # output
    setv("output", "dir", g("output_dir"))
    setv("output", "split", g("split"))
    setv("output", "page_numbers", g("page_numbers"))
    setv("output", "page_separators", g("page_separators"))
    setv("output", "normalize_line_breaks", g("normalize_line_breaks"))
    setv("output", "clean", g("clean"))
    setv("output", "clobber", g("clobber"))
    # cache
    setv("cache", "enabled", g("cache_enabled"))
    setv("cache", "refresh", g("refresh"))
    # workdir
    setv("workdir", "path", g("workdir"))
    setv("workdir", "keep_intermediates", g("keep_intermediates"))
    # bibtex
    setv("bibtex", "enabled", g("bibtex"))
    setv("bibtex", "append_to_document", g("bibtex_in_doc"))
    # net
    setv("net", "offline", g("offline"))

    return dict(sections)


def build_run_config(argv: list[str]):
    """Parse ``argv`` (already including a subcommand) into a resolved RunConfig."""
    parser = build_parser()
    args = parser.parse_args(argv)
    file_dict, _ = load_config_file(args.config)
    cli_sections = collect_cli_sections(args)
    cfg = resolve_config(
        command=args.command,
        input_arg=args.input,
        config_path=args.config,
        file_dict=file_dict,
        cli_sections=cli_sections,
        pages=getattr(args, "pages", None),
        verbose=args.verbose,
        quiet=args.quiet,
    )
    validate_structural(cfg)
    return cfg


def _normalize_argv(argv: list[str]) -> list[str]:
    """Inject the default ``run`` subcommand for bare ``inscriber INPUT`` usage."""
    if not argv:
        return argv
    if argv[0] in SUBCOMMANDS or argv[0] in ("-h", "--help", "--version"):
        return argv
    return ["run", *argv]


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    parser = build_parser()
    if not argv:
        parser.print_help(sys.stderr)
        return 2

    argv = _normalize_argv(argv)
    args = parser.parse_args(argv)  # --version / -h exit here via argparse actions
    setup_logging(args.verbose, args.quiet)
    logger = get_logger()

    try:
        file_dict, _ = load_config_file(args.config)
        cli_sections = collect_cli_sections(args)
        cfg = resolve_config(
            command=args.command,
            input_arg=args.input,
            config_path=args.config,
            file_dict=file_dict,
            cli_sections=cli_sections,
            pages=getattr(args, "pages", None),
            verbose=args.verbose,
            quiet=args.quiet,
        )
        validate_structural(cfg)
    except ConfigError as e:
        logger.error("%s", e)
        return 2

    from inscriber import pipeline

    try:
        if cfg.command == "run":
            written = pipeline.run(cfg)
        elif cfg.command == "ocr":
            written = pipeline.run_ocr(cfg)
        elif cfg.command == "describe":
            written = pipeline.describe(cfg)
        else:  # pragma: no cover - argparse guarantees a valid command
            parser.error(f"unknown command {cfg.command!r}")
            return 2
    except ConfigError as e:
        logger.error("%s", e)
        return 2
    except InscriberError as e:
        logger.error("%s", e)
        return 1
    except NotImplementedError as e:
        logger.error("not implemented yet: %s", e)
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.error("interrupted")
        return 130

    for path in written or []:
        print(path)
    return 0
