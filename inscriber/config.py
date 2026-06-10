"""Config loading, 3-layer merge, and validation (DESIGN §13, §16).

Precedence: **CLI flag > config file > built-in default** (DESIGN §13.1).

Validation is two-layer (review Fix 3):

* **structural** — enum membership, numeric ranges, types. Runs always, right
  after the merge. Raises :class:`ConfigError`.
* **path-existence** — llama binary, model/mmproj GGUFs. Command/stage-aware and
  invoked only just before a server actually launches (so ``--version`` / config
  errors never touch the filesystem, ``--ocr-endpoint`` bypasses OCR-model checks,
  and ``describe`` never validates ``[ocr].*``). This is DESIGN §16's "validate
  before any model loads".
"""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import platformdirs

from inscriber.errors import InscriberError
from inscriber.models import (
    BibtexConfig,
    CacheConfig,
    FigureConfig,
    InferenceConfig,
    LlamaConfig,
    NetConfig,
    OcrConfig,
    OutputConfig,
    ResolutionMode,
    RunConfig,
    TableConfig,
    VlmConfig,
    WorkdirConfig,
)

try:  # 3.11+ has tomllib in the stdlib; tomli is the <3.11 fallback (DESIGN §15)
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as tomllib  # type: ignore[no-redef]


class ConfigError(InscriberError):
    """Raised on any structural or path-existence configuration error."""


_FIGURE_DETECT = {"auto", "grounding", "none", "pdf-embedded"}
_FIGURE_MODE = {"describe-only", "describe-and-keep", "placeholder"}
_INFERENCE_MODE = {"sequential", "concurrent"}
_BIBTEX_MODE = {"off", "on", "auto"}
_RESOLUTIONS = {m.value for m in ResolutionMode}

# section name -> dataclass; the merge applies known fields only.
_SECTIONS: dict[str, type] = {
    "llama": LlamaConfig,
    "inference": InferenceConfig,
    "ocr": OcrConfig,
    "vlm": VlmConfig,
    "figure": FigureConfig,
    "table": TableConfig,
    "output": OutputConfig,
    "cache": CacheConfig,
    "workdir": WorkdirConfig,
    "bibtex": BibtexConfig,
    "net": NetConfig,
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def default_config_path() -> Path:
    """Platform config path (DESIGN §13.1) via platformdirs (never hardcoded)."""
    return Path(platformdirs.user_config_dir("inscriber")) / "config.toml"


def local_config_path() -> Path:
    """Project-local config path checked before the platform config."""
    return Path.cwd() / "config.toml"


def load_config_file(config_path: str | None) -> tuple[dict, Path | None]:
    """Load a TOML config file.

    If ``config_path`` is given it must exist (explicit request → hard error if
    missing). If omitted, ``./config.toml`` is used when present, otherwise the
    platform default is used. Missing implicit config files are fine (returns
    ``{}``).
    """
    if config_path is not None:
        p = Path(config_path).expanduser()
        if not p.is_file():
            raise ConfigError(f"Config file not found: {p}")
    else:
        p = local_config_path()
        if not p.is_file():
            p = default_config_path()
        if not p.is_file():
            return {}, None
    try:
        with open(p, "rb") as f:
            return tomllib.load(f), p
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {p}: {e}") from e


# --------------------------------------------------------------------------- #
# Merge
# --------------------------------------------------------------------------- #


def _build_section(cls: type, file_section: dict | None, cli_section: dict | None):
    """Construct one section dataclass: defaults < file < CLI (None = unset)."""
    valid = {f.name for f in dataclasses.fields(cls)}
    kwargs: dict = {}
    for src in (file_section or {}, cli_section or {}):
        for key, value in src.items():
            if key in valid and value is not None:
                kwargs[key] = value
    return cls(**kwargs)


def resolve_config(
    *,
    command: str,
    input_arg: str,
    config_path: str | None,
    file_dict: dict,
    cli_sections: dict[str, dict],
    pages: str | None = None,
    verbose: int = 0,
    quiet: bool = False,
) -> RunConfig:
    """Merge defaults < file < CLI into a fully-resolved :class:`RunConfig`."""
    sections = {
        name: _build_section(cls, file_dict.get(name), cli_sections.get(name))
        for name, cls in _SECTIONS.items()
    }
    return RunConfig(
        command=command,
        input=input_arg,
        config_path=config_path,
        pages=pages,
        verbose=verbose,
        quiet=quiet,
        **sections,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Validation — layer (a): structural
# --------------------------------------------------------------------------- #


def validate_structural(cfg: RunConfig) -> None:
    """Enum membership, numeric ranges, types. Raises :class:`ConfigError`."""
    # Lazy import to avoid an import cycle (registries → backends → config); the
    # registries are the single source of truth for known backend names.
    from inscriber.ocr.registry import known_ocr_backends
    from inscriber.vlm.registry import known_vlm_backends

    errors: list[str] = []

    if cfg.ocr.resolution not in _RESOLUTIONS:
        errors.append(
            f"ocr.resolution {cfg.ocr.resolution!r} invalid; "
            f"choose one of {sorted(_RESOLUTIONS)}"
        )
    if cfg.figure.detect not in _FIGURE_DETECT:
        errors.append(
            f"figure.detect {cfg.figure.detect!r} invalid; "
            f"choose one of {sorted(_FIGURE_DETECT)}"
        )
    if cfg.figure.mode not in _FIGURE_MODE:
        errors.append(
            f"figure.mode {cfg.figure.mode!r} invalid; "
            f"choose one of {sorted(_FIGURE_MODE)}"
        )
    if cfg.inference.mode not in _INFERENCE_MODE:
        errors.append(
            f"inference.mode {cfg.inference.mode!r} invalid; "
            f"choose one of {sorted(_INFERENCE_MODE)}"
        )
    if cfg.bibtex.mode not in _BIBTEX_MODE:
        errors.append(
            f"bibtex.mode {cfg.bibtex.mode!r} invalid; "
            f"choose one of {sorted(_BIBTEX_MODE)}"
        )
    if cfg.ocr.backend not in known_ocr_backends():
        errors.append(
            f"ocr.backend {cfg.ocr.backend!r} unknown; v1 supports {known_ocr_backends()}"
        )
    if cfg.vlm.backend not in known_vlm_backends():
        errors.append(
            f"vlm.backend {cfg.vlm.backend!r} unknown; v1 supports {known_vlm_backends()}"
        )

    # Numeric ranges — type-guarded first so a malformed TOML value (e.g.
    # ``port = "x"``) yields a clean ConfigError, not a raw TypeError on the
    # comparison (bool is excluded since it is an int subclass).
    def _is_int(v) -> bool:
        return isinstance(v, int) and not isinstance(v, bool)

    def _is_num(v) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    if not _is_int(cfg.llama.port):
        errors.append(f"llama.port must be an integer (got {cfg.llama.port!r})")
    elif cfg.llama.port < 0 or cfg.llama.port > 65535:
        errors.append(f"llama.port {cfg.llama.port} out of range [0, 65535]")
        # Concurrent mode runs two servers at once; a single fixed port can't host both.
    if cfg.inference.mode == "concurrent" and _is_int(cfg.llama.port) and cfg.llama.port != 0:
        errors.append(
            "concurrent mode requires an auto port (llama.port=0 / --port 0); a fixed "
            "port cannot host both the OCR and VLM servers simultaneously"
        )
    if not _is_int(cfg.llama.ctx_size):
        errors.append(f"llama.ctx_size must be an integer (got {cfg.llama.ctx_size!r})")
    elif cfg.llama.ctx_size <= 0:
        errors.append(f"llama.ctx_size must be > 0 (got {cfg.llama.ctx_size})")
    if not _is_int(cfg.llama.server_start_timeout):
        errors.append(
            f"llama.server_start_timeout must be an integer (got {cfg.llama.server_start_timeout!r})"
        )
    elif cfg.llama.server_start_timeout <= 0:
        errors.append(
            f"llama.server_start_timeout must be > 0 (got {cfg.llama.server_start_timeout})"
        )
    for label, val in (("ocr.n_gpu_layers", cfg.ocr.n_gpu_layers),
                       ("vlm.n_gpu_layers", cfg.vlm.n_gpu_layers)):
        if isinstance(val, str):
            if val not in ("auto", "all"):
                errors.append(f"{label} must be an integer, 'auto', or 'all' (got {val!r})")
        elif not _is_int(val):
            errors.append(f"{label} must be an integer, 'auto', or 'all' (got {val!r})")
        elif val < 0:
            errors.append(f"{label} must be >= 0 (got {val})")
    if not _is_num(cfg.figure.crop_padding):
        errors.append(f"figure.crop_padding must be a number (got {cfg.figure.crop_padding!r})")
    elif not (0.0 <= cfg.figure.crop_padding <= 1.0):
        errors.append(
            f"figure.crop_padding must be in [0, 1] (got {cfg.figure.crop_padding})"
        )
    if not _is_int(cfg.figure.context_chars):
        errors.append(f"figure.context_chars must be an integer (got {cfg.figure.context_chars!r})")
    elif cfg.figure.context_chars < 0:
        errors.append(
            f"figure.context_chars must be >= 0 (got {cfg.figure.context_chars})"
        )
    if not isinstance(cfg.table.refine, bool):
        errors.append(f"table.refine must be a boolean (got {cfg.table.refine!r})")

    if errors:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))


# --------------------------------------------------------------------------- #
# Validation — layer (b): path existence (command/stage-aware)
# --------------------------------------------------------------------------- #


def binary_filename(base_name: str, os_name: str | None = None) -> str:
    """Platform executable filename: appends ``.exe`` on Windows (DESIGN §5.2).

    ``os_name`` defaults to the live ``os.name``; tests pass it explicitly so the
    Windows branch can be exercised on POSIX without monkeypatching the global
    ``os.name`` (which would make ``pathlib.Path`` build an unusable ``WindowsPath``).
    """
    if os_name is None:
        os_name = os.name
    return f"{base_name}.exe" if os_name == "nt" else base_name


def find_binary(bin_dir: str, base_name: str, os_name: str | None = None) -> Path | None:
    """Resolve a llama.cpp executable (DESIGN §5.2).

    Appends ``.exe`` on Windows. If ``bin_dir`` is set, look there; otherwise fall
    back to ``shutil.which`` (which honors ``PATHEXT`` on Windows). ``os_name`` is
    forwarded to :func:`binary_filename` (see its note on why it is parameterized).
    """
    name = binary_filename(base_name, os_name=os_name)
    if bin_dir:
        candidate = Path(bin_dir).expanduser() / name
        return candidate if candidate.exists() else None
    found = shutil.which(base_name)
    return Path(found) if found else None


def _check_model_pair(label: str, model: str, mmproj: str, errors: list[str]) -> None:
    for kind, value in (("model", model), ("mmproj", mmproj)):
        if not value:
            errors.append(f"{label}.{kind} is not configured (set it in config or via CLI)")
        elif not Path(value).expanduser().is_file():
            errors.append(f"{label}.{kind} file not found: {value}")


def validate_ocr_paths(cfg: RunConfig) -> None:
    """Validate the llama binary + OCR model/mmproj (run/ocr, before OCR launch).

    Bypassed entirely when ``ocr.endpoint`` is set (no server is spawned).
    """
    if cfg.ocr.endpoint:
        return
    errors: list[str] = []
    if find_binary(cfg.llama.bin_dir, "llama-server") is None:
        errors.append(
            "llama-server binary not found "
            f"(llama.bin_dir={cfg.llama.bin_dir!r}; not on PATH either)"
        )
    _check_model_pair("ocr", cfg.ocr.model, cfg.ocr.mmproj, errors)
    if errors:
        raise ConfigError("OCR configuration error:\n  - " + "\n  - ".join(errors))


def validate_vlm_paths(cfg: RunConfig) -> None:
    """Validate the llama binary + VLM model/mmproj (run/describe, before VLM launch).

    Bypassed when ``vlm.endpoint`` is set.
    """
    if cfg.vlm.endpoint:
        return
    errors: list[str] = []
    if find_binary(cfg.llama.bin_dir, "llama-server") is None:
        errors.append(
            "llama-server binary not found "
            f"(llama.bin_dir={cfg.llama.bin_dir!r}; not on PATH either)"
        )
    _check_model_pair("vlm", cfg.vlm.model, cfg.vlm.mmproj, errors)
    if errors:
        raise ConfigError("VLM configuration error:\n  - " + "\n  - ".join(errors))
