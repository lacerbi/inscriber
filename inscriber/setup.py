"""`inscriber setup` — model download + config bootstrap (DESIGN §13.4).

Downloads the recommended GGUF pairs from Hugging Face into the platform data
dir and writes (or updates) the platform ``config.toml`` so a fresh install
becomes runnable in one command. Deliberately **outside the pipeline**: no
RunConfig, no servers, no caches — just httpx, hashlib, and the config writer.

Design points (discussed before implementation; see DESIGN §13.4):

* **Pinned content identities.** Every file carries the exact byte size and
  sha256 published by the Hugging Face API (the LFS ``oid``), so downloads are
  tamper-evident and a silent upstream re-upload fails loudly instead of
  producing a config that points at unverified weights.
* **Resumable, atomic downloads.** Streaming goes to ``{name}.part`` with a
  ``Range`` resume; only a size+hash-verified file is promoted (atomic
  ``Path.replace``) to its final name. Re-running ``setup`` is idempotent —
  complete files are verified and skipped, partial ones resumed.
* **The Gemma projector is renamed on disk.** unsloth ships it as the generic
  ``mmproj-BF16.gguf``; it is saved under a Gemma-specific name so model
  folders holding several families stay unambiguous (the README's manual
  advice, baked in).
* **No ``--offline`` interaction.** ``setup`` is inherently online and opt-in;
  it does not read any config (it *writes* one).
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx
import platformdirs

from inscriber.config import default_config_path, find_binary, local_config_path
from inscriber.errors import InscriberError
from inscriber.input.resolver import USER_AGENT
from inscriber.logging import get_logger

try:  # 3.11+ stdlib; tomli is the <3.11 fallback (DESIGN §15)
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised only on <3.11
    import tomli as tomllib  # type: ignore[no-redef]

logger = get_logger()

CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_EVERY_BYTES = 512 * 1024 * 1024
DISK_HEADROOM_BYTES = 1024 * 1024 * 1024  # require 1 GiB beyond the downloads
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, read=120.0)  # read applies per chunk

MANUAL_DOWNLOAD_HINT = (
    "the README's model table has the direct links for a manual download "
    "(then set the [ocr]/[vlm] paths in config.toml yourself)"
)


class SetupError(InscriberError):
    """Raised when a download or the config write cannot be completed."""


# --------------------------------------------------------------------------- #
# Model registry — pinned content identities
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelFile:
    """One GGUF to fetch: where from, what it must hash to, where it lands.

    ``size``/``sha256`` are the Hugging Face API's LFS ``size``/``oid`` for the
    exact upstream revision the README recommends (captured 2026-06-11). If
    upstream re-uploads a file these go stale and the download fails with a
    pointed message — update them together with the README model table.
    """

    role: str  # config key it fills: "ocr.model" | "ocr.mmproj" | "vlm.model" | "vlm.mmproj"
    local_name: str
    url: str
    size: int
    sha256: str


_HF = "https://huggingface.co"

DEEPSEEK_QUANTS: dict[str, tuple[ModelFile, ModelFile]] = {
    "bf16": (
        ModelFile(
            role="ocr.model",
            local_name="deepseek-ocr-bf16.gguf",
            url=f"{_HF}/sabafallah/DeepSeek-OCR-GGUF/resolve/main/deepseek-ocr-bf16.gguf?download=true",
            size=5_876_578_112,
            sha256="b36ae07925d06a3675b3409c8babbd8023c004c6ee2c87137046809e8f74f13c",
        ),
        ModelFile(
            role="ocr.mmproj",
            local_name="mmproj-deepseek-ocr-bf16.gguf",
            url=f"{_HF}/sabafallah/DeepSeek-OCR-GGUF/resolve/main/mmproj-deepseek-ocr-bf16.gguf?download=true",
            size=826_425_472,
            sha256="4caeed8b6c3c7d25dfebccfdb5cf34d6ae540ef4dc4fa2b9842b69cfa50ecbe2",
        ),
    ),
    "q8_0": (
        ModelFile(
            role="ocr.model",
            local_name="DeepSeek-OCR-Q8_0.gguf",
            url=f"{_HF}/ggml-org/DeepSeek-OCR-GGUF/resolve/main/DeepSeek-OCR-Q8_0.gguf?download=true",
            size=3_126_139_712,
            sha256="81ede3e256230707dccf7fa052570c3a939d57db99de655f43cbb1a830d14d92",
        ),
        ModelFile(
            role="ocr.mmproj",
            local_name="mmproj-DeepSeek-OCR-Q8_0.gguf",
            url=f"{_HF}/ggml-org/DeepSeek-OCR-GGUF/resolve/main/mmproj-DeepSeek-OCR-Q8_0.gguf?download=true",
            size=447_856_768,
            sha256="786c9b5159898de3d1d94a102836df559fed0bcf09f41a32f62c3219b0e278e0",
        ),
    ),
}

GEMMA_FILES: tuple[ModelFile, ModelFile] = (
    ModelFile(
        role="vlm.model",
        local_name="gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
        url=f"{_HF}/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf?download=true",
        size=4_215_693_760,
        sha256="b3052f962d6449b4eb2075733c068bdec1c51eadb7b237e6c3157bfbb7b1dae0",
    ),
    # Upstream name is the generic "mmproj-BF16.gguf" — saved Gemma-specific
    # so multi-family model folders stay unambiguous.
    ModelFile(
        role="vlm.mmproj",
        local_name="mmproj-gemma-4-E4B-it-qat-BF16.gguf",
        url=f"{_HF}/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/mmproj-BF16.gguf?download=true",
        size=991_552_320,
        sha256="7c9bafa27f82d658eda805c1d82ef62bb0368e1ff75f64f77de58ad318beaaf9",
    ),
)


def plan_files(deepseek_quant: str) -> tuple[ModelFile, ...]:
    """The four files a setup run manages, for the chosen DeepSeek quant."""
    if deepseek_quant not in DEEPSEEK_QUANTS:
        raise SetupError(
            f"unknown DeepSeek quant {deepseek_quant!r}; "
            f"choose one of {sorted(DEEPSEEK_QUANTS)}"
        )
    return (*DEEPSEEK_QUANTS[deepseek_quant], *GEMMA_FILES)


def default_models_dir() -> Path:
    """Platform data dir for downloaded models (sibling of the §8.6 cache)."""
    return Path(platformdirs.user_data_dir("inscriber")) / "models"


# --------------------------------------------------------------------------- #
# Download — resumable, verified, atomic
# --------------------------------------------------------------------------- #


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB"


def _verify_existing(file: ModelFile, dest: Path) -> bool:
    """True if ``dest`` already holds the pinned content (skip the download)."""
    actual_size = dest.stat().st_size
    if actual_size != file.size:
        raise SetupError(
            f"{dest} already exists but has the wrong size "
            f"({actual_size} bytes, expected {file.size}). Not overwriting a "
            f"file inscriber didn't finish writing — delete it (or use "
            f"--models-dir elsewhere) and re-run setup."
        )
    logger.info("verifying existing %s ...", dest.name)
    if _hash_file(dest) != file.sha256:
        raise SetupError(
            f"{dest} already exists but its content does not match the pinned "
            f"sha256. Delete it (or use --models-dir elsewhere) and re-run "
            f"setup; if this recurs, upstream may have re-uploaded the file — "
            f"{MANUAL_DOWNLOAD_HINT}."
        )
    logger.info("  %s: already downloaded (verified)", dest.name)
    return True


def download_model(
    file: ModelFile,
    dest_dir: Path,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Path:
    """Fetch one model file: verify-and-skip, or resume/stream to ``.part``.

    Only a size+hash-verified file is promoted to its final name; a hash
    mismatch deletes the partial so the retry starts clean.
    """
    dest = dest_dir / file.local_name
    if dest.exists():
        _verify_existing(file, dest)
        return dest

    part = dest_dir / (file.local_name + ".part")
    hasher = hashlib.sha256()
    start = 0
    if part.exists():
        start = part.stat().st_size
        if start > file.size:
            part.unlink()  # cannot be a prefix of the pinned content
            start = 0
        else:
            hasher = hashlib.sha256()
            with open(part, "rb") as f:
                while chunk := f.read(CHUNK_BYTES):
                    hasher.update(chunk)
            if start:
                logger.info("  %s: resuming at %s", file.local_name, _gb(start))

    headers = {"User-Agent": USER_AGENT}
    if start:
        headers["Range"] = f"bytes={start}-"

    try:
        with httpx.Client(
            follow_redirects=True, timeout=DOWNLOAD_TIMEOUT, transport=transport
        ) as client:
            with client.stream("GET", file.url, headers=headers) as resp:
                if start and resp.status_code == 200:
                    # Server ignored the Range; start over.
                    start = 0
                    hasher = hashlib.sha256()
                elif start and resp.status_code == 416:
                    # Nothing left to request — fall through to verification.
                    resp.read()
                elif resp.status_code not in (200, 206):
                    raise SetupError(
                        f"download of {file.local_name} returned HTTP "
                        f"{resp.status_code} ({file.url})"
                    )
                if resp.status_code in (200, 206):
                    mode = "ab" if start else "wb"
                    done = start
                    next_mark = (done // PROGRESS_EVERY_BYTES + 1) * PROGRESS_EVERY_BYTES
                    with open(part, mode) as out:
                        if not start:
                            out.truncate(0)
                        for chunk in resp.iter_bytes(CHUNK_BYTES):
                            out.write(chunk)
                            hasher.update(chunk)
                            done += len(chunk)
                            if done >= next_mark:
                                logger.info(
                                    "  %s: %s / %s",
                                    file.local_name, _gb(done), _gb(file.size),
                                )
                                next_mark += PROGRESS_EVERY_BYTES
    except httpx.HTTPError as e:
        raise SetupError(
            f"download of {file.local_name} failed: {e} — re-run setup to "
            f"resume from {_gb(part.stat().st_size if part.exists() else 0)}"
        ) from e

    actual_size = part.stat().st_size if part.exists() else 0
    if actual_size != file.size:
        raise SetupError(
            f"{file.local_name}: download ended at {actual_size} bytes, "
            f"expected {file.size} — re-run setup to resume"
        )
    if hasher.hexdigest() != file.sha256:
        part.unlink()
        raise SetupError(
            f"{file.local_name}: downloaded content does not match the pinned "
            f"sha256 (partial file deleted). If this recurs, upstream may have "
            f"re-uploaded the file — {MANUAL_DOWNLOAD_HINT}."
        )
    try:
        part.replace(dest)
    except OSError as e:
        # Windows raises PermissionError/WinError 32 when the target is open in
        # another process (a running llama-server with the GGUF mmap'd, etc.).
        raise SetupError(
            f"could not move {part.name} into place: {e} — {dest.name} may be "
            f"open in another program (a running llama-server?); close it and "
            f"re-run setup. The verified download is kept, so the retry is "
            f"instant."
        ) from e
    logger.info("  %s: done (%s)", file.local_name, _gb(file.size))
    return dest


def _check_disk_space(files: list[ModelFile], dest_dir: Path) -> None:
    """Fail early if the missing files can't fit (with 1 GiB headroom)."""
    needed = 0
    for f in files:
        if (dest_dir / f.local_name).exists():
            continue
        part = dest_dir / (f.local_name + ".part")
        needed += f.size - (part.stat().st_size if part.exists() else 0)
    if needed <= 0:
        return
    free = shutil.disk_usage(dest_dir).free
    if free < needed + DISK_HEADROOM_BYTES:
        raise SetupError(
            f"not enough disk space in {dest_dir}: {_gb(needed)} to download "
            f"but only {_gb(free)} free (1 GB headroom required). "
            f"Use --models-dir to pick another location."
        )


# --------------------------------------------------------------------------- #
# Config write — fresh minimal file, or parse-merge-emit update
# --------------------------------------------------------------------------- #


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    raise SetupError(f"cannot re-emit config value of type {type(value).__name__}: {value!r}")


def _emit_sections(data: dict, prefix: str = "") -> list[str]:
    """Re-emit parsed TOML (scalar/list values + nested tables). Lossy: no comments."""
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    for key, value in scalars.items():
        lines.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        full = f"{prefix}.{name}" if prefix else name
        if lines:
            lines.append("")
        lines.append(f"[{full}]")
        lines.extend(_emit_sections(table, full))
    return lines


def _write_config_text(target: Path, content: str) -> None:
    """Write the config, converting a file lock into an actionable SetupError.

    On Windows, an editor (or another process) holding ``target`` open can make
    the write raise PermissionError — surface a hint instead of a traceback.
    """
    try:
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    except OSError as e:
        raise SetupError(
            f"could not write config {target}: {e} — the file may be open in "
            f"another program (an editor that locks it?); close it and re-run "
            f"setup (the downloaded models are kept)."
        ) from e


_FRESH_TEMPLATE = """\
# inscriber configuration — written by `inscriber setup`.
# Only the keys setup manages are listed; everything else uses built-in
# defaults. Full reference: config.example.toml in the repository / README.md.

[llama]
{bin_dir_line}

[ocr]
model = {ocr_model}
mmproj = {ocr_mmproj}

[vlm]
model = {vlm_model}
mmproj = {vlm_mmproj}
"""


def write_setup_config(
    target: Path,
    *,
    model_paths: dict[str, Path],
    llama_bin_dir: str | None,
) -> Path:
    """Write (fresh) or update (parse-merge-emit) ``target`` with the managed keys.

    ``model_paths`` maps the registry roles ("ocr.model", ...) to local files.
    Paths are emitted ``as_posix()`` — forward slashes work on every platform
    and need no TOML escaping. Updating an existing file preserves all keys
    but **not comments** (logged); a fresh file carries a comment header.
    """
    paths = {role: p.as_posix() for role, p in model_paths.items()}
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        bin_dir_line = (
            f"bin_dir = {_toml_str(Path(llama_bin_dir).expanduser().as_posix())}"
            if llama_bin_dir
            else '# bin_dir = ""   # <- set this: folder containing llama-server[.exe]'
        )
        content = _FRESH_TEMPLATE.format(
            bin_dir_line=bin_dir_line,
            ocr_model=_toml_str(paths["ocr.model"]),
            ocr_mmproj=_toml_str(paths["ocr.mmproj"]),
            vlm_model=_toml_str(paths["vlm.model"]),
            vlm_mmproj=_toml_str(paths["vlm.mmproj"]),
        )
        _write_config_text(target, content)
        logger.info("wrote config: %s", target)
        return target

    try:
        with open(target, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise SetupError(
            f"existing config {target} is not valid TOML ({e}); fix or remove "
            f"it and re-run setup"
        ) from e
    for role, value in paths.items():
        section, key = role.split(".", 1)
        data.setdefault(section, {})[key] = value
    if llama_bin_dir:
        data.setdefault("llama", {})["bin_dir"] = (
            Path(llama_bin_dir).expanduser().as_posix()
        )
    header = (
        "# inscriber configuration — model paths updated by `inscriber setup`.\n"
        "# (Comments from the previous file are not preserved.)\n"
    )
    body = "\n".join(_emit_sections(data))
    _write_config_text(target, header + "\n" + body + "\n")
    logger.info("updated config: %s (existing keys preserved; comments are not)", target)
    return target


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_setup(
    *,
    config_path: str | None = None,
    models_dir: str | None = None,
    llama_bin_dir: str | None = None,
    deepseek_quant: str = "bf16",
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    """Download the recommended models and write/update the config.

    Returns the written paths (models + config) for the stdout contract (§16).
    ``transport`` is the test seam — threaded into the httpx client.
    """
    files = list(plan_files(deepseek_quant))
    dest_dir = (
        Path(models_dir).expanduser() if models_dir else default_models_dir()
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = (
        Path(config_path).expanduser() if config_path else default_config_path()
    )

    total = sum(f.size for f in files)
    logger.info(
        "setup: DeepSeek-OCR %s + Gemma 4 E4B (%s total) -> %s",
        deepseek_quant, _gb(total), dest_dir,
    )
    _check_disk_space(files, dest_dir)

    model_paths: dict[str, Path] = {}
    for f in files:
        model_paths[f.role] = download_model(f, dest_dir, transport=transport)

    if llama_bin_dir and find_binary(llama_bin_dir, "llama-server") is None:
        logger.warning(
            "llama-server not found in %s — the path is written to the config "
            "anyway; install llama.cpp (build >= 9587) there, or fix "
            "[llama].bin_dir later", llama_bin_dir,
        )

    write_setup_config(target, model_paths=model_paths, llama_bin_dir=llama_bin_dir)

    if not llama_bin_dir:
        logger.info(
            "one manual step left: download llama.cpp (build >= 9587) from "
            "https://github.com/ggml-org/llama.cpp/releases and set "
            "[llama].bin_dir in %s (or keep llama-server on PATH)", target,
        )
    local = local_config_path()
    if config_path is None and local.is_file() and local != target:
        logger.info(
            "note: %s exists and takes precedence over %s when running from "
            "this directory", local, target,
        )

    return [str(p) for p in (*model_paths.values(), target)]
