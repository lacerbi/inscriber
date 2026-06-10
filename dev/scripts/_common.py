"""Config-backed defaults shared by the dev scripts.

Machine-local paths (the llama.cpp bin dir, GGUF model paths) must never be
committed inside scripts — they live in the developer's gitignored
``config.toml``, discovered exactly like the CLI does (``./config.toml``,
then the platform config dir; DESIGN §13.1). Scripts declare the standard
flags with ``default=None`` and call :func:`fill_from_config` after parsing;
a flag always overrides the config file.

Importing this module also bootstraps ``sys.path`` so the scripts can import
``inscriber`` when run as ``python dev/scripts/<name>.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from inscriber.config import load_config_file  # noqa: E402

# argparse dest -> (config section, key)
CONFIG_KEYS = {
    "bin_dir": ("llama", "bin_dir"),
    "ocr_model": ("ocr", "model"),
    "ocr_mmproj": ("ocr", "mmproj"),
    "vlm_model": ("vlm", "model"),
    "vlm_mmproj": ("vlm", "mmproj"),
}


def fill_from_config(args, *, require: tuple[str, ...] = ()) -> None:
    """Fill ``None`` bin/model attrs on ``args`` from the discovered config.

    ``require`` names the attrs that must be set afterwards; anything still
    missing exits with a message naming both the flag and the config knob.
    ``args.config`` (when the script defines a ``--config`` flag) selects an
    explicit file, mirroring the CLI's ``-c/--config``.
    """
    file_dict, path = load_config_file(getattr(args, "config", None))
    missing: list[str] = []
    for dest, (section, key) in CONFIG_KEYS.items():
        if not hasattr(args, dest):
            continue
        if not getattr(args, dest):
            setattr(args, dest, (file_dict.get(section) or {}).get(key) or None)
        if dest in require and not getattr(args, dest):
            missing.append(
                f"--{dest.replace('_', '-')}  (or [{section}] {key} in config.toml)"
            )
    if missing:
        where = str(path) if path else "none found (./config.toml, then the platform config dir)"
        raise SystemExit(
            "missing required settings — pass the flag or set the config key:\n  "
            + "\n  ".join(missing)
            + f"\nconfig file searched: {where}"
        )
