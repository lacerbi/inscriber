"""OCR bundle — the portable two-step artifact (DESIGN §3.1, §8.5).

The bundle is the inspectable output of ``inscriber ocr`` and the input to
``inscriber describe``. It contains everything needed to run the VLM/assembly
stages later with **no OCR model required**::

    OUT/paper.inscriber-ocr/
    ├── manifest.json     # source meta + OCR config + per-page results
    ├── figures/          # cropped figure PNGs (fig_p{page}_{i}.png)
    └── pages/            # optional page rasters (--keep-intermediates)

The bundle is a superset of the OCR cache value: per page it adds the post-crop
``figures[]`` and the cropped PNGs (cache = step-3 boundary; bundle = step-4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from inscriber import __version__
from inscriber.errors import InscriberError
from inscriber.models import Figure, OcrPageResult, Region
from inscriber.serialize import (
    figure_from_dict,
    figure_to_dict,
    ocr_page_result_to_dict,
    region_from_dict,
)

BUNDLE_SCHEMA = 1  # the compatibility gate (DESIGN §8.5)
BUNDLE_SUFFIX = ".inscriber-ocr"


class BundleError(InscriberError):
    """Raised on a malformed/incompatible bundle or a missing crop."""


@dataclass
class BundlePage:
    page_number: int
    markdown: str
    regions: list[Region] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)


@dataclass
class Bundle:
    dir: Path
    source: dict
    ocr: dict
    figure_detect: str
    pages: list[BundlePage]

    @property
    def source_name(self) -> str:
        return self.source.get("name", "paper")


def bundle_dir_for(out_dir: str | Path, base_name: str) -> Path:
    return Path(out_dir) / f"{base_name}{BUNDLE_SUFFIX}"


def write_bundle(
    bundle_dir: Path,
    *,
    base_name: str,
    source: dict,
    ocr_meta: dict,
    figure_detect: str,
    page_results: list[OcrPageResult],
    page_figures: dict[int, list[Figure]],
    created_at: str = "",
) -> Path:
    """Write ``manifest.json`` (crops are expected already saved under ``figures/``).

    Returns the bundle directory.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    pages_json = []
    for res in page_results:
        figs = page_figures.get(res.page_number, [])
        pages_json.append(
            {
                **ocr_page_result_to_dict(res),
                "figures": [figure_to_dict(f) for f in figs],
            }
        )

    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "inscriber_version": __version__,
        "created_at": created_at,
        "source": {"name": base_name, **source},
        "ocr": ocr_meta,
        "figure_detect": figure_detect,
        "pages": pages_json,
    }
    (bundle_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return bundle_dir


def read_bundle(bundle_dir: str | Path) -> Bundle:
    """Load + validate a bundle (DESIGN §8.5).

    Refuses a ``bundle_schema`` higher than supported (never silently misparse) and
    validates that every referenced crop file exists.
    """
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / "manifest.json"
    if not manifest_path.is_file():
        raise BundleError(f"not an inscriber OCR bundle (no manifest.json): {bundle_dir}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise BundleError(f"invalid manifest.json: {e}") from e

    schema = manifest.get("bundle_schema")
    if not isinstance(schema, int):
        raise BundleError("manifest.json missing integer bundle_schema")
    if schema > BUNDLE_SCHEMA:
        raise BundleError(
            f"bundle_schema {schema} is newer than supported ({BUNDLE_SCHEMA}); "
            "upgrade inscriber to read this bundle"
        )

    pages: list[BundlePage] = []
    for p in manifest.get("pages", []):
        regions = [region_from_dict(r) for r in p.get("regions", [])]
        figures = [figure_from_dict(f) for f in p.get("figures", [])]
        for f in figures:
            if f.crop_path and not (bundle_dir / f.crop_path).is_file():
                raise BundleError(f"bundle missing referenced crop: {f.crop_path}")
        pages.append(
            BundlePage(
                page_number=p["page_number"],
                markdown=p["markdown"],
                regions=regions,
                figures=figures,
            )
        )

    return Bundle(
        dir=bundle_dir,
        source=manifest.get("source", {}),
        ocr=manifest.get("ocr", {}),
        figure_detect=manifest.get("figure_detect", "auto"),
        pages=pages,
    )
