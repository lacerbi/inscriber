"""OCR bundle — the portable two-step artifact (DESIGN §3.1, §8.5).

The bundle is the inspectable output of ``inscriber ocr`` and the input to
``inscriber describe``. It contains everything needed to run the VLM/assembly
stages later with **no OCR model required**::

    OUT/paper.inscriber-ocr/
    ├── manifest.json     # source meta + OCR config + per-page results
    ├── figures/          # cropped figure PNGs (fig_p{page}_{i}.png)
    └── pages/            # page rasters for pages with tables (page_NNNN.png)

The bundle is a superset of the OCR cache value: per page it adds the post-crop
``figures[]`` and the cropped PNGs (cache = step-3 boundary; bundle = step-4).

Pages containing ``<table>`` blobs also carry their **verbatim** page raster
(``raster_path``) so ``describe`` can run the VLM table-restructuring pass with
no PDF present; verbatim bytes keep table cache keys identical between ``run``
and ``describe``. The field is additive — old readers ignore it, so
``bundle_schema`` stays 1; old bundles without it skip table refinement.

Every page also carries ``raster_sha256`` (the verbatim raster's hash) and the
manifest a top-level ``figure_crop_padding`` (the ocr-time ``[figure]`` knob) —
together the figure-description cache-key material (DESIGN §9.6: keys are
``(raster, bbox, padding)``, immune to PNG-encoder churn), since the bundle
stores no rasters for figure-only pages. Both additive (``bundle_schema``
stays 1); ``describe`` falls back to hashing the stored crop bytes without
them. The crop PNGs themselves are **derived data**: hand-replacing one does
not change the cache key (use ``--refresh`` to recompute descriptions), and
hand-editing a figure's ``bbox_norm`` only re-keys the description without
re-cutting the crop (``describe`` sends the stored crop file; re-run ``ocr``
to change crops).
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
    raster_path: str | None = None  # relative page-raster path (table pages only)
    raster_sha256: str | None = None  # verbatim-raster hash (figure cache key, §9.6)


@dataclass
class Bundle:
    dir: Path
    source: dict
    ocr: dict
    figure_detect: str
    pages: list[BundlePage]
    # The ocr-time [figure].crop_padding — figure cache-key material (§9.6).
    # None on bundles predating the field (describe then keys on crop bytes).
    figure_crop_padding: float | None = None

    @property
    def source_name(self) -> str:
        return self.source.get("name", "paper")

    @property
    def original_url(self) -> str | None:
        """The source URL recorded at ``ocr`` time (BibTeX provenance, DESIGN §12)."""
        return self.source.get("original_url")


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
    page_rasters: dict[int, str] | None = None,
    page_raster_hashes: dict[int, str] | None = None,
    figure_crop_padding: float | None = None,
    created_at: str = "",
) -> Path:
    """Write ``manifest.json`` (crops are expected already saved under ``figures/``;
    likewise ``page_rasters`` maps page numbers to already-saved raster paths).
    ``page_raster_hashes`` + ``figure_crop_padding`` are the figure cache-key
    material (§9.6) — additive fields, omitted when not given.

    Returns the bundle directory.
    """
    bundle_dir = Path(bundle_dir)
    try:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise BundleError(f"could not create bundle directory {bundle_dir}: {e}") from e

    pages_json = []
    for res in page_results:
        figs = page_figures.get(res.page_number, [])
        page_json = {
            **ocr_page_result_to_dict(res),
            "figures": [figure_to_dict(f) for f in figs],
        }
        raster = (page_rasters or {}).get(res.page_number)
        if raster:
            page_json["raster_path"] = raster
        raster_hash = (page_raster_hashes or {}).get(res.page_number)
        if raster_hash:
            page_json["raster_sha256"] = raster_hash
        pages_json.append(page_json)

    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "inscriber_version": __version__,
        "created_at": created_at,
        "source": {"name": base_name, **source},
        "ocr": ocr_meta,
        "figure_detect": figure_detect,
        "pages": pages_json,
    }
    if figure_crop_padding is not None:
        manifest["figure_crop_padding"] = figure_crop_padding
    try:
        (bundle_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as e:
        # An unwritable manifest after a full OCR pass deserves an actionable
        # error, not a traceback (the file may be open in an editor).
        raise BundleError(
            f"could not write bundle manifest {bundle_dir / 'manifest.json'}: {e}"
        ) from e
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
        raster_path = p.get("raster_path")
        if raster_path and not (bundle_dir / raster_path).is_file():
            raise BundleError(f"bundle missing referenced page raster: {raster_path}")
        pages.append(
            BundlePage(
                page_number=p["page_number"],
                markdown=p["markdown"],
                regions=regions,
                figures=figures,
                raster_path=raster_path,
                raster_sha256=p.get("raster_sha256"),
            )
        )

    crop_padding = manifest.get("figure_crop_padding")
    return Bundle(
        dir=bundle_dir,
        source=manifest.get("source", {}),
        ocr=manifest.get("ocr", {}),
        figure_detect=manifest.get("figure_detect", "auto"),
        pages=pages,
        figure_crop_padding=float(crop_padding) if crop_padding is not None else None,
    )
