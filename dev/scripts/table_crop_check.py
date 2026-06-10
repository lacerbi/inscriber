"""Cropped-table-input real-hardware check (TODO: cropped table input; DESIGN §9.7).

Validates the table pass's cropped-input path end to end on real models, using
the production code paths (parser, matcher, cropper, prompts). Reusable — the
structure-damage-guard TODO item wants the same probe harness. Two phases:

1. **Crop completeness (OCR only).** OCR each requested page, match every
   ``<table>`` blob to its grounded ``table[[bbox]]`` region
   (``match_table_regions``), and write the page render + each table crop to
   the out dir. *Inspect the crops visually*: a crop that clips rows/columns
   would actively mislead the VLM (the silent-structure-damage failure mode),
   so completeness gates the prompt pin. Raw OCR outputs are saved too — a
   real table page should be captured as a committed parser fixture.

2. **Page-vs-crop VLM comparison** (with ``--vlm-model``/``--vlm-mmproj``).
   For each matched table, run the VLM twice — the validated whole-page
   prompt (baseline) and the cropped variant (candidate) — and write both
   pipe tables side by side for cell-by-cell diffing against the PDF.
   Reference baseline quality: 2 clean / 3 shape-wrong / 5 damaged on the 10
   PriorGuide tables (dev/notes/2026-06-10-e2e-quality-findings.md §Tables).

Servers run sequentially (OCR fully torn down before the VLM starts), like the
pipeline default. Outputs land in out-tablecrop/ (gitignored).

Usage::

    python dev/scripts/table_crop_check.py \
        --bin-dir "C:/Users/luigi/llms" \
        --ocr-model "C:/Users/luigi/llms/models/deepseek-ocr-bf16.gguf" \
        --ocr-mmproj "C:/Users/luigi/llms/models/mmproj-deepseek-ocr-bf16.gguf" \
        --paper %TEMP%/arxiv-2510.13763.pdf --pages 7,27,33,36,37 \
        [--vlm-model ".../gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf" \
         --vlm-mmproj ".../mmproj-BF16.gguf"] \
        [--target 2048]
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import fitz  # noqa: E402  (PyMuPDF)
from PIL import Image  # noqa: E402

from inscriber.llama.client import ChatClient  # noqa: E402
from inscriber.llama.server import LlamaServerManager, ServerSpec  # noqa: E402
from inscriber.models import PageImage, Region, ResolutionMode  # noqa: E402
from inscriber.ocr.base import HttpInferencer  # noqa: E402
from inscriber.ocr.deepseek import DeepSeekOcrBackend  # noqa: E402
from inscriber.pdf.crop import crop_region_bytes  # noqa: E402
from inscriber.postprocess.tables import (  # noqa: E402
    TABLE_CROP_PADDING,
    blob_is_refinable,
    find_table_blobs,
    match_table_regions,
    sanitize_table_output,
    table_page_context,
)
from inscriber.vlm.gemma import GemmaVlmBackend  # noqa: E402

OUT = REPO / "out-tablecrop"


@dataclass
class TableJob:
    page_number: int
    index: int  # 1-based blob index on the page
    blob_count: int  # all blobs on the page (locator material)
    blob: str
    context: str
    region: Region | None  # matched table region (None = whole-page fallback only)
    raster_png: bytes
    crop_png: bytes | None = None


def render(pdf_bytes: bytes, page_number: int, target_px: int) -> PageImage:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page = doc[page_number - 1]
        zoom = target_px / max(page.rect.width, page.rect.height)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        return PageImage(
            page_number=page_number,
            png_bytes=pix.tobytes("png"),
            width_px=pix.width,
            height_px=pix.height,
        )


def png_size(png: bytes) -> str:
    img = Image.open(io.BytesIO(png))
    return f"{img.width}x{img.height}px"


def ocr_phase(args, pages: list[int], pdf: bytes) -> list[TableJob]:
    """OCR each page → match blobs↔regions → write crops + raws; return jobs."""
    backend = DeepSeekOcrBackend()  # production prompt/sampling/flags
    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=args.timeout, log_dir=OUT)
    spec = ServerSpec(
        model=args.ocr_model, mmproj=args.ocr_mmproj, n_gpu_layers=args.ngl,
        ctx_size=args.ctx, extra_flags=backend.server_flags(),
        chat_template=backend.chat_template("server"), label="tablecrop-ocr",
    )
    jobs: list[TableJob] = []
    with mgr.serve(spec) as url:
        inf = HttpInferencer(url)
        for pg_num in pages:
            page = render(pdf, pg_num, args.target)
            (OUT / f"page_p{pg_num}.png").write_bytes(page.png_bytes)
            t0 = time.monotonic()
            result = backend.ocr_page(inf, page, ResolutionMode.GUNDAM)
            secs = time.monotonic() - t0
            raw = getattr(inf, "last_raw", "")
            (OUT / f"page_p{pg_num}_raw.txt").write_text(raw, encoding="utf-8")
            finish = getattr(inf.client, "last_finish_reason", None)

            spans = find_table_blobs(result.markdown)
            refinable = [
                (i, blob) for i, (_, _, blob) in enumerate(spans, 1)
                if blob_is_refinable(blob)
            ]
            n_regions = sum(1 for r in result.regions if r.label.lower() == "table")
            print(
                f"\n--- page {pg_num} ({page.width_px}x{page.height_px}px, {secs:.0f}s, "
                f"finish={finish}) ---"
            )
            print(
                f"  {len(spans)} <table> blob(s), {len(refinable)} refinable, "
                f"{n_regions} grounded table region(s)"
            )
            matches = match_table_regions([b for _, b in refinable], result.regions)
            context = table_page_context(result.markdown)
            for (index, blob), region in zip(refinable, matches, strict=True):
                job = TableJob(
                    page_number=pg_num, index=index, blob_count=len(spans),
                    blob=blob, context=context, region=region, raster_png=page.png_bytes,
                )
                tag = f"table_p{pg_num}_{index}"
                (OUT / f"{tag}_blob.txt").write_text(blob, encoding="utf-8")
                if region is None:
                    print(f"  {tag}: NO MATCH -> whole-page fallback only")
                else:
                    job.crop_png = crop_region_bytes(
                        page.png_bytes, region.bbox_norm, padding=TABLE_CROP_PADDING
                    )
                    if job.crop_png is None:
                        # Probe-only divergence: production keeps the raw blob
                        # here; the harness demotes to whole-page so phase 2
                        # still yields an output to inspect.
                        print(f"  {tag}: degenerate crop (bbox={region.bbox_norm})")
                        job.region = None
                    else:
                        (OUT / f"{tag}_crop.png").write_bytes(job.crop_png)
                        bbox = ", ".join(f"{v:.3f}" for v in region.bbox_norm)
                        print(
                            f"  {tag}: matched [{bbox}] -> crop "
                            f"{png_size(job.crop_png)}  ** INSPECT {tag}_crop.png "
                            f"for completeness **"
                        )
                jobs.append(job)
    return jobs


def vlm_phase(args, jobs: list[TableJob]) -> None:
    """Run each matched table through the VLM both ways; write outputs side by side."""
    backend = GemmaVlmBackend()
    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=args.timeout, log_dir=OUT)
    spec = ServerSpec(
        model=args.vlm_model, mmproj=args.vlm_mmproj, n_gpu_layers=args.vlm_ngl,
        ctx_size=args.ctx, extra_flags=backend.server_flags(),
        chat_template=None, label="tablecrop-vlm",
    )
    with mgr.serve(spec) as url:
        backend.client = ChatClient(url)
        for job in jobs:
            tag = f"table_p{job.page_number}_{job.index}"
            variants: list[tuple[str, bytes, bool]] = [("page", job.raster_png, False)]
            if job.crop_png is not None:
                variants.append(("crop", job.crop_png, True))
            print(f"\n--- {tag} ---")
            for name, image, cropped in variants:
                prompt = backend.build_table_prompt(
                    job.blob, job.context,
                    table_index=job.index, table_count=job.blob_count, cropped=cropped,
                )
                t0 = time.monotonic()
                try:
                    raw = backend.restructure_table(image, prompt)
                except Exception as e:  # noqa: BLE001 - probe script: report, go on
                    print(f"  {name:>4}-input: ERROR {e}")
                    continue
                secs = time.monotonic() - t0
                finish = getattr(backend.client, "last_finish_reason", None)
                table_md = sanitize_table_output(raw)
                out_file = OUT / f"{tag}_{name}.md"
                out_file.write_text(
                    table_md if table_md is not None else f"(REJECTED)\n{raw or ''}",
                    encoding="utf-8",
                )
                rows = len(table_md.splitlines()) if table_md else 0
                status = "OK" if table_md is not None else "REJECTED/truncated"
                print(
                    f"  {name:>4}-input: {secs:5.1f}s finish={finish} "
                    f"sanitize={status} rows={rows} -> {out_file.name}"
                )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--ocr-model", required=True)
    p.add_argument("--ocr-mmproj", required=True)
    p.add_argument("--paper", required=True, help="local PDF with the probe tables")
    p.add_argument("--pages", required=True,
                   help="comma-separated 1-indexed page numbers (e.g. 7,27,33)")
    p.add_argument("--target", type=int, default=2048,
                   help="render long edge in px (default: the gundam 2048)")
    p.add_argument("--vlm-model", default=None,
                   help="optional: also run the page-vs-crop VLM comparison")
    p.add_argument("--vlm-mmproj", default=None)
    p.add_argument("--ngl", default="auto")
    p.add_argument("--vlm-ngl", default="auto")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--timeout", type=float, default=180.0)
    args = p.parse_args()

    # Windows console is cp1252; model output can carry fullwidth glyphs etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    OUT.mkdir(exist_ok=True)
    pages = [int(x) for x in args.pages.split(",") if x.strip()]
    pdf = Path(args.paper).expanduser().read_bytes()

    jobs = ocr_phase(args, pages, pdf)
    matched = [j for j in jobs if j.region is not None]
    print(
        f"\n=== OCR phase done: {len(jobs)} refinable table(s), "
        f"{len(matched)} matched to a grounded region ==="
    )
    if args.vlm_model and args.vlm_mmproj:
        if jobs:
            vlm_phase(args, jobs)
    else:
        print("(no --vlm-model/--vlm-mmproj: skipping the page-vs-crop comparison)")
    print(f"\nartifacts saved under {OUT}/ — diff *_page.md vs *_crop.md against the PDF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
