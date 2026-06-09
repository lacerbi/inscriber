"""Pipeline orchestrator (DESIGN §3, §3.1).

Three entry points mirror the three subcommands:

* :func:`run`        — full pipeline (OCR → describe → write) in one process.
* :func:`run_ocr`    — resolve → rasterize → OCR → crop → write OCR bundle, stop.
* :func:`describe`   — OCR bundle → VLM describe → assemble → write.

Key design decision (DESIGN §3): sequential, single-model-resident inference — the
OCR server is fully torn down before the VLM server starts (default). ``run`` is
``ocr``-then-``describe`` sharing in-memory objects (no bundle I/O).

M2 emits a minimal ``{base}.md`` (figures injected); the polished
main/appendix/backmatter split set is delivered in M3.
"""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from inscriber.bibtex.semantic_scholar import generate_bibtex
from inscriber.bundle import bundle_dir_for, read_bundle, write_bundle
from inscriber.cache import (
    OcrCache,
    VlmCache,
    file_identity,
    make_ocr_key,
    make_vlm_key,
    sha256_bytes,
)
from inscriber.config import ConfigError, find_binary
from inscriber.input.resolver import resolve_input
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager, ServerSpec, endpoint_or_serve
from inscriber.logging import get_logger
from inscriber.models import (
    Figure,
    OcrPageResult,
    PageImage,
    ResolutionMode,
    ResolvedInput,
    RunConfig,
)
from inscriber.ocr.base import HttpInferencer
from inscriber.ocr.registry import get_ocr_backend
from inscriber.output import (
    copy_figures,
    sanitize_base_name,
    write_full_document,
    write_split_documents,
    write_text_file,
)
from inscriber.pdf.crop import crop_figures
from inscriber.pdf.figures import select_figure_regions
from inscriber.pdf.rasterize import rasterize
from inscriber.postprocess.inject import (
    PLACEHOLDER_RE,
    ensure_placeholders,
    inject_descriptions,
)
from inscriber.postprocess.prompt import build_page_context
from inscriber.postprocess.splitter import (
    extract_title,
    prepare_formatted_sections,
    split_markdown_content,
)
from inscriber.postprocess.stitch import (
    dehyphenate,
    ensure_image_description_spacing,
    normalize_line_breaks,
    stitch_pages,
    strip_running_headers_footers,
)
from inscriber.vlm.registry import get_vlm_backend

logger = get_logger()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


@dataclass
class _Page:
    """Per-page working state for the describe/assemble stages."""

    page_number: int
    markdown: str  # with figure placeholders ensured
    page_text: str  # placeholders stripped, for figure context
    figures: list[Figure] = field(default_factory=list)


@contextmanager
def _workdir(cfg: RunConfig):
    """Per-run work dir (server logs, intermediate rasters/crops) (DESIGN §15).

    Deleted on success (unless ``--keep-intermediates``); **kept on failure** for
    debugging. An explicit ``--workdir`` is user-managed and never auto-deleted.
    """
    explicit = bool(cfg.workdir.path)
    if explicit:
        path = Path(cfg.workdir.path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
    else:
        path = Path(tempfile.mkdtemp(prefix="inscriber-"))
    ok = False
    try:
        yield path
        ok = True
    finally:
        if explicit:
            pass
        elif ok and not cfg.workdir.keep_intermediates:
            shutil.rmtree(path, ignore_errors=True)
        elif not ok:
            logger.warning("keeping work dir for debugging: %s", path)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_ocr_backend(cfg: RunConfig):
    # Grounding prompt only for auto/grounding; none and pdf-embedded use plain OCR.
    figures_enabled = cfg.figure.detect in ("auto", "grounding")
    return get_ocr_backend(cfg.ocr.backend, figures_enabled=figures_enabled)


def _ocr_identities(cfg: RunConfig, cache: OcrCache) -> tuple[str, str]:
    """Stable model/mmproj identities for the OCR cache key (DESIGN §8.6)."""
    if cfg.ocr.endpoint:
        return f"endpoint:{cfg.ocr.model}", f"endpoint:{cfg.ocr.mmproj}"
    for label, path in (("ocr.model", cfg.ocr.model), ("ocr.mmproj", cfg.ocr.mmproj)):
        if not path or not Path(path).expanduser().is_file():
            raise ConfigError(f"{label} file not found: {path}")
    # Under --no-cache, don't persist the hash sidecar either (no cache writes at all).
    disk = cache.hash_disk_cache if cache.enabled else None
    return (
        file_identity(cfg.ocr.model, hash_disk_cache=disk),
        file_identity(cfg.ocr.mmproj, hash_disk_cache=disk),
    )


def run_ocr_pass(
    cfg: RunConfig, resolved: ResolvedInput, work_dir: str | Path
) -> tuple[list[PageImage], list[OcrPageResult]]:
    """OCR every selected page → ``(pages, results)`` (DESIGN §8, §8.6).

    Consults the per-page cache first; only launches the OCR server when at least
    one page is uncached. Resilient: a page that errors becomes an empty page.
    """
    backend = _build_ocr_backend(cfg)
    mode = ResolutionMode(cfg.ocr.resolution)

    logger.info("rasterizing pages (resolution=%s)…", mode.value)
    pages = rasterize(resolved.pdf_bytes, mode, pages=cfg.pages)
    logger.info("rasterized %d page(s)", len(pages))

    cache = OcrCache(enabled=cfg.cache.enabled, refresh=cfg.cache.refresh)
    pdf_hash = sha256_bytes(resolved.pdf_bytes)
    model_identity, mmproj_identity = _ocr_identities(cfg, cache)
    prompt = backend.prompt()
    # The cache key's sampling includes max_tokens (a hard generation guard, §8.6).
    sampling = {**backend.sampling(), "max_tokens": backend.max_tokens()}

    keys: dict[int, str] = {}
    results_by_page: dict[int, OcrPageResult] = {}
    todo: list[PageImage] = []
    for pg in pages:
        key = make_ocr_key(
            pdf_hash=pdf_hash,
            page_number=pg.page_number,
            backend_name=backend.name,
            model_identity=model_identity,
            mmproj_identity=mmproj_identity,
            resolution_mode=mode.value,
            render_long_edge_px=mode.long_edge_px,
            prompt=prompt,
            sampling=sampling,
        )
        keys[pg.page_number] = key
        cached = cache.get(key)
        if cached is not None:
            logger.info("OCR page %d: cache hit", pg.page_number)
            results_by_page[pg.page_number] = cached
        else:
            todo.append(pg)

    if todo:
        if not cfg.ocr.endpoint and find_binary(cfg.llama.bin_dir, "llama-server") is None:
            raise ConfigError(
                "llama-server binary not found "
                f"(llama.bin_dir={cfg.llama.bin_dir!r}; not on PATH either)"
            )
        mgr = LlamaServerManager(
            cfg.llama.bin_dir,
            server_start_timeout=cfg.llama.server_start_timeout,
            log_dir=work_dir,
        )
        spec = ServerSpec(
            model=cfg.ocr.model,
            mmproj=cfg.ocr.mmproj,
            host=cfg.llama.host,
            port=cfg.llama.port,
            ctx_size=cfg.llama.ctx_size,
            n_gpu_layers=cfg.ocr.n_gpu_layers,
            extra_flags=backend.server_flags(),
            chat_template=backend.chat_template("server"),
            label="ocr",
        )
        with endpoint_or_serve(mgr, cfg.ocr.endpoint, spec) as url:
            inf = HttpInferencer(url)
            for i, pg in enumerate(todo, 1):
                logger.info("OCR page %d/%d (doc page %d)…", i, len(todo), pg.page_number)
                try:
                    res = backend.ocr_page(inf, pg, mode)
                except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
                    logger.warning(
                        "OCR failed on page %d: %s; emitting empty page", pg.page_number, e
                    )
                    res = OcrPageResult(page_number=pg.page_number, markdown="", regions=[])
                cache.put(keys[pg.page_number], res, raw_output=getattr(inf, "last_raw", ""))
                results_by_page[pg.page_number] = res
    else:
        logger.info(
            "all %d page(s) served from OCR cache; OCR server not launched", len(pages)
        )

    results = [results_by_page[pg.page_number] for pg in pages]
    return pages, results


def _crop_pages(
    cfg: RunConfig,
    backend,
    resolved_pdf: bytes,
    pages: list[PageImage],
    results: list[OcrPageResult],
    figures_dir: Path,
) -> list[_Page]:
    """Detect + crop figures per page; return working pages with placeholders ensured."""
    out: list[_Page] = []
    for pg, res in zip(pages, results, strict=True):
        regions = select_figure_regions(
            cfg.figure.detect,
            supports_grounding=backend.supports_grounding,
            result=res,
            pdf_bytes=resolved_pdf,
            page_number=pg.page_number,
        )
        figs = crop_figures(
            pg, regions, crop_padding=cfg.figure.crop_padding, figures_dir=figures_dir
        )
        md = ensure_placeholders(res.markdown, figs)
        out.append(
            _Page(
                page_number=pg.page_number,
                markdown=md,
                page_text=PLACEHOLDER_RE.sub("", md).strip(),
                figures=figs,
            )
        )
    return out


def _vlm_identities(cfg: RunConfig, vlm_cache: VlmCache) -> tuple[str, str]:
    if cfg.vlm.endpoint:
        return f"endpoint:{cfg.vlm.model}", f"endpoint:{cfg.vlm.mmproj}"
    for label, path in (("vlm.model", cfg.vlm.model), ("vlm.mmproj", cfg.vlm.mmproj)):
        if not path or not Path(path).expanduser().is_file():
            raise ConfigError(f"{label} file not found: {path}")
    hash_cache = (vlm_cache.dir / "hashes.json") if vlm_cache.enabled else None
    return (
        file_identity(cfg.vlm.model, hash_disk_cache=hash_cache),
        file_identity(cfg.vlm.mmproj, hash_disk_cache=hash_cache),
    )


def _vlm_describe(
    cfg: RunConfig,
    pages: list[_Page],
    crop_base: Path,
    work_dir: Path,
    *,
    vlm_endpoint: str | None = None,
) -> dict[str, str]:
    """Describe every figure (DESIGN §9). Cache-first; launch VLM server only on miss.

    ``vlm_endpoint`` (concurrent mode) points at an already-running VLM server so no
    second server is launched.
    """
    tasks: list[tuple[Figure, str, bytes]] = []
    for pg in pages:
        context = build_page_context(pg.page_number, pg.page_text, cfg.figure.context_chars)
        for fig in pg.figures:
            if not fig.crop_path:
                continue
            crop_bytes = (Path(crop_base) / fig.crop_path).read_bytes()
            tasks.append((fig, context, crop_bytes))
    if not tasks:
        return {}

    proto = get_vlm_backend(cfg.vlm.backend)  # client=None — used for prompts/keys
    vlm_cache = VlmCache(enabled=cfg.cache.enabled, refresh=cfg.cache.refresh)
    model_id, mmproj_id = _vlm_identities(cfg, vlm_cache)

    descriptions: dict[str, str] = {}
    keys: dict[str, str] = {}
    todo: list[tuple[Figure, str, bytes]] = []
    for fig, context, crop_bytes in tasks:
        prompt = proto.build_prompt(context)
        key = make_vlm_key(
            figure_crop_hash=sha256_bytes(crop_bytes),
            vlm_backend_name=proto.name,
            vlm_model_identity=model_id,
            vlm_mmproj_identity=mmproj_id,
            full_assembled_prompt=prompt,
            sampling=proto.sampling(),
        )
        keys[fig.id] = key
        cached = vlm_cache.get(key)
        if cached is not None:
            logger.info("figure %s: description cache hit", fig.id)
            descriptions[fig.id] = cached
        else:
            todo.append((fig, context, crop_bytes))

    if todo:
        endpoint = vlm_endpoint or cfg.vlm.endpoint
        if not endpoint and find_binary(cfg.llama.bin_dir, "llama-server") is None:
            raise ConfigError(
                "llama-server binary not found "
                f"(llama.bin_dir={cfg.llama.bin_dir!r}; not on PATH either)"
            )
        mgr = LlamaServerManager(
            cfg.llama.bin_dir,
            server_start_timeout=cfg.llama.server_start_timeout,
            log_dir=work_dir,
        )
        spec = ServerSpec(
            model=cfg.vlm.model,
            mmproj=cfg.vlm.mmproj,
            host=cfg.llama.host,
            port=cfg.llama.port,
            ctx_size=cfg.llama.ctx_size,
            n_gpu_layers=cfg.vlm.n_gpu_layers,
            extra_flags=proto.server_flags(),
            chat_template=None,
            label="vlm",
        )
        with endpoint_or_serve(mgr, endpoint, spec) as url:
            backend = get_vlm_backend(cfg.vlm.backend, client=ChatClient(url))
            for i, (fig, context, crop_bytes) in enumerate(todo, 1):
                logger.info("describing figure %d/%d (%s)…", i, len(todo), fig.id)
                try:
                    desc = backend.describe(crop_bytes, context)
                except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
                    logger.warning("figure %s description failed: %s", fig.id, e)
                    desc = ""  # injection → [figure description unavailable]
                if desc:
                    vlm_cache.put(keys[fig.id], desc)
                descriptions[fig.id] = desc
    return descriptions


def _assemble(cfg: RunConfig, pages: list[_Page], descriptions: dict[str, str]) -> str:
    """Stitch + clean + inject descriptions into the full document (DESIGN §10)."""
    figures_by_id = {f.id: f for pg in pages for f in pg.figures}

    # (b) new cleanup on per-page text: strip recurring headers/footers (DESIGN §10.3b).
    page_texts = [pg.markdown for pg in pages]
    if cfg.output.clean:
        page_texts = strip_running_headers_footers(page_texts)

    combined = stitch_pages(
        list(zip([pg.page_number for pg in pages], page_texts, strict=True)),
        page_numbers=cfg.output.page_numbers,
        page_separators=cfg.output.page_separators,
        normalize=cfg.output.normalize_line_breaks,
    )
    if cfg.output.clean:
        combined = dehyphenate(combined)

    # Inject figure descriptions (or strip placeholders when figures are off).
    if cfg.figure.detect == "none":
        combined = PLACEHOLDER_RE.sub("", combined)
    else:
        combined = inject_descriptions(
            combined, descriptions=descriptions, figures=figures_by_id, mode=cfg.figure.mode
        )

    # (a) ported cleanup: spacing around descriptions + final blank-line collapse.
    combined = ensure_image_description_spacing(combined)
    if cfg.output.normalize_line_breaks:
        combined = normalize_line_breaks(combined)
    return combined.strip() + "\n"


def _bibtex_outputs(
    cfg: RunConfig, full_md: str, out_dir: Path, base: str
) -> tuple[str | None, list[str]]:
    """Fetch BibTeX (opt-in, online) → write ``{base}.bib``; optionally a fenced
    block to prepend into the document (DESIGN §12). Never fails the run."""
    if not cfg.bibtex.enabled:
        return None, []
    if cfg.net.offline:
        logger.warning("--offline: skipping BibTeX (requires network)")
        return None, []
    title = extract_title(full_md)
    logger.info("fetching BibTeX for: %s", title)
    bibtex = generate_bibtex(title)
    bib_path = write_text_file(out_dir / f"{base}.bib", bibtex + "\n", clobber=cfg.output.clobber)
    block = f"```\n{bibtex}\n```\n\n---\n\n" if cfg.bibtex.append_to_document else None
    return block, [str(bib_path)]


def _write_documents(
    cfg: RunConfig, base: str, full_md: str, out_dir: Path, *, bibtex_block: str | None = None
) -> list[str]:
    """Write the full doc (always) + main/appendix/backmatter when split (DESIGN §14).

    When ``bibtex_block`` is given (``--bibtex-in-doc``), it is prepended to the
    full document and the main split (DESIGN §12 — full/main/allparts only).
    """
    full_out = (bibtex_block + full_md) if bibtex_block else full_md
    written: list[Path] = [write_full_document(out_dir, base, full_out, clobber=cfg.output.clobber)]
    if cfg.output.split:
        sections = split_markdown_content(full_md)  # split the clean doc, not the prepended one
        main, appendix, backmatter = prepare_formatted_sections(sections)
        if bibtex_block:
            main = bibtex_block + main
        written += write_split_documents(
            out_dir, base, main=main, appendix=appendix, backmatter=backmatter,
            clobber=cfg.output.clobber,
        )
    return [str(p) for p in written]


def _ocr_meta(cfg: RunConfig, backend, model_id: str, mmproj_id: str) -> dict:
    mode = ResolutionMode(cfg.ocr.resolution)
    return {
        "backend": cfg.ocr.backend,
        "model_identity": model_id,
        "mmproj_identity": mmproj_id,
        "resolution": cfg.ocr.resolution,
        "render_long_edge_px": mode.long_edge_px,
        "prompt": backend.prompt(),
        "sampling": backend.sampling(),
    }


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #


def run_ocr(cfg: RunConfig) -> list[str]:
    """OCR-only: produce an inspectable OCR bundle (DESIGN §3.1, §8.5)."""
    resolved = resolve_input(cfg.input, offline=cfg.net.offline)
    base = sanitize_base_name(resolved.suggested_name)
    out_dir = Path(cfg.output.dir)
    bdir = bundle_dir_for(out_dir, base)
    figures_dir = bdir / "figures"
    backend = _build_ocr_backend(cfg)

    with _workdir(cfg) as work:
        pages, results = run_ocr_pass(cfg, resolved, work)
        cache = OcrCache(enabled=cfg.cache.enabled, refresh=cfg.cache.refresh)
        model_id, mmproj_id = _ocr_identities(cfg, cache)
        working = _crop_pages(cfg, backend, resolved.pdf_bytes, pages, results, figures_dir)
        page_results = [
            OcrPageResult(page_number=p.page_number, markdown=p.markdown, regions=r.regions)
            for p, r in zip(working, results, strict=True)
        ]
        page_figures = {p.page_number: p.figures for p in working}
        source = {
            "source": resolved.source,
            "original_url": resolved.original_url,
            "pdf_sha256": sha256_bytes(resolved.pdf_bytes),
        }
        write_bundle(
            bdir,
            base_name=base,
            source=source,
            ocr_meta=_ocr_meta(cfg, backend, model_id, mmproj_id),
            figure_detect=cfg.figure.detect,
            page_results=page_results,
            page_figures=page_figures,
            created_at=_now(),
        )
    logger.info("wrote OCR bundle: %s", bdir)
    return [str(bdir)]


def describe(cfg: RunConfig) -> list[str]:
    """Describe a previously produced OCR bundle (DESIGN §3.1, §8.5)."""
    bundle = read_bundle(cfg.input)
    base = sanitize_base_name(bundle.source_name)
    out_dir = Path(cfg.output.dir)
    pages = [
        _Page(
            page_number=p.page_number,
            markdown=p.markdown,
            page_text=PLACEHOLDER_RE.sub("", p.markdown).strip(),
            figures=p.figures,
        )
        for p in bundle.pages
    ]

    with _workdir(cfg) as work:
        descriptions: dict[str, str] = {}
        if cfg.figure.detect != "none" and cfg.figure.mode != "placeholder":
            descriptions = _vlm_describe(cfg, pages, bundle.dir, work)
        full_md = _assemble(cfg, pages, descriptions)
        bibtex_block, bib_written = _bibtex_outputs(cfg, full_md, out_dir, base)
        written = _write_documents(cfg, base, full_md, out_dir, bibtex_block=bibtex_block)
        written += bib_written
        if cfg.figure.mode == "describe-and-keep" and cfg.figure.detect != "none":
            all_figs = [f for pg in pages for f in pg.figures]
            written += [str(p) for p in copy_figures(all_figs, src_base=bundle.dir, out_dir=out_dir)]
    return written


def _run_body(cfg: RunConfig, resolved, base, out_dir, work, *, vlm_endpoint=None) -> list[str]:
    backend = _build_ocr_backend(cfg)
    needs_vlm = cfg.figure.detect != "none" and cfg.figure.mode != "placeholder"

    pages, results = run_ocr_pass(cfg, resolved, work)
    figures_dir = Path(work) / "figures"
    working = _crop_pages(cfg, backend, resolved.pdf_bytes, pages, results, figures_dir)

    descriptions: dict[str, str] = {}
    if needs_vlm:
        descriptions = _vlm_describe(cfg, working, work, work, vlm_endpoint=vlm_endpoint)
    full_md = _assemble(cfg, working, descriptions)
    bibtex_block, bib_written = _bibtex_outputs(cfg, full_md, out_dir, base)
    written = _write_documents(cfg, base, full_md, out_dir, bibtex_block=bibtex_block)
    written += bib_written
    if cfg.figure.mode == "describe-and-keep" and cfg.figure.detect != "none":
        all_figs = [f for pg in working for f in pg.figures]
        written += [str(p) for p in copy_figures(all_figs, src_base=work, out_dir=out_dir)]
    return written


def run(cfg: RunConfig) -> list[str]:
    """Full end-to-end pipeline (DESIGN §3) — ocr-then-describe, no bundle I/O.

    Sequential by default (one model resident at a time). In ``concurrent`` mode
    (DESIGN §5.4) the VLM server is pre-launched so its load overlaps the OCR pass —
    both models are resident at once (the documented VRAM caveat).
    """
    resolved = resolve_input(cfg.input, offline=cfg.net.offline)
    base = sanitize_base_name(resolved.suggested_name)
    out_dir = Path(cfg.output.dir)
    needs_vlm = cfg.figure.detect != "none" and cfg.figure.mode != "placeholder"

    with _workdir(cfg) as work:
        if cfg.inference.mode == "concurrent" and needs_vlm and not cfg.vlm.endpoint:
            # Pre-launch the VLM server alongside OCR (VRAM caveat). The OCR cache is
            # still consulted inside run_ocr_pass before any OCR server launches.
            for label, path in (("vlm.model", cfg.vlm.model), ("vlm.mmproj", cfg.vlm.mmproj)):
                if not path or not Path(path).expanduser().is_file():
                    raise ConfigError(f"{label} file not found: {path}")
            if find_binary(cfg.llama.bin_dir, "llama-server") is None:
                raise ConfigError(
                    f"llama-server binary not found (llama.bin_dir={cfg.llama.bin_dir!r})"
                )
            logger.info("concurrent mode: pre-launching VLM server alongside OCR")
            proto = get_vlm_backend(cfg.vlm.backend)
            vlm_mgr = LlamaServerManager(
                cfg.llama.bin_dir,
                server_start_timeout=cfg.llama.server_start_timeout,
                log_dir=work,
            )
            vlm_spec = ServerSpec(
                model=cfg.vlm.model, mmproj=cfg.vlm.mmproj, host=cfg.llama.host,
                port=cfg.llama.port, ctx_size=cfg.llama.ctx_size,
                n_gpu_layers=cfg.vlm.n_gpu_layers, extra_flags=proto.server_flags(),
                chat_template=None, label="vlm",
            )
            with vlm_mgr.serve(vlm_spec) as vlm_url:
                return _run_body(cfg, resolved, base, out_dir, work, vlm_endpoint=vlm_url)
        return _run_body(cfg, resolved, base, out_dir, work)
