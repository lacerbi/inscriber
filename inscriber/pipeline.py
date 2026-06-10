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
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from inscriber.bibtex.chain import citable_provenance, generate_bibtex_auto
from inscriber.bibtex.probe import ProbeResult, parse_probe_response
from inscriber.bibtex.semantic_scholar import generate_bibtex
from inscriber.bundle import bundle_dir_for, read_bundle, write_bundle
from inscriber.cache import (
    OcrCache,
    VlmCache,
    file_identity,
    make_bibtex_probe_key,
    make_ocr_key,
    make_table_key,
    make_vlm_key,
    sha256_bytes,
)
from inscriber.config import ConfigError, find_binary
from inscriber.input.resolver import resolve_input
from inscriber.llama.client import ChatClient
from inscriber.llama.server import (
    LlamaServerManager,
    ServerSpec,
    build_number,
    endpoint_or_serve,
    llama_build_identity,
)
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
from inscriber.postprocess.notice import append_transcription_notice
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
from inscriber.postprocess.tables import (
    blob_is_refinable,
    find_table_blobs,
    sanitize_table_output,
    splice_tables,
    table_page_context,
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
    raster_png: bytes | None = None  # verbatim page render (table restructuring input)


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


def _ocr_identities(cfg: RunConfig, cache: OcrCache) -> tuple[str, str, str]:
    """Stable model/mmproj/server identities for the OCR cache key (DESIGN §8.6).

    The server identity (the llama.cpp build) is probed via ``--version`` — no
    server launch — or, in endpoint mode, from the running server's ``/props``.
    """
    if cfg.ocr.endpoint:
        # Key on the endpoint URL too — the model path may be empty/unchanged while
        # the endpoint serves a different model, which would otherwise collide.
        ep = cfg.ocr.endpoint
        server_id = llama_build_identity(cfg.llama.bin_dir, endpoint=ep)
        return f"endpoint:{ep}:{cfg.ocr.model}", f"endpoint:{ep}:{cfg.ocr.mmproj}", server_id
    for label, path in (("ocr.model", cfg.ocr.model), ("ocr.mmproj", cfg.ocr.mmproj)):
        if not path or not Path(path).expanduser().is_file():
            raise ConfigError(f"{label} file not found: {path}")
    # Under --no-cache, don't persist the hash sidecar either (no cache writes at all).
    disk = cache.hash_disk_cache if cache.enabled else None
    return (
        file_identity(cfg.ocr.model, hash_disk_cache=disk),
        file_identity(cfg.ocr.mmproj, hash_disk_cache=disk),
        llama_build_identity(cfg.llama.bin_dir),
    )


def _check_server_build(backend, server_identity: str) -> None:
    """Enforce the backend's minimum llama.cpp build (DESIGN §2.2/§8.2).

    Model-side preprocessing changes across builds — DeepSeek-OCR's grounding
    coordinate frame switched from padded-square to per-axis between builds 9028
    and 9587 — so running a backend against an older server than its pinned
    behavior silently corrupts output (shifted figure crops). An unparseable
    identity (an endpoint without ``/props`` ``build_info``) warns instead of
    blocking: the user manages that server.
    """
    min_build = getattr(backend, "min_server_build", None)
    if not min_build:
        return
    num = build_number(server_identity)
    if num is None:
        logger.warning(
            "cannot determine the llama.cpp build from %r; backend %r requires "
            "build >= %d — grounding coordinates may be wrong on an older server",
            server_identity, backend.name, min_build,
        )
        return
    if num < min_build:
        raise ConfigError(
            f"llama.cpp build {num} is too old for OCR backend {backend.name!r} "
            f"(requires >= {min_build}: the grounding coordinate frame changed "
            f"upstream — see dev/docs/build-9587-verification.md). "
            f"Use a llama.cpp build >= {min_build} for the OCR server "
            f"(llama.bin_dir, or the server behind --ocr-endpoint)."
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
    model_identity, mmproj_identity, server_identity = _ocr_identities(cfg, cache)
    _check_server_build(backend, server_identity)
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
            server_identity=server_identity,
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
                    # Only cache SUCCESSFUL pages — a transient failure must not
                    # poison the cache with an empty page (recoverable only via
                    # --refresh otherwise).
                    cache.put(keys[pg.page_number], res, raw_output=getattr(inf, "last_raw", ""))
                except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
                    logger.warning(
                        "OCR failed on page %d: %s; emitting empty page (not cached)",
                        pg.page_number, e,
                    )
                    res = OcrPageResult(page_number=pg.page_number, markdown="", regions=[])
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


def _vlm_configured(cfg: RunConfig) -> bool:
    """Whether any VLM is set up at all (endpoint or a model/mmproj pair)."""
    return bool(cfg.vlm.endpoint or (cfg.vlm.model and cfg.vlm.mmproj))


class _VlmSession:
    """Lazily-launched VLM server shared by the table and figure passes.

    Owns the ONE backend instance (``proto``), the shared :class:`VlmCache`, and
    the model identities — the prompts/sampling/template kwargs that feed the
    cache keys come from the same object that performs the inference, so a key
    can never drift from the request actually sent (DESIGN §9.2/§9.6).

    Nothing is launched until the first :meth:`backend` call, so a fully-cached
    (or empty) pass never starts a server; both passes reuse one server when
    either has a miss. ``close()`` tears it down.
    """

    def __init__(
        self, cfg: RunConfig, work_dir: str | Path, endpoint_override: str | None = None
    ) -> None:
        self.cfg = cfg
        self.work_dir = work_dir
        self.endpoint = endpoint_override or cfg.vlm.endpoint
        self.proto = get_vlm_backend(cfg.vlm.backend)  # client attached on first miss
        self.cache = VlmCache(enabled=cfg.cache.enabled, refresh=cfg.cache.refresh)
        self._identities: tuple[str, str, str] | None = None
        self._stack: ExitStack | None = None

    def identities(self) -> tuple[str, str, str]:
        """Model/mmproj/server identities for cache keys (probes the binary's
        ``--version`` or the endpoint's ``/props``; no server launch needed)."""
        if self._identities is None:
            self._identities = _vlm_identities(self.cfg, self.cache)
        return self._identities

    def backend(self):
        """The client-connected VLM backend (the same ``proto`` instance the
        cache keys are built from), launching the server on first use."""
        if self.proto.client is not None:
            return self.proto
        cfg = self.cfg
        if not self.endpoint and find_binary(cfg.llama.bin_dir, "llama-server") is None:
            raise ConfigError(
                "llama-server binary not found "
                f"(llama.bin_dir={cfg.llama.bin_dir!r}; not on PATH either)"
            )
        mgr = LlamaServerManager(
            cfg.llama.bin_dir,
            server_start_timeout=cfg.llama.server_start_timeout,
            log_dir=self.work_dir,
        )
        spec = ServerSpec(
            model=cfg.vlm.model,
            mmproj=cfg.vlm.mmproj,
            host=cfg.llama.host,
            port=cfg.llama.port,
            ctx_size=cfg.llama.ctx_size,
            n_gpu_layers=cfg.vlm.n_gpu_layers,
            extra_flags=self.proto.server_flags(),
            chat_template=None,
            label="vlm",
        )
        stack = ExitStack()
        try:
            url = stack.enter_context(endpoint_or_serve(mgr, self.endpoint, spec))
        except BaseException:
            stack.close()
            raise
        self._stack = stack
        self.proto.client = ChatClient(url)
        return self.proto

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self.proto.client = None


def _refine_tables(cfg: RunConfig, pages: list[_Page], session: _VlmSession) -> int:
    """Restructure DeepSeek ``<table>`` blobs via the VLM, before figure description
    (dev/docs/table-reconstruction-findings.md).

    Cache-first; the shared ``session`` launches the VLM server only on a miss.
    Any per-table failure (error, truncation, unusable output) keeps the original
    OCR blob — it still holds every value. Updates ``pg.markdown``/``pg.page_text``
    in place (so figure context sees clean tables) and returns the number of
    tables restructured.
    """
    if not cfg.table.refine:
        return 0

    # (page, page-context, blob-count-on-page, [(index, start, end, blob), ...])
    work: list[tuple[_Page, str, int, list[tuple[int, int, int, str]]]] = []
    for pg in pages:
        spans = find_table_blobs(pg.markdown)
        if not spans:
            continue
        if pg.raster_png is None:
            logger.warning(
                "page %d has %d table(s) but no page raster (pre-table-pass bundle?); "
                "keeping raw OCR tables",
                pg.page_number, len(spans),
            )
            continue
        entries = [
            (i, start, end, blob)
            for i, (start, end, blob) in enumerate(spans, 1)
            if blob_is_refinable(blob)
        ]
        if entries:
            # Context computed once per page against the pre-splice markdown (all
            # blobs + placeholders stripped), exactly as in the validated prompt.
            work.append((pg, table_page_context(pg.markdown), len(spans), entries))
    if not work:
        return 0
    if not _vlm_configured(cfg):
        logger.warning(
            "table refinement skipped: no VLM configured (set [vlm] model/mmproj or "
            "--vlm-endpoint, or pass --no-table-refine to silence this)"
        )
        return 0

    proto = session.proto  # one instance: cache-key material AND inference (§9.2)
    vlm_cache = session.cache
    model_id, mmproj_id, server_id = session.identities()

    total = sum(len(entries) for _, _, _, entries in work)
    done = 0
    refined = 0
    for pg, context, blob_count, entries in work:
        replacements: list[tuple[int, int, str]] = []
        for index, start, end, blob in entries:
            done += 1
            prompt = proto.build_table_prompt(
                blob, context, table_index=index, table_count=blob_count
            )
            key = make_table_key(
                page_image_hash=sha256_bytes(pg.raster_png),
                vlm_backend_name=proto.name,
                vlm_model_identity=model_id,
                vlm_mmproj_identity=mmproj_id,
                server_identity=server_id,
                full_assembled_prompt=prompt,
                sampling=proto.sampling(),
                chat_template_kwargs=proto.chat_template_kwargs(),
            )
            cached = vlm_cache.get(key)
            if cached is not None:
                logger.info("table %d/%d (page %d): cache hit", done, total, pg.page_number)
                replacements.append((start, end, cached))
                refined += 1
                continue
            logger.info("refining table %d/%d (page %d)…", done, total, pg.page_number)
            try:
                # The key's prompt string is the one sent — assembled exactly once.
                raw = session.backend().restructure_table(pg.raster_png, prompt)
            except ConfigError:
                raise  # missing binary is a setup error, not a per-table failure
            except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
                logger.warning(
                    "table %d/%d (page %d) restructure failed: %s; keeping raw OCR table",
                    done, total, pg.page_number, e,
                )
                continue
            table_md = sanitize_table_output(raw)
            if table_md is None:
                logger.warning(
                    "table %d/%d (page %d): truncated/unusable output; keeping raw OCR table",
                    done, total, pg.page_number,
                )
                continue
            vlm_cache.put(key, table_md)
            replacements.append((start, end, table_md))
            refined += 1
        if replacements:
            pg.markdown = splice_tables(pg.markdown, replacements)
            pg.page_text = PLACEHOLDER_RE.sub("", pg.markdown).strip()
    return refined


def _vlm_identities(cfg: RunConfig, vlm_cache: VlmCache) -> tuple[str, str, str]:
    if cfg.vlm.endpoint:
        ep = cfg.vlm.endpoint
        server_id = llama_build_identity(cfg.llama.bin_dir, endpoint=ep)
        return f"endpoint:{ep}:{cfg.vlm.model}", f"endpoint:{ep}:{cfg.vlm.mmproj}", server_id
    for label, path in (("vlm.model", cfg.vlm.model), ("vlm.mmproj", cfg.vlm.mmproj)):
        if not path or not Path(path).expanduser().is_file():
            raise ConfigError(f"{label} file not found: {path}")
    hash_cache = (vlm_cache.dir / "hashes.json") if vlm_cache.enabled else None
    return (
        file_identity(cfg.vlm.model, hash_disk_cache=hash_cache),
        file_identity(cfg.vlm.mmproj, hash_disk_cache=hash_cache),
        llama_build_identity(cfg.llama.bin_dir),
    )


def _vlm_describe(
    cfg: RunConfig,
    pages: list[_Page],
    crop_base: Path,
    session: _VlmSession,
) -> dict[str, str]:
    """Describe every figure (DESIGN §9). Cache-first; the shared ``session``
    launches the VLM server only on a miss (and reuses the table pass's server)."""
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

    proto = session.proto  # one instance: cache-key material AND inference (§9.2)
    vlm_cache = session.cache
    model_id, mmproj_id, server_id = session.identities()

    descriptions: dict[str, str] = {}
    keys: dict[str, str] = {}
    todo: list[tuple[Figure, str, bytes]] = []  # carries the key's prompt verbatim
    for fig, context, crop_bytes in tasks:
        prompt = proto.build_prompt(context)
        key = make_vlm_key(
            figure_crop_hash=sha256_bytes(crop_bytes),
            vlm_backend_name=proto.name,
            vlm_model_identity=model_id,
            vlm_mmproj_identity=mmproj_id,
            server_identity=server_id,
            full_assembled_prompt=prompt,
            sampling=proto.sampling(),
            chat_template_kwargs=proto.chat_template_kwargs(),
        )
        keys[fig.id] = key
        cached = vlm_cache.get(key)
        if cached is not None:
            logger.info("figure %s: description cache hit", fig.id)
            descriptions[fig.id] = cached
        else:
            todo.append((fig, prompt, crop_bytes))

    if todo:
        backend = session.backend()
        for i, (fig, prompt, crop_bytes) in enumerate(todo, 1):
            logger.info("describing figure %d/%d (%s)…", i, len(todo), fig.id)
            try:
                desc = backend.describe(crop_bytes, prompt)
            except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
                logger.warning("figure %s description failed: %s", fig.id, e)
                desc = ""  # injection → [figure description unavailable]
            if desc:
                vlm_cache.put(keys[fig.id], desc)
            descriptions[fig.id] = desc
    return descriptions


def _bibtex_probe(
    cfg: RunConfig, pages: list[_Page], session: _VlmSession, original_url: str | None
) -> ProbeResult | None:
    """The BibTeX citability/metadata probe (DESIGN §12, auto mode).

    Runs INSIDE the open VLM session (after the figure pass) because the server
    is torn down before ``_bibtex_outputs`` runs. Cache-first like every VLM
    pass; a failed/truncated/unparseable probe is treated as "citability
    unknown" and is never cached. Never fails the run.
    """
    if cfg.bibtex.mode != "auto" or not pages:
        return None
    # Recognized repository provenance settles citability, and when online the
    # by-ID/S2 sources don't need the probe's metadata either — skip the VLM
    # call entirely. (Offline still probes: best-effort needs the metadata.)
    if not cfg.net.offline and citable_provenance(original_url):
        logger.info("BibTeX (auto): provenance recognized; probe skipped")
        return None
    if not _vlm_configured(cfg):
        logger.warning(
            "BibTeX probe skipped: no VLM configured (set [vlm] model/mmproj or "
            "--vlm-endpoint)"
        )
        return None
    # The first PROCESSED page: with a --pages range that excludes page 1 this
    # is body text and the abstain-biased probe will typically answer no.
    page = pages[0]
    proto = session.proto  # one instance: cache-key material AND inference (§9.2)
    prompt = proto.build_bibtex_probe_prompt(page.page_text)
    model_id, mmproj_id, server_id = session.identities()
    key = make_bibtex_probe_key(
        vlm_backend_name=proto.name,
        vlm_model_identity=model_id,
        vlm_mmproj_identity=mmproj_id,
        server_identity=server_id,
        full_assembled_prompt=prompt,
        sampling=proto.sampling(),
        chat_template_kwargs=proto.chat_template_kwargs(),
    )
    cached = session.cache.get(key)
    if cached is not None:
        logger.info("BibTeX probe: cache hit")
        return parse_probe_response(cached)
    logger.info("BibTeX probe: reading front matter (page %d)…", page.page_number)
    try:
        # The key's prompt string is the one sent — assembled exactly once.
        raw = session.backend().probe_metadata(prompt)
    except ConfigError:
        raise  # missing binary is a setup error, not a per-document failure
    except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16)
        logger.warning("BibTeX probe failed: %s; treating citability as unknown", e)
        return None
    result = parse_probe_response(raw)
    if result is None:
        logger.warning(
            "BibTeX probe returned unusable output; treating citability as unknown"
        )
        return None
    session.cache.put(key, result.raw)  # never cache a failure
    return result


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
    cfg: RunConfig,
    full_md: str,
    out_dir: Path,
    base: str,
    *,
    probe: ProbeResult | None = None,
    original_url: str | None = None,
) -> tuple[str | None, list[str]]:
    """Produce + write ``{base}.bib`` (and optionally a fenced block to prepend
    into the document) per ``bibtex.mode`` (DESIGN §12). Never fails the run.

    ``on`` is the frozen paper2llm-parity path (title search + mock fallback,
    network required). ``auto`` walks the citability → source chain with
    ``probe`` and ``original_url`` (provenance), degrading gracefully.
    """
    if cfg.bibtex.mode == "off":
        return None, []
    source = ""
    try:
        if cfg.bibtex.mode == "on":
            if cfg.net.offline:
                logger.warning("--offline: skipping BibTeX (requires network)")
                return None, []
            title = extract_title(full_md)
            logger.info("fetching BibTeX for: %s", title)
            bibtex = generate_bibtex(title)
        else:  # auto: citability → source chain
            bibtex, source = generate_bibtex_auto(
                probe,
                original_url=original_url,
                online_allowed=not cfg.net.offline,
                fallback_title=extract_title(full_md),
            )
            if bibtex is None:
                if source == "not-citable":
                    logger.info("BibTeX (auto): document judged not citable; skipping")
                else:
                    logger.info("BibTeX (auto): skipped: %s", source)
                return None, []
    except Exception as e:  # noqa: BLE001 - resilience (DESIGN §16): e.g. a
        # malformed-but-HTTP-200 API body; BibTeX never fails the run.
        logger.warning("BibTeX generation failed: %s; skipping", e)
        return None, []
    bib_path = write_text_file(out_dir / f"{base}.bib", bibtex + "\n", clobber=cfg.output.clobber)
    if cfg.bibtex.mode == "auto":
        logger.info("BibTeX (auto): wrote entry via %s", source)
    block = f"```\n{bibtex}\n```\n\n---\n\n" if cfg.bibtex.append_to_document else None
    return block, [str(bib_path)]


def _write_documents(
    cfg: RunConfig,
    base: str,
    full_md: str,
    out_dir: Path,
    *,
    bibtex_block: str | None = None,
    vlm_tables: bool = False,
) -> list[str]:
    """Write the full doc (always) + main/appendix/backmatter when split (DESIGN §14).

    When ``bibtex_block`` is given (``--bibtex-in-doc``), it is prepended to the
    full document and the main split (DESIGN §12 — full/main/allparts only).
    ``vlm_tables`` notes VLM-restructured tables in the transcription notice.
    """
    full_out = (bibtex_block + full_md) if bibtex_block else full_md
    if cfg.output.notice:
        full_out = append_transcription_notice(full_out, vlm_tables=vlm_tables)
    written: list[Path] = [write_full_document(out_dir, base, full_out, clobber=cfg.output.clobber)]
    if cfg.output.split:
        sections = split_markdown_content(full_md)  # split the clean doc, not the prepended one
        main, backmatter, appendix = prepare_formatted_sections(sections)
        if bibtex_block:
            main = bibtex_block + main
        if cfg.output.notice:
            main = append_transcription_notice(main, vlm_tables=vlm_tables)
            if appendix is not None:
                appendix = append_transcription_notice(appendix, vlm_tables=vlm_tables)
            if backmatter is not None:
                backmatter = append_transcription_notice(backmatter, vlm_tables=vlm_tables)
        written += write_split_documents(
            out_dir, base, main=main, appendix=appendix, backmatter=backmatter,
            clobber=cfg.output.clobber,
        )
    return [str(p) for p in written]


def _ocr_meta(cfg: RunConfig, backend, model_id: str, mmproj_id: str, server_id: str) -> dict:
    mode = ResolutionMode(cfg.ocr.resolution)
    return {
        "backend": cfg.ocr.backend,
        "model_identity": model_id,
        "mmproj_identity": mmproj_id,
        "server_identity": server_id,  # additive (bundle_schema stays 1, DESIGN §8.5)
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
        model_id, mmproj_id, server_id = _ocr_identities(cfg, cache)
        working = _crop_pages(cfg, backend, resolved.pdf_bytes, pages, results, figures_dir)
        page_results = [
            OcrPageResult(page_number=p.page_number, markdown=p.markdown, regions=r.regions)
            for p, r in zip(working, results, strict=True)
        ]
        page_figures = {p.page_number: p.figures for p in working}
        # Pages with restructurable <table> blobs carry their page raster so a
        # later `describe` can run the VLM table pass with no PDF present. The
        # bytes are written VERBATIM — run and describe then share table cache keys.
        page_rasters: dict[int, str] = {}
        for pg, p in zip(pages, working, strict=True):
            spans = find_table_blobs(p.markdown)
            if not any(blob_is_refinable(blob) for _, _, blob in spans):
                continue
            rel = f"pages/page_{pg.page_number:04d}.png"
            target = bdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(pg.png_bytes)
            page_rasters[pg.page_number] = rel
        source = {
            "source": resolved.source,
            "original_url": resolved.original_url,
            "pdf_sha256": sha256_bytes(resolved.pdf_bytes),
        }
        write_bundle(
            bdir,
            base_name=base,
            source=source,
            ocr_meta=_ocr_meta(cfg, backend, model_id, mmproj_id, server_id),
            figure_detect=cfg.figure.detect,
            page_results=page_results,
            page_figures=page_figures,
            page_rasters=page_rasters,
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
            raster_png=(bundle.dir / p.raster_path).read_bytes() if p.raster_path else None,
        )
        for p in bundle.pages
    ]
    figures_need_vlm = cfg.figure.detect != "none" and cfg.figure.mode != "placeholder"

    with _workdir(cfg) as work:
        descriptions: dict[str, str] = {}
        probe: ProbeResult | None = None
        session = _VlmSession(cfg, work)
        try:
            tables_refined = _refine_tables(cfg, pages, session)
            if figures_need_vlm:
                descriptions = _vlm_describe(cfg, pages, bundle.dir, session)
            probe = _bibtex_probe(cfg, pages, session, bundle.original_url)
        finally:
            session.close()
        full_md = _assemble(cfg, pages, descriptions)
        bibtex_block, bib_written = _bibtex_outputs(
            cfg, full_md, out_dir, base, probe=probe, original_url=bundle.original_url
        )
        written = _write_documents(
            cfg, base, full_md, out_dir,
            bibtex_block=bibtex_block, vlm_tables=tables_refined > 0,
        )
        written += bib_written
        if cfg.figure.mode == "describe-and-keep" and cfg.figure.detect != "none":
            all_figs = [f for pg in pages for f in pg.figures]
            written += [str(p) for p in copy_figures(
                all_figs, src_base=bundle.dir, out_dir=out_dir, clobber=cfg.output.clobber)]
    return written


def _run_body(cfg: RunConfig, resolved, base, out_dir, work, *, vlm_endpoint=None) -> list[str]:
    backend = _build_ocr_backend(cfg)
    figures_need_vlm = cfg.figure.detect != "none" and cfg.figure.mode != "placeholder"

    pages, results = run_ocr_pass(cfg, resolved, work)
    figures_dir = Path(work) / "figures"
    working = _crop_pages(cfg, backend, resolved.pdf_bytes, pages, results, figures_dir)
    for w, pg in zip(working, pages, strict=True):
        w.raster_png = pg.png_bytes  # table restructuring input (verbatim render)

    descriptions: dict[str, str] = {}
    probe: ProbeResult | None = None
    session = _VlmSession(cfg, work, endpoint_override=vlm_endpoint)
    try:
        tables_refined = _refine_tables(cfg, working, session)
        if figures_need_vlm:
            descriptions = _vlm_describe(cfg, working, work, session)
        probe = _bibtex_probe(cfg, working, session, resolved.original_url)
    finally:
        session.close()
    full_md = _assemble(cfg, working, descriptions)
    bibtex_block, bib_written = _bibtex_outputs(
        cfg, full_md, out_dir, base, probe=probe, original_url=resolved.original_url
    )
    written = _write_documents(
        cfg, base, full_md, out_dir,
        bibtex_block=bibtex_block, vlm_tables=tables_refined > 0,
    )
    written += bib_written
    if cfg.figure.mode == "describe-and-keep" and cfg.figure.detect != "none":
        all_figs = [f for pg in working for f in pg.figures]
        written += [str(p) for p in copy_figures(
            all_figs, src_base=work, out_dir=out_dir, clobber=cfg.output.clobber)]
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
    # Tables may use the VLM too, but only pre-launch for them when a VLM is
    # actually configured (table.refine degrades gracefully without one).
    needs_vlm = (cfg.figure.detect != "none" and cfg.figure.mode != "placeholder") or (
        cfg.table.refine and _vlm_configured(cfg)
    )

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
            # Gate on the OCR backend's minimum build BEFORE loading the VLM —
            # otherwise a too-old build wastes a full VLM model load just to be
            # refused inside run_ocr_pass.
            _check_server_build(
                _build_ocr_backend(cfg),
                llama_build_identity(cfg.llama.bin_dir, endpoint=cfg.ocr.endpoint),
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
