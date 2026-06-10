"""Canonical dataclasses & enums shared across the pipeline (DESIGN §7, §8.2).

This module is the single home for the data shapes the whole pipeline passes
around (review Fix 2): ``Region`` / ``OcrPageResult`` are imported from here by the
OCR layer, the cache, and the bundle, so their (de)serialization is defined once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# --------------------------------------------------------------------------- #
# Resolution modes (DESIGN §2.2, §7)
# --------------------------------------------------------------------------- #


class ResolutionMode(str, Enum):
    """DeepSeek-OCR native resolution ladder + the dense-page Gundam opt-in.

    The ``long_edge_px`` is the rasterizer's render target (long edge, in px).
    ``gundam`` was meant as the model-side tiling mode, but the pinned llama.cpp
    build does NOT tile (dev/docs/gundam-findings.md) — it is currently a strict
    alias of ``large``; raising its render target is a pending decision (TODO.md).
    """

    TINY = "tiny"
    SMALL = "small"
    BASE = "base"
    LARGE = "large"
    GUNDAM = "gundam"

    @property
    def long_edge_px(self) -> int:
        return {
            ResolutionMode.TINY: 512,
            ResolutionMode.SMALL: 640,
            ResolutionMode.BASE: 1024,
            ResolutionMode.LARGE: 1280,
            ResolutionMode.GUNDAM: 1280,
        }[self]


# --------------------------------------------------------------------------- #
# OCR / figure data shapes (DESIGN §7, §8.2, §8.5)
# --------------------------------------------------------------------------- #


@dataclass
class PageImage:
    """One rasterized PDF page (DESIGN §7).

    ``width_px``/``height_px`` are the ORIGINAL rendered page dimensions and are
    the reference frame for ``Region.bbox_norm`` and cropping.
    """

    page_number: int  # 1-indexed
    png_bytes: bytes
    width_px: int
    height_px: int


@dataclass
class Region:
    """A detected layout region in the ORIGINAL-PAGE [0,1] frame (DESIGN §8.2)."""

    label: str  # e.g. "figure", "table", "text", "title"
    # x1,y1,x2,y2 in [0,1], relative to the original rendered page image
    bbox_norm: tuple[float, float, float, float]
    text: str | None = None  # caption / inline text for this region, if any

    @property
    def is_figure(self) -> bool:
        return self.label.lower() in FIGURE_LABELS


# Figure-class labels (DESIGN §8.3 step 4): grounding spans with these labels
# become ``⟦INSCRIBER_FIG:{id}⟧`` placeholders; others are stripped-but-kept.
FIGURE_LABELS = frozenset(
    {"figure", "image", "picture", "chart", "diagram", "plot"}
)

# The placeholder token spliced where a figure was (DESIGN §8.3). The id format
# is ``fig_p{page}_{i}`` (1-indexed page, 1-indexed figure-on-page).
FIG_PLACEHOLDER_PREFIX = "⟦INSCRIBER_FIG:"
FIG_PLACEHOLDER_SUFFIX = "⟧"


def fig_placeholder(fig_id: str) -> str:
    return f"{FIG_PLACEHOLDER_PREFIX}{fig_id}{FIG_PLACEHOLDER_SUFFIX}"


@dataclass
class OcrPageResult:
    """The pre-crop OCR boundary for one page (DESIGN §8.2).

    ``markdown`` is clean markdown where figure regions are represented by
    ``⟦INSCRIBER_FIG:{id}⟧`` placeholders. ``regions`` are all detected regions
    with bboxes already in the original-page [0,1] frame.
    """

    page_number: int  # 1-indexed
    markdown: str
    regions: list[Region] = field(default_factory=list)


@dataclass
class Figure:
    """A cropped figure (post step-4 boundary, DESIGN §8.4 / §8.5)."""

    id: str  # fig_p{page}_{i}
    page: int
    bbox_norm: tuple[float, float, float, float]
    crop_path: str | None = None  # relative path, e.g. "figures/fig_p3_1.png"
    caption: str | None = None


@dataclass
class ResolvedInput:
    """Output of input resolution (DESIGN §6)."""

    pdf_bytes: bytes
    source: str  # "file" | "url"
    original_url: str | None
    suggested_name: str  # base name (no extension), pre-sanitization


# --------------------------------------------------------------------------- #
# Resolved run configuration (DESIGN §13) — nested by TOML section.
# --------------------------------------------------------------------------- #


@dataclass
class LlamaConfig:
    bin_dir: str = ""  # folder containing llama-server[.exe]
    host: str = "127.0.0.1"
    port: int = 0  # 0 = auto-select a free port
    server_start_timeout: int = 120
    # The single size knob: prompt + generation share this window. 16384 leaves
    # room for the table pass (~2-4k prompt tokens of page image + page text +
    # OCR blob, plus ~6-8k of VLM thinking + answer).
    ctx_size: int = 16384


@dataclass
class InferenceConfig:
    mode: str = "sequential"  # "sequential" | "concurrent"


@dataclass
class OcrConfig:
    backend: str = "deepseek-ocr"
    model: str = ""
    mmproj: str = ""
    resolution: str = "large"  # tiny | small | base | large | gundam
    # -ngl for the OCR server: "auto" (let llama.cpp fit as many layers as VRAM
    # allows — its own default), "all", or an explicit integer. 0 forces CPU.
    n_gpu_layers: int | str = "auto"
    endpoint: str = ""  # if set, talk to this URL; don't spawn a server


@dataclass
class VlmConfig:
    backend: str = "gemma"
    model: str = ""
    mmproj: str = ""
    # -ngl for the VLM server: "auto" (default) | "all" | integer (0 = CPU).
    n_gpu_layers: int | str = "auto"
    endpoint: str = ""


@dataclass
class FigureConfig:
    detect: str = "auto"  # auto | grounding | none | pdf-embedded
    mode: str = "describe-only"  # describe-only | describe-and-keep | placeholder
    crop_padding: float = 0.02  # fraction of page dims (ocr-stage)
    context_chars: int = 2000  # whole-page context truncation cap (describe-stage)


@dataclass
class TableConfig:
    """VLM table restructuring (describe-stage; dev/docs/table-reconstruction-findings.md).

    Independent of figure settings: ``--no-figures`` does not disable it. There is
    no per-table token budget — generation is bounded by ``llama.ctx_size`` (the
    single size knob); complex tables need ~6-8k tokens of thinking + answer.
    """

    refine: bool = True  # restructure DeepSeek <table> blobs via the VLM


@dataclass
class OutputConfig:
    dir: str = "."
    split: bool = True
    page_numbers: bool = False
    page_separators: bool = False
    normalize_line_breaks: bool = True
    clean: bool = True
    clobber: bool = True
    notice: bool = True


@dataclass
class CacheConfig:
    enabled: bool = True  # False <=> --no-cache (no read, no write)
    refresh: bool = False  # True <=> --refresh (recompute + overwrite)


@dataclass
class WorkdirConfig:
    path: str = ""  # "" = OS temp dir; else explicit dir
    keep_intermediates: bool = False


@dataclass
class BibtexConfig:
    enabled: bool = False  # online; opt-in
    append_to_document: bool = False


@dataclass
class NetConfig:
    offline: bool = False  # hard-disable all network use


@dataclass
class RunConfig:
    """Fully-resolved configuration for a single invocation (DESIGN §3, §13).

    Built by ``config.resolve_config`` from defaults < config file < CLI flags.
    """

    command: str  # "run" | "ocr" | "describe"
    input: str  # PDF path / URL (run, ocr) or bundle dir (describe)
    config_path: str | None = None
    pages: str | None = None  # "1-10", "3", "5-", "-12", "all" (run, ocr)
    verbose: int = 0
    quiet: bool = False

    llama: LlamaConfig = field(default_factory=LlamaConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    vlm: VlmConfig = field(default_factory=VlmConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    table: TableConfig = field(default_factory=TableConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    workdir: WorkdirConfig = field(default_factory=WorkdirConfig)
    bibtex: BibtexConfig = field(default_factory=BibtexConfig)
    net: NetConfig = field(default_factory=NetConfig)
