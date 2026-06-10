"""Gundam-mode real-hardware check (TODO: "Gundam coordinate frame").

Two facts to establish on the pinned build (follow-up to M1A-FINDINGS Q2):

1. ``ResolutionMode.GUNDAM.long_edge_px`` is currently **1280 — identical to
   ``large``** — so inscriber's gundam mode feeds the model byte-identical
   images. Any model-side tiling can only be triggered by *bigger* inputs.
   This script renders the calibration page at several long-edge targets
   (1280 control, then genuinely larger) and checks, per render:
     - does the ``LABEL[[bbox]]`` grounding format still parse,
     - does the emitted ``image`` box stay at the **scale-invariant
       padded-square prediction** ``[312, 250, 687, 649]`` (global frame) or
       diverge (tile-relative frame),
     - response length + finish_reason (loop/truncation spotting).

2. Optionally re-runs a real paper page at a larger render (``--paper`` /
   ``--paper-page`` / ``--paper-target``) — e.g. the page that looped at 1280
   (dev/notes/2026-06-10-equation-fidelity-findings.md) — to see whether a different input
   size escapes the loop.

Outputs land in out-gundam/ (gitignored): raw model outputs + the llama-server
log, whose mtmd lines reveal whether the image was tiled (token counts).

Bin dir and model paths come from the discovered ``config.toml`` (the same
lookup the CLI uses); the flags override. Usage::

    python dev/scripts/gundam_check.py \
        [--targets 1280,1664,2048,2560] \
        [--paper some_paper.pdf --paper-page 5 --paper-target 2048]
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from _common import REPO, fill_from_config  # bootstraps sys.path for inscriber

import fitz  # noqa: E402  (PyMuPDF)

from inscriber.llama.server import LlamaServerManager, ServerSpec  # noqa: E402
from inscriber.models import PageImage  # noqa: E402
from inscriber.ocr.base import HttpInferencer  # noqa: E402
from inscriber.ocr.deepseek import DeepSeekOcrBackend  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
OUT = REPO / "out-gundam"

# Scale-invariant calibration predictions (tests/fixtures/calibration.json):
# box (150,200,450,520) pt on a 600x800 pt page.
PAD_PREDICTION = [312, 250, 687, 649]
REF_PREDICTION = [250, 250, 749, 649]

DET_RE = re.compile(r"(?P<label>[A-Za-z_]+)\[\[(?P<coords>\d+,\s*\d+,\s*\d+,\s*\d+)\]\]")


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


def parse_dets(text: str) -> list[tuple[str, list[int]]]:
    out = []
    for m in DET_RE.finditer(text):
        coords = [int(c) for c in m.group("coords").replace(" ", "").split(",") if c]
        out.append((m.group("label"), coords))
    return out


def closeness(a: list[int], b: list[int]) -> float:
    if len(a) != len(b):
        return 1e9
    return sum(abs(x - y) for x, y in zip(a, b, strict=False)) / len(a)


def loopiness(text: str) -> tuple[int, str]:
    """(max consecutive repeats of one line, that line) — crude loop detector."""
    best, best_line, run, prev = 1, "", 1, None
    for ln in [ln for ln in text.splitlines() if ln.strip()]:
        run = run + 1 if ln == prev else 1
        if run > best:
            best, best_line = run, ln
        prev = ln
    return best, best_line[:60]


def report(tag: str, raw: str, finish: str | None, secs: float, w: int, h: int) -> None:
    dets = parse_dets(raw)
    images = [(lab, c) for lab, c in dets if lab == "image"]
    repeats, rep_line = loopiness(raw)
    print(f"\n--- {tag} ({w}x{h}px, {secs:.0f}s, {len(raw)} chars, finish={finish}) ---")
    print(f"  grounding spans: {len(dets)} total, {len(images)} 'image'")
    if repeats >= 5:
        print(f"  !! LOOP suspicion: a line repeats {repeats}x: {rep_line!r}")
    for lab, c in images or dets[:3]:
        d_pad = closeness(c, PAD_PREDICTION)
        d_ref = closeness(c, REF_PREDICTION)
        pick = "PADDED-SQUARE/global" if d_pad < d_ref else "REFERENCE(?)"
        print(f"  [{lab}] {c}  d_pad={d_pad:.1f} d_ref={d_ref:.1f} -> {pick}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="config file (default: ./config.toml, then the platform dir)")
    p.add_argument("--bin-dir", default=None, help="default: [llama] bin_dir from config")
    p.add_argument("--ocr-model", default=None, help="default: [ocr] model from config")
    p.add_argument("--ocr-mmproj", default=None, help="default: [ocr] mmproj from config")
    p.add_argument("--ngl", default="auto")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--req-timeout", type=float, default=900.0)
    p.add_argument("--targets", default="1280,1664,2048,2560")
    p.add_argument("--paper", default=None, help="optional real PDF for the loop check")
    p.add_argument("--paper-page", default="5",
                   help="page number, or comma-separated list (e.g. 3,5,22)")
    p.add_argument("--paper-target", type=int, default=2048)
    args = p.parse_args()
    fill_from_config(args, require=("bin_dir", "ocr_model", "ocr_mmproj"))

    # Windows console is cp1252; server logs contain fullwidth ｜ etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    OUT.mkdir(exist_ok=True)
    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    calib = (FIXTURES / "calibration.pdf").read_bytes()
    backend = DeepSeekOcrBackend()  # production prompt/sampling/flags

    jobs: list[tuple[str, PageImage]] = [
        (f"calibration@{t}", render(calib, 1, t)) for t in targets
    ]
    if args.paper:
        pdf = Path(args.paper).expanduser().read_bytes()
        for pg in [int(x) for x in str(args.paper_page).split(",") if x.strip()]:
            jobs.append(
                (f"paper_p{pg}@{args.paper_target}",
                 render(pdf, pg, args.paper_target))
            )

    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=args.timeout, log_dir=OUT)
    spec = ServerSpec(
        model=args.ocr_model, mmproj=args.ocr_mmproj, n_gpu_layers=args.ngl,
        ctx_size=args.ctx, extra_flags=backend.server_flags(),
        chat_template=None, label="gundam-check",
    )
    print(f"calibration padded-square prediction (scale-invariant): {PAD_PREDICTION}")
    with mgr.serve(spec) as url:
        inf = HttpInferencer(url)
        for tag, pg in jobs:
            t0 = time.monotonic()
            raw = inf.infer(
                pg, backend.prompt(), sampling=backend.sampling(),
                chat_template=None, max_tokens=args.max_tokens,
                timeout_s=args.req_timeout,
            )
            secs = time.monotonic() - t0
            finish = getattr(inf.client, "last_finish_reason", None)
            (OUT / f"{tag.replace('@', '_')}_raw.txt").write_text(raw, encoding="utf-8")
            report(tag, raw, finish, secs, pg.width_px, pg.height_px)

    # mtmd preprocessing evidence: image token counts per render reveal tiling.
    print("\n=== server-log image/mtmd lines ===")
    for log in sorted(OUT.glob("*.log")):
        for ln in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if re.search(r"(?i)mtmd|image|slice|encoding|n_tokens|tokeniz", ln):
                print(f"  {ln.strip()}")
    print(f"\nraws + server log saved under {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
