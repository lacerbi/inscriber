"""DeepSeek-OCR-2 real-hardware spike (TODO: "DeepSeek-OCR 2 spike").

The three gating unknowns before adoption (TODO.md; research record
dev/notes/2026-06-10-upstream-watch.md §2):

1. **Grounding format + coordinate frame under real tiling.** v2 in llama.cpp
   (PR #20975, included in the pinned build 9587) ships multi-tile dynamic
   resolution from day one: a 1024-px global view + a grid of 768-px tiles
   (inputs <= 768 px stay a single global view — the upstream test's own
   comment). The v1 questions return for real: does the ``LABEL[[bbox]]``
   block format survive, and is the frame per-axis (v1 on 9587), padded-square
   (v1 on 9028), tile-relative, or something new? This script renders the
   calibration page (known box at (150,200,450,520) pt on a 600x800 pt page)
   below and above the tiling threshold and compares the emitted box against
   both scale-invariant predictions.
2. **Loop behavior on the known-bad page** — PriorGuide p. 5 (triple-underbrace
   equation array; looped v1 at BF16 + DRY + temp 0,
   dev/notes/2026-06-10-equation-fidelity-findings.md). Run it via ``--paper``.
3. **Real-page format capture** — raws land in out-dsocr2/ for fixture pinning
   if adoption proceeds (the M1a capture -> compare -> re-pin discipline).

Server invocation is pinned to PR #20975's example (verified against the PR
2026-06-10): ``--chat-template deepseek-ocr --no-jinja`` (v2 REQUIRES the
template on the server path — the opposite of v1), ``--flash-attn off``,
``--no-warmup``, and v2's own DRY tuning (clears the default sequence
breakers). The grounding prompt is UNCHANGED from v1 — confirmed on the
official deepseek-ai/DeepSeek-OCR-2 model card ("Main Prompts": ``<image>\n
<|grounding|>Convert the document to markdown.``).

Bin dir comes from the discovered ``config.toml``; the v2 GGUF pair defaults
to ``deepseek-ocr-2-bf16.gguf`` / ``mmproj-deepseek-ocr-2-bf16.gguf`` next to
the config's (v1) ``[ocr] model`` — override with flags. Usage::

    python dev/scripts/dsocr2_check.py \
        [--targets 640,1024,1280,2048] \
        [--paper out/priorguide-G4I23g5Ugh.pdf --paper-page 1,5,27 --paper-target 2048]
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
from inscriber.ocr.deepseek import GROUNDING_PROMPT  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
OUT = REPO / "out-dsocr2"

# Scale-invariant calibration predictions (tests/fixtures/calibration.json):
# box (150,200,450,520) pt on a 600x800 pt page.
PER_AXIS_PREDICTION = [250, 250, 749, 649]  # v1 frame on build 9587
PADDED_PREDICTION = [312, 250, 687, 649]  # v1 frame on build <= 9028

# v2 server flags, verbatim from PR #20975's llama-server example. The chat
# template itself goes through ServerSpec.chat_template; everything else here.
# (--temp 0 / -n are per-request in inscriber, not server flags.)
V2_EXTRA_FLAGS = [
    "--no-jinja",
    "--flash-attn", "off",
    "--no-warmup",
    "--dry-multiplier", "0.8",
    "--dry-base", "1.75",
    "--dry-allowed-length", "2",
    "--dry-penalty-last-n", "-1",
    "--dry-sequence-breaker", "none",
]
V2_CHAT_TEMPLATE = "deepseek-ocr"

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


def report(tag: str, raw: str, finish: str | None, secs: float, w: int, h: int) -> list[list[int]]:
    """Print the per-render verdicts; return the 'image' boxes for invariance checks."""
    dets = parse_dets(raw)
    images = [(lab, c) for lab, c in dets if lab == "image"]
    repeats, rep_line = loopiness(raw)
    print(f"\n--- {tag} ({w}x{h}px, {secs:.0f}s, {len(raw)} chars, finish={finish}) ---")
    print(f"  grounding spans: {len(dets)} total, {len(images)} 'image'")
    if not dets:
        print("  !! NO grounding markers — v2 may use a different layout format")
        print(f"  head: {raw[:200]!r}")
    if repeats >= 5:
        print(f"  !! LOOP suspicion: a line repeats {repeats}x: {rep_line!r}")
    if finish is not None and finish != "stop":
        print(f"  !! TRUNCATED (finish_reason={finish}) — hit the token cap")
    for lab, c in images or dets[:3]:
        d_axis = closeness(c, PER_AXIS_PREDICTION)
        d_pad = closeness(c, PADDED_PREDICTION)
        if min(d_axis, d_pad) > 50:
            pick = "NEITHER — tile-relative or new frame?"
        elif d_axis < d_pad:
            pick = "PER-AXIS (v1@9587 frame)"
        else:
            pick = "PADDED-SQUARE (v1@9028 frame)"
        print(f"  [{lab}] {c}  d_axis={d_axis:.1f} d_pad={d_pad:.1f} -> {pick}")
    return [c for _, c in images]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="config file (default: ./config.toml, then the platform dir)")
    p.add_argument("--bin-dir", default=None, help="default: [llama] bin_dir from config")
    p.add_argument("--ocr-model", default=None,
                   help="v2 model GGUF (default: deepseek-ocr-2-bf16.gguf beside the "
                        "config's [ocr] model)")
    p.add_argument("--ocr-mmproj", default=None,
                   help="v2 mmproj GGUF (default: mmproj-deepseek-ocr-2-bf16.gguf "
                        "beside the config's [ocr] model)")
    p.add_argument("--ngl", default="auto")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--req-timeout", type=float, default=900.0)
    p.add_argument("--targets", default="640,1024,1280,2048",
                   help="calibration long-edge renders; 640 is below the 768 "
                        "tiling threshold (single global view), the rest tile")
    p.add_argument("--no-calibration", action="store_true",
                   help="skip the calibration renders (paper-only run)")
    p.add_argument("--paper", default=None, help="optional real PDF (loop check / capture)")
    p.add_argument("--paper-page", default="5",
                   help="page number, or comma-separated list (e.g. 1,5,27)")
    p.add_argument("--paper-target", type=int, default=2048)
    args = p.parse_args()

    # fill_from_config fills ocr_model/ocr_mmproj with the config's V1 paths;
    # unless the user passed v2 paths explicitly, those only seed the default
    # *directory* the v2 pair is looked up in.
    user_model, user_mmproj = args.ocr_model, args.ocr_mmproj
    fill_from_config(args, require=("bin_dir",))
    if user_model is None:
        if not args.ocr_model:
            raise SystemExit("pass --ocr-model (no [ocr] model in config to "
                             "derive the default v2 location from)")
        args.ocr_model = str(Path(args.ocr_model).parent / "deepseek-ocr-2-bf16.gguf")
    if user_mmproj is None:
        args.ocr_mmproj = str(
            Path(args.ocr_model).parent / "mmproj-deepseek-ocr-2-bf16.gguf"
        )
    for f in (args.ocr_model, args.ocr_mmproj):
        if not Path(f).is_file():
            raise SystemExit(f"model file not found: {f}")

    # Windows console is cp1252; server logs contain fullwidth ｜ etc.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    OUT.mkdir(exist_ok=True)
    jobs: list[tuple[str, PageImage]] = []
    if not args.no_calibration:
        calib = (FIXTURES / "calibration.pdf").read_bytes()
        for t in [int(t) for t in args.targets.split(",") if t.strip()]:
            jobs.append((f"calibration@{t}", render(calib, 1, t)))
    if args.paper:
        pdf = Path(args.paper).expanduser().read_bytes()
        for pg in [int(x) for x in str(args.paper_page).split(",") if x.strip()]:
            jobs.append(
                (f"paper_p{pg}@{args.paper_target}",
                 render(pdf, pg, args.paper_target))
            )
    if not jobs:
        raise SystemExit("nothing to do (--no-calibration and no --paper)")

    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=args.timeout, log_dir=OUT)
    spec = ServerSpec(
        model=args.ocr_model, mmproj=args.ocr_mmproj, n_gpu_layers=args.ngl,
        ctx_size=args.ctx, extra_flags=V2_EXTRA_FLAGS,
        chat_template=V2_CHAT_TEMPLATE, label="dsocr2-check",
    )
    print(f"model:  {args.ocr_model}")
    print(f"mmproj: {args.ocr_mmproj}")
    print(f"prompt: {GROUNDING_PROMPT!r} (unchanged from v1 — official v2 model card)")
    print(f"calibration predictions: per-axis {PER_AXIS_PREDICTION}, "
          f"padded-square {PADDED_PREDICTION}")
    calib_boxes: dict[str, list[list[int]]] = {}
    with mgr.serve(spec) as url:
        inf = HttpInferencer(url)
        for tag, pg in jobs:
            t0 = time.monotonic()
            raw = inf.infer(
                pg, GROUNDING_PROMPT, sampling={"temperature": 0, "seed": 0},
                chat_template=None, max_tokens=args.max_tokens,
                timeout_s=args.req_timeout,
            )
            secs = time.monotonic() - t0
            finish = inf.last_finish_reason
            (OUT / f"{tag.replace('@', '_')}_raw.txt").write_text(
                raw, encoding="utf-8", newline="\n"
            )
            boxes = report(tag, raw, finish, secs, pg.width_px, pg.height_px)
            if tag.startswith("calibration"):
                calib_boxes[tag] = boxes

    # Render-size invariance (the v1@9587 frame is invariant; tile-relative
    # coords would diverge between the single-view and tiled renders).
    if len(calib_boxes) >= 2:
        print("\n=== render-size invariance (first 'image' box per render) ===")
        firsts = {t: bs[0] for t, bs in calib_boxes.items() if bs}
        for t, b in firsts.items():
            print(f"  {t}: {b}")
        vals = list(firsts.values())
        if vals and all(closeness(v, vals[0]) <= 8 for v in vals):
            print("  -> INVARIANT across renders (matches v1@9587 behavior)")
        elif vals:
            print("  -> DIVERGES across renders — frame is render-size-dependent")

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
