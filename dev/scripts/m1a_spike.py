"""M1a real-hardware de-risk spike (DESIGN §2.1, §2.2, §8.3; PLAN M1a).

Answers the two highest-risk empirical unknowns on the pinned llama.cpp build,
using the actual inscriber inference layer:

  Q1. Does a base64 image round-trip through DeepSeek-OCR via llama-server
      ``/v1/chat/completions`` (open issue #21022 may break it)? If broken,
      fall back to llama-mtmd-cli.
  Q2. Is the grounding 0–999 coordinate frame the reference (per-axis, original
      image) or the padded-1024² square? Determined by running on the calibration
      page and matching the emitted <|det|> coords to the committed predictions.

Usage (bin dir + model paths come from the discovered config.toml; flags
override), e.g.::

    python dev/scripts/m1a_spike.py --ngl 99 --resolution large [--paper some_paper.pdf]

Writes raw outputs to tests/fixtures/ and prints a coordinate-frame verdict.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _common import REPO, fill_from_config  # bootstraps sys.path for inscriber

from inscriber.models import PageImage, ResolutionMode  # noqa: E402
from inscriber.ocr.base import HttpInferencer, MtmdCliInferencer  # noqa: E402
from inscriber.pdf.rasterize import rasterize  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
GROUNDING_PROMPT = "<|grounding|>Convert the document to markdown."

# M1a finding: this build emits grounding as ``LABEL[[x1, y1, x2, y2]]`` (NOT the
# DESIGN's ``<|ref|>..<|/ref|><|det|>..<|/det|>``).
DET_RE = re.compile(
    r"(?P<label>[A-Za-z_]+)\[\[(?P<coords>\d+,\s*\d+,\s*\d+,\s*\d+)\]\]"
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


def infer_all_server(args, pages: list[PageImage]) -> tuple[list[str] | None, str]:
    """Run every page through one llama-server session. Returns (raws|None, note)."""
    from inscriber.llama.client import ChatError
    from inscriber.llama.server import LlamaServerManager, ServerError, ServerSpec

    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=args.timeout)
    spec = ServerSpec(
        model=args.ocr_model, mmproj=args.ocr_mmproj, n_gpu_layers=args.ngl,
        ctx_size=args.ctx, chat_template=None, label="ocr",  # server uses built-in template
    )
    try:
        with mgr.serve(spec) as url:
            inf = HttpInferencer(url)
            raws = [
                inf.infer(pg, GROUNDING_PROMPT, sampling={"temperature": 0, "seed": 0},
                          chat_template=None, max_tokens=args.max_tokens,
                          timeout_s=args.req_timeout)
                for pg in pages
            ]
        return raws, "server HTTP path OK"
    except (ChatError, ServerError) as e:
        return None, f"server path FAILED: {e}"


def infer_all_mtmd(args, pages: list[PageImage]) -> tuple[list[str] | None, str]:
    """Fallback: one-shot llama-mtmd-cli per page (DESIGN §2.1)."""
    inf = MtmdCliInferencer(args.bin_dir, args.ocr_model, args.ocr_mmproj,
                            n_gpu_layers=args.ngl, ctx_size=args.ctx)
    try:
        raws = [
            inf.infer(pg, GROUNDING_PROMPT, sampling={"temperature": 0, "seed": 0},
                      chat_template="deepseek-ocr", max_tokens=args.max_tokens,
                      timeout_s=args.req_timeout)
            for pg in pages
        ]
        return raws, "mtmd-cli fallback OK"
    except Exception as e:
        return None, f"mtmd-cli path FAILED: {e}"


def verdict(raw: str, mode: str) -> None:
    meta = json.loads((FIXTURES / "calibration.json").read_text(encoding="utf-8"))
    pred = meta["modes"][mode]
    ref, pad = pred["predicted_grid_reference"], pred["predicted_grid_padded_square"]
    dets = parse_dets(raw)
    print(f"\n=== Q2: coordinate frame (mode={mode}) ===")
    print(f"  reference prediction     : {ref}")
    print(f"  padded-square prediction : {pad}")
    if not dets:
        print("  !! no LABEL[[bbox]] grounding spans found — inspect the raw output.")
        return
    # The figure region is labeled 'image'; use it for the frame check.
    figs = [(lab, c) for lab, c in dets if lab == "image"] or dets
    for label, coords in figs:
        d_ref, d_pad = closeness(coords, ref), closeness(coords, pad)
        pick = "REFERENCE" if d_ref < d_pad else "PADDED-SQUARE"
        print(f"  emitted [{label}] {coords}  -> closer to {pick} "
              f"(d_ref={d_ref:.1f}, d_pad={d_pad:.1f})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None,
                   help="config file (default: ./config.toml, then the platform dir)")
    p.add_argument("--bin-dir", default=None, help="default: [llama] bin_dir from config")
    p.add_argument("--ocr-model", default=None, help="default: [ocr] model from config")
    p.add_argument("--ocr-mmproj", default=None, help="default: [ocr] mmproj from config")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--resolution", default="large", choices=[m.value for m in ResolutionMode])
    p.add_argument("--timeout", type=float, default=180.0, help="server /health wait")
    p.add_argument("--req-timeout", type=float, default=900.0, help="per-request wait")
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--paper", default=None, help="optional real PDF to capture pages from")
    p.add_argument("--paper-pages", default="1-3", help="page range to capture from --paper")
    args = p.parse_args()
    fill_from_config(args, require=("bin_dir", "ocr_model", "ocr_mmproj"))

    mode = ResolutionMode(args.resolution)
    pages = rasterize((FIXTURES / "calibration.pdf").read_bytes(), mode)  # [calibration]
    labels = ["calibration"]
    if args.paper:
        paper_pages = rasterize(Path(args.paper).read_bytes(), mode, pages=args.paper_pages)
        pages += paper_pages
        labels += [f"paper_p{pg.page_number}" for pg in paper_pages]

    print("=== Q1: image round-trip ===")
    raws, note = infer_all_server(args, pages)
    path_used = "server"
    if raws is None:
        print(f"  {note}\n  → trying mtmd-cli fallback…")
        raws, note = infer_all_mtmd(args, pages)
        path_used = "mtmd-cli"
    print(f"  {note}")
    if raws is None:
        print("  !! both paths failed — see errors above.")
        return 1

    print(f"  image round-trip SUCCEEDED via {path_used} for {len(raws)} page(s).")
    for label, raw in zip(labels, raws, strict=False):
        out = FIXTURES / f"deepseek_{label}_raw.txt"
        out.write_text(raw, encoding="utf-8")
        print(f"  saved {out.name} ({len(raw)} chars)")

    verdict(raws[0], args.resolution)  # calibration page drives the frame check
    print(f"\nNEXT: record the chosen path ({path_used}) and frame in dev/notes/2026-06-09-m1a-findings.md, "
          f"then lock test_deepseek_parser.py to these fixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
