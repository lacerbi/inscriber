"""M1b vertical-slice check on real hardware (PLAN M1b verification).

Runs the OCR pass on a real PDF twice: the first launches the server and OCRs each
page; the second must be served entirely from the per-page cache (server NOT
relaunched). Prints the assembled per-page markdown (with figure placeholders).

    python dev/scripts/m1b_check.py --pdf tests/fixtures/sample_paper.pdf
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from inscriber.input.resolver import resolve_local_pdf  # noqa: E402
from inscriber.logging import setup_logging  # noqa: E402
from inscriber.models import RunConfig  # noqa: E402
from inscriber.pipeline import run_ocr_pass  # noqa: E402

MODELS = "C:/Users/luigi/llms/models"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", default=str(REPO / "tests/fixtures/sample_paper.pdf"))
    p.add_argument("--bin-dir", default="C:/Users/luigi/llms/new")  # >= 9587 (gate)
    p.add_argument("--ocr-model", default=f"{MODELS}/DeepSeek-OCR-Q8_0.gguf")
    p.add_argument("--ocr-mmproj", default=f"{MODELS}/mmproj-deepseek-ocr-q8_0.gguf")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--resolution", default="large")
    p.add_argument("--no-cache", action="store_true")
    args = p.parse_args()

    # Windows consoles default to cp1252; the ⟦⟧ placeholder needs UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    setup_logging(verbose=1)
    cfg = RunConfig(command="ocr", input=args.pdf)
    cfg.llama.bin_dir = args.bin_dir
    cfg.ocr.model = args.ocr_model
    cfg.ocr.mmproj = args.ocr_mmproj
    cfg.ocr.n_gpu_layers = args.ngl
    cfg.ocr.resolution = args.resolution
    cfg.cache.enabled = not args.no_cache

    resolved = resolve_local_pdf(args.pdf)
    with tempfile.TemporaryDirectory(prefix="inscriber-m1b-") as work:
        print("\n========== PASS 1 (cold) ==========")
        pages, results = run_ocr_pass(cfg, resolved, work)
        for r in results:
            print(f"\n----- page {r.page_number} ({len(r.regions)} regions) -----")
            print(r.markdown)

        print("\n========== PASS 2 (warm — expect cache hits, no server) ==========")
        run_ocr_pass(cfg, resolved, work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
