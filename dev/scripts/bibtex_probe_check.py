"""BibTeX probe validation on real hardware (PLAN-bibtex-auto B4).

Runs the pinned citability/metadata probe prompt against a real Gemma server on
(a) the sample paper fixture, (b) a real arXiv paper, and (c) two synthetic
non-citable PDFs (slides, an invoice) generated on the fly. Page-1 text is
extracted with PyMuPDF as a proxy for the production input (OCR markdown).

    python dev/scripts/bibtex_probe_check.py

Outcomes are recorded in dev/notes/2026-06-10-bibtex-probe-findings.md; once recorded, the
prompt is pinned (table-pass discipline).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import fitz  # noqa: E402  (PyMuPDF)

from inscriber.bibtex.local import best_effort_bibtex  # noqa: E402
from inscriber.bibtex.probe import parse_probe_response  # noqa: E402
from inscriber.llama.client import ChatClient  # noqa: E402
from inscriber.llama.server import LlamaServerManager, ServerSpec  # noqa: E402
from inscriber.logging import setup_logging  # noqa: E402
from inscriber.vlm.gemma import GemmaVlmBackend  # noqa: E402

MODELS = "C:/Users/luigi/llms/models"


def _make_slides_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=720, height=540)  # 4:3 slide
    page.insert_text((60, 120), "Quarterly Engineering Review", fontsize=36)
    page.insert_text((60, 170), "Q2 2026 - All Hands", fontsize=18)
    page.insert_text((60, 240), "- Roadmap status: green", fontsize=20)
    page.insert_text((60, 280), "- Hiring update: 3 open roles", fontsize=20)
    page.insert_text((60, 320), "- Infra costs down 12%", fontsize=20)
    page.insert_text((60, 490), "Internal - do not distribute", fontsize=12)
    doc.save(str(path))
    doc.close()


def _make_invoice_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 80), "INVOICE #2026-0142", fontsize=24)
    page.insert_text((72, 120), "Billed to: Example Corp, 1 Main Street, Helsinki", fontsize=12)
    page.insert_text((72, 140), "Date: 2026-06-10    Due: 2026-07-10", fontsize=12)
    page.insert_text((72, 190), "Qty   Description               Unit      Total", fontsize=12)
    page.insert_text((72, 210), "2     Consulting day            800.00    1600.00", fontsize=12)
    page.insert_text((72, 230), "1     Travel expenses           240.50    240.50", fontsize=12)
    page.insert_text((72, 270), "Total due: EUR 1840.50", fontsize=12)
    page.insert_text((72, 310), "Payment within 30 days. VAT ID: FI12345678.", fontsize=12)
    doc.save(str(path))
    doc.close()


def _page1_text(pdf_path: Path) -> str:
    with fitz.open(str(pdf_path)) as doc:
        return doc[0].get_text("text")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", default="C:/Users/luigi/llms/new")
    p.add_argument("--vlm-model", default=f"{MODELS}/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf")
    p.add_argument("--vlm-mmproj", default=f"{MODELS}/gemma-4-E4B-it-mmproj-BF16.gguf")
    p.add_argument("--ngl", default="auto")
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--pdf", action="append", default=[],
                   help="extra PDF(s) to probe (no expected verdict)")
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    setup_logging(verbose=1)

    with tempfile.TemporaryDirectory(prefix="inscriber-probe-") as work:
        work_dir = Path(work)
        slides = work_dir / "slides.pdf"
        invoice = work_dir / "invoice.pdf"
        _make_slides_pdf(slides)
        _make_invoice_pdf(invoice)

        docs: list[tuple[str, Path, bool | None]] = [
            ("sample_paper fixture", REPO / "tests/fixtures/sample_paper.pdf", True),
            ("arXiv 2510.09477v2 (real paper)", REPO / "out/2510.09477v2.pdf", True),
            ("synthetic slides", slides, False),
            ("synthetic invoice", invoice, False),
        ]
        docs += [(f"extra: {x}", Path(x), None) for x in args.pdf]

        mgr = LlamaServerManager(args.bin_dir, server_start_timeout=180, log_dir=work_dir)
        ngl = args.ngl if args.ngl in ("auto", "all") else int(args.ngl)
        spec = ServerSpec(
            model=args.vlm_model, mmproj=args.vlm_mmproj, host="127.0.0.1", port=0,
            ctx_size=args.ctx, n_gpu_layers=ngl, extra_flags=[], chat_template=None,
            label="vlm",
        )
        failures = 0
        with mgr.serve(spec) as url:
            client = ChatClient(url)
            backend = GemmaVlmBackend(client=client)
            for label, pdf, expected in docs:
                print(f"\n========== {label} ==========")
                if not pdf.is_file():
                    print(f"SKIP: {pdf} not found")
                    continue
                text = _page1_text(pdf)
                prompt = backend.build_bibtex_probe_prompt(text)
                raw = backend.probe_metadata(prompt)
                tokens = client.last_completion_tokens
                print(f"--- raw response ({tokens} completion tokens, "
                      f"finish={client.last_finish_reason}):\n{raw}")
                result = parse_probe_response(raw)
                if result is None:
                    print("--- parse: UNUSABLE (would be treated as unknown, not cached)")
                    verdict_ok = expected is None
                else:
                    print(f"--- parsed: citable={result.citable} title={result.title!r}")
                    print(f"            authors={result.authors} year={result.year!r} "
                          f"venue={result.venue!r}")
                    if result.citable:
                        print(f"--- best-effort entry:\n{best_effort_bibtex(result)}")
                    verdict_ok = expected is None or result.citable == expected
                print(f"--- expected citable={expected}: {'PASS' if verdict_ok else 'FAIL'}")
                failures += 0 if verdict_ok else 1
        print(f"\n========== done: {failures} failure(s) ==========")
        return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
