# BibTeX probe — real-hardware validation (PLAN-bibtex-auto B4)

> **Date:** 2026-06-10 · **Status:** concluded — prompt frozen (DESIGN §12); re-run `dev/scripts/bibtex_probe_check.py` and update this note before changing it.

Hardware: RTX 4060 Laptop 8GB
llama.cpp: build **9587 (d2e22ed97)**, Clang 20.1.8 Windows x86_64 (`C:/Users/.../llms`)
Model: `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` + `gemma-4-E4B-it-mmproj-BF16.gguf`, `-ngl auto`, ctx 16384
Script: `dev/scripts/bibtex_probe_check.py` (repeatable; synthesizes the non-citable PDFs on the fly)

## What was validated

The pinned probe prompt (`inscriber/bibtex/probe.py::PROBE_PROMPT_TEMPLATE`),
sent **text-only** via `ChatClient.chat()` with the production settings:
`temperature 0`, `seed 0`, `chat_template_kwargs={"enable_thinking": true}`,
no `max_tokens`. Page-1 text was extracted with PyMuPDF `get_text()` as a
proxy for the production input (OCR markdown); front matter content is
equivalent for this purpose.

## Results — 4/4 PASS, zero prompt tuning needed

| document                                  | expected    | verdict             | extraction                                                    | tokens (incl. thinking) | finish |
| ----------------------------------------- | ----------- | ------------------- | ------------------------------------------------------------- | ----------------------- | ------ |
| `tests/fixtures/sample_paper.pdf`         | citable     | ✅ `citable: true`  | title + 2 authors (no year/venue on page — correctly omitted) | 427                     | stop   |
| `out/2510.09477v2.pdf` (real arXiv paper) | citable     | ✅ `citable: true`  | title + all 9 authors + year 2026 + venue ICLR                | 689                     | stop   |
| synthetic slides (quarterly review deck)  | not citable | ✅ `citable: false` | no fields emitted                                             | 349                     | stop   |
| synthetic invoice                         | not citable | ✅ `citable: false` | no fields emitted                                             | 400                     | stop   |

## Observed behaviors (drove/confirmed design choices)

- **Code fence in the wild**: the real-paper response arrived wrapped in
  ` ```json … ``` ` (the fixture's didn't) — the fence-tolerant strict parsing
  in `parse_probe_response` is **required**, not defensive overengineering.
- **Abstain shape**: non-citable answers were the minimal `{"citable": false}`
  with no hallucinated metadata — decision 5 (transcription, not recall) holds
  unprompted on this model.
- **Omission works**: the fixture paper has no year/venue on page 1 and the
  model omitted those fields rather than guessing.
- **Typographic transcription**: the real paper's title came back in the
  page's ALL-CAPS typography (faithful transcription). Acceptable: the
  best-effort entry is explicitly marked "verify before use", and
  `generate_citation_key` lowercases anyway (`hassan2026efficient`).
- **Cost**: 350–700 completion tokens per probe (thinking included),
  `finish_reason: stop` in all cases — comfortably inside the 16384 window.

## Verdict

**The prompt is FROZEN as shipped in `probe.py`** (table-pass discipline:
changing it requires re-running this script and updating this record; the
discriminator phrase "bibliographic metadata" must survive any future tuning —
the test mocks dispatch on it). The default flip to `bibtex.mode = "auto"`
is cleared by this validation.
