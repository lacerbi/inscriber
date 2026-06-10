## What this is

`inscriber` is a cross-platform CLI that converts academic PDFs into LLM-friendly
text-only Markdown **entirely locally** via llama.cpp (DeepSeek-OCR for text +
figure grounding; a Gemma 4 VLM for figure descriptions and table restructuring).
It is a Python port of the cloud web app `paper2llm`. No ML libraries in the
package — all inference is a llama.cpp subprocess driven over its
OpenAI-compatible HTTP API.

## Commands

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; source .venv/bin/activate elsewhere
pip install -e ".[dev]"

pytest                            # full suite; mocked inference, no GPU/models needed
pytest tests/test_tables.py -k locator   # single file / keyword
ruff check                        # lint (config in pyproject.toml)

python -m inscriber run paper.pdf -o out/   # real runs need llama.cpp + GGUFs
```

Real runs read `./config.toml` (gitignored, machine-local) — `config.example.toml`
is the tracked template. The maintainer's setup is llama.cpp build 9587 on an
RTX 4060 8GB. Builds **older than 9587 are refused for OCR**
(`DeepSeekOcrBackend.min_server_build` — the grounding coordinate frame changed
upstream; DESIGN §2.2, `dev/notes/2026-06-10-build-9587-verification.md`).

## Where truth lives (read before changing behavior)

- **`DESIGN.md` is the authoritative, living spec** — it states the confirmed
  model behavior directly (OCR grounding format/coordinate frame in §2.1–2.2 and
  §8.3; the table pass and its pinned prompt in §9.7). Code comments cite its
  sections (`§9.7` etc.). When you change behavior, update DESIGN.md, README.md,
  and `config.example.toml` in the same change — this repo treats docs as
  first-class.
- `dev/notes/` holds the **dated lab notes** (`YYYY-MM-DD-name.md`) — the
  empirical evidence records behind those sections. Consult them before
  changing model-facing behavior; when new real-hardware findings land, add a
  new dated note (or an addendum/status line on an existing one) rather than
  rewriting history.
- `TODO.md` tracks concrete pending items (real-hardware verifications, code
  debts, blocked refinements) — add to it rather than burying TODOs in spec
  prose; longer-horizon future work stays in DESIGN §22.
- `dev/` is developer-only material (never user-facing): scripts, the dated
  lab notes above, and `dev/plans/` — executed feature plans and build
  roadmaps, archived as design records.

## Architecture

Pipeline (`inscriber/pipeline.py` orchestrates; DESIGN §3): resolve input →
rasterize (PyMuPDF) → per-page OCR → figure crop → **VLM table restructuring →
VLM figure description → BibTeX citability probe** (in that order — figure
context must see clean tables; the text-only probe shares the open VLM
session) → stitch/clean → split (main/appendix/backmatter) → BibTeX (default
`auto`: provenance/probe citability → source chain; DESIGN §12) → write.

- **Three subcommands**: `run` (end-to-end), `ocr` (writes a portable _bundle_:
  `manifest.json` + `figures/` crops + `pages/` rasters for table pages), and
  `describe` (bundle → VLM + assembly, no OCR model). `run` = `ocr` + `describe`
  sharing in-memory objects. The bundle's `bundle_schema` int is the
  compatibility gate; new manifest fields must be additive or bump it.
- **Sequential single-model-resident by default**: the OCR server is torn down
  before the VLM server starts. Both VLM passes (tables, figures) share one
  lazily-launched server (`_VlmSession`) that only starts on the first cache
  miss. `--mode concurrent` pre-launches the VLM server instead.
- **Backends own their model's quirks**: `OcrBackend.ocr_page` owns the whole
  per-page inference (prompt, calls, parsing, coordinate mapping into the
  original-page `[0,1]` frame); `VlmBackend` owns `describe` and
  `restructure_table`. Registries map names → classes; adding a backend must
  require zero pipeline changes.
- **Caching is content-addressed and load-bearing** (`inscriber/cache.py`):
  per-page OCR cache + shared VLM store (figure descriptions and restructured
  tables, disjoint key payloads). Keys include model+mmproj _content_ identities,
  the llama.cpp **build identity** (`llama_build_identity` in `llama/server.py`
  — `llama-server --version`, or the endpoint's `/props` `build_info`), the
  fully assembled prompt, sampling, and `chat_template_kwargs`. **Anything
  that changes model output must become key material.** `--refresh` recomputes
  and overwrites; `--no-cache` neither reads nor writes. Never cache a failed
  result. One deliberate nuance: a *truncated* OCR page (repetition loop hit
  the token cap) IS cached, flagged `truncated`, and re-warned on every hit —
  its key pins every output-determining knob, so a recompute could only
  reproduce the loop (DESIGN §8.6; the table pass differs because its key
  excludes `ctx_size`).

## Invariants and gotchas

- `⟦INSCRIBER_FIG:{id}⟧` placeholders are the **only** anchors tying figures to
  their position in page markdown (DeepSeek emits no `![]()`); never strip them
  without injecting their replacement.
- **`ctx_size` is the single size knob.** VLM calls send no `max_tokens`
  (generation bounded by the context window; truncation detected via
  `finish_reason != "stop"`). The one exception is DeepSeek-OCR's internal 8192
  cap — it is an anti-repetition-loop guard (llama.cpp lacks the model's n-gram
  penalty), not a verbosity knob. Keep it. Also keep DeepSeek at BF16/F16
  weights (Q4_K_M loops) and `temperature: 0` everywhere.
- Gemma 4 is a thinking model; thinking is explicitly activated per request via
  `chat_template_kwargs: {"enable_thinking": true}` (verified to toggle on build
  9028).
- The table pass falls back to the original `<table>` blob on **any** failure
  (error, truncation, commentary, empty) — the blob still holds every value.
  Bundle page rasters are written **verbatim** so `run` and `describe` share
  table cache keys.
- **Many behaviors are deliberate verbatim ports from paper2llm** (the figure
  prompt, `> **Image description.**` header strings, splitter regexes, BibTeX
  mock/warning text, the allparts section reordering). DESIGN §23–24 maps each
  to its TypeScript source — check there before "fixing" something that looks
  odd.
- Cross-platform rules (DESIGN §15): `pathlib` everywhere, subprocess with list
  args (never `shell=True`), `platformdirs` for config/cache dirs, `.exe`
  suffixing via `config.binary_filename`, and **always write text files with
  `encoding="utf-8", newline="\n"`**.

## Testing conventions

- Tests mock at the **chat-client boundary**: monkeypatch `ChatClient.chat_image`
  AND `ChatClient.chat` (the text-only BibTeX probe lands on `chat`), and fake
  `LlamaServerManager.serve` (see `tests/test_pipeline_mocked.py`,
  `tests/test_tables.py`). Mock prompts are discriminated by content
  (`"<|grounding|>"` / `"Convert the document to markdown"` for OCR,
  `"reconstructing ONE table"` for tables, `"bibliographic metadata"` for the
  BibTeX probe, else figure). Probe fakes default to `{"citable": false}` so
  default-`auto` runs stay inert and network-free in tests.
- Use the `hermetic_cache` fixture pattern (monkeypatch `cache.default_cache_dir`
  / `default_vlm_cache_dir` into tmp) — never let tests touch the real
  platformdirs cache.
- `tests/fixtures/deepseek_paper_p1_raw.txt` is the golden real-output format the
  DeepSeek parser is pinned to; extend it rather than inventing new shapes.
- Changes to llama.cpp-facing behavior (prompts, template kwargs, server flags)
  cannot be proven by mocked tests — verify on real hardware (see
  `dev/scripts/` for prior spike patterns) and record findings in `dev/notes/`.
