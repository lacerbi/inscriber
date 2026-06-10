# DeepSeek-OCR-2 spike — real-hardware findings

> **Date:** 2026-06-10 · **Status:** spike complete — all three TODO gating
> questions answered. **Verdict: NOT adoptable as the default OCR backend
> yet** — v2 clearly beats v1 on equations/loops/reading order, but loses
> **more than half the numeric values in dense tables** (silently, on both
> the server and mtmd-cli paths), breaking the §9.7 raw-blob fallback
> premise. Re-test on future llama.cpp builds.
>
> Hardware/build: llama.cpp build 9587 (`d2e22ed97`), RTX 4060 laptop 8 GB.
> Models: `sabafallah/DeepSeek-OCR-2-GGUF` **bf16** (5.6 GB model + 886 MB
> mmproj; q8_0/Q4_K_M untested). Harness:
> `dev/scripts/dsocr2_check.py` (the `gundam_check.py` calibration discipline
> adapted to v2's server flags); raw outputs under `out-dsocr2/` (gitignored).
> Test document: PriorGuide (`openreview.net/pdf?id=G4I23g5Ugh`) — chosen
> because `dev/benchmarks/openreview-G4I23g5Ugh/` pins the gold transcription
> AND the v1 error inventory on the same pages.

## Setup facts (re-verified upstream before running, 2026-06-10)

- **Server invocation** confirmed against PR #20975's own example: v2
  *requires* `--chat-template deepseek-ocr --no-jinja` on the server path
  (the opposite of v1, which must NOT pass a template), plus
  `--flash-attn off`, `--no-warmup`, and its own DRY tuning
  `--dry-multiplier 0.8 --dry-base 1.75 --dry-allowed-length 2
  --dry-penalty-last-n -1 --dry-sequence-breaker none` (clears the default
  `\n` sequence breakers).
- **Prompt unchanged from v1**: the official `deepseek-ai/DeepSeek-OCR-2`
  model card lists `<|grounding|>Convert the document to markdown.` as the
  document prompt (and `Free OCR.` as the ungrounded one).
- **Tiling threshold** (from the upstream test's own comment): inputs whose
  dims are ≤ 768 px stay a single 1024 global view; larger inputs get the
  global view + a grid of 768-px tiles.

## Q1 — Grounding format + coordinate frame under tiling: SAME FRAME, small format deltas

Calibration page (known box at (150,200,450,520) pt on 600×800 pt;
per-axis prediction `[250,250,749,649]`, padded-square `[312,250,687,649]`)
rendered at 640 / 1024 / 1280 / 2048 long edge:

| render | encoding (slot n_tokens) | emitted `image` box | Δ per-axis | Δ padded |
| --- | --- | --- | --- | --- |
| 640 | 336 — single global view | `[245,245,752,650]` | 3.5 | 34.5 |
| 1024 | 1200 — global + tiles | `[241,244,752,650]` | 4.8 | 35.8 |
| 1280 | 1200 | `[241,244,752,650]` | 4.8 | 35.8 |
| 2048 | 1200 | `[241,243,752,651]` | 5.2 | 36.2 |

- **Frame: per-axis 0–999, identical to v1 on 9587** (Δ within v1's own
  Δ≈4–6 calibration band; padded-square clearly rejected). **Tiling does NOT
  change the frame** — coords are global-image-relative on both sides of the
  tiling threshold, and **render-size-invariant** 640→2048. `grid_to_norm`
  carries over unchanged.
- **Token count saturates by ~1024 long edge** for a 3:4 page (the tile grid
  looks aspect-ratio-driven): 1024/1280/2048 all encode to 1200 tokens.
  Unlike v1 (where ≥1664 px input flips to a larger saturated encoding —
  the reason `gundam` renders 2048), **rendering above ~1024–1280 buys v2
  nothing**. Outputs still differ slightly across those renders (resampled
  pixel content), but the encoding budget doesn't.
- **Block format: same `LABEL[[x1,y1,x2,y2]]` block layout as v1** — parses
  with the v1 `MARKER_RE`; `title` carries `#`, `sub_title` carries `##`,
  `image` is an empty block, math arrives as `\(…\)`/`\[…\]`.
- **Format deltas a v2 backend must handle** (vs `ocr/deepseek.py`):
  1. **Caption label is `figure_title`** — for BOTH figure and table
     captions (v1 emits `image_caption` / `table_caption`). Not in v1's
     `CAPTION_LABELS`.
  2. **Blocks come in document order**, and the **`table` block carries its
     own `<table>` HTML** (v1@9587: `table` is empty and the *following*
     `table_caption` carries caption + HTML — the §9.7 matcher anchors on
     that). In v2 the caption block *precedes* the table; a figure caption
     *follows* its `image` block. Caption pairing must check both neighbors.
  3. **Side-by-side panels can split into multiple `image` blocks** sharing
     one `figure_title` (PriorGuide Fig. 2 → two boxes, one caption). v1
     emitted one box per figure on the same page.

## Q2 — Loop check on the known-bad page: PASSED (and then some)

PriorGuide p. 5 — the Eq. 6–9 array whose three adjacent underbraces sent v1
into an ~80× verbatim loop to the 8192 cap (~6 min, everything after Eq. 9
silently lost; `2026-06-10-equation-fidelity-findings.md`):

- v2 @2048: **`finish_reason=stop`, 24 s, 7,059 chars, no loop** — the page
  is complete including the two paragraphs + footnote after Eq. 9.
- **Each array row emitted as its own `equation` block with its own
  `(6)`–`(9)` tag** — v1 collapsed all 5 multi-row arrays in this paper to a
  single tag (a whole error class from the benchmark inventory gone).
- Residual quality on the hardest row (Eq. 9): first underbrace intact and
  correctly annotated, second garbled (annotation displaced, "prior ratio" →
  `p r i r a t i o`); a new cosmetic artifact class —
  **letter-spaced `\mathrm{}` text** (`\mathrm{t r a i n}`) — appears in
  hard math regions. v1's systematic subscript slips persist in different
  form (`θ_i` for `θ_t`, `p_traint`/`p_taint` in body text), and v1's
  hyphen-spacing artifact (`parameter- data`, `VE - SDE`) is **gone**.
- ⚠️ The repetition problem is **not gone — it morphed** (see Q4): a dense
  table at 2048 degenerated into a *self-terminating* spam run.

## Q3 — Real-page format capture: DONE

Raws under `out-dsocr2/`: pages 1, 2, 5, 9, 27 @2048; page 9 also at
640/1024/1280; mtmd-cli run on p. 9 @2048. Highlights:

- p. 1 (title/authors/abstract): block shapes identical to v1's
  `deepseek_paper_p1_raw.txt` fixture pattern.
- p. 27 vs the v1 fixture `deepseek_paper_table_p27_raw.txt` (same page,
  same build): near-identical bboxes (table `[337,128,658,223]` vs v1
  `[333,128,663,226]` — per-axis frame confirmed on real content); the
  simple Table A1 blob is complete in both, and v2's is *richer* (proper
  `p_{\text{train}}(θ)` subscript where v1 read `p train(θ)`).
- Parser fixtures are NOT pinned yet — that's backend-implementation work,
  deferred with adoption.

## Q4 — NEW FINDING: dense-table value loss (the adoption blocker)

PriorGuide Tables 1+2 (the two wide multi-level-header results tables;
**v1's blobs held all 216 numeric cells, verified** — the benchmark's
error inventory, where the v1 damage was purely structural):

| render | Table 1 cells | Table 2 cells | retained |
| --- | --- | --- | --- |
| 640 (global view only) | 0 | 1 | ~0% |
| 1024 | 57 | 57 | **53%** (best) |
| 1280 | 49 | 23 | 33% |
| 2048 | 61 | 0 — see below | 28% |
| 2048, mtmd-cli | 56 | 9 | 30% |

- v2 emits **structured `<td rowspan/colspan>` HTML** (v1: flat concatenated
  text) — structurally richer but **value-lossy at every render**: whole cell
  runs missing, garbled values (`18(0.07)` for `0.18(0.07)`), words truncated
  mid-cell (`PriorGuid`, `PriorGu`).
- At 2048, Table 2's blob degenerated into a growing-whitespace
  `\( \downarrow \)` spam run that **self-terminated with
  `finish_reason=stop`** — invisible to the truncation detector (the
  DESIGN §2.2 known-limitation class: a loop that ends below the cap) AND to
  a consecutive-identical-line heuristic (it is one giant line).
- **mtmd-cli cross-check (same image, PR flags): same ballpark** (65/216) —
  this is NOT a server-path bug (cf. issue #22785's Qwen-VL server-vs-cli
  degradation — not supported here); it is how v2-in-llama.cpp behaves on
  dense tables. (Side finding: `llama-mtmd-cli` no longer crashes on 9587 —
  the build-9028 `STATUS_STACK_BUFFER_OVERRUN` of M1a is gone.)
- **Consequence for inscriber:** §9.7's load-bearing fallback premise — *"on
  any failure keep the blob; the blob still holds every value"* — **fails on
  v2**. The digit-coverage guard cannot catch it either: it compares VLM
  output against the blob, and here the *blob itself* is the lossy artifact.
  Dense-table value loss would be **silent end-to-end**.

## Adoption assessment

| dimension | v2 vs v1 |
| --- | --- |
| equations / multi-row arrays | **v2 much better** (no loop on the known-bad page; per-row tags) |
| reading order / block order | **v2 better** (document order, matches the paper's claims) |
| hyphen/spacing artifacts | **v2 better** (v1's `word- word` class gone) |
| simple tables | comparable; v2 blob richer (LaTeX in cells) |
| dense/wide tables | **v2 far worse — silent >47% value loss at best render** |
| speed (4060, per page @2048) | 12–29 s — comparable or faster; no 6-min loop pages |
| glyph-level misreads | different profile, neither clean (`\sim`→`-`/`<`; letter-spaced `\mathrm`) |

**Decision: defer adoption.** The dense-table failure is silent, hits exactly
the content class the table pass exists for, and has no detector.

**The loss is most likely a llama.cpp implementation issue, not the model**
(unproven — circumstantial): (a) the paper's OmniDocBench gains would be
impossible if the true model dropped half of dense-table cells; (b) the PR's
own parity gate is "intentionally loose" (one low-quality test image, no
table content); (c) the PR discussion flags the new **Qwen2-encoder
attention mask** (non-causal image tokens vs causal query tokens, prepared
CPU-side) as the subtle part — a slightly-wrong mask degrades dense
fine-grained content exactly like this while leaving simple pages intact;
(d) both server and mtmd-cli are affected, consistent with a bug in the
shared encoder/preprocessing rather than either front-end. The
discriminating test, when worth running, is the **HF reference
implementation on the same p9 PNG** (`deepseek-ai/DeepSeek-OCR-2` `infer()`
with `crop_mode=True`) — if HF retains the cells, file it upstream.
No dsocr2-targeted fix has landed upstream as of 2026-06-10 (checked
`tools/mtmd` history through #24357); the build was re-pinned to 9587 only
hours before this spike, so the sensible posture is to **wait for upstream
maturation** and re-run this spike (the harness is ready) on future builds.
If a future build fixes table retention, the backend work is well-scoped:
a `deepseek-ocr-2` `OcrBackend` with its own `server_flags()` /
`chat_template()` (plumbing verified — `ServerSpec.chat_template` +
`extra_flags` carry everything), the Q1 parser deltas (caption label,
caption-before-table, table-carries-blob, multi-`image` figures), pinned
fixtures from `out-dsocr2/`, and a **1024–1280 render target** (encoding
saturates; 2048 buys nothing and hit the worst degeneration observed).
