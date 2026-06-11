---
name: inscribe
description: Convert an academic PDF (local path or paper-repository URL) to Markdown with the inscriber CLI, then verify the transcription against the source PDF with parallel subagents and apply the important fixes. Use when the user asks to inscribe, convert, or transcribe a paper or PDF.
---

# Inscribe a paper (convert + verify)

The user's request holds the input (PDF path or URL) plus options, stated either
as inscriber flags or in plain words. The pipeline: run `inscriber`, then —
unless the user said to skip it — verify the output against the source PDF with
subagents and apply the fixes that matter.

## 1. Read the docs first

**Read `README.md` at the repo root before anything else.** It defines the CLI
surface, the supported URL repositories, and every option flag — map the
user's plain-language options onto flags from there, never from memory.
Real runs need the machine-local `./config.toml` (llama.cpp + model paths); if
the run fails on configuration or the minimum llama.cpp build, surface the
error verbatim and stop.

## 2. Run inscriber

- Python: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (POSIX),
  from the repo root.
- Default output dir `out/` unless the user gave one:
  `… -m inscriber run <INPUT> -o out/ [flags]`
- OCR + VLM on real hardware takes minutes: run it with a long timeout or as a
  hidden background process with stdout/stderr logs, then monitor it to
  completion. Watch stderr for warnings — especially `truncated` page warnings
  (note the page numbers; they get extra scrutiny in step 4). Do not finish
  while an `inscriber` process needed for the request is still running.
- On success it prints the written files to stdout: the full `<base>_full.md`,
  splits (`<base>_main/_appendix/_backmatter.md` unless `--no-split`),
  `<base>.bib`, `figures/`. Take `<base>` from that printed list, not from the
  input filename — by default the BibTeX citation key (e.g.
  `chang2025amortized`) names the outputs when an entry was produced.

## 3. Make the source PDF available for verification

The verifiers need the actual PDF next to the outputs:

- **Local path input** — use the user's file directly, in place. It is theirs:
  never move, rename, or delete it.
- **URL input** — download the PDF to `out/<base>.source.pdf` (transform to
  the direct-PDF link per the repository rules in README, e.g.
  `arxiv.org/abs/X` → `arxiv.org/pdf/X`). Only this downloaded copy is
  temporary: delete it in step 6 unless the user asked to keep it.

Get the page count (e.g. PyMuPDF one-liner via the venv python). If the run
used `--pages`, only those pages exist in the output — verify only those.

## 4. Verification (default ON — skip only if the user said so)

Partition the processed pages into chunks of **at most 10 pages**. If the user
has explicitly requested or confirmed subagents/parallel verification, spawn a
`worker` subagent for each chunk and launch chunks in parallel, in multiple
rounds if the paper is long. Omit the model override by default; for mechanical
chunk checks where a cheaper model is appropriate, use `gpt-5.4-mini`. If the
user has not explicitly authorized subagents, ask once before spawning them.
Each subagent prompt must contain, explicitly:

1. The absolute PDF path and its page range. Tell the subagent to inspect only
   that range, using local PDF tools such as PyMuPDF rendering/extraction from
   the repo venv when needed. Born-digital PDFs carry an exact text layer:
   instruct the subagent to cross-check table cells, SEM digits, and
   subscripts against PyMuPDF `page.get_text()` output instead of reading
   tiny numbers off the rendered page — rendered-page reads have produced
   confidently-wrong verdicts in both directions (claimed-blank cells that
   hold values; "confirmed" digits that were wrong).
2. The absolute paths of the Markdown outputs to check (the split files when
   they exist, else `<base>_full.md`) and a note that there are no page markers —
   locate the chunk's content by matching headings/text.
3. Any pages flagged `truncated` in its range (these likely end in a
   repetition loop with content missing after it).
4. A note that the subagent is not alone in the codebase and must be read-only:
   do not edit files, revert changes, delete outputs, or modify caches.
5. **The typical failure modes, in priority order — paste this list into the
   subagent prompt:**
   - **Tables (highest risk — verify the header row AND every numeric cell).**
     Column-attribution damage is the most dangerous class because the values
     are usually correct while the labels lie: mislabeled or invented column
     headers (`TNP-A (Set 1)/(Set 2)` over four different methods), a phantom
     extra column shifting every label one place, an entire column silently
     dropped, colspan group headers collapsed so sub-labels (`AR`/`Ind`)
     vanish — compare the header tokens and the column COUNT against the PDF,
     not just the cells. Also: header subscript mangling
     (`q_mild`/`q_strong`/`q_mixture` → `q&out`); cell drift in sparse rows
     (rows of mostly `—` with 1–2 values placed in the wrong column);
     single-cell value slips (`±0.24` for `±0.19`); silently dropped rows. A
     raw HTML `<table>` blob (instead of a pipe table) means the
     restructuring fell back — its values may be fused together
     (`159.99346.68300.4`); report the correct segmentation.
   - **Subscripts and short words in text/equations.** `θ_t` → `θ_i`,
     `p_train` → `p_min`, `Fail` → `Full` — the misread is always plausible,
     so compare symbol-by-symbol against the PDF wherever a claim depends on
     the identifier.
   - **Equations.** Dense display equations (underbraces, multi-line arrays)
     are loop-prone: check completeness and LaTeX fidelity.
   - **Truncated pages.** Verify where the content stops, what is missing
     after the loop point, and report the missing span.
   - **Figure descriptions** (`> **Image description.**` blockquotes): the
     description must match the actual figure — axis labels, panel structure,
     trends, and any numbers it quotes. Also check the caption transcription.
   - **Boundary damage.** Dropped lines at page boundaries (header/footer
     stripping can overreach), bad de-hyphenation joins, and missing or
     duplicated content at the main/appendix/backmatter split boundaries.
   - **References.** Garbled author names, years, venues.
6. The report format: **read-only — do not edit any file.** Return only
   important fixes (wrong values, wrong symbols, missing/garbled content —
   not style), each as: target file · nearest heading · exact current text
   (quoted, long enough to be a unique match) · corrected text · severity
   (critical/minor) · confidence. Say explicitly if a section could not be
   verified (e.g. unreadable PDF page).

## 5. Review and apply fixes

- Review every reported fix yourself; for anything surprising or
  low-confidence, check the PDF page directly before accepting. Reject
  "fixes" that re-style rather than correct.
- Verifier reports err in BOTH directions: quoted "current text" that does
  not exist in the file (grep for it before editing — fabricated quotes
  happen), claimed-blank PDF cells that actually hold values, and cells
  "confirmed correct" that are wrong. Re-check every accepted table/math fix
  against the PDF's extracted text layer (PyMuPDF `get_text()`), not the
  rendered image — it is exact for born-digital PDFs and catches both
  failure modes cheaply.
- Apply accepted fixes **to the split files** (`*_main.md`, `*_appendix.md`,
  `*_backmatter.md`) — never to `<base>_full.md` directly. Then regenerate the
  full document from the corrected splits:
  `… -m inscriber join out/<base>`
- With `--no-split`, edit `<base>_full.md` directly (no join needed).
- If a `truncated` page is missing real content the subagent transcribed from
  the PDF, splice the transcription in.

## 6. Report and clean up

If (and only if) step 3 downloaded `out/<base>.source.pdf`, delete that copy
(unless the user asked to keep it). A user-provided PDF is never touched.
Then report:
the output files, fixes applied (grouped by kind, with counts), suggestions
rejected and why, and remaining risks (e.g. truncated pages, tables kept as
raw HTML).
