# End-to-end quality check — full paper on build 9587

> **Date:** 2026-06-10 · **Status:** concluded — action items tracked in `TODO.md`.

> First full-paper quality measurement on the re-pinned llama.cpp build 9587
> (same-day re-pin: `2026-06-10-build-9587-verification.md`). Document: **PriorGuide**
> (ICLR 2026, `openreview.net/pdf?id=G4I23g5Ugh` ≡ arXiv 2510.13763) — chosen
> deliberately: it is the same paper as the 9028-era
> `2026-06-10-equation-fidelity-findings.md`, giving a direct before/after comparison.
>
> **Verdict:** pipeline mechanics flawless (per-axis crops confirmed on a real
> multi-figure paper; zero placeholder/description failures; no loops); OCR
> equation quality matches the 9028 error profile (same vision-level classes,
> runaway loop gone); **tables are the weak spot** — all 10 captured and
> restructured, but 5 of 10 carry real structure damage on dense multi-header
> layouts. Action items tracked in `TODO.md`.

## Setup

- llama.cpp **build 9587 (d2e22ed97)**, RTX 4060 Laptop 8 GB; DeepSeek-OCR
  BF16 + BF16 mmproj at `large` (1280 px); Gemma 4 E4B QAT Q4_K_XL + BF16
  mmproj; default config (figure mode `describe-only`, table refine on,
  ctx 16384, temp 0). `inscriber run <url> -o out`, 38 pages.
- OCR ~28 s/page; **all 38 pages `finish_reason: stop`** (no truncation), no
  loop symptoms in any cached raw output (max consecutive-repeated-line = 1
  on every page).
- Method: three independent reviews (equations / tables / figures), each
  comparing the output markdown cell-by-cell / symbol-by-symbol against
  PDF pages rendered at 2000–2200 px. Render/crop artifacts were written
  under `out-check/` (gitignored, ephemeral) — the key evidence is quoted
  inline below.

## Pipeline mechanics — all clean

- 9 figures detected → 9 crops → 9 descriptions; 0 `⟦INSCRIBER_FIG⟧` leaks,
  0 `[figure description unavailable]`, captions adjacent to every block.
- **Crop placement: no per-axis frame errors.** Every crop (reproduced
  exactly: 1280 long-edge render × `bbox_norm` + 0.02 padding) contains the
  complete figure — all panels, axes, legends — including the hard cases
  (p37's full-page triangular corner grid, p38's two corner plots). This is
  the first real-paper confirmation of the ≥9587 per-axis mapping.
- 10 `<table>` blobs detected → 10 restructured (no fallback-to-blob events).
- Splitter produced main/appendix/backmatter correctly.

## Figures — excellent

7/9 descriptions fully accurate (figure type, panels, axes, legends, trends).
Two minor description flaws (Gemma-side, not crop-side):

- p10 (Fig 3): MMTV hallucinated as "Mean Maximum Test Value" (paper: Mean
  Marginal Total Variation).
- p37 (Fig A4): "121 plots in an 11×11 matrix" — actually a lower-triangular
  corner grid (~66 panels).

(p11's EU/EuroHPC funding logos were detected and described as such —
acceptable; no caption, so no `Figure N` block.)

## Equations — same error profile as 9028, loop gone

~26 of 37 numbered display equations content-exact (13 exact including
equation-number tags). All error classes from the 9028 study **recur on the
same equations**, confirming they are model-inherent vision misreads, not
build regressions:

| class | 9028 (equation-fidelity-findings) | 9587 (this run) |
| --- | --- | --- |
| Eq (9) triple-underbrace array | **runaway loop**, ~80× row repeat to the 8192 cap, rest of page lost | no loop; one duplicated row + "prior ratio" underbrace label lost |
| Eq (A22) big fraction | denominator dropped | denominator dropped again (34× `\qquad` filler in its place) |
| `θ_t → θ_i` subscript swap | systematic in A2–A13 | same (27× in A5–A13), plus `Σ̃ → S̃` (19×) |
| `p_train` tiny-subscript misread | `p_min`/`p_sim` clusters | `p_min` ×8 through the B.2 proof |
| equation-label collapse | all 5 multi-row arrays kept one tag | ~8 arrays keep one tag (e.g. (6)–(9)→(9), (A6)–(A10)→(A10)) |

New one-off glyph errors: `σ(t)²I → σ(t)zI` (Eq 4), `μ^new → μ^prior`
(Eq 16), `q(θ) → π(θ)` (Eq 13 prose), `N_points → N_beam` (Turin).
Delimiters remain uniformly `\(…\)`/`\[…\]` in OCR text (the few `$…$`
spans are inside VLM-generated description/table regions — harmless).

**Takeaway:** the 9587 upgrade removed the catastrophic failure mode
(runaway loop destroying page remainders) without changing the underlying
misread rate. The "no normalization pass" decision stands; the
loop/truncation-detection TODO stands (this run had nothing to detect, but
the guard is still absent).

## Tables — the weak spot

All 10 tables present (10/10 vs the PDF; none missing, none invented —
every output digit traces to the page). But: **2 clean · 3 value-perfect
with wrong shape · 5 with real damage.** Two distinct failure layers:

**(b) Structure damage (VLM restructuring, dense multi-header layouts):**

1. **Table 1: an entire column group silently dropped** (`q_mixture`,
   3 cols × 18 rows) — output is a syntactically clean 6-column table that
   looks complete. The scariest failure mode observed.
2. **Table A6:** 11×7 grid drifted ~1 cell/row — only row 1 aligned.
3. **Table A5:** cascading row-label misalignment (each method row holds a
   different method's numbers) + one value dropped.
4. Table 2: full method/column transpose (all 36 values correct, layout
   wrong); A3/A4: SIR rows shifted one column; A7/A8: spurious columns from
   multi-level headers.

**(a) OCR digit damage (DeepSeek-side, present in the blob before the VLM):**

- Dropped leading digits: `9346.6 → 346.6`, `8300.4 → 830.4` (A8),
  `1.03 → 0.03`, `0.82 → 0.02` (Table 1).
- Digit duplication: `10 → 1010`, `20 → 2020` (A1 `dim(x)`).
- `Fail → Full` ×6 (A4/A5) — meaning-flipping.
- `#GMM components` `2/20/200 → 2/5/10` (A7) — corrupts the ablation's
  independent variable.

**Takeaway:** simple tables come through clean; wide/dense multi-header
appendix tables exceed what Gemma can resolve from the whole page downscaled
to ~896 px (the known blocker on the cropped-table-input TODO item) and what
DeepSeek reads reliably at 1280 px. The restructuring contract held in one
important sense — **no hallucinated values** — but "looks clean" is no
guarantee of completeness (Table 1). New TODO items filed: a guard against
silent structure damage; the equation-tag/table question also feeds the
gundam render-target decision (more vision tokens may help both).
