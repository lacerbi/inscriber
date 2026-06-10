# Cropped-table-input validation (page-vs-crop on real hardware)

> **Date:** 2026-06-10 · **Status:** concluded — cropped prompt **validated and
> frozen** (DESIGN §9.7); digit-coverage guard shipped (`MIN_DIGIT_COVERAGE`);
> residual shape/label issues folded into the structure-damage TODO item.

> Validation run for the cropped-table-input feature (committed earlier the
> same day, `dev/plans/PLAN-cropped-table-input.md`): does sending the VLM a
> crop of the grounded `table[[bbox]]` region beat the validated whole-page
> input? Harness: `dev/scripts/table_crop_check.py`. Document: **PriorGuide**
> (arXiv 2510.13763) — the same paper as
> `2026-06-10-e2e-quality-findings.md`, whose §Tables verdict (2 clean /
> 3 value-perfect-wrong-shape / 5 damaged at 1280-OCR + whole-page input) is
> the baseline. All 10 tables, pages 9, 27, 32, 33, 35, 36, 37.

## Setup

- llama.cpp build 9587 (d2e22ed97), RTX 4060 Laptop 8 GB; DeepSeek-OCR BF16 at
  2048 px (the gundam default); Gemma 4 E4B QAT Q4_K_XL + BF16 mmproj;
  ctx 16384, temp 0, thinking on. Each table run **twice in one session**:
  whole-page input + validated prompt vs table crop + the cropped prompt
  variant. All 20 calls `finish_reason: stop`, all outputs sanitize-clean.
- Note when comparing against the e2e baseline: the page-input arm here uses
  the **2048 px** raster (the e2e run's table pass used 1280 px pages), so the
  whole-page numbers below are themselves a slightly stronger baseline.

## Discovery 1 — the blob lives in the CAPTION block (matcher fix)

The first harness round matched **0/10** blobs despite 10/10 pages emitting
grounded `table` regions. Raw output shows why — tables mirror the
`image`/`image_caption` pairing exactly:

```text
table[[333, 128, 663, 226]]          ← EMPTY block (like `image`)
table_caption[[334, 101, 664, 117]]
Table A1: Characteristics of the simulator models.

<table>Modeldim(θ)dim(x)p train(θ)…</table>
```

The `table[[bbox]]` block has **no text of its own**; the following
`table_caption` block carries the caption line AND the `<table>` HTML.
Consistent across all 10 tables (and the e2e note's 1280 baselines).
`match_table_regions` now anchors on the region's own text **or the
immediately following region's text**; the real capture is committed as
`tests/fixtures/deepseek_paper_table_p27_raw.txt` with a golden
parse→blob→match→bbox test. Recorded in DESIGN §2.2.

## Discovery 2 — crop completeness: 10/10

Every crop (bbox + 0.02 padding from the 2048 raster) contains the complete
table — headers, all rows, both rules — plus harmless padding slivers (a
caption tail above, a line of body text below). Bbox quality is not the risk
it was feared to be on this build.

## Page-vs-crop results (per table, cell-by-cell against the PDF)

| table | whole-page input | cropped input |
| --- | --- | --- |
| 1 (p9, 18×9, 3-level header) | all 162 values present; headers fused (`RMSEC\|2ST`); Simformer rows mislabeled with simulator names | ⚠️ **6 value-rows silently dropped** (54 values): each block keeps Simformer's values under an "ACCE" label, true ACE values gone. The silent-data-loss mode. |
| 2 (p9, 2-level header) | invented `RMSE 1…MMD 6` headers + 6 phantom empty columns | **values 36/36 exact**, bold preserved, structure right; group labels degraded to the blob's `q&out` misread (see below) |
| A1 (p27) | clean — incl. correct `10\|10`, `20\|20` splits | clean; more faithful header typography (`p_train(θ)`) |
| A2 (p32) | clean | clean + bold preserved |
| A3 (p33) | header exploded into a 16-column doubled set | **values 60/60 in order**; one header split (`C2\|ST`) drifts two column labels |
| A4 (p33) | headers mangled (`SEC\|STMM\|TV`); rows aligned | headers perfect; the 2 SIR rows drop the `—` under Acc Rate → ESS/k̂ shifted one cell |
| A5 (p33) | **2 ACE rows lost** + digit dropped (`10.61→0.61`) + SIR cells scrambled | all 12 rows present, ACE values correct; SIR cells still scrambled (`~1` glyphs lost, a spurious `0.001`) |
| A6 (p35, 11×7) | one-cell-per-row drift catastrophe + bogus headers (the historical failure, again) | **structurally perfect grid**; ONE ± slip (`±0.24` for `±0.19` — blob had it right); `0.00→0.0` reformat |
| A7 (p36) | `C2ST MMTV` → `C2S\|TMM\|TV` + a spurious column of repeated junk | **perfect** |
| A8 (p37, the fusion probe) | fused `159.99346.68300.4` split WRONG (`346.6`, `830.4` — leading digits lost) | **perfect**: `159.9 \| 9346.6 \| 8300.4` |

**Score: crop better on 7, tied on 2, catastrophically worse on 1 (Table 1).**
The crop directly fixes the two error classes resolution couldn't touch
(fusion mis-splits, row/column drift) — but on the densest table it produced
the *worst possible* failure: a clean-looking table missing a third of its
data. (Table 1 has never been transcribed safely by any mode: the e2e
whole-page run dropped its `q_mixture` column group.)

Timing: crop-input is usually faster (A7 29 s vs 58 s; A8 24 s vs 39 s) —
less thinking spent untangling page context.

## The digit-coverage guard (shipped with this note)

The Table 1 failure has a clean, cheap detector: the **digit stream** (all
digits concatenated, in order of appearance). It is invariant under correct
re-segmentation — the A8 fusion fix preserves every digit — while dropped
rows delete a visible chunk. Measured over all 20 captured outputs
(blob digits counted after stripping tags + entities):

| output | digit-coverage ratio |
| --- | --- |
| Table 1 crop (the 6-row drop) | **0.664** |
| every other output (19/19) | 0.976 – 1.208 |

`digit_coverage_ok` (`tables.py`, `MIN_DIGIT_COVERAGE = 0.8`) rejects a
restructured table below the floor → raw blob kept, never cached. The old
value-*count* check stays rejected (merged cells make counts unreliable);
counting raw *digits* sidesteps segmentation entirely. One-sided by design:
added digits (split labels, duplicated cells) are not data loss. With the
guard, Table 1 degrades to its raw blob — every value present.

## Residual issues (the honest list — now in the structure-damage TODO item)

1. **Blob header misreads propagate** — DeepSeek read all three
   `q_mild/q_strong/q_mixture(θ)` subscripts as `q&out(θ)` (Tables 1–2 blobs);
   the crop pass faithfully copied the blob per its contract, and neither
   input mode recovered the names from image or caption. OCR-side subscript
   limit, not a VLM regression.
2. **Sparse-row drift** — rows that are mostly `—` (the SIR/RS rows in A4/A5)
   still misplace 1–2 cells; below the guard's radar by construction.
3. **One-cell value slips** survive (A6's `±0.24` for `±0.19`) — no count- or
   coverage-based check can see these.

## Decisions (2026-06-10)

1. **Cropped input stays the default; the cropped prompt variant is validated
   and FROZEN** (DESIGN §9.7 pinned-prompt rule — do not reword without
   re-validating). 7/10 better, 2/10 equal; both fusion-split probes and the
   drift probe fixed.
2. **`MIN_DIGIT_COVERAGE = 0.8` guard shipped** as part of output sanitation —
   it converts the one observed catastrophic mode into the standard
   keep-the-blob fallback. This discharges the "at minimum" tier of the
   structure-damage TODO item; the investigation tier (shape/label damage the
   guard can't see) stays open with the residual list above as its evidence.
3. The caption-carried blob shape is **confirmed model behavior** (DESIGN
   §2.2) with a committed fixture + golden test.

Raw artifacts: `out-tablecrop/` (gitignored, ephemeral) — per-table
`*_blob.txt`, `*_page.md`, `*_crop.md`, crops, page renders, raws; the key
evidence is quoted above.
