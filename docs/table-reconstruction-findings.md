# Table reconstruction findings (post-v1 investigation)

> **Status:** Exploration / not yet implemented in the pipeline. Captures what was
> learned about improving table quality, and the prompt that works, so the work
> isn't lost. Tested against arXiv `2510.09477v2` (39-page ICLR paper) on
> llama.cpp build 9028, RTX 4060 8GB.

## The problem

DeepSeek-OCR (the v1 OCR backend) emits tables as **degenerate HTML** — it wraps
the region in `<table>…</table>` with a few stray `<td colspan>` / `<br>` tags but
**no `<tr>` rows and most cell boundaries missing**, so adjacent cells concatenate
(`Dep. Variable:CCSR-squared:0.616`). The content (all values) is present but the
grid is gone, and it is **not post-fixable** (no separators to split on). This is
the model's native output under `<|grounding|>Convert the document to markdown.`,
confirmed from the raw cache (`raw_output`), not an inscriber transform.

## Alternatives explored

1. **GLM-OCR as a text backend** (added experimentally, `ocr/glm.py`). It emits
   clean Markdown pipe tables (a real win) and clean hyphenation, but in llama.cpp
   it produces **no `#` headings and no figure boxes** — GLM-OCR's document
   structure comes from a separate layout stage (PP-DocLayout-V3, a PaddlePaddle
   model **not in llama.cpp**), exactly like the figure-grounding gap (DESIGN
   §22.1). So standalone it breaks the splitter and loses figures. It also drops
   columns on complex multi-level headers. Kept as an experimental backend, not a
   drop-in replacement.
2. **PP-DocLayout-V3** — rejected: pulls a full second ML runtime (PaddlePaddle)
   into a tool whose design is "llama.cpp does all inference" (DESIGN §18.1).
3. **Gemma (the existing VLM) re-structures each DeepSeek `<table>` blob** — the
   chosen direction below.

## Chosen direction: Gemma cleans DeepSeek tables

Keep DeepSeek for everything it does well (headings, figure boxes, reading order),
and for each `<table>` blob run a Gemma pass that **restructures** it. The split of
labor is the key: DeepSeek's blob already has the **values** (it just lost the
grid); Gemma **sees the layout** in the page image and arranges the known values.
That makes Gemma's task low-risk *structuring*, not from-scratch re-OCR (so it
doesn't hallucinate numbers — it even copies DeepSeek's typos verbatim).

### The prompt that works

Three ingredients, each added after a failed simpler version:

1. **Count-aware locator** — "This page contains N tables; reconstruct the i-th
   (the one whose values match the OCR below)." Disambiguates which table on a
   multi-table page (whole-page input, no crop needed). Adjust wording for N=1.
2. **Correct-when-certain** — tell Gemma the OCR may have *merged* adjacent labels
   or values, tables may be *irregular* (column groups with differing sub-column
   counts), and it may **fix clear OCR mistakes when certain** (split run-together
   labels/values), but never invent data or drop values to look uniform. This is
   what recovered a ragged column a plain "preserve every value" instruction could
   not.
3. **Page-text context** — pass the rest of the page's prose (tables and figure
   placeholders stripped) so the caption/body can supply correct spellings for
   merged header labels. Fixes labels that appear in the prose; inert otherwise.

### What it achieves (over Tables 1, A6, A7 of the test paper)

- **Data values: reliably correct** — 63/63 across a flat table, a 2-level-header
  table, and a ragged 11-column / 3-level-header table; all correctly grouped.
- **Headers: mostly correct** — the page-text context fixes most merged-label
  errors (e.g. `ARInd → AR | Ind`, `JenaCali → Jena | Cali.`). Residual mis-splits
  only when a term is illegible in the downscaled image *and* absent from the
  prose — minor, human-recoverable, never silent data loss.
- Handles ragged tables; does not over-structure simple flat tables.

### Rejected guard

A **value-count check** (compare blob value count to output cell count) was
considered and **rejected**: DeepSeek *merges* cells, so the blob's value count is
not a reliable baseline (false positives and negatives).

## Notes

- **Gemma 4 is a thinking model.** Hard reconstructions spend ~2k reasoning tokens
  before answering (stripped from `content`, not surfaced in `reasoning_content`).
  The VLM pass runs at `max_tokens=4096`, which absorbs think+answer; tables needed
  ~6–8k. The truncation marker (`[...]`) flags responses that hit the cap.
- **Cropping** would sharpen headers further (Gemma downscales the whole page to
  ~896px, so small header glyphs are lost) — but DeepSeek does not ground tables
  with boxes, so there is no clean table bbox to crop to. Open problem.

## Next steps (if implemented)

1. A post-OCR `table_merge` pass: per page, for each DeepSeek `<table>` blob, run
   the Gemma prompt above (image + page-text context) and splice the result in.
2. Reuses the VLM server already up for figure description; cache per table.
3. Decide whole-page (works now) vs. a future cropped path for crisper headers.
