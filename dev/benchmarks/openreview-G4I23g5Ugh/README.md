# OCR benchmark: PriorGuide (openreview-G4I23g5Ugh)

A real-paper gold transcription for comparing OCR/VLM approaches against the
current pipeline. **Not CI material** — producing a candidate transcription
needs real models and hardware; this directory only pins the target.

## Source document

- **OpenReview (the version the gold was verified against):**
  https://openreview.net/pdf?id=G4I23g5Ugh
- arXiv mirror: https://arxiv.org/abs/2510.13763
- "PriorGuide: Test-Time Prior Adaptation for Simulation-Based Inference",
  ICLR 2026, 38 pages. Math-heavy: dense display equations with underbraces,
  two wide multi-level-header results tables, many appendix tables with
  sparse `—` rows, algorithm listings, figure-description targets.

The PDF itself is **not committed** (gitignored here) — download it from the
OpenReview URL above when needed.

## Contents

- `openreview-G4I23g5Ugh.{main,appendix,backmatter}.md` — the **gold
  splits**: an inscriber run (DeepSeek-OCR + Gemma 4, 2026-06-10) verified
  against the source PDF in ≤10-page chunks by four review subagents plus
  manual PDF checks, with all ~45 confirmed errors fixed by hand.
  `inscriber join <this-dir>/openreview-G4I23g5Ugh` regenerates the combined
  document.
- `openreview-G4I23g5Ugh.verification-fixes.md` — every error the 2026-06-10
  pipeline made on this document (exact before→after strings, grouped by
  failure mode, with a per-category tally). This is the **baseline
  scorecard**: a candidate approach can be checked against precisely these
  sites for fixes and regressions.

## How to compare a candidate approach

1. Run it on the source PDF (e.g.
   `inscriber run "https://openreview.net/pdf?id=G4I23g5Ugh" -o /tmp/cand/`
   with the model/flags under test).
2. Diff the three splits against the gold splits.
3. Read the diff with these rules:
   - **Exact-match regions** — body text, equations, tables, references,
     headings. Differences here are real errors (in the candidate or, if it
     out-transcribes the gold, fix the gold and log it in the fixes record).
   - **Semantic-match-only regions** — the `> **Image description.**`
     blockquotes are VLM prose; a different model writes different-but-valid
     text. Judge them against the actual figure (axes, panels, trends,
     quoted numbers), not against the gold wording.
   - **Known don't-care gaps in the gold** (see "Known remaining gaps" in the
     fixes record): bold significance markers in tables are not reproduced;
     appendix-B multi-line derivations have collapsed per-line equation
     labels; Table A8's colspan header is flattened. A candidate that gets
     these *right* beats the gold — update the gold accordingly.
4. Score against the baseline tally at the bottom of the fixes record (table
   structure and plausible math-symbol substitutions were the dominant error
   classes; numeric table values were almost entirely correct).
