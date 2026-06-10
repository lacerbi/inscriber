# Verification record: openreview-G4I23g5Ugh (PriorGuide)

Reference list of every transcription issue found and fixed during the
`/inscribe` verification pass, kept as ground truth for comparing other OCR
approaches on the same document.

- **Source:** https://openreview.net/pdf?id=G4I23g5Ugh — "PriorGuide:
  Test-Time Prior Adaptation for Simulation-Based Inference" (ICLR 2026), 38 pages.
- **Pipeline:** inscriber (DeepSeek-OCR + Gemma 4 table/figure passes), full run.
- **Verification:** 2026-06-10. Four Claude subagents (Sonnet) over ≤10-page
  chunks comparing the Markdown splits against the PDF, every accepted fix
  re-checked against the PDF page by the orchestrator. Fixes were applied to
  the split files and `inscriber join` regenerated the full document.
- Line references below are approximate (pre-fix positions); the
  before/after strings are exact.

## 1. Tables

### Table 1 + Table 2 (main, §4.2/§4.3) — restructuring fell back to raw HTML

The VLM table pass failed on both main results tables; the output kept
DeepSeek-OCR's degenerate `<table>` blobs. All 216 numeric cells in the blobs
were verified correct against the PDF — the damage was structural:

| issue | before (in blob) | after |
| --- | --- | --- |
| All 3 column-group headers garbled, both tables | `q&amp;out(θ)` ×3 | `q_mild(θ)`, `q_strong(θ)`, `q_mixture(θ)` |
| "Simformer" row label missing in every task group (6× in Table 1, 2× in Table 2 it was present) | `<td rowspan="3">Two Moons0.39(0.19)...` | task and method as separate cells |
| Method name misread, Table 1 only (6×) | `ACCE` | `ACE` |
| Table 1 metric header row truncated | `...RMSEC2ST` (final `MMTV` missing) | 9 metric columns |
| Table 2 metric header row had a phantom 4th group | `RMSEMMDxRMSEMMDxRMSEMMDxRMSEMMDx` | 3 groups |

Fix applied: both blobs rebuilt as Markdown pipe tables (values transcribed
from the verified blob). **Not reproduced:** the PDF's bold
"significantly best" markers (the blob never had them; captions still
mention bolding).

### Table A3 (appendix D.3.1) — phantom column from a split metric name

- Header `| RMSE | C2 | ST | MMTV | ... |` → `C2ST` is one metric; the split
  created a 9-column header over 8-column data.
- 7 of 10 data rows carried an extra trailing `—` to fill the phantom column.
- Header `k` → `$\hat{k}$`.

### Table A4 (appendix D.3.1) — row drift in sparse rows

Both `Diffusion + SIR` rows had their values shifted one column left (the `—`
under "Acc Rate (%)" was dropped, so ESS/k̂ values landed under Acc/ESS):

- `| 0.22 ± 0.03 | 0.95 ± 0.04 | 0.63 ± 0.32 | 0.01 ± 0.01 | 1.28 ± 0.98 | — |`
  → `| ... | — | 0.01 ± 0.01 | 1.28 ± 0.98 |`
- `| 0.25 ± 0.03 | 0.95 ± 0.04 | 0.79 ± 0.31 | 0.00 ± 0.00 | 2.61 ± 1.78 | — |`
  → `| ... | — | 0.00 ± 0.00 | 2.61 ± 1.78 |`

### Table A5 (appendix D.3.1) — dropped cells + row drift

Both `Diffusion + SIR` rows lost their two `~ 1` entries (C2ST, MMTV) and
shifted: `| 0.15 ± 0.02 | — | — | 0.00 ± 0.00 | 10.85 ± 1.32 | |`
→ `| 0.15 ± 0.02 | ~ 1 | ~ 1 | — | 0.00 ± 0.00 | 10.85 ± 1.32 |`
(same for the (M) row with 10.61 ± 1.12). RS rows' empty last cell → `—`.

### Table A6 (appendix D.4) — single-cell value slip

- Row `2.85`: RMSE `1.25 ± 0.20` → `1.25 ± 0.19` (classic ±-digit slip).
- Header `$\|\mu_q - \mu_p\|$` → `$\|\mu^q - \mu^p\|$` (PDF uses superscripts).

### Table A7 (appendix D.5) — digit misread, both groups

- Gaussian Linear 10D, middle row: `10` GMM components → `20`.
- BCI, middle row: `10` → `20` (same misread; PDF shows 2 / 20 / 200 in both).

### Table A8 (appendix D.8)

Values all correct (`159.9 | 9346.6 | 8300.4 | 1037.4 | 8.8` etc.). The PDF's
two-row colspan header was flattened into two pipe rows — left as-is (pipe
tables cannot express colspan).

## 2. Equations

### Eqs. (6)–(9) (main, §3.1) — the paper's central derivation

- The 4-line array carried a single label `(8)`; PDF numbers each line
  (6)–(9). Text references "Eq. (7)" / "Eq. (9)" pointed at nothing.
- Eq. (9)'s two underbraces were mangled into a spurious fraction:
  `\underbrace{\mathbb{E}_{p(\theta_0|\theta_t,x)}\left[\frac{r(\theta_0)}{\mathrm{reversekernel}}\right]}_{\mathrm{reversekernel}}\mathrm{prior~ratio}`
  → restored: expectation subscript underbraced "reverse kernel", `r(θ₀)`
  underbraced "prior ratio", no fraction.

### Algorithms 1–2 (appendix A.2)

- Alg 1 line 3: `\mathrm{FirTMM}` → `\mathrm{FitGMM}` (subroutine name).
- Alg 1 line 12: `\eta \frac{\sigma(t)\sigma(t)}{2}` →
  `\eta \frac{\dot{\sigma}(t)\sigma(t)}{2}` (missing time-derivative dot).
- Alg 1 line 17: `\sqrt{2\dot{\sigma}(t)\sigma(t)}\Delta t\epsilon` →
  `\sqrt{2\dot{\sigma}(t)\sigma(t)\Delta t}\epsilon` (Δt belongs inside the root).
- Alg 2 line 7: `(\pmb{\theta}_{t_N}, \mathbf{x}_{t_N}^{*})` →
  `(\pmb{\theta}_{t_N}^{*}, \mathbf{x}_{t_N}^{*})` (missing star).
- Alg 2 lines 14, 19 and Eq. (A1): `\hat{\sigma}` → `\dot{\sigma}`
  (hat-for-dot misread, 5 occurrences) + Δt into the root in line 19.

### Appendix B

- B.1: the (A2)/(A3) array carried only the `(A3)` label → `(A2)` restored on
  the first row.
- Eq. (A47): `\log \mathrm{it}p_{\mathrm{same}}` →
  `\mathrm{logit}\, p_{\mathrm{same}}` ("logit" split across tokens).

## 3. Subscripts and short identifiers in text

- C.3.1 (4 spots): `\sigma_i^{\mathrm{mid}}`, `q_{\mathrm{mid}}`,
  `\pmb{\Sigma}^{\mathrm{mid}}` (incl. diag entries) → `mild`
  ("mild" truncated to "mid" — changes the prior-family name).
- C.3.1, Eq. (A52): `q_{\mathrm{mixure}}` → `q_{\mathrm{mixture}}`.
- D.4: `(\mu^d)` → `(\mu^q)`; `(\mu^r)` → `(\mu^p)`; `(d^r)` → `(d')`
  3× (d-prime misread as a superscript-r — flips the metric's identity).
- D.4: `\mathcal{N}(\mathbf{0},0.1\cdot \mathbf{1}_{10})` → `\mathbf{I}_{10}`.
- D.3.1: "where `k > 0.7`" → "`\hat{k} > 0.7`".
- §2.3 (main): "preforming" → "performing".
- E (software): `sbib` → `sbi`; `PyVBMCC` → `PyVBMC` (package names; the
  neighboring `sbibm` was correct and kept).

## 4. Figure descriptions (VLM, not OCR)

- Figure 3 description (main, §4.4): hallucinated expansion
  "MMTV (Mean Time to Maximum Variance)" → "MMTV (Mean Marginal Total
  Variation)". All other checked descriptions (Figs. 1–3, A1–A5) matched the
  actual figures.

## 5. Structure / boundary damage

- Section heading text duplicated and fused into the first body line, 3×:
  `C.4.2 SIGNIFICANCE TESTINGTo assess...`,
  `D ADDITIONAL EXPERIMENTAL RESULTSWe present here...`,
  `D.1 ILLUSTRATION ... ON TWO MOONSThis section provides...` → prefix removed
  (the proper `##` headings were already present above).

## 6. References (backmatter)

- Gelman et al.: "volume 3nd edition" → "3rd edition".
- Pedersen: "room electromagnetic" → "room electromagnetics".
- Thornton et al.: "Louis Bethune" → "Louis Béthune".
- Vaswani et al.: "Lukasz Kaiser" → "Łukasz Kaiser".

## Rejected verifier suggestions

- One verifier claimed Table A6's header `d'` should be `d^r`, citing the body
  text — backwards. The PDF (pp. 34–35) uses `d'` everywhere; the body text
  was the misread side. (The verifier had drifted outside its assigned page
  range; the pages-31–38 verifier and a direct PDF check settled it.)
- "PriorGuide provides provides meaningful approximations" (D.4) — the
  duplication exists in the published PDF itself; kept for fidelity.

## Known remaining gaps in the gold

- Bold significance markers in Tables 1, 2, A2 are not reproduced (the OCR
  blobs carry no emphasis, and bolding could not be read reliably off the
  page scans during verification). **Recovering bold is wanted** — marginally
  important, tracked in TODO.md; a transcription that restores it correctly
  should be folded into the gold.
- Appendix B multi-line derivations have collapsed equation labels (e.g. one
  block tagged `(A9)` spans (A6)–(A10); `(A24)` spans (A22)–(A25)), so text
  references like Eq. (A25)/(A29)/(A35) point into unlabeled blocks. Content
  is intact; only intermediate numbers are missing.
- Table A8's colspan header is flattened.

## Failure-mode tallies (for comparing OCR approaches)

| category | count (fix sites) | notes |
| --- | --- | --- |
| Table structure (headers/labels/drift) | 5 tables + 2 blob fallbacks | highest-impact class |
| Table numeric values | 3 cells (1.25±0.19, 20, 20) | out of ~400 checked — value accuracy was high |
| Equation structure/labels | 4 (eq 6–9, A2/A3, underbraces, logit) | plus systemic label collapse in App. B |
| Math symbol misreads | 12 (σ̇/σ̂, stars, d′, μ^q/μ^p, mid/mild, I₁₀, k̂) | always-plausible substitutions |
| Word-level typos | 4 (preforming, mixure, sbib, PyVBMCC) | |
| Figure-description errors (VLM) | 1 (MMTV expansion) | hallucinated acronym expansion |
| Heading/body fusion | 3 | |
| Reference garbling | 4 (2 diacritics, 1 ordinal, 1 plural) | |
