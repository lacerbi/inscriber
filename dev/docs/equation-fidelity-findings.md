# Equation fidelity — real-hardware findings (2026-06-10)

> Closes the TODO item *"Equation fidelity: check DeepSeek-OCR's LaTeX/math
> output quality on real papers; decide whether a normalization pass is
> needed."* **Verdict: no normalization pass** (§Conclusions). A new, more
> important finding fell out of the same session: a **silent repetition loop on
> a real page at BF16** (§The page-5 loop).

## Setup

- llama.cpp build 9028, `llama-server`, RTX 4060 8 GB (maintainer machine).
- DeepSeek-OCR **BF16** + BF16 mmproj, resolution `large` (1280 px long edge),
  grounded prompt, `temperature 0`, `seed 0`, `max_tokens 8192`, DRY flags
  (the §2.2 partial mitigation) — i.e. the recommended production configuration.
- Paper: arXiv 2510.13763 (*PriorGuide*, ICLR 2026) — math-heavy SBI/diffusion
  paper. ~65 s/page.
- Pages compared symbol-by-symbol against the PDF: **3–5** (main text,
  Eqs. 1–9) and **20–22** (appendix derivations A2–A25) — ~29 display equations
  plus dozens of inline spans:

  ```
  inscriber run <pdf> -o out --pages 20-22 --figure-mode placeholder --no-table-refine
  ```

## What is reliably correct

The large majority of math survives verbatim. Confirmed-correct constructs,
each across many instances:

- Delimiters: inline `\(...\)`, display `\[...\]` with `\begin{array}` for
  multi-row derivations — **fully consistent**, no mixed `$`/`$$` forms.
- Fractions (incl. stacked, e.g. `\frac{1}{p_0(\theta_t)+\sigma^2 C_1(\theta_t)}`),
  integrals, sums with limits (`\sum_{j=1}^{M_q}`), products of Gaussians.
- Nested sub/superscripts: `\mu_{0|t}(\cdot)`, `\{\theta^{(i)}\}_{i=1}^{M_p}`,
  `\ell_j^\prime`, `\nabla^2_{\mathbf{b}(\mathbf{s})}`.
- Accents and decorations: `\dot{\sigma}(t)`, `\widetilde{\Sigma}_i`,
  `\tilde{w}_i`, `\widehat{r}`, `\bar·`, transposes (`^\top`), norms
  (`\|\cdot\|_2^2`, `||\mathbf{s}||^2`), Landau `O(\sigma^4)`.
- Operators: `\propto`, `\approx`, `\sim`, `\sqrt{\cdot}`, `\exp`, `\mathbb{E}`
  with full subscripts, `\mathcal{N}`, `\mathrm{d}\theta` differentials.
- Parenthetical annotations inside arrays survive: `(chain rule)`,
  `(since \nabla f = f\nabla\log f)`, `(posterior)`/`(posterior predictive)`.
- Even a paper typo ("preforming") was copied faithfully — the model
  transcribes rather than "corrects".

## Error classes observed

All errors below are **vision-level misreads** — no text-level post-processing
could detect or fix them.

| class | examples (OCR vs. truth) | frequency |
| --- | --- | --- |
| Small-subscript glyph swaps `i↔t` | `\nabla_{\theta_i}` for `∇_{θ_t}` throughout A2–A13; conversely `\{(\theta_t,x_t)\}_{t=1}^N` for `{(θ_i,x_i)}_{i=1}^N` on p. 3 | systematic per region (~10 eqs affected) |
| Tiny-subscript word misreads | `p_{\mathrm{min}}` and `p_{\mathrm{sim}}` for `p_train` (B.2 region ×4, p. 3 ×1) | localized clusters |
| Single-glyph confusions | `\pmb{\alpha}` for bold `a` (symmetry property); `\sigma(t)\mathbf{z}\mathbf{I}` for `σ(t)²I` (Eq. 4); `p(\cdot)` for `p(t)` | rare (3 across 6 pages) |
| Lost sub-expressions | Eq. A22 lost its entire denominator `/(p_0(θ_t)+σ²C_1(θ_t))`; A24's σ-term lost `/p_0(θ_t)` | 2 of ~29 display eqs |
| Equation-label collapse | multi-row `array` blocks keep only **one** `(A10)`-style tag; standalone equations keep theirs | all 5 multi-row arrays |
| Underbraces | partially survive: A19's `\underbrace{...}_{=C_1(\theta_t)}` kept, its `=0` sibling dropped; Eq. 9's three close-set braces fused wrongly (annotation became a denominator) and **triggered the loop below** | the hardest construct observed |
| Word garbling | "multinoufl" for "multi-round" | 1 |
| Cosmetics | stray space before punctuation after inline math (`\(t\) ,`), spurious all-`\qquad` filler rows in one big array, dropped ¨ ("Hyvarinen" — though "Müller" kept it), `□` tombstone kept on p. 4, dropped on p. 21 | scattered |

## The page-5 repetition loop (new finding — acts on DESIGN §2.2/§16)

Page 5's Eq. 6–9 array (four rows, the last carrying **three adjacent
underbrace annotations**) sent the model into a runaway loop: the final row
repeated verbatim ~80× until the `max_tokens 8192` cap cut generation.
**First in-the-wild loop on the recommended configuration** — BF16 weights +
grounded prompt + DRY flags + temperature 0 (M1a's loops were Q4_K_M or the
`OCR` prompts).

Consequences observed:

1. The cap **worked as the bounding guard** (run continued; ~6 min page).
2. The loop was **completely silent**: `DeepSeekOcrBackend.ocr_page` never
   checks `finish_reason`, so no warning was logged — despite DESIGN §16
   promising "best-effort parse, **log**, move on" for looping pages.
3. The degenerate page was **cached as a success** — everything after Eq. 9 on
   the page (two paragraphs + a footnote) is silently lost, and every future
   cached run reproduces the damage. "Never cache a failed result" holds only
   because the failure was never detected.
4. Determinism means `--refresh` will NOT fix such a page (same input → same
   loop). A different render size is NOT a reliable remedy either: the same
   page re-rendered at 2048 px long edge also degenerated, just in a different
   shape (`gundam-findings.md`). The realistic fix is hand-editing the bundle
   markdown (the two-step workflow).

Follow-up tracked in `TODO.md`: detect `finish_reason != "stop"` on OCR pages →
warn loudly + do not cache (mirror the table pass, which already treats
truncation as failure).

## Conclusions

1. **No LaTeX normalization pass.** Delimiters are already uniform; symbol
   fidelity is high; every substantive error class is a vision-level misread
   that text post-processing cannot detect, let alone fix. The only candidates
   a pass could address are cosmetic (spacing before punctuation, `\qquad`
   filler) — not worth the risk of touching correct math. Decision: **rejected**;
   revisit only if a downstream consumer chokes on `\(...\)` delimiters
   (a trivial, separate transform).
2. **Users of math-heavy output should be told the failure modes**: subscript
   swaps and lost denominators look plausible and are invisible without the
   source. The existing transcription-notice footer covers this in spirit.
3. **The actionable engineering item is loop/truncation detection**, not
   equation cleanup (see TODO).
