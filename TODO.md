# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/notes/`.

Legend: `[ ]` todo · `[!]` blocked.

## Pending real-hardware verifications

- [x] ~~**Verify llama.cpp build 9587 before trusting real runs**~~ — done
      2026-06-10 (`dev/notes/2026-06-10-build-9587-verification.md`). **FAILED as a
      drop-in: the grounding coordinate frame changed from padded-square to
      per-axis** (calibration box `[242, 243, 753, 653]` ≈ the per-axis
      prediction; Δ≈37 off padded-square) → `grid_to_norm` silently shifts
      every figure crop. Passed everything else: format parses, still no
      tiling (283/431-token single-slice saturation), thinking kwarg toggles,
      m1b + the new build-identity cache keys work live — and the PriorGuide
      p. 5 loop is GONE (37 s, `finish_reason: stop`, lost content
      recovered). v1 stays pinned on 9028 (`llms/` root); adoption is the
      item below.
- [x] ~~**Adopt llama.cpp ≥9587: per-axis grounding frame decision**~~ — done
      2026-06-10: **re-pinned on ≥ 9587, single frame** (no dual-frame
      maintenance). `grid_to_norm` is per-axis only;
      `DeepSeekOcrBackend.min_server_build = 9587` and the pipeline's
      `_check_server_build` refuse older spawned servers (an endpoint without
      `/props` `build_info` warns instead). Golden fixtures re-captured on
      9587; verified live end-to-end (calibration crop within 1% of ground
      truth). Evidence: `dev/notes/2026-06-10-build-9587-verification.md`.
- [ ] **Gundam render target** (`dev/notes/2026-06-10-gundam-findings.md`,
      `dev/notes/2026-06-10-build-9587-verification.md`): neither build 9028 nor 9587
      tiles, so gundam is currently a strict alias of `large` (both render
      1280). Rendering ≥1664 buys the saturated visual encoding (431 vs 283
      prompt tokens on 9587; ~3× encode time). Decide whether
      `ResolutionMode.GUNDAM.long_edge_px` becomes 2048 — validate OCR quality
      on dense pages first; concrete probes now exist in
      `dev/notes/2026-06-10-e2e-quality-findings.md` (equation-tag collapse, table digit
      errors like `9346.6→346.6` / `Fail→Full` — re-run those pages at ≥1664
      and diff). (The coordinate frame itself is resolved: per-axis on ≥9587
      at every input size, golden-tested.) Consider simply waiting for
      upstream v1 tiling instead (`dev/notes/2026-06-10-upstream-watch.md` §1) — real
      tiling would change the question's shape entirely.
- [ ] **Equation-number tag collapse in multi-row arrays** — DeepSeek keeps
      only one `(N)` tag per `\begin{array}` block (~8 arrays affected in one
      real paper; content otherwise faithful —
      `dev/notes/2026-06-10-e2e-quality-findings.md` §Equations). Vision-level: the tags
      are absent from the raw output, so no text post-processing can recover
      them. Decide: accept as a documented limitation (the transcription
      notice already warns about equations), or test whether a ≥1664 render
      preserves the tags (fold into the gundam render-target probe above).
- [ ] **OCR loop/truncation detection** (`dev/notes/2026-06-10-equation-fidelity-findings.md`):
      a real page looped at BF16 + grounded prompt + DRY and was **silently
      cached** with half its text missing — `DeepSeekOcrBackend.ocr_page` never
      checks `finish_reason`. Detect `finish_reason != "stop"` → warn loudly +
      don't cache the page (DESIGN §16 already promises the logging; mirror the
      table pass's truncation handling). Note `--refresh` can't fix such a page
      (deterministic); suggest a different `--ocr-resolution` in the warning.
      (That specific page no longer loops on build 9587 —
      `dev/notes/2026-06-10-build-9587-verification.md` §4 — but the detection gap remains
      for whatever page loops next.)

## Table-restructuring pass (DESIGN §9.7)

- [!] **Cropped table input** for crisper headers (Gemma downscales the whole
      page to ~896px, losing small header glyphs) — blocked: DeepSeek does not
      ground tables with boxes, so there is no clean table bbox to crop to
      (`dev/notes/2026-06-10-table-reconstruction-findings.md` §Notes). The cost is now
      quantified: 5 of 10 tables on a real paper carry structure damage,
      concentrated in dense multi-header layouts
      (`dev/notes/2026-06-10-e2e-quality-findings.md` §Tables).
- [ ] **Guard against silent structure damage** in restructured tables —
      the worst observed failure is a syntactically clean table that silently
      dropped an entire column group (Table 1 in
      `dev/notes/2026-06-10-e2e-quality-findings.md`; also per-row cell drift, row-label
      misalignment). The old value-count check stays rejected (DeepSeek merges
      cells, so the blob is no baseline), but options remain: prompt-level
      column-count echo (pinned prompt — re-validate on real hardware before
      touching, §9.7), a header-width vs page-image consistency probe, or at
      minimum flagging wide/multi-header tables as low-confidence in the
      transcription notice. Investigation item — no chosen design yet.
- [ ] **System/user message split** of the table prompt (static instructions as
      a system message → llama-server prefix-cache reuse, possible adherence
      gain). The validated prompt is a single user message — re-validate on
      real hardware before adopting.

## Upstream llama.cpp watch (researched 2026-06-10 — `dev/notes/2026-06-10-upstream-watch.md`)

- [ ] **DeepSeek-OCR 2 spike** — upstream support landed (llama.cpp PR #20975,
      2026-05-29; GGUFs at `sabafallah/DeepSeek-OCR-2-GGUF`, bf16 5.9+0.9 GB).
      Paper: +3.73% OmniDocBench, reading order 0.085→0.057, repetition rate
      ~⅓ lower (our loop class), and the llama.cpp impl ships WITH multi-tile
      dynamic resolution. Gated on: (i) grounding format + coordinate frame
      under tiling — full M1a calibration discipline, `gundam_check.py`
      reusable; (ii) loop check on the known-bad page (PriorGuide p. 5);
      (iii) real-page format capture → fixtures. The pinned build 9587
      already includes v2 support (no build upgrade needed); requires a new
      `deepseek-ocr-2` backend (different server template/flags —
      `--chat-template deepseek-ocr --no-jinja`, `--flash-attn off`, its own
      DRY tuning); zero pipeline changes (§8).
- [ ] **v1 Gundam tiling** — descoped from #17400; the follow-up PR #24300 was
      closed 2026-06-09 in favor of a generic batching API (PR #24384, WIP
      draft). Watch #24384 + the DSOCR re-adaptation on top of it; stall risk
      noted in the watch doc. When it lands, revisit the gundam render-target
      item above.

## Planned features

- [x] ~~**BibTeX `auto` mode**~~ — done 2026-06-10
      (`dev/plans/PLAN-bibtex-auto.md` B0–B4): `bibtex.mode` tri-state
      (legacy `enabled` alias), cached
      text-only citability/metadata probe (pinned prompt validated on real
      hardware — `dev/notes/2026-06-10-bibtex-probe-findings.md`, 4/4 PASS), provenance-
      first chain (S2-by-arXiv-ID preferring the published version → arXiv
      export API → S2 title search → local best-effort `@misc`), default
      flipped to `auto`. Spec: DESIGN §12. Deferred refinements
      (`--bibtex-source`, Crossref, by-DOI, type inference, …): DESIGN §22.2.
- [ ] **Publish to PyPI** — the name `inscriber` was verified available
      (DESIGN §18) but nothing is published yet; README documents source
      install until then.
- [ ] **Model auto-download helper** — optional command to fetch the recommended
      GGUFs from Hugging Face (kept out of the core pipeline; opt-in, online —
      README.md's model table has the direct links to start from).
- [ ] **Alternate BibTeX sources** — Crossref / arXiv API as fallbacks to
      Semantic Scholar (unauthenticated 429s are common in practice; the
      degrade path is DESIGN §12), or fully-offline extraction from the paper's
      own reference list. The arXiv half shipped 2026-06-10 with BibTeX `auto`
      (S2-by-arXiv-ID + the arXiv export API fallback, DESIGN §12.1); Crossref,
      S2-by-DOI for bioRxiv/medRxiv, and reference-list extraction remain
      (DESIGN §22.2).

## Code debts (2026-06-10 implementation review)

- [x] ~~**llama.cpp build identity is not cache-key material**~~ — done
      2026-06-10: `llama_build_identity` (`llama-server --version`, or the
      endpoint's `/props` `build_info`) is now in the OCR + VLM keys
      (DESIGN §8.6/§9.6). The `VlmCache` value-field rename
      (`"description"` → `"text"`, `VLM_VALUE_SCHEMA` 2) rode the same
      all-keys-bust, as planned.
- [ ] **`<table>` inside a fenced code block** would be mis-spliced (blob
      detection doesn't see fences). Unobserved in DeepSeek output — handle if
      another OCR backend can emit fenced HTML. (The companion edge case, a
      nested `<table>`, is now guarded in `blob_is_refinable`; the VLM-pass
      scaffolding consolidation from this review also landed — one backend
      instance + prompt-assembled-once, DESIGN §9.2.)
