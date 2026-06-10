# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/notes/`.

Legend: `[ ]` todo · `[!]` blocked.

## Pending real-hardware verifications

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

- [ ] **Cropped table input** for crisper VLM reading — **UNBLOCKED
      2026-06-10 and the top table-quality item.** The blocker ("DeepSeek does
      not ground tables with boxes",
      `dev/notes/2026-06-10-table-reconstruction-findings.md` §Notes) was a
      build-9028 fact: **on 9587 DeepSeek emits `table[[bbox]]` +
      `table_caption[[bbox]]` regions, at 1280 and 2048 alike**
      (`dev/notes/2026-06-10-e2e-quality-findings.md` §Render-size experiment).
      Sketch: parser keeps table regions (additive — unknown labels already
      pass through); crop the table bbox (+padding) from the page raster —
      now 2048px by default — and send the crop to the VLM instead of the
      whole ~896px-downscaled page; mirrors the figure-crop path; cache keys
      on the crop hash; bundle already carries table-page rasters. Motivation
      is quantified: 5 of 10 tables on a real paper carried structure damage,
      and the digit-fusion/segmentation errors (`159.99346.6…` splits,
      `1010` cell merges) are NOT resolution-sensitive — a crisp crop the
      VLM can actually read is the remaining lever. Validate prompt shape on
      real hardware (§9.7 pinned-prompt rule).
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

- [ ] **`<table>` inside a fenced code block** would be mis-spliced (blob
      detection doesn't see fences). Unobserved in DeepSeek output — handle if
      another OCR backend can emit fenced HTML. (The companion edge case, a
      nested `<table>`, is now guarded in `blob_is_refinable`; the VLM-pass
      scaffolding consolidation from this review also landed — one backend
      instance + prompt-assembled-once, DESIGN §9.2.)
