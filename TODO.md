# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/notes/`.

Legend: `[ ]` todo · `[!]` blocked.

## Table-restructuring pass (DESIGN §9.7)

- [ ] **Cropped table input — validate on real hardware** (the prompt-pin
      gate). The mechanics shipped 2026-06-10 (DESIGN §9.7): blobs are
      content-matched to grounded `table[[bbox]]` regions, the crop (+0.02
      padding) is cut from the verbatim page raster and sent to the VLM with a
      new **cropped prompt variant** (locator → crop preamble; shared tail
      pinned by test); unmatched blobs fall back to the validated whole-page
      path (INFO line); cache key = raster hash + bbox + padding (conditional —
      whole-page keys preserved). Per the §9.7 pinned-prompt rule the cropped
      variant is **not validated yet** — run `dev/scripts/table_crop_check.py`
      against the PriorGuide table pages and:
      (i) **inspect every crop for completeness** — a clipped crop contradicts
      "never drop values" and is the dangerous case;
      (ii) diff page-input vs crop-input outputs cell-by-cell against the PDF
      (baseline: 2 clean / 3 value-perfect-wrong-shape / 5 damaged,
      `dev/notes/2026-06-10-e2e-quality-findings.md` §Tables — the fusion
      splits `159.99346.6…` and `1010` merges are the target);
      (iii) capture a real table-page raw output as a committed parser fixture
      (no current fixture contains `table[[bbox]]`);
      (iv) record a dated note and mark the cropped prompt validated in DESIGN
      §9.7 — or revert to whole-page input if it regresses.
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

- [ ] **Loop-breaking retry for truncated OCR pages** — detection shipped
      2026-06-10 (`finish_reason != "stop"` → page flagged `truncated`, kept
      best-effort, cached WITH the flag, re-warned on every hit; DESIGN §8.6 —
      the flag gives a repair pass its target list). Two rungs, both
      model-facing (real-hardware spike + dated note before adoption):
      (a) **per-request stronger-DRY / seed-jitter retry** — DRY params are
      per-request sampler params, so on detection re-run the page once with
      stronger DRY; risks corrupting legitimately repetitive content (table
      rows), and the same attractor may recur;
      (b) **prefix-prefill + deflection** — find the repeated suffix text-side
      (deterministic Python), prefill the assistant message up to just before
      the first repetition, and deflect the first continuation token
      (`logit_bias` ban / brief temperature burst / `n_probs`-driven
      client-side decoding at the loop point). Gated on verifying that
      llama-server supports assistant-message continuation with multimodal
      input + DeepSeek-OCR's built-in template on build 9587.
      Open: a repaired page came from different effective sampling — decide
      its cache-key material. Reminder: a loop that self-terminates under the
      cap emits `finish_reason: "stop"` and is invisible to the detector
      (DESIGN §2.2 known limitation).
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
