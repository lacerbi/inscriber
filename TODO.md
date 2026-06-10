# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/notes/`.

Legend: `[ ]` todo · `[!]` blocked.

## Table-restructuring pass (DESIGN §9.7)

- [ ] **Guard against silent SHAPE damage** in restructured tables — the
      data-loss half of this item shipped 2026-06-10: the **digit-coverage
      guard** (`MIN_DIGIT_COVERAGE = 0.8`, DESIGN §9.7) rejects an output that
      lost a chunk of the blob's digit stream (calibrated: healthy ≥ 0.976,
      the one silent 6-row drop 0.664 —
      `dev/notes/2026-06-10-cropped-table-validation.md`). What the guard
      CANNOT see, per the validation run's residual list: blob header
      misreads faithfully propagated (`q_mild/q_strong/q_mixture` → `q&out`
      ×3 — an OCR subscript limit), sparse-row cell drift (the mostly-`—`
      SIR/RS rows misplace 1–2 cells), single-cell value slips (`±0.24` for
      `±0.19`), and digit-neutral shape damage (transposes, phantom empty
      columns). Options remain: prompt-level column-count echo (pinned
      prompt — re-validate on real hardware before touching, §9.7), a
      header-width vs image consistency probe, or flagging wide/multi-header
      tables as low-confidence in the transcription notice. Investigation
      item — no chosen design yet.
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
