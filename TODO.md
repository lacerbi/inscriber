# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/docs/`.

Legend: `[ ]` todo · `[!]` blocked.

## Pending real-hardware verifications

- [ ] **Gundam coordinate frame** (DESIGN §2.2/§8.3; M1A-FINDINGS Q2): when
      `--ocr-resolution gundam` is first exercised, determine empirically
      whether grounding coords are relative to the 1024 global view or the
      tiles; extend the `grid_to_norm` golden tests with the answer.
- [ ] **BF16 DeepSeek weights** (M1A-FINDINGS "remaining confirmations"):
      confirm the recommended bf16 GGUFs produce the same `LABEL[[bbox]]`
      format and padded-square frame as the tested Q8_0.
- [ ] **Equation fidelity**: check DeepSeek-OCR's LaTeX/math output quality on
      real papers (inline `\(…\)` observed in M1a); decide whether a
      normalization pass is needed.

## Table-restructuring pass (DESIGN §9.7)

- [!] **Cropped table input** for crisper headers (Gemma downscales the whole
      page to ~896px, losing small header glyphs) — blocked: DeepSeek does not
      ground tables with boxes, so there is no clean table bbox to crop to
      (`dev/docs/table-reconstruction-findings.md` §Notes).
- [ ] **System/user message split** of the table prompt (static instructions as
      a system message → llama-server prefix-cache reuse, possible adherence
      gain). The validated prompt is a single user message — re-validate on
      real hardware before adopting.

## Planned features

- [ ] **Model auto-download helper** — optional command to fetch the recommended
      GGUFs from Hugging Face (kept out of the core pipeline; opt-in, online).
- [ ] **Alternate BibTeX sources** — Crossref / arXiv API as fallbacks to
      Semantic Scholar (unauthenticated 429s are common in practice; the
      degrade path is DESIGN §12), or fully-offline extraction from the paper's
      own reference list.

## Code debts (2026-06-10 implementation review)

- [ ] **Consolidate the duplicated VLM-pass scaffolding** between
      `_refine_tables` and `_vlm_describe` (each builds its own keys-only
      backend, `VlmCache`, and model identities); the table prompt is assembled
      twice (once for the cache key, once inside the backend call) — unify, or
      assert the two stay equal.
- [ ] **`VlmCache` value field naming**: restructured tables are stored under a
      JSON field literally named `"description"` — harmless (key payloads are
      disjoint) but misleading; rename/generalize (bump `VLM_VALUE_SCHEMA`).
- [ ] **Blob-detection edge cases** that would mis-splice: a genuinely *nested*
      `<table>` (the non-greedy match leaves an orphan tail) and a `<table>`
      inside a fenced code block. Both unobserved in DeepSeek output — handle
      if another OCR backend can emit them.
