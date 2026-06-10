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

- [ ] **Publish to PyPI** — the name `inscriber` was verified available
      (DESIGN §18) but nothing is published yet; README documents source
      install until then.
- [ ] **Model auto-download helper** — optional command to fetch the recommended
      GGUFs from Hugging Face (kept out of the core pipeline; opt-in, online —
      README.md's model table has the direct links to start from).
- [ ] **Alternate BibTeX sources** — Crossref / arXiv API as fallbacks to
      Semantic Scholar (unauthenticated 429s are common in practice; the
      degrade path is DESIGN §12), or fully-offline extraction from the paper's
      own reference list.

## Code debts (2026-06-10 implementation review)

- [ ] **`VlmCache` value field naming**: restructured tables are stored under a
      JSON field literally named `"description"` — harmless (key payloads are
      disjoint) but misleading. Fold the rename into the next
      `VLM_VALUE_SCHEMA` bump made for a real reason; a standalone bump would
      needlessly invalidate every user's cached descriptions/tables.
- [ ] **`<table>` inside a fenced code block** would be mis-spliced (blob
      detection doesn't see fences). Unobserved in DeepSeek output — handle if
      another OCR backend can emit fenced HTML. (The companion edge case, a
      nested `<table>`, is now guarded in `blob_is_refinable`; the VLM-pass
      scaffolding consolidation from this review also landed — one backend
      instance + prompt-assembled-once, DESIGN §9.2.)
