# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/docs/`.

Legend: `[ ]` todo · `[!]` blocked.

## Pending real-hardware verifications

- [ ] **Gundam render target** (`dev/docs/gundam-findings.md`): build 9028
      does not tile, so gundam is currently a strict alias of `large` (both
      render 1280). Rendering ≥1664 buys the saturated 421-token visual
      encoding (vs 273 at 1280; ~3× encode time). Decide whether
      `ResolutionMode.GUNDAM.long_edge_px` becomes 2048 — validate OCR quality
      on dense pages first. (The coordinate frame itself is resolved:
      padded-square at every input size, golden-tested.) Consider simply
      waiting for upstream v1 tiling instead
      (`dev/docs/upstream-watch.md` §1) — real tiling would change the
      question's shape entirely.
- [ ] **OCR loop/truncation detection** (`dev/docs/equation-fidelity-findings.md`):
      a real page looped at BF16 + grounded prompt + DRY and was **silently
      cached** with half its text missing — `DeepSeekOcrBackend.ocr_page` never
      checks `finish_reason`. Detect `finish_reason != "stop"` → warn loudly +
      don't cache the page (DESIGN §16 already promises the logging; mirror the
      table pass's truncation handling). Note `--refresh` can't fix such a page
      (deterministic); suggest a different `--ocr-resolution` in the warning.

## Table-restructuring pass (DESIGN §9.7)

- [!] **Cropped table input** for crisper headers (Gemma downscales the whole
      page to ~896px, losing small header glyphs) — blocked: DeepSeek does not
      ground tables with boxes, so there is no clean table bbox to crop to
      (`dev/docs/table-reconstruction-findings.md` §Notes).
- [ ] **System/user message split** of the table prompt (static instructions as
      a system message → llama-server prefix-cache reuse, possible adherence
      gain). The validated prompt is a single user message — re-validate on
      real hardware before adopting.

## Upstream llama.cpp watch (researched 2026-06-10 — `dev/docs/upstream-watch.md`)

- [ ] **DeepSeek-OCR 2 spike** — upstream support landed (llama.cpp PR #20975,
      2026-05-29; GGUFs at `sabafallah/DeepSeek-OCR-2-GGUF`, bf16 5.9+0.9 GB).
      Paper: +3.73% OmniDocBench, reading order 0.085→0.057, repetition rate
      ~⅓ lower (our loop class), and the llama.cpp impl ships WITH multi-tile
      dynamic resolution. Gated on: (i) grounding format + coordinate frame
      under tiling — full M1a calibration discipline, `gundam_check.py`
      reusable; (ii) loop check on the known-bad page (PriorGuide p. 5);
      (iii) real-page format capture → fixtures. Requires a build newer than
      pinned 9028 and a new `deepseek-ocr-2` backend (different server
      template/flags — `--chat-template deepseek-ocr --no-jinja`,
      `--flash-attn off`, its own DRY tuning); zero pipeline changes (§8).
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
      own reference list.

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
