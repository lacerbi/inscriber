# TODO

Concrete, actionable pending items — collected here so they don't get lost
inside spec prose. Longer-horizon feature work stays in `DESIGN.md` §22; the
empirical evidence records cited below live in `dev/docs/`.

Legend: `[ ]` todo · `[!]` blocked.

## Pending real-hardware verifications

- [x] ~~**Verify llama.cpp build 9587 before trusting real runs**~~ — done
      2026-06-10 (`dev/docs/build-9587-verification.md`). **FAILED as a
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
      truth). Evidence: `dev/docs/build-9587-verification.md`.
- [ ] **Gundam render target** (`dev/docs/gundam-findings.md`,
      `dev/docs/build-9587-verification.md`): neither build 9028 nor 9587
      tiles, so gundam is currently a strict alias of `large` (both render
      1280). Rendering ≥1664 buys the saturated visual encoding (431 vs 283
      prompt tokens on 9587; ~3× encode time). Decide whether
      `ResolutionMode.GUNDAM.long_edge_px` becomes 2048 — validate OCR quality
      on dense pages first. (The coordinate frame itself is resolved:
      per-axis on ≥9587 at every input size, golden-tested.) Consider simply
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
      (That specific page no longer loops on build 9587 —
      `dev/docs/build-9587-verification.md` §4 — but the detection gap remains
      for whatever page loops next.)

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

- [ ] **BibTeX `auto` mode** — classify citability (provenance heuristics +
      a cached LLM probe) → source chain (Semantic Scholar by arXiv ID,
      preferring the published version when one exists → arXiv export API
      fallback → S2 title search → local best-effort entry). Network intent
      comes from the existing `net.offline` knob (`--offline` ⇒ local
      best-effort only). Full design + phased roadmap (B0–B4):
      `PLAN-bibtex-auto.md`. Subsumes the arXiv half of the
      alternate-sources item below.
- [ ] **Publish to PyPI** — the name `inscriber` was verified available
      (DESIGN §18) but nothing is published yet; README documents source
      install until then.
- [ ] **Model auto-download helper** — optional command to fetch the recommended
      GGUFs from Hugging Face (kept out of the core pipeline; opt-in, online —
      README.md's model table has the direct links to start from).
- [ ] **Alternate BibTeX sources** — Crossref / arXiv API as fallbacks to
      Semantic Scholar (unauthenticated 429s are common in practice; the
      degrade path is DESIGN §12), or fully-offline extraction from the paper's
      own reference list. The arXiv-by-ID source is planned as phase B3 of
      `PLAN-bibtex-auto.md`; Crossref and reference-list extraction remain.

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
