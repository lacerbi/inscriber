# Upstream llama.cpp watch — v1 tiling & DeepSeek-OCR 2 (researched 2026-06-10)

> Research record (GitHub/paper lookup, **not** local-hardware findings): where
> upstream is on the two things the pinned build 9028 lacks — Gundam tiling for
> DeepSeek-OCR v1, and DeepSeek-OCR 2 — and what any llama.cpp upgrade forces
> on this repo. Action items live in `TODO.md`. Re-check this page's facts
> before acting on them; upstream was moving **daily** when researched.

## 1. v1 Gundam tiling: deliberately cut from #17400; follow-up in limbo

- [PR #17400](https://github.com/ggml-org/llama.cpp/pull/17400) (the v1
  support we run, merged 2026-03-25) **descoped Gundam on purpose** — the
  maintainer (ngxson) asked to split it out; the author (sfallah — the same
  account whose HF repo hosts our BF16 GGUFs) agreed: *"I'll finish the Gundam
  implementation in a follow-up PR."* This is why `gundam-findings.md` measured
  one slice / no tiling: that is the shipped state, not an accident.
  Technical blockers discussed there: the view-separator is an embedding (not
  a vocab token), and tile batching needs a batch dim in the vision graph.
- Groundwork merged since: [#18014](https://github.com/ggml-org/llama.cpp/pull/18014)
  (llava_uhd resize generalization, 2025-12),
  [#24352](https://github.com/ggml-org/llama.cpp/pull/24352)
  (`build_vit` batching, **2026-06-09**).
- The actual Gundam PR,
  [#24300](https://github.com/ggml-org/llama.cpp/pull/24300) ("DeepSeek-OCR
  multi-tile dynamic resolution batched encoding", unified v1/v2 dynamic-res
  preprocessor), was **closed unmerged 2026-06-09**: ngxson is replacing it
  with a generic, model-agnostic batching API —
  [#24384](https://github.com/ggml-org/llama.cpp/pull/24384), a WIP draft.
  The DSOCR-specific tiling then has to be re-adapted on top of it.
- **Stall risk:** sfallah wrote *"this will most probably be the last thing
  that I will implement for DSOCR"* — if he disengages after #24384 lands,
  the v1-specific half may sit unowned.

**Consequence for the `TODO.md` "Gundam render target" decision:** consider
waiting — if upstream tiling lands, `gundam` becomes real tiling and the
"bump the render to 2048 for 421 tokens" question changes shape entirely.

## 2. DeepSeek-OCR 2: merged upstream 2026-05-29

[PR #20975](https://github.com/ggml-org/llama.cpp/pull/20975) (also sfallah)
added DeepSeek-OCR-2 — this **invalidated** DESIGN's earlier "no llama.cpp
path for v2" claim (§2.2/§22.2 updated 2026-06-10). Needs a build newer than
the pinned 9028.

**Paper claims** ([arXiv 2601.20552](https://arxiv.org/pdf/2601.20552),
DeepEncoder V2 "visual causal flow" — the encoder selects/orders visual tokens
by content rather than fixed scan order):

| metric | v1 | v2 |
| --- | --- | --- |
| OmniDocBench v1.5 overall | ~87.4% | **91.09%** (+3.73) |
| reading-order edit distance | 0.085 | **0.057** |
| repetition rate (user-log imgs) | 6.25% | **4.17%** |
| repetition rate (PDF production) | 3.69% | **2.88%** |

The repetition reduction targets exactly the loop class we hit in the wild
(`equation-fidelity-findings.md`); reading order matters directly for stitched
markdown.

**llama.cpp implementation facts (from the PR):**

- Ships **with multi-tile dynamic resolution from day one** (1024 global view
  + grid of 768-px tiles, InternVL-style) — the very thing v1 lacks.
- GGUFs: [sabafallah/DeepSeek-OCR-2-GGUF](https://huggingface.co/sabafallah/DeepSeek-OCR-2-GGUF)
  — bf16 5.88 + 0.93 GB (same VRAM class as v1; 4060-OK), q8_0 3.13 + 0.51 GB,
  Q4_K_M 1.95 GB (untested for loops — v1's Q4_K_M warning may or may not
  carry over).
- **Different backend, not a drop-in.** On `llama-server` v2 requires
  `--chat-template deepseek-ocr --no-jinja` (v1 must NOT pass a template on
  the server path), plus `--flash-attn off`, `--no-warmup`, and a different
  DRY tuning (`0.8 / base 1.75 / allowed-length 2 / penalty-last-n -1 /
  sequence-breaker none`). Maps cleanly onto a new `deepseek-ocr-2`
  `OcrBackend` (own `server_flags()` / `chat_template()`) — zero pipeline
  changes by design (DESIGN §8).
- DRY remains an *approximation* of the HF `no_repeat_ngram_size` — the loop
  guard story (cap + detection TODO) still applies.
- **Maturity caveat:** merged two weeks before this research; the PR's own
  regression gate is "intentionally loose" (hard test image, weak CER/chrF
  parity signal vs the HF reference).

**Gating unknowns before adoption (the v2 spike in `TODO.md`):**

1. Grounding format + **coordinate frame under real tiling** — the question
   that dissolved for v1 returns for real here, compounded by causal-flow
   token reordering. Full M1a calibration discipline;
   `dev/scripts/gundam_check.py` is the ready-made tool.
2. Loop behavior on the known-bad page (PriorGuide p. 5, triple-underbrace
   array).
3. Real-page format capture → new parser fixtures if the block format differs.

## 3. What ANY llama.cpp upgrade off 9028 forces

- [PR #23345](https://github.com/ggml-org/llama.cpp/pull/23345) (merged
  2026-05-20) already changed **v1's image preprocessing / resize padding** —
  outputs and grounding coords may shift on a newer build. The M1a re-capture
  discipline applies: capture, compare, re-pin fixtures (DESIGN §22.2).
- ⚠️ **Cache-correctness gap surfaced by this:** the OCR/VLM cache keys do NOT
  include any llama.cpp build identity — an upgrade that changes preprocessing
  changes model outputs *without busting the cache* (stale entries served
  silently). Tracked as a code debt in `TODO.md`; the interim rule is
  **`--refresh` after any llama.cpp upgrade**.
