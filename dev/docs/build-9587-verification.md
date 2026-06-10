# llama.cpp build 9587 verification — DeepSeek-OCR frame change (2026-06-10)

> Executes the TODO item *"Verify llama.cpp build 9587 before trusting real
> runs"*. **Verdict: NOT a drop-in — the grounding coordinate frame changed
> from padded-square to per-axis** (§2 below); `grid_to_norm` would silently
> mislocate every figure crop. Everything else passed, and the PriorGuide
> p. 5 loop is **gone** on this build (§4).
>
> **Adoption (same day): re-pinned on ≥ 9587, single frame.** `grid_to_norm`
> is per-axis only; `DeepSeekOcrBackend.min_server_build = 9587` makes the
> pipeline refuse older spawned servers (endpoints without `/props`
> `build_info` warn instead). Golden fixtures re-captured on 9587 (calibration
> @1280, calibration @2048, sample_paper p1). Verified live end-to-end:
> `inscriber ocr` on the calibration page yields
> `bbox_norm = (0.2422, 0.2432, 0.7538, 0.6537)` vs ground truth
> `(0.25, 0.25, 0.75, 0.65)` — within 1% per coordinate; the crop PNG visually
> contains the full box with the 2% margin. The §Consequences options below
> are kept as the record of the decision space.

## Environment

- llama.cpp build **9587 (d2e22ed97)** (`llms/new`, Windows CUDA prebuilt) vs
  the pinned build **9028 (d6e7b033a)** (`llms/` root). RTX 4060 Laptop 8 GB.
- DeepSeek-OCR BF16 + BF16 mmproj; production prompt/sampling/DRY flags via
  `DeepSeekOcrBackend`; ctx 16384. Gemma 4 E4B QAT Q4_K_XL + BF16 mmproj for
  the thinking check.
- Scripts: `gundam_check.py` (calibration @1280/1664/2048/2560 + PriorGuide
  p. 5 @1280), `m1b_check.py` (sample_paper fixture, cold + warm pass),
  `verify_thinking_spike.py`. Raw artifacts in `out-gundam/` (gitignored).
- Build-behavior note: 9587 caps the slot context at the model's
  `n_ctx_train` (8192 for DeepSeek-OCR; "slot context (16384) exceeds the
  training context") and defaults to `n_parallel = 4` — neither affected
  these runs, but ctx semantics differ from 9028.

## Results

### 1. Grounding format: unchanged ✓

The `LABEL[[x1, y1, x2, y2]]` block layout parses everywhere; labels as
pinned (`title`/`sub_title`/`text`/`image`/`image_caption`). `equation`
blocks now appear at 1280 renders too (9028 emitted them only at ≥2048);
the parser keeps unknown labels' text verbatim — no change needed.

### 2. Coordinate frame: CHANGED ✗ — per-axis, not padded-square

| render | 9028 `image[[…]]` (golden) | 9587 `image[[…]]` | Δ vs padded-square | Δ vs per-axis |
| --- | --- | --- | --- | --- |
| calibration @1280 | `[305, 245, 690, 653]` | `[242, 243, 753, 653]` | 36.8 | **5.8** |
| calibration @1664–2560 | `[305, 245, 689–690, 650–651]` | `[244, 244, 751, 651]` | 35.0 | **4.0** |

Predictions for the calibration box: padded-square `[312, 250, 687, 649]`,
per-axis `[250, 250, 749, 649]`. The real fixture page agrees:
`image[[280, 501, 707, 757]]` (9028) → `[240, 500, 755, 757]` (9587) —
y identical, x widened, exactly the portrait pad axis.

So this build's preprocessing no longer pads the image to a square: grid
coords are **relative to the original image, per axis** — the
reference-implementation mapping that M1a measured as *wrong* on 9028.
Consequences: `grid_to_norm`'s padded-square mapping mislocates boxes by
~7% of the long edge on A4 portrait pages, **silently** (text OCR is
unaffected; only crops shift). The frame is still render-size-invariant
(identical coords at 1664–2560).

### 3. Tiling: still none ✓

Exactly one `processing image...` per request at every size; prompt tokens
283 @1280 → 431 @≥1664, flat through 2560 — the same single-slice
saturation shape as 9028's 273/421 (the +10 is likely template
accounting). Encode ~0.8–3.1 s. Upstream Gundam tiling remains unshipped
for DeepSeek-OCR v1 (see `upstream-watch.md`).

### 4. PriorGuide p. 5 loop: gone on this build ✓ (improvement)

The page that ran to the 8192-token cap on 9028 (~80× repeated array row,
~6 min) completes in **37 s, 2012 tokens, `finish_reason: stop`**. The
Eq. 6–9 triple-underbrace array is transcribed (one spuriously duplicated
final row and a "revers kernel" misread remain), and the content 9028
lost — the two paragraphs + footnote after Eq. 9 — is present. The loop
*class* is improved, not proven gone: keep the loop/truncation-detection
TODO item.

### 5. m1b vertical slice + cache keys ✓

Cold pass OCRs and parses end-to-end (11 regions, `⟦INSCRIBER_FIG⟧`
placeholder spliced, caption kept); the warm pass serves from cache with
no server launch — the first live exercise of the build-identity cache
keys (landed the same day). Minor text drift vs the 9028 golden on this
clean page: `\(\mathrm{p}(y \mid x)\)` → `\(\mathsf{p}(\mathsf{y}\mid \mathsf{x})\)` —
build-dependent output at identical model/prompt/sampling, i.e. exactly
why the build is now key material.

(Found and fixed in passing: `m1b_check.py`'s `REPO` constant resolved to
`dev/` instead of the repo root, breaking its default `--pdf`.)

### 6. Gemma `enable_thinking` toggle ✓

`enable_thinking=True` → 112 completion tokens for a one-word answer
(thinking active); `False` → 2 tokens; omitted → 111 (this build's
default is thinking-ON). inscriber always sends the kwarg explicitly, so
the default is moot.

## Consequences

1. **Do not run inscriber against 9587 with the current code.** Text OCR
   is fine, but every figure crop shifts on the padded axis. Point
   `config.toml` `bin_dir` back to `llms/` (build 9028) until adoption.
2. **Adoption options** (decision tracked in `TODO.md`):
   - **Build-gated mapping** — `llama_build_identity()` is already probed
     for cache keys; `grid_to_norm` could select padded-square vs per-axis
     by build number. The changeover is bounded in (9028, 9587] — likely
     the preprocessing rework flagged in `upstream-watch.md` (PR #23345,
     unconfirmed); bisect or just gate at 9587. Needs per-frame golden
     fixtures.
   - **Re-pin v1 on ≥9587 per-axis only** — simplest code, drops verified
     9028 support; the no-loop result argues for this, but one page is
     thin evidence for OCR-quality parity.
   - **Fold into the DeepSeek-OCR-2 spike** — it needs ≥9587 anyway and
     redoes the calibration discipline from scratch; v1-model support on
     new builds could be decided there.
3. README's "any recent build should work" was wrong and is corrected —
   model-facing behavior is build-sensitive; the calibration page catches
   it in seconds (`gundam_check.py`).
