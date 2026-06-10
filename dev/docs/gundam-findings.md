# Gundam-mode findings — coordinate frame & tiling (2026-06-10)

> Closes the TODO item *"Gundam coordinate frame: determine empirically whether
> grounding coords are relative to the 1024 global view or the tiles."*
> **Answer: the question dissolves — llama.cpp build 9028 does not tile.**
> Every input size is encoded as ONE slice, and the grounding frame is the same
> **padded-square global frame** M1a pinned for `large`. Evidence captured by
> `dev/scripts/gundam_check.py`; golden fixture
> `tests/fixtures/deepseek_calibration_gundam2048_raw.txt`.

## Environment

Same pinned setup as `M1A-FINDINGS.md` (build 9028, RTX 4060 8 GB) with
**BF16** weights, production prompt/sampling/DRY flags via
`DeepSeekOcrBackend`, ctx 16384, `max_tokens` 8192.

## Method

The M1a calibration page (600×800 pt, box at (150,200,450,520) pt) rendered at
long-edge targets **1280 / 1664 / 2048 / 2560** and sent through one
`llama-server` session. The padded-square grid prediction is scale-invariant
(`[312, 250, 687, 649]`), so frame changes show up as coordinate divergence.

## Results

| render | image tokens | emitted `image[[…]]` | Δ vs padded-square | Δ vs per-axis ref |
| ------ | ------------ | --------------------- | ------------------ | ----------------- |
| 960×1280 (control = `large`) | **273** | `[305, 244, 690, 653]` | **5.0** | 31.0 |
| 1248×1664 | **421** | `[305, 245, 689, 650]` | **3.8** | 30.2 |
| 1536×2048 | **421** | `[305, 245, 690, 650]` | **4.0** | 30.0 |
| 1920×2560 | **421** | `[305, 245, 689, 651]` | **4.0** | 30.5 |

1. **No tiling.** The server log shows `clip load_hparams: image_size: 1024`
   and exactly one `encoding image slice... / decoding image batch 1/1` per
   request at every size. Upstream's Gundam dynamic tiling (n×640 tiles +
   global view) is **not implemented** in this build's DeepSeek-OCR path.
2. **Vision tokens saturate**: 273 tokens at 1280 long edge → **421 tokens at
   ≥1664**, flat through 2560. Bigger renders do buy more encoder capacity,
   but only up to that ceiling (encode time ~2.4 s → ~7.5 s/page).
3. **Frame: padded-square global at every size** (Δ≈4–5 vs prediction, vs
   Δ≈30 for the per-axis reference). Grid coords are render-size-invariant.
   `grid_to_norm` is correct unchanged for any render; golden test added
   (`test_parse_calibration_gundam2048_fixture_same_global_frame`).
4. The 1280 control reproduces M1a's Q8_0 coords at BF16
   (`[305, 244, 690, 653]` vs `[305, 245, 690, 653]`) — format/frame parity
   across quants, again.

## Consequences for inscriber

- **`gundam` is currently a strict alias of `large`** on this build:
  `ResolutionMode.GUNDAM.long_edge_px == 1280 == LARGE`, identical raster in,
  identical (deterministic) output. DESIGN §7's "model tiles it" does not
  happen here.
- The only real lever a bigger render buys is **273 → 421 vision tokens**.
  Candidate change (tracked in `TODO.md`): bump gundam's render target to
  ≥1664 (e.g. 2048) so the mode actually buys the saturated encoding —
  **needs quality validation** (more tokens ≠ proven better OCR) and costs
  ~3× image-encode time.
- **New grounding label observed at high res:** `equation[[…]]` (paper page
  rendered at 2048 grounded display equations as their own blocks). The parser
  already keeps unknown labels' text verbatim — no change needed; extends the
  M1A label list.

## Upstream status (researched the same day)

Tiling was **deliberately descoped** from llama.cpp PR #17400 ("I'll finish the
Gundam implementation in a follow-up PR"), and that follow-up (#24300) was
closed unmerged on 2026-06-09 in favor of a generic batching API (#24384, WIP).
So the no-tiling result above is the intended shipped state, with real tiling
pending upstream. Full picture — including DeepSeek-OCR 2, which ships *with*
tiling — in `upstream-watch.md`.

## Loop check (follow-up to equation-fidelity-findings)

The page that looped at 1280 (PriorGuide p. 5, the triple-underbrace Eq. 6–9
array) was re-run at a 1583×2048 render: **still degenerates** —
`finish_reason: length`, this time collapsing mid-array into `:eq:eq:eq…`
token spam inside one 10.9k-char equation block (everything before that block
was clean, correctly grounded, padded-square coords). So the loop is
**content-triggered, not resolution-fixable**, and the DRY flags
(0.5/1.75/30/90) did not break either loop shape — the `max_tokens` cap
remains the only effective bound. This strengthens the case for the
loop/truncation-detection TODO item.
