# M1A findings — DeepSeek-OCR de-risk spike

> **Date:** 2026-06-09 · **Status:** partially superseded — the server-path and block-format facts stand (DESIGN §2.1–2.2); the padded-square coordinate frame was build-9028-scoped, re-determined as per-axis on ≥9587 (`2026-06-10-build-9587-verification.md`).

> **Purpose (PLAN M1a / DESIGN §2.1, §2.2, §8.3):** pin the two highest-risk
> empirical unknowns on the real, pinned llama.cpp build before M1b locks the OCR
> parser + coordinate mapping. *"Nothing else can be trusted until this lands."*
>
> **Status: RESOLVED on real hardware (2026-06-09).** Several answers diverge from
> the DESIGN's expected-but-unverified assumptions — see ⚠️ rows.

## Environment (pinned)

| Item | Value |
| --- | --- |
| llama.cpp build | **9028 (`d6e7b033a`)** — `C:\Users\luigi\llms\llama-server.exe` |
| Backend | CUDA 12.4; **NVIDIA RTX 4060 Laptop, 8 GB VRAM** |
| OS / Python | Windows 11; CPython 3.12.6 |
| OCR model (tested) | `DeepSeek-OCR-Q8_0.gguf` + `mmproj-deepseek-ocr-q8_0.gguf`, `-ngl 99` |
| OCR model (recommended) | `deepseek-ocr-bf16.gguf` + `mmproj-deepseek-ocr-bf16.gguf` (DESIGN §2.2) |
| VLM model | `gemma-4-E4B-it-Q4_K_M.gguf` + `mmproj-BF16.gguf` (M2) |

**8 GB VRAM** justifies the **sequential** default (DESIGN §5.4).

## Q1 — Image round-trip → ✅ SERVER HTTP PATH WORKS

- A base64 image **round-trips successfully** through DeepSeek-OCR via
  `llama-server` `/v1/chat/completions` on build 9028. **Issue #21022 does NOT
  affect this build.**
- **Decision: v1 ships the `llama-server` HTTP path.** (`HttpInferencer`.)
- **Gemma 4 E4B VLM round-trip CONFIRMED (M2, 2026-06-09):** `gemma-4-E4B-it-Q4_K_M` +
  `mmproj-BF16` describes a figure crop over the server HTTP path (image-first), end to
  end through `inscriber describe`. Output quality is high (accurate chart description).
- ⚠️ The **`llama-mtmd-cli` fallback currently CRASHES** on this build
  (`STATUS_STACK_BUFFER_OVERRUN` during model warmup; also note its valid flags
  differ — no `--no-display-prompt`). Not needed since the server path works; the
  `MtmdCliInferencer` code remains as a documented, currently-broken fallback.

## ⚠️ Q1b — Content ordering: IMAGE MUST PRECEDE TEXT

- **Grounding only activates when the image content-part is sent BEFORE the text
  prompt.** Text-first → plain markdown, figures silently dropped, **zero** layout
  boxes. Image-first → full grounded layout output.
- **Fix applied:** `ChatClient.chat_image(image_first=True)` (default).

## ⚠️ Q2 — Coordinate frame → PADDED-SQUARE (not the DESIGN default)

- The 0–999 grid is relative to the image **padded to a square of side = the long
  edge**, with the **short axis centered** — NOT the reference per-axis/original-
  image mapping the DESIGN defaulted to.
- **Evidence (calibration page, `large`, render 960×1280, box at pts
  (150,200,450,520)):** emitted `image[[305, 245, 690, 653]]`.
  - padded-square prediction `[312, 250, 687, 649]` → Δ≈4.8 ✅
  - reference prediction `[250, 250, 749, 649]` → Δ≈30.8 ✗
  - the `title` box `x0=197` ≈ padded's predicted **200** (reference predicts 100).
- **Conversion to original-page [0,1] (lock this in `DeepSeekOcrBackend`):**

  ```text
  L = max(W_px, H_px)            # long edge of the rendered page, in px
  pad_x = (L - W_px) / 2;  pad_y = (L - H_px) / 2
  px = grid_x / 999 * L - pad_x  # → original-image pixels
  py = grid_y / 999 * L - pad_y
  x_norm = clamp(px / W_px, 0, 1);  y_norm = clamp(py / H_px, 0, 1)
  ```
- Gundam frame: **resolved 2026-06-10** — no tiling on this build; the same
  padded-square frame applies at every input size (`2026-06-10-gundam-findings.md`).
- ⚠️ **Superseded (build ≥ 9587):** everything above is 9028-specific — newer
  builds emit **per-axis** coords (the "reference prediction" above!) and the
  project re-pinned accordingly (`2026-06-10-build-9587-verification.md`).

## ⚠️ Q3 — Grounding OUTPUT FORMAT (not `<|ref|>`/`<|det|>`)

This build emits a **block layout list**, one region per block:

```text
LABEL[[x1, y1, x2, y2]]
<region markdown text, until the next LABEL[[…]] or a blank line>
```

- **NOT** the DESIGN's expected `<|ref|>LABEL<|/ref|><|det|>[[…]]<|/det|>`.
- **Labels observed:** `title`, `sub_title` (headings — text already carries `##`),
  `text` (body / author line), `image` (figure; **no text of its own**),
  `image_caption` (caption, wrapped `<center>…</center>`). Real papers will add
  more (`table`, …) — the parser keeps any non-figure label's text verbatim.
- **Figure-class label = `image`** (within DESIGN's figure set; `image` is the one
  this build uses). The caption is the `image_caption` block that immediately
  follows the `image` block → use it for `Region.text`.
- Math returns as inline `\(…\)` LaTeX. Headings already markdown-formatted.

### M1b parser implication

Parse the output into ordered `LABEL[[bbox]]` blocks (not inline-span regex).
Rebuild clean page markdown by concatenating each block's text in order, replacing
each `image` block with a `⟦INSCRIBER_FIG:{id}⟧` placeholder (DESIGN §8.3 step 4),
and attaching the following `image_caption` text as the figure's caption.

## Prompt / sampling (confirmed)

- Prompt **`<|grounding|>Convert the document to markdown.`** ✅ (grounded layout).
- `<|grounding|>OCR` and plain `OCR` → **runaway repetition loops** (`3.1.1.1…`) —
  do not use. Plain `Convert the document to markdown.` → ungrounded clean text
  (use when figures disabled, but note it won't ground).
- `temperature: 0`, `seed: 0`; hard `max_tokens` cap retained as the repetition
  guard (DESIGN §2.2).

## Fixtures captured (committed)

- `tests/fixtures/calibration.{pdf,json}` — generated; `large` predictions baked in.
- `tests/fixtures/deepseek_calibration_raw.txt` — real grounded output (Q8, large).
- `tests/fixtures/sample_paper.pdf` + `tests/fixtures/deepseek_paper_p1_raw.txt` —
  realistic multi-block grounded page (title/text/image/caption/math) — the M1b
  golden-parser fixture.

## Exit gate → CLEARED

M1b's parser + coordinate mapping are locked to: **server HTTP path**, **image-first**,
**`LABEL[[bbox]]` block format**, **padded-square frame**. Remaining confirmations
(non-blocking): bf16 weight produces identical format/frame; Gundam coord frame.

> **Update (2026-06-10):** the **bf16 confirmation is closed** — all subsequent
> real runs (the 39-page test paper incl. grounded figures/tables, and the table
> verification) used `deepseek-ocr-bf16.gguf` per the working config and produced
> the same `LABEL[[bbox]]` format with correct padded-square crops. The **Gundam
> frame is closed too** (later the same day): no tiling on this build, frame
> render-size-invariant — see `2026-06-10-gundam-findings.md`, which also adds `equation`
> to the observed-labels list (emitted at high-res renders).
