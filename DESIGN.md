# inscriber — Design Document

> **Status:** Implemented (v1 complete per `PLAN-inscriber-v1.md`); this document
> is the **authoritative, living specification** and is kept in sync with the
> code. Where the original pre-implementation draft made assumptions that real
> hardware later contradicted, the text below states the **confirmed** behavior
> directly; the empirical evidence records live in `dev/docs/`
> (`M1A-FINDINGS.md` for the OCR facts in §2.1–2.2/§8.3,
> `table-reconstruction-findings.md` for §9.7).
>
> **Audience:** A developer who has never seen this project (or its sibling,
> `paper2llm`). It is written to be read entirely standalone — every concept,
> dependency, and external quirk needed to build v1 is described here.
>
> **Last updated:** 2026-06-10 (added §9.7 — VLM table restructuring; aligned
> §2.1–2.2, §8.3, §25 with the M1a-confirmed facts)

---

## 1. What this project is

**`inscriber`** is a cross-platform command-line tool that converts academic
PDFs into clean, LLM-friendly **text-only Markdown** — running **entirely on the
user's own machine** using local models served by
[**llama.cpp**](https://github.com/ggml-org/llama.cpp). No cloud APIs are
required for the core pipeline.

It is the local, offline-first reimagining of an existing web app called
[**`paper2llm`**](https://github.com/lacerbi/paper2llm). `paper2llm` does the
same job but relies on cloud APIs (Mistral OCR for text extraction; Mistral /
OpenAI / Gemini / Anthropic vision models for figure description). The cloud
model landscape changes constantly and is tedious to track. `inscriber` trades
that churn for local control: the user points the tool at llama.cpp plus a
couple of GGUF model files and gets the same kind of output, reproducibly,
without sending documents to third parties.

### 1.1 What "the same job" means (pipeline parity with paper2llm)

For a given PDF, the output is:

1. A **full Markdown file** — the paper's text, tables, and equations, with each
   figure replaced by a generated **textual description** of that figure.
2. **Split files** (unless disabled): the document divided into `main`,
   `appendix`, and `backmatter` parts (see §11).
3. Optionally, a **BibTeX entry** for the paper (this single feature requires
   network access; see §12).

### 1.2 Goals

- Fully local core pipeline (OCR + figure description) — works with no internet.
- Runs on **Windows, Linux, and macOS**.
- Input is a **PDF file path or a URL**; output mirrors `paper2llm`.
- A **config file** specifies the llama.cpp binary location and model paths;
  **every config value is overridable from the CLI.**
- **Pluggable OCR backends** behind a stable interface. **v1 implements one:
  DeepSeek-OCR** — the only currently-supported model that locates figures itself
  in llama.cpp, which the figure→description pipeline requires (§2.4). Other
  SOTA text-OCR models (GLM-OCR, PaddleOCR-VL, Dots.OCR, …) are **deferred**
  pending a figure-detection solution (§22.1); the abstraction makes adding them
  purely additive.
- Pluggable **VLM backends** for figure description; first target is the
  **Gemma 4** family (Apache-2.0, multimodal, supported by llama.cpp).
- **Two execution modes** (§3.1): **end-to-end by default** (one command), or a
  **two-step `ocr` → `describe`** flow that materializes an inspectable _OCR
  bundle_ (§8.6) so you can run/compare different VLMs on the **same OCR + figure
  crops** without re-running OCR.

### 1.3 Non-goals (v1)

- No GUI / web interface. CLI only.
- No bundling or downloading of model weights — the user supplies GGUFs.
- No training, fine-tuning, or quantization of models.
- No attempt to perfectly reconstruct multi-page tables/equations that straddle
  a page break (documented limitation, §10.3).
- No OCR of scanned-handwriting or non-document images beyond what the chosen
  OCR model supports.

---

## 2. Background: external facts the design depends on

These were verified in June 2026. A future dev should re-verify against current
llama.cpp before relying on exact token strings.

### 2.1 llama.cpp multimodal support

llama.cpp exposes multimodal (vision) inference two ways, both relevant here:

- **`llama-server`** — a long-running HTTP server with an **OpenAI-compatible**
  `/v1/chat/completions` endpoint and a `/health` endpoint. Images are passed as
  base64 data URLs in the chat message content (the standard OpenAI
  `image_url` content-part shape). **This is what `inscriber` uses.**
- **`llama-mtmd-cli`** — a one-shot CLI for a single image+prompt. Reloads the
  model on every call (slow), so it is **not the primary path** — but it is kept
  as a **documented fallback** behind the same backend abstraction (see the
  ⚠️ note below and §8.2), because the server image path has had model-specific
  bugs.

> ✅ **Resolved (M1a, build 9028 — `dev/docs/M1A-FINDINGS.md` Q1).** A base64
> image **round-trips successfully** through DeepSeek-OCR via `llama-server`
> `/v1/chat/completions` — llama.cpp issue #21022 ("number of bitmaps (1) does
> not match number of markers (0)") does **not** affect this build. **v1 ships
> the `llama-server` HTTP path.** The Gemma 4 VLM round-trip over the same path
> is likewise confirmed. The `llama-mtmd-cli` fallback **crashes on this build**
> (`STATUS_STACK_BUFFER_OVERRUN` during warmup); because the fallback is not
> HTTP, the inference path stays abstracted behind an `Inferencer` (HTTP-server
> impl + mtmd-cli-subprocess impl, §8.2) — `MtmdCliInferencer` remains as a
> documented, currently-broken fallback should a future build regress the server
> path.
>
> ⚠️ **One ordering requirement the OpenAI shape doesn't suggest:** DeepSeek-OCR
> grounding **only activates when the image content-part precedes the text
> prompt** (M1a Q1b). Text-first silently degrades to plain markdown with zero
> layout boxes. `ChatClient.chat_image(image_first=True)` is the default for
> this reason.

A multimodal model in llama.cpp is **two files**:

- the **text model** GGUF (loaded with `-m` / `--model`), and
- a **multimodal projector** GGUF, conventionally named `mmproj-*.gguf` (loaded
  with `--mmproj`), which encodes images into embeddings the text model
  consumes.

So **every** model `inscriber` uses (OCR and VLM) is configured as a
`(model_gguf, mmproj_gguf)` pair.

### 2.2 DeepSeek-OCR (the v1 OCR backend)

- Support was **merged into llama.cpp `master`** via PR #17400 (merged
  2026-03-25). Requires a `deepseek-ocr` model GGUF + `mmproj-deepseek-ocr`
  projector GGUF; reference GGUFs live in the `ggml-org/DeepSeek-OCR-GGUF` HF
  collection.
- **Version note (settled).** A successor, **DeepSeek-OCR-2** (official; arXiv
  2601.20552 "Visual Causal Flow", deepseek-ai, ~27 Jan 2026, Apache-2.0, new
  DeepEncoder V2), exists and is real — but **llama.cpp supports only the
  _original_ DeepSeek-OCR** (arXiv 2510.18234, the DeepSeek3B-MoE-A570M decoder).
  **PR #17400 targets the original**, all ready-to-use GGUFs are for the original,
  and there is **no llama.cpp PR/issue for v2** as of June 2026. So **v1 targets
  the original DeepSeek-OCR**; DeepSeek-OCR-2 is future work, gated on upstream
  llama.cpp support (§22.2).
- **Quirks (must be respected):**
  - Use **f16** weights. **Q4_K_M causes runaway repetition loops** because the
    upstream model uses an **n-gram repetition penalty (ngram_size≈30,
    window≈90)** that llama.cpp **does not implement**. There is no exact
    equivalent flag: llama.cpp's `--repeat-penalty`/`--repeat-last-n` are
    token-level (not n-gram), and the DRY sampler (`--dry-multiplier`) is the
    closest analog — offer these via `server_flags()` as a _partial mitigation_,
    but the **real guards are f16 + a hard `max_tokens` cap + a per-request
    wall-clock timeout + soft-failure** on a looping/truncated page (§5.3, §16).
  - Drive OCR **deterministically**: `temperature: 0` + fixed seed (part of the
    cache key, §8.6).
  - **Chat template is path-dependent.** With **`llama-server`**, do **not** pass
    `--chat-template deepseek-ocr` — the server applies the model's built-in
    template. With the **`llama-mtmd-cli` fallback** (§2.1), the template flag
    _is_ used (the upstream examples pass `--chat-template deepseek-ocr --temp 0`
    to mtmd-cli). So the template choice is **per-path** — see `chat_template(path)`
    in §8.2, not a flat bool. M1 should confirm the server path's behavior.
  - **Prompt.** **`<|grounding|>Convert the document to markdown.`** — confirmed
    in M1a as the working grounded-layout prompt. ⚠️ `<|grounding|>OCR` and plain
    `OCR` (despite being reported working in the llama.cpp guide) produce
    **runaway repetition loops** on this build — do not use them. Plain
    `Convert the document to markdown.` yields clean **ungrounded** text and is
    what `inscriber` sends when figures are disabled (§8.3).
  - **Resolution modes.** DeepSeek-OCR's documented native modes are
    **Tiny (512px)**, **Small (640px)**, **Base (1024px)**, **Large (1280px)**,
    plus a dynamic tiling mode informally called **"Gundam"** (multiple ~640px
    tiles **plus** a 1024px global view) — highest quality, slowest, best for
    dense/multi-column pages. There is **no "standard" mode** (an earlier draft
    invented one). `inscriber` **defaults to `large`**, exposes the full ladder,
    and `gundam` as the dense-page opt-in (§7, §13). See §7 for the mode→render
    mapping, and note that **Gundam tiling is model-side behavior**: the
    rasterizer renders a single high-res page image and the model tiles it — the
    grounding-coordinate frame for Gundam must be confirmed during M1
    (coords may be relative to the 1024 global view, not the tiles).

> ✅ **Grounding format & coordinate frame (CONFIRMED — M1a, build 9028;
> evidence in `dev/docs/M1A-FINDINGS.md` Q2–Q3, locked in
> `tests/test_deepseek_parser.py` golden fixtures).** Upstream DeepSeek-VL docs
> describe inline `<|ref|>LABEL<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>` spans,
> but **this build emits a block layout list** instead — one region per block:
>
> ```text
> LABEL[[x1, y1, x2, y2]]
> <region markdown text, until the next LABEL[[…]] or a blank line>
> ```
>
> Labels observed: `title`, `sub_title` (text already carries `##`), `text`,
> `image` (the figure-class label this build uses; no text of its own),
> `image_caption` (wrapped `<center>…</center>`, immediately follows its
> `image` block). Math arrives as inline `\(…\)` LaTeX.
>
> Coordinates are on a **0–999 grid relative to the page image padded to a
> square** of side `L = max(W, H)` with the short axis centered — **not** the
> per-axis/original-image mapping reference implementations use (the two
> disagree on non-square pages; the padded-square prediction matched the
> calibration box to Δ≈5 grid units vs Δ≈31 for per-axis):
>
> ```text
> pad = (L - dim) / 2     per axis
> px  = grid / 999 * L - pad   →   norm = clamp(px / dim, 0, 1)
> ```
>
> The mapping lives in `DeepSeekOcrBackend` (`grid_to_norm`), keeping
> `bbox_norm` original-page-relative for the rest of the pipeline (§8.2). The
> **Gundam** mode's frame (global view vs. tiles) remains unconfirmed (tracked
> in `TODO.md`); re-verify format + frame on any llama.cpp upgrade.

### 2.3 Gemma 4 (first VLM backend)

- Released April 2026, **Apache-2.0** licensed. Variants: `E2B`, `E4B`
  (multimodal, efficient), `12B`, a `26B-A4B` MoE, and `31B` dense.
- The `E2B`/`E4B` variants are supported as multimodal models in llama.cpp and
  are the recommended figure-description models for `inscriber` (small, fast,
  permissively licensed). Larger variants work if the user has the hardware.
- **GGUF filenames in this doc (e.g. `gemma-4-e4b-f16.gguf`) are placeholders** —
  the user supplies the actual paths; real distributions use their own casing and
  quant suffixes (e.g. unsloth `gemma-4-E4B-it-GGUF`).
- Used as a **vision→text** describer (image in, prose out) for figures (§9) and
  as the table restructurer (§9.7). It does not need grounding or special prompts
  beyond the description/table prompts.
- **Gemma 4 is a thinking model.** Hard tasks spend reasoning tokens before the
  answer; llama-server strips the thought channel from `content`. `inscriber`
  activates thinking **explicitly** per request via
  `chat_template_kwargs: {"enable_thinking": true}` (needs the server's jinja
  templating; a no-op kwarg falls back to the model default). No `max_tokens` is
  sent on VLM calls — generation is bounded by `ctx_size`, and hitting the window
  yields `finish_reason: "length"` (the truncation signal).

### 2.4 OCR model landscape and why v1 is DeepSeek-OCR-only

Several SOTA OCR models are merged into llama.cpp and run via
`llama-server`/`llama-mtmd-cli` as `(model, mmproj)` pairs. **The decisive
difference for _this_ tool is whether the model locates figures itself** — because
the whole point of `inscriber` is converting figures into text descriptions, and
that requires knowing where the figures are.

| backend                      | llama.cpp PR | text/markdown OCR  | **native figure grounding?**                                                                    | in `inscriber`                   |
| ---------------------------- | ------------ | ------------------ | ----------------------------------------------------------------------------------------------- | -------------------------------- | --- | --- | -------------------- | ----------------------- |
| **DeepSeek-OCR**             | #17400       | ✅                 | ✅ inline `<                                                                                    | ref                              | >/< | det | >` boxes, 0–999 grid | **v1 (default & only)** |
| **PaddleOCR-VL** (1.5, 0.9B) | #18825       | ✅ (markdown/JSON) | ⚠️ **not in llama.cpp** — layout/detection is a _separate Paddle model_ (PP-DocLayout)          | **deferred (§22.1)**             |
| **GLM-OCR**                  | #19677       | ✅                 | ❌ **text-only by design** — doesn't predict coordinates; upstream pairs it with PP-DocLayoutV3 | **deferred (§22.1)**             |
| Dots.OCR                     | #17575       | ✅                 | ✅ JSON layout _with_ boxes                                                                     | future grounding-capable backend |
| HunyuanOCR                   | #21395       | ✅                 | (tbd)                                                                                           | future                           |

**Bottom line: DeepSeek-OCR is the only currently-supported model that delivers
the full figure→description pipeline standalone in llama.cpp**, so it is the sole
implemented backend in v1. GLM-OCR and PaddleOCR-VL are excellent at the _text_
half (SOTA), but in llama.cpp they emit **no figure boxes** — their detection
stage lives in an external PaddlePaddle model. They would only catch figures via
a raster-image fallback that **misses the vector figures common in LaTeX papers**
(matplotlib/TikZ → PDF). Rather than ship a half-working figure path for them,
**they are deferred until figure detection is solved** — see §22.1, which keeps
the capability comparison and lists candidate solutions. The `OcrBackend`
abstraction (§8) is built so adding them later is purely additive.

---

## 3. High-level architecture

```
                         ┌──────────────────────────────────────────┐
                         │                  CLI                      │
                         │  (argparse) parse args + load config      │
                         └───────────────┬──────────────────────────┘
                                         │ resolved RunConfig
                                         ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                              Pipeline orchestrator                          │
│                                                                             │
│  1. Input resolution   (PDF path | URL → local PDF bytes)   [§6]            │
│  2. Rasterize pages    (PDF → page PNGs, page-range applied) [§7,§13]       │
│  3. OCR pass           (each page PNG → markdown + figure bboxes) [§8]       │
│        └─ via OcrBackend (DeepSeekOcrBackend) over a managed llama-server    │
│  4. Figure crop        (bboxes → cropped figure PNGs)        [§8.4]          │
│  5. VLM pass: tables   (each <table> blob + page image → pipe table) [§9.7]  │
│  6. VLM pass: figures  (each figure crop + context → <img_desc>) [§9]        │
│        └─ both via VlmBackend (GemmaVlmBackend) over ONE managed llama-server│
│  7. Assemble + clean   (stitch pages, strip headers, inject descriptions)[§10]│
│  8. Split              (main / appendix / backmatter)        [§11]           │
│  9. BibTeX (optional, online)                               [§12]           │
│ 10. Write outputs                                           [§14]           │
└───────────────────────────────────────────────────────────────────────────┘
        │                               │
        ▼                               ▼
  LlamaServerManager              OcrCache (disk)
  (spawn/health/teardown) [§5]    (per-page OCR memoization) [§8.6]
```

**Key design decision — sequential, single-model-resident inference.** OCR and
VLM are different models. To keep peak RAM/VRAM to **one model at a time**, the
orchestrator runs **the entire OCR pass first** (OCR server up), tears that
server down, **then** brings up the VLM server for the entire figure pass. A
power user with plenty of memory can opt into keeping both up concurrently
(§5.4), but sequential is the default.

The OCR cache (§8.6) makes this design especially valuable: re-running with
different VLM settings reuses cached OCR and skips the expensive OCR pass
entirely.

### 3.1 Execution modes: end-to-end vs. two-step

The pipeline above is **end-to-end by default**, but it cleanly factors at the
OCR/VLM boundary (the OCR pass is independent of which VLM describes the figures).
`inscriber` exposes that boundary as three subcommands (§13.2):

- **`inscriber run INPUT`** (default; `inscriber INPUT` is shorthand) — the full
  pipeline, OCR through write, in one process.
- **`inscriber ocr INPUT`** — steps 1–4 only (resolve → rasterize → OCR → figure
  crop), then **write an _OCR bundle_** (§8.6) and stop. No VLM is loaded.
- **`inscriber describe BUNDLE`** — steps 5–10 (VLM table restructuring + figure
  description → assemble → split → BibTeX → write), reading a previously produced
  OCR bundle. No OCR is loaded.

**Why this is more than the cache.** The OCR cache (§8.6) is an internal,
content-addressed optimization for `run`. The OCR bundle is a **portable,
inspectable, user-facing artifact**. The motivating use case — _test/compare
several VLMs on the identical document and figure crops_ — is then just:

```
inscriber ocr paper.pdf -o out/                       # once
inscriber describe out/paper.inscriber-ocr --vlm-model gemma-4-e4b.gguf  ...
inscriber describe out/paper.inscriber-ocr --vlm-model qwen3-vl.gguf     ...
```

Each `describe` reuses the same OCR text and the same cropped figure PNGs, so
differences are attributable purely to the VLM. As a bonus, the bundle's per-page
markdown is **hand-editable** before `describe` (fix an OCR glitch once, then try
N VLMs). `run` is semantically `ocr` immediately followed by `describe`, sharing
the same serialization (§8.6).

---

## 4. Project layout & language

**Language: Python (3.10+).** Chosen because the local PDF/raster/imaging
ecosystem (PyMuPDF, Pillow) is best-in-class there, llama.cpp is consumed as a
subprocess + HTTP, and the reusable logic from `paper2llm` (splitting, BibTeX,
domain handling, the figure-description prompt) ports cleanly.

```
inscriber/
├── pyproject.toml              # packaging, deps, console entry point
├── README.md
├── DESIGN.md                   # this document
├── LICENSE                     # MIT
├── inscriber/
│   ├── __init__.py
│   ├── __main__.py             # enables `python -m inscriber`
│   ├── cli.py                  # argparse, wires CLI→RunConfig→pipeline
│   ├── config.py               # TOML load/merge/validate → RunConfig
│   ├── models.py               # dataclasses: Region, Figure, OcrPage, etc.
│   ├── pipeline.py             # orchestrator: run / ocr / describe (§3.1)
│   ├── input/
│   │   ├── resolver.py         # PDF path or URL → local bytes
│   │   └── domain_handlers.py  # 7 config-driven repo handlers (§6)
│   ├── pdf/
│   │   ├── rasterize.py        # PyMuPDF: PDF → page images, page count
│   │   ├── figures.py          # figure-detection strategies (§8.4)
│   │   └── crop.py             # crop figure regions from page images
│   ├── llama/
│   │   ├── server.py           # LlamaServerManager (spawn/health/teardown)
│   │   └── client.py           # OpenAI-compatible chat client (httpx)
│   ├── ocr/
│   │   ├── base.py             # OcrBackend ABC + shared dataclasses
│   │   ├── registry.py         # name → backend class
│   │   └── deepseek.py         # DeepSeekOcrBackend (grounding, §8.3)
│   │   # paddleocr_vl.py / glm.py — deferred (§22.1)
│   ├── vlm/
│   │   ├── base.py             # VlmBackend ABC
│   │   ├── registry.py
│   │   └── gemma.py            # GemmaVlmBackend
│   ├── postprocess/
│   │   ├── stitch.py           # multi-page join, header/footer & hyphen cleanup
│   │   ├── splitter.py         # main/appendix/backmatter (ported heuristics)
│   │   └── prompt.py           # figure-description prompt template + extractor
│   ├── bibtex/
│   │   └── semantic_scholar.py # optional online BibTeX (title→entry)
│   ├── bundle.py               # OCR bundle read/write (two-step, §8.5)
│   ├── cache.py                # OcrCache: content-addressed per-page store
│   ├── output.py               # writes full + splits + bibtex + figures/
│   └── logging.py              # progress + structured logging
└── tests/
    ├── fixtures/               # tiny sample PDF + recorded OCR/VLM responses
    ├── test_config.py
    ├── test_deepseek_parser.py # grounding parse + padding (golden, §17)
    ├── test_bundle_roundtrip.py # ocr→describe two-step (§8.5)
    ├── test_splitter.py
    ├── test_stitch.py
    ├── test_pipeline_mocked.py # full pipeline with mocked servers
    └── ...
```

---

## 5. llama.cpp server lifecycle (`llama/server.py`)

### 5.1 Ownership model

By default, **`inscriber` owns the server process**: it launches `llama-server`
with the right model/projector/flags, waits for readiness, runs the pass, and
terminates it. The user never hand-manages servers — they only configure the
binary directory and model paths.

A power-user escape hatch: if `--ocr-endpoint URL` (or `--vlm-endpoint URL`) is
given, `inscriber` **does not spawn** a server and instead talks to the
already-running endpoint at that URL. (Useful for remote/GPU boxes or shared
servers.)

### 5.2 Locating the binary (cross-platform)

`llama_cpp_bin_dir` from config points at the folder containing llama.cpp
binaries. To resolve the server executable:

```python
name = "llama-server.exe" if os.name == "nt" else "llama-server"
exe = Path(llama_cpp_bin_dir) / name
```

Resolve with `pathlib`; never rely on `PATH` unless `llama_cpp_bin_dir` is unset
(then fall back to `shutil.which("llama-server")`).

### 5.3 Launch, health, teardown

- **Launch:** `subprocess.Popen([exe, "-m", model, "--mmproj", mmproj, "--host",
"127.0.0.1", "--port", port, "-c", ctx, "-ngl", n_gpu_layers, ...])`.
  - Always use a **list of args** (never `shell=True`).
  - Bind to `127.0.0.1` on an **ephemeral free port** chosen by `inscriber`
    (probe with a socket bind, then pass it to `--port`). Note this is a small
    TOCTOU race — another process could grab the port between probe and the
    server's bind; on a `/health` timeout, retry with a fresh port.
  - **Do NOT add `--chat-template`** for DeepSeek-OCR (§2.2).
  - **Generation-safety flags** (per §2.2): pass repetition-penalty flags via
    `backend.server_flags()`; per-request, send `max_tokens` and `temperature: 0`
    from the client (§8.2). Capture stdout/stderr to a log file under the run dir.
- **Health:** poll `GET /health` until ready or timeout (`server_start_timeout`,
  default 120s). Contract: llama-server returns **503 while the model loads** and
  **200 when ready** — treat 503 as "keep waiting," not fatal. (Under load it can
  also return 200 with `"no slot available"`; for this single-client tool that
  won't occur, but don't assume every 200 means idle.) On timeout, surface a
  clear error including the last lines of the server log.
- **Teardown:** `proc.terminate()`, wait briefly, `proc.kill()` if needed.
  - Register an `atexit`/`finally` + signal handler so a Ctrl-C or crash never
    leaves an orphaned server. Use a `contextmanager`:
    ```python
    with server_manager.serve(ocr_model_spec) as endpoint:
        ... run OCR pass ...
    # server guaranteed down here
    ```
- **Cross-platform termination:** `Popen.terminate()` maps to `TerminateProcess`
  on Windows and `SIGTERM` on POSIX — both fine. Avoid `os.killpg`/process
  groups (POSIX-only). If a process group is needed for child cleanup, branch on
  `os.name`.

### 5.4 Concurrency mode

Config `inference.mode`:

- `sequential` (default) — one server at a time; the OCR pass fully completes and
  the server is torn down before the VLM server starts.
- `concurrent` — both servers up simultaneously (faster wall-clock). The real
  constraint is **VRAM**, not just RAM: each server gets its own `-ngl`, so allow
  an independent GPU-layer setting per server rather than a single global value.
  Even in `concurrent` mode, **consult the OCR cache before launching the OCR
  server** (§8.6) — a fully-cached document needs no OCR server at all. There is
  no automatic "do both models fit?" detection in v1; it is the user's
  responsibility, documented as a VRAM caveat.

---

## 6. Input resolution (`input/`)

Input is one positional argument: a **local PDF path** or an **http(s) URL**.

- **Path:** validate it exists, is readable, and has a `%PDF` magic header.
- **URL (requires network):**
  - Run it through **domain handlers** (ported from `paper2llm`). ⚠️ Reality
    check on the source: paper2llm has **no per-site handler classes and no
    generic fallback handler**. The directory `core/domain-handlers/` contains
    only `base-handler.ts`, `generic-handler.ts`, `index.ts` — a single
    **config-driven `GenericDomainHandler`** instantiated once per repository from
    a regex-based config (URL-match + PDF-URL transform + filename rule), wired up
    by `createAllRepositoryHandlers()` in `index.ts`. (Correction, verified
    2026-06-09: a `core/domain-handler-registry.ts` **does** exist one level up — a
    thin `DefaultDomainHandlerRegistry` singleton whose `getHandler(url)` is just
    find-first-`canHandle` over that list. The Python port needs only a list +
    first-match; a registry class is optional.) URLs not matching any config are
    simply **not handled** (no catch-all).
  - It ships **seven** repository configs — port all of them (pin each transform
    as a fixture, don't reverse-engineer):
    - **arXiv** `…/abs/{id}` → `…/pdf/{id}`
    - **bioRxiv / medRxiv** (identical rule) `…/content/(10.1101/{id})(vN)?…` →
      `…/content/{id}{vN}.full.pdf`
    - **NeurIPS/NIPS** `…/hash/{x}-Abstract.html` → `…/file/{x}-Paper.pdf`
    - **MLR Press (PMLR)** `…/vN/{id}` → `…/vN/{id}/{id}.pdf`
    - **ACL Anthology** — append `.pdf`
    - **OpenReview** — see special case below.
  - **OpenReview special case:** handled by a **host-level branch in
    `normalizePdfUrl` _before_ the generic transform rules** — it sets the path to
    `/pdf` while `URL.toString()` preserves the `?id=…` query (a plain path rewrite
    that dropped the query would break it). Its **filename** also reads
    `?id=` → `openreview-{id}.pdf` (fallback `openreview-paper.pdf`). Port the
    host-level branch, not just a per-rule replacement.
  - The Python shape can stay a small interface (method names are a free
    re-spelling of paper2llm's `canHandle` / `normalizePdfUrl` / `getFileName`):
    ```python
    class DomainHandler(Protocol):
        def can_handle(self, url: str) -> bool: ...
        def normalize_pdf_url(self, url: str) -> str: ...
        def file_name(self, url: str) -> str: ...
    ```
    …but the **reusable asset is the 7 regex configs**, not hand-written classes.
  - Download with `httpx`, following redirects, with a timeout and a
    descriptive User-Agent. Validate the downloaded bytes are a PDF.
- Output of this stage: a `ResolvedInput(pdf_bytes, source, original_url,
suggested_name)`.

> **Privacy note:** URL input and BibTeX (§12) are the _only_ features that
> touch the network. The OCR + VLM core is fully offline. The README must state
> this clearly. A `--offline` flag hard-disables all network use (URL input then
> errors early).

---

## 7. PDF rasterization (`pdf/rasterize.py`)

**Library: PyMuPDF (`pymupdf`).** Chosen specifically for cross-platform ease —
it ships prebuilt wheels for Windows/macOS/Linux with **no system dependency**
(unlike `pdf2image`, which needs poppler installed separately, painful on
Windows).

Responsibilities:

- **Page count** — needed to validate/clamp the page range.
- **Page range** — config/CLI `pages` as a **1-indexed inclusive** range,
  clamped to `[1, page_count]`. paper2llm only supports `{startPage, endPage}`;
  the open-ended/shorthand forms (`"1-10"`, `"3"`, `"5-"`, `"-12"`, `all`) are an
  **inscriber convenience, not ported behavior**.
- **Render** each selected page to a PNG at the long-edge pixel target for the
  OCR resolution mode. The zoom matrix is `fitz.Matrix(zoom, zoom)` with
  **`zoom = target_px / max(page_pt_w, page_pt_h)`** — PyMuPDF points are already
  1/72 inch and the matrix is a unit scale, so there is **no `* 72`** (an earlier
  draft had a `*72` that would render ~72× too large).

  | mode     | long-edge target                              | notes                                                                                                              |
  | -------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
  | `tiny`   | 512px                                         | fastest, lowest quality                                                                                            |
  | `small`  | 640px                                         |                                                                                                                    |
  | `base`   | 1024px                                        |                                                                                                                    |
  | `large`  | **1280px (default)**                          | balanced; good for most papers                                                                                     |
  | `gundam` | render high-res (≥1280px); **model tiles it** | densest pages; the rasterizer does _not_ tile — it renders one large page and DeepSeek-OCR tiles internally (§2.2) |

- Return `[PageImage(page_number, png_bytes, width_px, height_px)]`. The
  `(width_px, height_px)` are the **original rendered page** dimensions and are
  the reference frame for `bbox_norm` (§8.2) and cropping (§8.4).

Page images and crops are kept in a per-run **work directory** (under the OS temp
dir or `--workdir`); deleted on **success** unless `--keep-intermediates`, and
**kept on failure/Ctrl-C** for debugging (§15).

---

## 8. OCR pass & the `OcrBackend` abstraction (`ocr/`)

### 8.1 Why an abstraction

Different OCR models emit different grounding/layout formats, need different
prompts, and may even need a different _number of calls_. The pipeline must not
know these details. So OCR is hidden behind an interface; **v1 implements one
backend, `DeepSeekOcrBackend`** (§8.3), and the deferred text-OCR models (§22.1)
and future grounding models (Dots.OCR, …) are "write a new adapter + register
it", with **zero pipeline changes**. For that promise to actually hold, three
things below are non-obvious and deliberate: (a) the **backend owns the inference
call**, not just the prompt/parse; (b) `bbox_norm` is defined against a **fixed,
explicit frame**; and (c) a backend **declares whether it can ground figures**
(`supports_grounding`), which the figure step (§8.4) reads to choose grounding
vs. the (deferred) fallback path.

### 8.2 The interface (`ocr/base.py`)

```python
@dataclass
class Region:
    label: str                 # e.g. "figure", "table", "text", "title"
    # x1,y1,x2,y2 in [0,1], RELATIVE TO THE ORIGINAL RENDERED PAGE IMAGE
    # (the PageImage width_px/height_px from §7) — NOT the model's padded/tiled
    # frame. The backend is responsible for converting into this frame.
    bbox_norm: tuple[float, float, float, float]
    text: str | None = None    # caption/inline text for this region, if any

@dataclass
class OcrPageResult:
    page_number: int           # 1-indexed
    markdown: str              # clean markdown; figure regions are represented
                               # by ⟦INSCRIBER_FIG:{id}⟧ placeholders (§8.3)
    regions: list[Region]      # all detected regions (figures, tables, etc.)

class Inferencer(Protocol):
    """One multimodal (image+prompt → text) call. Two implementations:
       - HttpInferencer  → llama-server /v1/chat/completions (base64 image)
       - MtmdCliInferencer → one-shot `llama-mtmd-cli` subprocess (fallback, §2.1)
       Backends depend on this, NOT on an HTTP client directly, so the mtmd-cli
       fallback is implementable without changing any signatures."""
    def infer(self, image: PageImage, prompt: str, *, sampling: dict,
              chat_template: str | None, max_tokens: int, timeout_s: float) -> str: ...

class OcrBackend(ABC):
    name: str                  # registry key, e.g. "deepseek-ocr"

    @abstractmethod
    def ocr_page(self, inf: Inferencer, image: PageImage,
                 mode: ResolutionMode) -> OcrPageResult: ...
    """Own the WHOLE inference for one page: build prompt(s), call `inf` (possibly
       more than once, or expecting JSON layout), and return clean markdown +
       regions in the original-page frame. Single-call grounding backends
       (DeepSeek) and multi-call / JSON-layout backends both fit."""

    # capability: can this model locate figures from its own output?
    supports_grounding: bool = False   # DeepSeek-OCR → True; GLM/Paddle → False

    def server_flags(self) -> list[str]: return []      # e.g. DRY/repeat-penalty
    def sampling(self) -> dict: return {"temperature": 0}  # OCR determinism
    # chat template is PATH-AWARE (§2.2): the value (or None) to use on the
    # llama-server path vs the mtmd-cli path — they differ for DeepSeek-OCR.
    def chat_template(self, path: Literal["server", "mtmd-cli"]) -> str | None:
        return None
```

When `supports_grounding` is `False`, `ocr_page` returns `regions = []` (text
only) and figure detection falls to the experimental PyMuPDF-embedded path
(§8.4) — relevant only to the deferred backends (§22.1), not v1.

The orchestrator, per page, calls `backend.ocr_page(inf, image, mode)` and gets
an `OcrPageResult` whose bboxes are already in the original-page `[0,1]` frame —
so cropping (§8.4) is genuinely model-agnostic and the coordinate-frame mapping
(§8.3) lives **inside** each backend where it belongs.

> Why not the old `prompt()` + orchestrator-owned `client.describe()` + `parse()`
> split? Because it bakes in "exactly one text-returning call per page," which
> JSON-layout and two-call OCR models violate, and it would force per-model
> coordinate-frame logic into the shared crop step. Letting the backend own the
> call is what makes "second backend, zero pipeline changes" true rather than
> aspirational.

### 8.3 The v1 backend: `DeepSeekOcrBackend` (`ocr/deepseek.py`)

(The deferred text-OCR backends and their figure-detection problem are in §22.1.)

- `name = "deepseek-ocr"`; `supports_grounding = True`; `sampling()` sets
  `temperature: 0` + fixed seed and a `max_tokens` cap; `chat_template()` is
  path-aware (None on server; `"deepseek-ocr"` on the mtmd-cli fallback) (§2.2).
- Prompt: `"<|grounding|>Convert the document to markdown."` (§2.2; grounding on,
  for the figure boxes). When figures are disabled (`figure.detect = none`, §13),
  use the plain `"Convert the document to markdown."` prompt.
- **`ocr_page` algorithm** (single grounding call → clean text **and** boxes;
  this is the "exact parsing, single pass" decision; format/frame are the
  **M1a-confirmed** facts of §2.2):
  1. Call `inf` once with the grounding prompt (image content-part **before**
     the text — §2.1).
  2. Split the output into ordered **`LABEL[[x1, y1, x2, y2]]` blocks**
     (`MARKER_RE`): each marker line is followed by that region's markdown text,
     up to the next marker. Coords are on the **0–999 padded-square** grid.
  3. **Convert coordinates into the original-page `[0,1]` frame** via the
     padded-square mapping (`grid_to_norm`, §2.2): `L = max(W, H)`,
     `pad = (L − dim)/2` per axis, `px = grid/999 · L − pad`,
     `norm = clamp(px/dim, 0, 1)`. The mapping is **encapsulated in the
     backend** so `bbox_norm` is always original-page-relative (§8.2); for
     **Gundam**, the frame is still unconfirmed (§2.2).
  4. **Build clean markdown by _replacing_ each figure block, not blindly
     deleting it.** ⚠️ Critical: for **figure-class** blocks (`label` ∈
     {figure, image, picture, chart, diagram, plot}; this build emits `image`),
     emit a `⟦INSCRIBER_FIG:{id}⟧` placeholder token (`id = fig_p{page}_{i}`) in
     the block's position so the description can be injected at the figure's
     real position later (§10.2). The caption is the `image_caption` block that
     immediately follows the figure block — it becomes `Region.text` (used for
     the `{caption_or_label}` in `describe-and-keep`, §10.2) while its text also
     stays in the markdown. For non-figure blocks (text/title/table/…), keep the
     text verbatim. **Do not** strip everything — the placeholder is the only
     anchor and there is no inline `![]()` to fall back on (unlike paper2llm;
     see §10.2, B-note).
- **Robustness:** if grounding markup is malformed/absent, fall back to treating
  the whole output as plain markdown with `regions = []` (no figures described,
  pipeline still succeeds). Log a warning.

> ✅ **M1a (was the highest risk in the design) — DONE.** Real DeepSeek-OCR
> output was captured and committed as golden fixtures
> (`tests/fixtures/deepseek_paper_p1_raw.txt`, `deepseek_calibration_raw.txt`),
> `test_deepseek_parser.py` is pinned to them, and the coordinate frame was
> **determined empirically as padded-square** via a calibration page with a box
> at a known location (`dev/docs/M1A-FINDINGS.md` Q2). Re-run that discipline —
> capture, compare, re-pin — on any llama.cpp or model upgrade (§22.2).

### 8.4 Figure detection & cropping (`pdf/figures.py`, `pdf/crop.py`)

Figure detection is a **separate step from OCR text** (so future text-only
backends can plug in a different detector, §22.1). Config `figure.detect`:

- **`auto`** (default) — use OCR-backend grounding when
  `backend.supports_grounding`. In v1 that means **DeepSeek grounding**.
- **`grounding`** — force OCR-backend grounding; **error** if the backend can't.
- **`none`** — no figure detection/description (pure text OCR). `--no-figures`
  is an alias for `figure.detect = none` (there is no separate `enabled` flag —
  one knob, no redundancy).
- **`pdf-embedded`** — _experimental, mainly for the deferred text-only backends
  (§22.1)_: use **PyMuPDF** to extract embedded raster images + their page rects
  (`page.get_images()` + `page.get_image_rects()`) → `bbox_norm`. Catches raster
  figures only, **misses the vector figures common in LaTeX papers** — which is
  exactly why GLM/Paddle are deferred rather than shipped on this path. It ships
  in v1 only as an **experimental escape hatch** (it's just PyMuPDF; the test in
  §17 covers it); `auto` never selects it while DeepSeek grounds.

**Placeholder positioning:** grounding splices the `⟦INSCRIBER_FIG:{id}⟧`
placeholder at the figure's real position in the page markdown (§8.3 step 4).
(For the experimental `pdf-embedded` path there is no text anchor, so per-page
placeholders are appended after that page's text, ordered by rect `y0`.)

**Cropping** (bboxes already in the original-page `[0,1]` frame, §8.2): pixel box
= `(x1*W, y1*H, x2*W, y2*H)` against the page image (`W,H` = the `PageImage`
dims, §7); add a `figure.crop_padding` margin (default 0.02); clamp; skip
near-zero-area boxes; crop with Pillow; save `figures/fig_p{page}_{i}.png` keyed
by the placeholder `{id}`.

### 8.5 OCR bundle — the two-step artifact (`bundle.py`)

The OCR bundle is the **portable, inspectable output of `inscriber ocr`** and the
**input to `inscriber describe`** (§3.1). It contains everything needed to run
the VLM/assembly stages later, with **no OCR model required**. A directory:

```
OUT/paper.inscriber-ocr/
├── manifest.json     # source meta + OCR config + per-page results
├── figures/          # cropped figure PNGs (fig_p{page}_{i}.png)
└── pages/            # page rasters for table pages (page_NNNN.png, §9.7)
```

Pages whose markdown contains a restructurable `<table>` blob carry a per-page
`raster_path` (e.g. `"pages/page_0003.png"`) — the **verbatim** rendered page
PNG, so `describe` can run the VLM table-restructuring pass (§9.7) with no PDF
present, and `run`/`describe` share table cache keys (the key hashes the image
bytes). The field is additive: old readers ignore it (`bundle_schema` stays 1),
and a bundle without it simply skips table refinement with a warning.

`manifest.json`:

```jsonc
{
  "bundle_schema": 1, // integer; the compatibility gate (see below)
  "inscriber_version": "0.1.0", // informational only
  "created_at": "2026-06-09T...Z",
  "source": {
    "name": "paper",
    "source": "url",
    "original_url": "https://arxiv.org/abs/...",
    "pdf_sha256": "...",
  },
  "ocr": {
    "backend": "deepseek-ocr",
    "model_identity": "...",
    "mmproj_identity": "...",
    "resolution": "large",
    "render_long_edge_px": 1280,
    "prompt": "<|grounding|>Convert the document to markdown.",
    "sampling": { "temperature": 0 },
  },
  "figure_detect": "grounding",
  "pages": [
    {
      "page_number": 3,
      "markdown": "## 3. Method\n...\n⟦INSCRIBER_FIG:fig_p3_1⟧\n...",
      "regions": [
        {
          "label": "figure",
          "bbox_norm": [0.1, 0.24, 0.88, 0.61],
          "text": "Figure 1: ...",
        },
      ],
      "figures": [
        {
          "id": "fig_p3_1",
          "page": 3,
          "bbox_norm": [0.1, 0.24, 0.88, 0.61],
          "crop_path": "figures/fig_p3_1.png",
          "caption": "Figure 1: ...",
        },
      ],
    },
  ],
}
```

Notes:

- **Bundle vs. cache (they are NOT the same serialization).** The OCR **cache**
  (§8.6) stores the _pre-crop_ `OcrPageResult` (markdown + regions) at the OCR
  boundary. The **bundle** is a _superset_: per page it adds the post-crop
  `figures[]` (id, `crop_path`, caption) and the cropped PNGs on disk. So:
  cache = step-3 boundary; bundle = step-4 boundary. `run` threads in-memory
  objects, consults the cache, and skips bundle I/O entirely.
- `manifest.json` is **human-editable**: fix an OCR glitch in a page's `markdown`
  (keeping the `⟦INSCRIBER_FIG⟧` placeholders) once, then run `describe` with N
  different VLMs.
- **`bundle_schema` versioning:** `describe` accepts `bundle_schema <= SUPPORTED`
  and **refuses a higher value** with a clear error (never silently misparse).
  `inscriber_version` is informational and is **not** the gate (it churns every
  release). The §17 round-trip test asserts on `bundle_schema`.
- **What config `describe` honors** (it has no PDF and no OCR model):
  - **Applies:** `[vlm].*`, `[table].*` (§9.7), `[figure].mode`,
    `[figure].context_chars`, `[output].*`, `[bibtex].*`, `[net].offline`, and
    `[llama].*` + `[inference]` (it still launches a VLM server).
  - **Ignores (baked into the bundle at `ocr` time):** all `[ocr].*`,
    `[figure].detect`, `[figure].crop_padding`.
  - `figure.detect = none` / `--no-figures` at describe time **skips description**
    of bundled figures (leaves the figure out, or as a bare image ref if
    `describe-and-keep` — define as: drop the description, keep nothing).
  - **Output base name** comes from `manifest.source.name` (no PDF to derive from).
- `describe` also validates that every referenced `crop_path` exists.

### 8.6 OCR cache (`cache.py`)

Per-page OCR is the expensive step; cache it.

- **Key:** hash of `(pdf_content_hash, page_number, ocr_backend_name,
model_identity, mmproj_identity, resolution_mode, render_long_edge_px, prompt,
sampling_params)`. Each item matters:
  - `mmproj_identity` — the projector changes outputs too; hashing only the text
    model (an earlier draft's mistake) misses mmproj swaps.
  - `render_long_edge_px` — a different rendered resolution = a different input
    image even at the same mode name.
  - `sampling_params` — temperature/seed/`max_tokens` (§2.2/§8.2).
  - `*_identity` = file path + size + **content hash** (the hash itself cached by
    path+size+mtime so it's computed once). Keying on bare `mtime` is fragile:
    a re-download/copy that preserves content but changes mtime busts the cache
    spuriously, and `touch` without change wouldn't. Hash the content.
- **Value:** the **pre-crop** `OcrPageResult` (JSON; markdown with placeholders +
  regions) **plus** raw model output (debugging) and a `value_schema` integer so
  a future backend's richer result can't collide with a v1 entry. **No crops are
  stored** — cropping is recomputed each run from `figure.crop_padding` (which is
  therefore _not_ in the OCR key); the VLM cache's `figure_crop_hash` (§9.6) is
  what protects correctness when crops change.
- **Location:** `platformdirs.user_cache_dir("inscriber")/ocr/`. **Written
  per-page as each page completes** (not batched at the end), so an interrupted
  `run`/`ocr` resumes from the last completed page. The VLM cache (§9.6) is
  likewise written per-figure.
- On a re-run that changes only VLM settings, the entire OCR pass is served from
  cache → the OCR server is never even launched.
- **`--refresh`** ignores existing entries, recomputes, and **overwrites** them.
  **`--no-cache`** neither reads nor writes the cache (pure passthrough). These
  are distinct (§13).

---

## 9. VLM pass & the `VlmBackend` abstraction (`vlm/`)

### 9.1 Purpose

Each cropped figure is sent to a vision-language model with **surrounding text
as context**, producing a prose description that replaces the figure in the
final Markdown. This is exactly what `paper2llm` does with cloud vision models;
here it's a local VLM (Gemma 4).

### 9.2 Interface (`vlm/base.py`)

```python
class VlmBackend(ABC):
    name: str
    @abstractmethod
    def describe(self, image_png: bytes, context_text: str | None) -> str: ...
    """Return the cleaned description text (already extracted from tags)."""
```

`GemmaVlmBackend.describe` builds the prompt (§9.3), calls the chat client with
the image as a base64 data URL, then extracts the description from the
`<img_desc>…</img_desc>` tags (§9.4).

### 9.3 The figure-description prompt (`postprocess/prompt.py`)

Ported verbatim from `paper2llm` (it is model-agnostic and well-tuned). The
template, with a `{contextText}` placeholder:

```
# Task

Please describe the visual content of this image in detail, focusing on all
visible elements, text, and relevant information.

- Focus primarily on visual elements directly observable in the image: shapes,
  colors, objects, arrangements, and any visible text. When appropriate, include
  reasonable interpretation of what these elements represent based on their
  visual context.
- For academic or technical visuals: Identify the specific type (bar chart, line
  graph, flow diagram, etc.). Describe axes, labels, data points, and visual
  patterns exactly as they appear in the image.
- For any text visible in the image: Provide an accurate transcription,
  maintaining the original layout where meaningful.
- For images with multiple panels: Describe each panel separately based on its
  visual appearance. Note any panel labels if present. If the composition is
  unusual or the panels interact in a non-standard way, explain their
  relationship.
{contextText}

# Format

- Begin with a concise overview sentence identifying the type of image (e.g., "A
  line graph showing...", "A diagram illustrating...", "A photograph of...").
- Then provide specific details in a well-structured format. Use multiple
  paragraphs if necessary to organize different aspects of complex images.
- For complex visuals, you may use bullet points or numbered lists to clearly
  separate distinct elements.
- Adjust the length of your description based on the complexity of the image -
  simple images may need only a paragraph, while complex diagrams might require
  more detailed explanations.

IMPORTANT: You must wrap your entire description inside <img_desc> and
</img_desc> XML tags like this:

<img_desc>Your detailed description goes here.</img_desc>

Do not include anything else outside these tags.
```

When context is available, `{contextText}` is replaced with:

```
# Context

Context for reference:

<context>
{context}
</context>

Use this to correctly identify technical terms and provide reasonable
interpretations of what you can see in the image.
Your image description should still focus primarily on the visual aspects of the
figure and not be a mere repetition of the image caption or provided context.
```

When no context is available, the placeholder is removed.

### 9.4 Response extraction

Extract the substring between `<img_desc>` and `</img_desc>`. If the closing tag
is missing (truncated output), take everything after the opening tag. If the
opening tag is missing entirely, treat the whole (trimmed) response as the
description but log a warning (the model didn't follow format). Ported from
`paper2llm`'s `extractDescriptionFromTags`.

### 9.5 Context extraction

**Baseline behavior is ported from `paper2llm`** (`markdown-processor.ts`
→ `buildImageContextMap` / `extractImageContext`): it uses the **entire page's
text** as the figure's context — not a narrow window — prefixed with a short
preamble and **capped at ~2000 characters** to avoid overwhelming the model:

```
This image appears on page {N}. The surrounding page content follows.

{page_text, truncated to ~2000 chars}
```

This whole-page text becomes the `{context}` injected in §9.3.
**`figure.context_chars` is the truncation cap on the whole-page text, default
`2000`** (paper2llm truncates at `substring(0, 1997) + "..."` only when the page
exceeds 2000 chars) — it is **not** a "window around the figure." A narrow window
is an optional future refinement, but the default must reproduce paper2llm's
whole-page behavior.

Two precision notes for the implementer:

- **The preamble page number is a paper2llm _bug_ — inscriber fixes it.** paper2llm
  does `image.id.split("-")[0]`, but the Mistral image id is like `img-0.jpeg`, so
  this yields `"img"` (or `"unknown"`), **never** a real page number — its preamble
  is effectively always "This image appears on page img." inscriber has the real
  page, so use `N` directly (correcting, not reproducing, the behavior). Do **not**
  port `.split("-")[0]`.
- paper2llm does **not** extract captions separately for context — context is
  purely the whole-page text, and any caption is included only because it lives
  in that text. (`Region.text` from §8.3 feeds the `{caption_or_label}` in
  `describe-and-keep` output, §10.2 — a distinct use from context.)

### 9.6 VLM caching

Same scheme as §8.6, keyed on `(figure_crop_hash, vlm_backend_name,
vlm_model_identity, vlm_mmproj_identity, full_assembled_prompt, sampling_params)`.
The key uses the **fully assembled prompt — context text included** — not just a
template name; otherwise changing `context_chars` or the page text would serve a
stale description. Lets you re-run the document (e.g. to re-split or re-fetch
BibTeX) without re-describing figures.

### 9.7 Table restructuring (`postprocess/tables.py`) — tables before figures

> Validated post-v1 in `dev/docs/table-reconstruction-findings.md`; that note holds
> the experiment history and the prompt rationale. This section is the
> implemented behavior.

**Problem.** DeepSeek-OCR emits tables as **degenerate HTML** — `<table>…</table>`
with most cell boundaries missing, so adjacent cells concatenate
(`Dep. Variable:CCSR-squared:0.616`). All values are present but the grid is
gone, and it is not post-fixable from the text alone.

**Fix.** For each `<table>` blob, ask the VLM to **restructure** it: the blob
supplies the values, the **whole page image** supplies the layout, and the rest
of the page's text supplies correct spellings for merged labels. Low-risk
*structuring*, not re-OCR — the model copies the blob's values (even its typos).
The prompt is the validated one from the findings note, verbatim (count-aware
locator + correct-when-certain + page-text context), assembled by
`format_table_prompt()` and sent as a single user message, image first.
⚠️ **Treat the prompt text and message shape as pinned**: every ingredient was
added after a simpler version failed (history in the findings note) — do not
reword or restructure it without re-validating on real hardware.

Mechanics, in pipeline order (step 5, **before** figure description so figure
context already sees clean tables):

- **Detection** — well-formed `<table>…</table>` spans only (non-greedy regex;
  an unclosed tag never matches). GLM-OCR emits pipe tables, so it is a natural
  no-op there.
- **Guards** — a blob containing a `⟦INSCRIBER_FIG⟧` placeholder is left alone
  (splicing would destroy the anchor); an empty/value-less blob is left alone
  (nothing to anchor on → the task would degrade to re-OCR).
- **Output sanitation** — tolerate a wrapping code fence; reject anything that
  is not purely a pipe table. **Any failure — error, truncation
  (`finish_reason != "stop"`), commentary, empty — keeps the original blob**,
  which still holds every value. (A value-count check was considered and
  rejected: DeepSeek merges cells, so the blob's count is not a baseline.)
- **One VLM server for both passes** — the orchestrator's lazy `_VlmSession`
  starts the server on the first cache miss from either pass and shares it.
- **Caching** — per table, same store as §9.6, keyed on
  `(page_image_hash, backend, model/mmproj identities, full assembled prompt,
  sampling, chat_template_kwargs)` plus a `kind` discriminator.
- **Two-step** — `ocr` saves the verbatim page raster for table pages
  (`raster_path`, §8.5); `describe` reads it. Bundles without rasters skip with
  a warning.
- **Config** — `[table] refine = true` (default **on**), CLI `--no-table-refine`.
  Describe-stage; **independent of figure settings** (`--no-figures` does not
  disable it, and a run with tables but no VLM configured skips with a warning
  rather than failing).
- **No token budget** — generation is bounded by `ctx_size` alone (the single
  size knob; default 16384 leaves ~6–8k for the VLM's thinking + answer on top
  of the ~2–4k prompt). Gemma 4's thinking is activated explicitly per request
  via `chat_template_kwargs: {"enable_thinking": true}` (§2.3).

**Open refinements** (deliberately not in this pass — a cropped-table input
path and a system/user prompt split) are tracked in `TODO.md`.

---

## 10. Assembly & post-processing (`postprocess/stitch.py`)

### 10.1 Page stitching

OCR is per-page, so the document is reassembled by concatenating per-page
markdown in order. paper2llm exposes **two independent** page options
(`MarkdownOptions.addPageNumbers` / `addPageSeparators`) that `inscriber` keeps:

- **page numbers** — insert `#### Page {n}` before each page's content;
- **page separators** — insert a `---` horizontal rule between pages.

Both default off. **Note:** the splitter (§11) recognizes `#### Page N` markers
and shifts split boundaries around them, so keep the heading shape consistent
(`#### Page N`). Also port `normalizeLineBreaks` (collapse excess blank lines) as
part of the cleanup pass (§10.3).

### 10.2 Figure injection

Replace each `⟦INSCRIBER_FIG:{id}⟧` placeholder (spliced in at §8.3 step 4) with
the assembled figure block. The `<img_desc>…</img_desc>` tags are only the
model's _response envelope_ — they are **stripped** (§9.4) — and the extracted
text is rendered as a **Markdown blockquote with a bold header**, every line
prefixed with `> ` (including blank lines, which become `>` so the blockquote
doesn't break across paragraphs/lists in the description).

⚠️ **Port the _format_, not the mechanism.** paper2llm's `enhanceImageReferences`
works by regex-matching the inline `![alt](src)` image syntax that Mistral OCR
emits and keying on image id. DeepSeek-OCR grounding produces **no inline
`![]()`** — which is exactly why §8.3 splices a `⟦INSCRIBER_FIG:{id}⟧` placeholder
where each figure was. So reuse only the blockquote/header **formatting** from
`enhanceImageReferences`; the `![]()`-matching loop does not apply.

The **exact header string matters** (the `ensureImageDescriptionSpacing` regex
and downstream tooling depend on it), and paper2llm uses **two different**
headers:

- a real description → **`> **Image description.**`** (`markdown-processor.ts:298`);
- the no-description placeholder → **`> **Image.** [not displayed]`**
  (`markdown-processor.ts:329`).

Config `figure.mode` (mirrors paper2llm's `MarkdownOptions`):

- **`describe-only`** (**default — matches paper2llm**, whose `keepOriginalImages`
  defaults off, i.e. the image is _replaced_ by the description): emit just
  ```markdown
  > **Image description.** {description}
  ```
- **`describe-and-keep`** (paper2llm's `keepOriginalImages = true`; recommended
  for inscriber since we save crops to `figures/` anyway) — keep an image
  reference **and** the description:

  ```markdown
  ![{caption_or_label}](figures/{id}.png)

  > **Image description.** {description}
  ```

- **`placeholder`** (`replaceImagesWithPlaceholder`): emit
  `> **Image.** [not displayed]` (note: `Image.`, not `Image description.`).

Match paper2llm's trailing newline exactly: each emitted block ends with a single
`\n` (`markdown-processor.ts:312/:315/:329`) so `ensureImageDescriptionSpacing`
(§10.3) behaves identically.

> Do **not** leave raw `<img_desc>` tags in the output — they are an internal
> protocol with the VLM, not part of the document.

### 10.3 Cleanup pass

Two tiers: the **light normalization paper2llm already does** (port verbatim),
plus **new cleanup that local per-page OCR requires** (paper2llm got this for
free from Mistral's whole-document OCR).

**(a) Ported from paper2llm** (`markdown-processor.ts`) — always on:

- **`normalizeLineBreaks`** — collapse 3+ consecutive newlines to a single blank
  line (`\n{3,}` → `\n\n`).
- **`ensureImageDescriptionSpacing`** — guarantee a blank line **before and
  after** each description blockquote (`> **Image description.** …`, and the
  `> **Image.** [not displayed]` placeholder), and around any `Figure …` caption
  line that immediately follows an image block. Operates line-by-line; the real
  regex (`markdown-processor.ts:112`) is
  `^> \*\*(?:Image description|Image Description|Image)\.\*\*` (it tolerates all
  three header spellings — keep it as-is) and `^Figure `. This keeps descriptions
  from fusing into adjacent text.

**(b) New for inscriber** (per-page OCR artifacts) — heuristic, conservative
(never delete content we're unsure about), toggled by `--no-clean`:

- **Running headers/footers & page numbers:** detect short lines that recur at
  the same relative page position across many pages and strip them. Threshold-
  based; log what was removed.
- **De-hyphenation across page/line breaks:** join `word-\nword` → `word`, and
  merge sentences split by a page break when the next page starts mid-sentence
  (lowercase continuation). Conservative rules only.
- **Known limitation:** tables and equations that span a page boundary may not
  reassemble cleanly. Documented, not fixed in v1.

---

## 11. Splitting (`postprocess/splitter.py`)

Ported from `paper2llm`'s `markdown-splitter`. Splits the full document into up
to three parts by detecting section boundaries via heading regexes (case-
insensitive, any heading level `#+`):

- **Backmatter start** — first match of acknowledgments / author contributions /
  funding / impact statements / ethics, **or** references/bibliography:
  - `Acknowledgments?` / `Acknowledgements?`
  - `Author Contributions`, `Funding`
  - `Impact Statement`, `Broader Impact`, `Societal Impact`,
    `Ethical Considerations`
  - `References`, `Bibliography`, `Works Cited`, `Literature Cited`,
    `Citations`, `References and Notes`, `References Cited`, `Cited Works`,
    `Cited Literature`
- **Appendix start**:
  - `Appendix` / `Appendices`
  - `Supplementary|Supporting (Material|Materials|Information|Data)`
  - `Supplemental …`, `SI …`, `S1.`/`S2.` style headings
  - `A ` / `A. ` style appendix headings — **only accepted if they occur after
    the acknowledgments match** (guards against false positives like "A " in
    body text).
- Title is extracted from the first `# Title` heading, with paper2llm's
  fallbacks: if absent, try a BibTeX `title={…}` field, else default to
  `"Untitled_Paper"` (`markdown-splitter.ts` `extractTitle`).
- If a page marker immediately precedes a split boundary, the boundary is moved
  before it so page markers don't dangle. The marker regex tolerates **H3 or
  H4** (`^#{3,4}\s+Page\s+\d+\s*$`), though inscriber emits `#### Page N`.

Outputs `MarkdownSections(main_content, backmatter | None, appendix | None,
title)`. Positionally in the source, backmatter (acknowledgments/references)
usually precedes the appendix, so the regions are: `main = [0, backmatter_start)`
(or to appendix if no backmatter), `backmatter = [backmatter_start,
appendix_start)`, `appendix = [appendix_start, end)`.

**Standalone split files** must carry paper2llm's section framing (don't just
dump the raw slice): the **main** file's first H1 is normalized to the canonical
title (`prepareFormattedSections`), and standalone **appendix**/**backmatter**
files are prefixed with `# {title} - Appendix` / `# {title} - Backmatter` +
`\n\n---\n\n` (`content-utils.ts` `getSectionContent`, `markdown-splitter.ts`).

**Combined / "allparts" assembly** (paper2llm's `getSectionContent("allparts")`):
the parts can also be re-joined into a single document where appendix and
backmatter are reintroduced under derived headings. ⚠️ Note the **deliberate
reordering**: although backmatter precedes appendix _positionally_ in the source,
`allparts` re-emits in order **main → appendix → backmatter**
(`content-utils.ts:43-66`). This is faithful — don't "fix" it.

```markdown
{main_content}

# {title} - Appendix

---

{appendix}

# {title} - Backmatter

---

{backmatter}
```

This is the basis for the standalone full file (§14) and the
append-BibTeX-to-document option (§12).

---

## 12. BibTeX (optional, **online**) (`bibtex/semantic_scholar.py`)

Ported from `paper2llm`. **This is the one core-adjacent feature that requires
network access** and is therefore **opt-in** (`--bibtex` / config
`bibtex.enabled = true`).

- Extract the paper **title** from the document (`# Title`, §11).
- Query the **Semantic Scholar** API and take the **first result** (`results[0]`)
  as the best match. Exact call (verified 2026-06-09):
  `GET https://api.semanticscholar.org/graph/v1/paper/search?query={url-encoded title}&limit=3&fields=title,authors,venue,year,abstract,externalIds,url`,
  response taken from `data.data[0]`. Generate a citation key
  `{firstAuthorLastName}{year}{firstSubstantiveTitleWord}` where: author part = the
  last whitespace-token of the first author, lowercased; the title word is the first
  one that is `>2` chars and not a skip-word after stripping non-alphanumerics
  (skip-words, verbatim: `["a","an","the","on","in","of","for","and","or"]`), else
  fall back to the first word; year = paper year or current year. Note Semantic
  Scholar is **rate-limited** for unauthenticated use — degrade gracefully on 429.
  ⚠️ The source has **no explicit 429 handling** (any HTTP error → `[]`); inscriber
  **adds** the clean degrade-and-skip path.
- **No result / API error → mock fallback** (don't just drop it). ⚠️ Source
  precision: `bibtex-generator.ts`'s own `generateMockBibTeXEntry` is **discarded**
  — `generateBibTeXFromTitle` returns **`bibtex === ""`** (empty string) on
  failure, and that sentinel is what drives the include/retry path. The
  user-visible mock — the literal `@article{unknownYear, …, author={Unknown
Author}, journal={Unknown Journal}, …}` prefixed with `% WARNING: This is a
fallback mock citation.` — is assembled in **`content-utils.ts`**
  (`getContentWithOptionalBibtex`), **not** in `bibtex-generator.ts`. **Port the
  `content-utils` mock text and the empty-string sentinel** (not the discarded
  generator mock).
- **Title validation:** compare document title vs. returned title under a
  normalized comparison (`BibTeXTitleValidation`). Exact rules (verified 2026-06-09):
  normalize = lowercase → strip everything but `[a-z ]` → collapse whitespace → trim;
  titles whose normalized length is `<10` chars require an **exact** normalized match;
  longer titles match when the word-overlap ratio
  `commonWords / max(origWordCount, bibtexWordCount)` is **strictly `> 0.75`**.
  On mismatch, still emit the entry but prepend
  paper2llm's **exact 4-line** warning (note the trailing `% ` line):
  ```
  % WARNING: The retrieved citation title may not match the paper title.
  % Paper title: "{original_title}"
  % Citation title: "{bibtex_title}"
  %
  ```
  (paper2llm also has a slightly different mismatch wording —
  `% WARNING: The paper title does not match the citation title.` — inside the
  _mock_ branch; inscriber **standardizes on the one 4-line form above** for both
  paths, intentionally.)
- **Placement** (`content-utils.ts` `getContentWithOptionalBibtex`):
  - write a standalone `paper.bib` (default); **and/or**
  - **inject the entry into the document** (`bibtex.append_to_document`). ⚠️
    paper2llm **prepends** it (before the content) and wraps it in a **fenced
    code block** with a `---` separator — not a bare append:

    ````
    ```
    {bibtex, incl. any % WARNING lines}
    ```

    ---

    {document content}
    ````

    Only for `section ∈ {full, main, allparts}`.

- Respects `--offline` (skips with a clear message) and network failure (warns,
  continues — never fails the whole run for BibTeX).
- **On `retryBibtexGeneration` (§24 row 17):** in paper2llm this is an
  _interactive UI affordance_ (re-run when the user ticks the include-BibTeX box
  after a prior failure). A one-shot CLI has no such surface, so it is **not a
  faithful pipeline port** — model it as "re-running with `--bibtex` (cache makes
  this cheap) re-attempts the lookup," and it is listed under reclassified items,
  not as a literal feature.

---

## 13. Configuration & CLI

### 13.1 Config file (TOML)

Default discovery checks the current working directory first:

- `./config.toml`

If no local config exists, the fallback location is resolved via
**`platformdirs`**:

- Linux: `~/.config/inscriber/config.toml`
- macOS: `~/Library/Application Support/inscriber/config.toml`
- Windows: `%APPDATA%\inscriber\config.toml`

Overridable with `--config PATH`. **Every field is overridable by a CLI flag.**
Precedence: **CLI flag > config file > built-in default.**

```toml
[llama]
bin_dir = "/opt/llama.cpp/build/bin"   # folder containing llama-server[.exe]
host = "127.0.0.1"
port = 0                               # 0 = auto-select a free port
server_start_timeout = 120             # seconds to wait for /health
ctx_size = 16384                       # -c; the single size knob (prompt +
                                       #   generation share it; 16384 leaves room
                                       #   for the table pass, §9.7)

[inference]
mode = "sequential"                    # "sequential" | "concurrent"

[ocr]
backend = "deepseek-ocr"               # v1: deepseek-ocr only (others §22.1)
model = "/models/deepseek-ocr-f16.gguf"
mmproj = "/models/mmproj-deepseek-ocr-f16.gguf"
resolution = "large"                   # tiny | small | base | large | gundam
n_gpu_layers = "auto"                  # -ngl for the OCR server (per-server):
                                       #   "auto" (default; llama.cpp fits VRAM) |
                                       #   "all" | integer (0 = CPU)
endpoint = ""                          # if set, use this URL; don't spawn server

[vlm]
backend = "gemma"
model = "/models/gemma-4-e4b-f16.gguf" # placeholder name; user-supplied (§2.3)
mmproj = "/models/mmproj-gemma-4-e4b.gguf"
n_gpu_layers = "auto"                  # -ngl for the VLM server (per-server); see [ocr]
endpoint = ""

[figure]
detect = "auto"                        # auto | grounding | none | pdf-embedded(exp.)
                                       #   none = no figures (--no-figures alias)
mode = "describe-only"                 # describe-only (paper2llm default) |
                                       #   describe-and-keep | placeholder
crop_padding = 0.02                    # fraction of page dims (ocr-stage)
context_chars = 2000                   # whole-page context truncation cap (describe-stage, §9.5)

[table]
refine = true                          # VLM-restructure DeepSeek <table> blobs (§9.7;
                                       #   describe-stage, independent of [figure])

[output]
dir = "."                              # output directory
split = true                           # also write main/appendix/backmatter
page_numbers = false                   # insert "#### Page N" before each page
page_separators = false                # insert "---" between pages
normalize_line_breaks = true           # collapse excess blank lines
clean = true                           # header/footer + de-hyphenation pass
clobber = true                         # overwrite existing outputs
notice = true                          # append compact OCR/VLM caveat footer

[cache]
enabled = true                         # false ⇔ --no-cache (no read, no write)
refresh = false                        # true ⇔ --refresh (recompute + overwrite)

[workdir]
path = ""                              # "" = OS temp dir; else explicit dir
keep_intermediates = false             # keep page/crop images on success

[bibtex]
enabled = false                        # online; opt-in
append_to_document = false             # also inject (prepend, fenced) into doc

[net]
offline = false                        # hard-disable all network use
```

### 13.2 CLI surface (`cli.py`, argparse subparsers)

Three subcommands (§3.1). `run` is the default — bare `inscriber INPUT` ≡
`inscriber run INPUT`. Flags below are grouped by the stage they affect; each
subcommand accepts only the groups relevant to it.

```
inscriber run     INPUT [options]     # end-to-end (default)
inscriber ocr     INPUT [ocr-options] # OCR + crop → write OCR bundle, stop
inscriber describe BUNDLE [vlm-options]# OCR bundle → VLM + assemble + write

  # --- common ---
  INPUT                         PDF file path or http(s) URL   (run, ocr)
  BUNDLE                        path to a *.inscriber-ocr dir   (describe)
  -c, --config PATH             config file (default: ./config.toml, then platform config dir)
  -o, --output-dir DIR          output directory (default: cwd)
      --pages RANGE             1-indexed inclusive, e.g. "1-10","3","5-","-12","all" (run, ocr)

  # --- shared inference (run, ocr, describe — all launch a server) ---
      --llama-bin-dir DIR
      --host HOST               llama-server bind host (default 127.0.0.1)
      --port N                  fixed port (default 0 = auto)
      --ctx N                   context size
      --server-timeout SEC      seconds to wait for /health
      --mode {sequential,concurrent}   (run only; ocr/describe use one server)

  # --- OCR stage (run, ocr) ---
      --ocr-backend NAME        v1: deepseek-ocr (others deferred, §22.1)
      --ocr-model PATH
      --ocr-mmproj PATH
      --ocr-resolution MODE     tiny|small|base|large|gundam
      --ocr-ngl N               GPU layers for the OCR server (auto|all|int; default auto)
      --ocr-endpoint URL        use running server; don't spawn
      --figure-detect MODE      auto|grounding|none|pdf-embedded(exp.)
      --no-figures              alias for --figure-detect none
      --crop-padding FRAC       figure crop margin (fraction of page dims)

  # --- VLM / describe stage (run, describe) ---
      --vlm-backend NAME
      --vlm-model PATH
      --vlm-mmproj PATH
      --vlm-ngl N               GPU layers for the VLM server (auto|all|int; default auto)
      --vlm-endpoint URL
      --figure-mode {describe-only,describe-and-keep,placeholder}
      --context-chars N         whole-page context truncation cap
      --no-table-refine         keep raw OCR tables (skip VLM restructuring, §9.7)

  # --- output / assembly (run, describe) ---
      --no-split                write only the full document
      --page-numbers            insert "#### Page N" before each page
      --page-separators         insert "---" between pages
      --no-clean                skip header/footer + de-hyphenation cleanup
      --no-normalize-breaks     skip blank-line collapsing
      --no-clobber              error instead of overwriting existing outputs
      --no-notice               omit the OCR/VLM caveat footer
      --bibtex                  fetch BibTeX (requires network)
      --bibtex-in-doc           also inject the BibTeX entry into the document
      --offline                 disable ALL network use (URL input + bibtex)

  # caching / debugging
      --no-cache                neither read nor write caches
      --refresh                 ignore + recompute + overwrite caches
      --workdir DIR             where intermediate page/crop images go
      --keep-intermediates      don't delete the work dir on success
  -v, --verbose / -q, --quiet
      --version
```

> **`--no-figures` (= `--figure-detect none`) semantics** differ from paper2llm:
> here it means "don't detect or describe figures at all" (no crops, no VLM
> server, figure regions stripped from the markdown). paper2llm has no true off
> switch — with vision model "None" it still routes through
> `replaceImagesWithPlaceholder` and emits `> **Image.** [not displayed]` for
> every detected image. To reproduce _that_, use `--figure-mode placeholder`
> (detect + placeholder), not `--no-figures`.

### 13.3 Config ↔ CLI mapping (the "every field is overridable" contract)

| config key                                             | CLI flag                                                                          |
| ------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `llama.bin_dir`                                        | `--llama-bin-dir`                                                                 |
| `llama.host` / `llama.port`                            | `--host` / `--port`                                                               |
| `llama.ctx_size`                                       | `--ctx`                                                                           |
| `llama.server_start_timeout`                           | `--server-timeout`                                                                |
| `ocr.backend` / `ocr.model` / `ocr.mmproj`             | `--ocr-backend` / `--ocr-model` / `--ocr-mmproj`                                  |
| `ocr.resolution` / `ocr.n_gpu_layers` / `ocr.endpoint` | `--ocr-resolution` / `--ocr-ngl` / `--ocr-endpoint`                               |
| `vlm.*`                                                | `--vlm-backend` / `--vlm-model` / `--vlm-mmproj` / `--vlm-ngl` / `--vlm-endpoint` |
| `inference.mode`                                       | `--mode` (run only)                                                               |
| `figure.detect`                                        | `--figure-detect` (`--no-figures` ⇒ `none`)                                       |
| `figure.mode`                                          | `--figure-mode`                                                                   |
| `figure.crop_padding` / `figure.context_chars`         | `--crop-padding` / `--context-chars`                                              |
| `table.refine`                                         | `--no-table-refine` (sets false)                                                  |
| `output.dir`                                           | `-o/--output-dir`                                                                 |
| `output.split`                                         | `--no-split` (sets false)                                                         |
| `output.page_numbers` / `output.page_separators`       | `--page-numbers` / `--page-separators`                                            |
| `output.normalize_line_breaks`                         | `--no-normalize-breaks` (sets false)                                              |
| `output.clean`                                         | `--no-clean` (sets false)                                                         |
| `output.clobber`                                       | `--no-clobber` (sets false)                                                       |
| `output.notice`                                        | `--no-notice` (sets false)                                                        |
| `cache.enabled` / `cache.refresh`                      | `--no-cache` / `--refresh`                                                        |
| `workdir.path` / `workdir.keep_intermediates`          | `--workdir` / `--keep-intermediates`                                              |
| `bibtex.enabled` / `bibtex.append_to_document`         | `--bibtex` / `--bibtex-in-doc`                                                    |
| `net.offline`                                          | `--offline`                                                                       |
| (page range — inscriber-only, §7)                      | `--pages`                                                                         |

Every config field now has a CLI override (the §1.2 promise holds literally);
`--server-timeout` and `--no-normalize-breaks` were added for that reason.
`[figure]` straddles stages: `detect`/`crop_padding` are **ocr-stage** (baked
into the bundle), `mode`/`context_chars` are **describe-stage** (§8.5).

---

## 14. Output layout (`output.py`)

Given `INPUT` resolving to a base name `paper` and output dir `OUT`:

```
OUT/
├── paper.md                  # full document (always)
├── paper.main.md             # if split = true and split succeeded
├── paper.appendix.md         # if an appendix section was detected
├── paper.backmatter.md       # if a backmatter section was detected
├── paper.bib                 # if --bibtex and an entry was found
└── figures/                  # if figure-mode keeps images
    ├── fig_p1_1.png
    └── ...
```

- Base name: for `run`/`ocr`, the PDF filename **stem** (`Path(...).stem`) or the
  domain handler's `file_name(url)`; for `describe`, `manifest.source.name`
  (no PDF present, §8.5). Sanitize so a source literally named `paper.main.pdf`
  can't collide with the `paper.main.md` split output.
- `paper.md` is the **full** document (the enhanced, stitched markdown).
- **Two distinct `figures/` dirs:** the **bundle** always has one (crops are made
  at `ocr` time, before `mode` is chosen — §8.5); the **output** dir gets one only
  when `figure.mode = describe-and-keep` (the only mode that references crops),
  else only under `--keep-intermediates`.
- All files written **UTF-8 explicitly**, with `\n` newlines (don't let Windows
  inject `\r\n`).
- Default overwrites existing outputs (`output.clobber = true`), logging each
  file written; `--no-clobber` makes a pre-existing target a hard error instead.

---

## 15. Cross-platform requirements (Win / Linux / macOS)

These are hard requirements, not nice-to-haves:

- **Paths:** `pathlib.Path` everywhere; never string-concatenate paths. Resolve
  user `~` with `Path.expanduser()`.
- **PDF rendering:** PyMuPDF (wheels, no system poppler). **Do not** introduce a
  dependency that needs a separate system install on Windows.
- **Binary discovery:** append `.exe` on `os.name == "nt"` (§5.2).
- **Subprocess:** list-args only, no `shell=True`; `Popen.terminate()` for
  teardown (works on all three). Avoid POSIX-only `os.killpg`/`preexec_fn`
  unless guarded by an `os.name` branch.
- **Config/cache/data dirs:** `platformdirs` (`user_config_dir`,
  `user_cache_dir`, `user_data_dir`) — never hardcode `~/.config`.
- **File encoding:** always `encoding="utf-8"`, `newline="\n"` when writing text.
- **Temp/work dir:** `tempfile.mkdtemp()` or `workdir.path`; managed by a
  contextmanager. **Delete on success** (unless `keep_intermediates`); **keep on
  failure/Ctrl-C** for debugging.
- **`tomli`** is a _conditional_ dependency only — declare it
  `tomli; python_version < "3.11"` and do `import tomllib` with a `tomli`
  fallback (3.11+ has `tomllib` in the stdlib). Don't add it unconditionally.
- **`shutil.which`** (the PATH fallback in §5.2) honors `PATHEXT` on Windows, so
  it finds `llama-server.exe` without manual suffixing.
- **`--offline` does not gate the local servers.** The OCR/VLM `llama-server`
  processes are loopback (`127.0.0.1`), not "network" in the privacy sense —
  `--offline` only disables URL input and BibTeX. Do **not** wrongly block server
  spawn behind `--offline`.
- **GPU backend** (Metal on macOS, CUDA/Vulkan/etc. on Win/Linux) is whatever
  the user's llama.cpp build supports. `inscriber` stays agnostic and only
  passes `-ngl`.
- **CI:** test on all three OSes in the matrix (§17). No GPU in CI → servers are
  mocked.

---

## 16. Error handling, logging, progress

- **Fail fast, fail clearly** on config errors (missing model files, missing
  binary, unreadable PDF) — validate everything in `config.py` before any model
  loads.
- **Per-stage progress** to stderr: rasterizing (n pages), OCR (page i/N), VLM
  (figure i/M), assembling, splitting, bibtex, writing. A simple counter is
  enough; a progress bar (e.g. `rich`/`tqdm`) is a nice-to-have.
- **Resilience:** a single figure that fails to describe should not kill the run
  — log it, insert a `[figure description unavailable]` placeholder, continue.
  Same for BibTeX network failure, and for an OCR page that loops/truncates
  (§2.2): best-effort parse what came back, log, move on.
- **stdout vs stderr:** progress/logs go to **stderr**; on completion print the
  **list of written file paths to stdout** (one per line) so the run is
  machine-parseable even under `-q`.
- **Server failures:** on a `/health` timeout or non-200 chat responses, include
  the tail of the captured server log in the error so the user can diagnose
  (wrong model/mmproj pairing, OOM, bad flags).
- **Logging:** standard `logging`; `-v` → DEBUG (includes raw model outputs when
  `--keep-intermediates`), default INFO, `-q` → WARNING.

---

## 17. Testing strategy (`tests/`)

The real models need a GPU/large RAM and aren't available in CI, so tests mock
the inference layer at the **chat-client boundary**.

- **`test_deepseek_parser.py`** — golden-string tests for the DeepSeek grounding
  parser (§8.3) using **recorded real outputs** as fixtures: tokens + the
  M1-confirmed coordinate-frame mapping. Highest-value test; the single-pass
  grounding design hinges on exact parsing. (Per-backend variants land with each
  deferred backend, §22.1.)
- **`test_bundle_roundtrip.py`** — `ocr` writes a bundle; `describe` loads it and
  produces output consistent with `run` (same base name from `manifest.source.name`,
  §8.5); a hand-edited page markdown survives; a `bundle_schema` higher than
  supported is rejected (§8.5).
- **`test_tables.py`** — the table-restructuring pass (§9.7): blob detection /
  guards / sanitation / splicing units, thinking-kwarg + `finish_reason`
  truncation, cache-key disjointness, and mocked `run` + `ocr`→`describe`
  integration (verbatim bundle rasters, old-bundle and no-VLM degradation,
  multi-table locators, concurrent mode).
- **`test_pdf_embedded_figures.py`** — `figure.detect = pdf-embedded` on a fixture
  PDF with an embedded raster figure yields a crop + appended placeholder (§8.4).
- **`test_splitter.py`** — section-detection on a battery of synthetic markdown
  docs (with/without appendix, backmatter, the `A ` edge case, page markers).
- **`test_stitch.py`** — header/footer stripping & de-hyphenation on crafted
  multi-page inputs.
- **`test_config.py`** — TOML load, CLI-override precedence, validation errors.
- **`test_pipeline_mocked.py`** — end-to-end on a tiny fixture PDF with the OCR
  and VLM clients **mocked** to return canned responses; asserts the full set of
  output files and figure injection.
- **`LlamaServerManager`** — unit-test launch-arg construction and the `.exe`
  suffix logic without actually spawning (mock `Popen`).
- A **manual/integration** test doc (not in CI) describes how to run against a
  real llama.cpp + real GGUFs, with a known sample PDF, for release validation.

`npm`-style smoke check equivalent: `inscriber --version` and
`inscriber sample.pdf --no-figures --offline` against a fixture should pass with
mocked servers.

---

## 18. Packaging & distribution

- **`pyproject.toml`** (PEP 621), build backend `hatchling` or `setuptools`.
- Console entry point: `inscriber = "inscriber.cli:main"`.
- **PyPI name: `inscriber`** (verified available on PyPI as of 2026-06; the
  `inscriber` GitHub _user_ exists but the repo will live in the maintainer's
  namespace, no conflict).
- Python `>=3.10`.
- License: **MIT** (matches `paper2llm`).
- Optional extras: `[bibtex]` could gate the Semantic Scholar dependency if it's
  more than `httpx`, but keep core deps minimal.

### 18.1 Dependencies (intended, minimal)

| Dependency          | Purpose                                                  |
| ------------------- | -------------------------------------------------------- |
| `pymupdf`           | PDF page count + rasterization (no system poppler)       |
| `pillow`            | Crop figure regions from page images                     |
| `httpx`             | llama-server chat client; URL download; Semantic Scholar |
| `platformdirs`      | Cross-platform config/cache/data dirs                    |
| `tomli` (py<3.11)   | TOML parsing (`tomllib` is stdlib from 3.11)             |
| `rich` _(optional)_ | Progress output / nicer logs                             |

No heavy ML libs in `inscriber` itself — all inference is delegated to
llama.cpp over HTTP.

---

## 19. Performance & resources

- **DeepSeek-OCR at f16 + a Gemma 4 VLM** are the main memory consumers. The
  **sequential** mode (§5.4) keeps only one resident at a time — the default for
  good reason.
- **Resolution** is the main speed/quality lever: `large` (default) is a good
  balance; `gundam` (model-side tiling) is best for dense two-column papers but
  noticeably slower; `base`/`small`/`tiny` are the speed escape hatches.
- **Caching** (§8.6/§9.6) makes iteration cheap — changing split/figure/bibtex
  options re-runs in seconds because OCR and VLM results are reused.
- **GPU offload** via `-ngl` is the biggest wall-clock win when available; left
  to the user's hardware/build.

---

## 20. Security & privacy

- **Local by default.** The only network egress is (a) downloading a PDF when
  the input is a URL and (b) the opt-in Semantic Scholar BibTeX lookup. Both are
  disabled by `--offline`. Documents and figures are **never** sent to any
  third-party model API — they go only to the user's own llama.cpp server on
  `127.0.0.1`.
- The server binds to **loopback** on an ephemeral port; it is not exposed.
- No telemetry. No persisted secrets (there are no API keys in the core flow).

---

## 21. Implementation milestones

1. **M0 — Skeleton.** Project layout, `pyproject.toml`, CLI argparse →
   `RunConfig`, TOML config load/merge/validate, logging. `inscriber --version`
   and config errors work.
2. **M1a — De-risk spike (do this first).** `Inferencer` (HTTP impl +
   mtmd-cli impl, §8.2) + `LlamaServerManager` + PyMuPDF rasterize, then **the two
   highest-risk unknowns**: (i) prove a base64 image round-trips through
   DeepSeek-OCR on `/v1/chat/completions` for the pinned llama.cpp build, _or_
   fall back to `llama-mtmd-cli` (§2.1); (ii) **capture real grounding output** to
   `tests/fixtures/` and **determine the coordinate frame empirically** (§8.3 step
   3 / §2.2). Nothing else can be trusted until this lands.
3. **M1b — OCR vertical slice.** `DeepSeekOcrBackend.ocr_page` with the parser +
   coordinate mapping **locked to the M1a fixtures**, the OCR cache, and per-page
   markdown (with `⟦INSCRIBER_FIG⟧` placeholders) for a real PDF. **Design the
   `OcrPageResult` (de)serialization once here** — it's reused by both the cache
   (M1b) and the bundle (M2), so don't pick a format the bundle must later migrate.
4. **M2 — Figures + two-step split.** Figure detection (§8.4: grounding for
   DeepSeek), cropping, VLM server + `GemmaVlmBackend`, prompt + extraction,
   whole-page context, blockquote injection (§10.2), VLM cache. **Land the
   `ocr`/`describe` subcommands and OCR-bundle read/write here** (§3.1, §8.5) —
   it falls out naturally once the OCR↔VLM boundary is serialized, and it's the
   workflow that makes VLM comparison cheap.
5. **M3 — Assembly & splitting.** Stitching, the ported light post-processing +
   new cleanup (§10.3), splitter with standalone-file headers (§11), output
   writer (full + splits + figures/).
6. **M4 — Inputs & BibTeX.** URL input + the 7 domain configs (§6), `--offline`,
   Semantic Scholar BibTeX with title validation, mock fallback, and
   prepend/fenced injection (§12). (GLM-OCR / PaddleOCR-VL are **not** here —
   post-v1, gated on figure detection, §22.1.)
7. **M5 — Hardening.** Cross-platform CI matrix, mocked end-to-end tests,
   `concurrent` mode, docs/README, packaging to PyPI.

---

## 22. Open questions / future work

> Concrete, near-term actionables (pending verifications, code debts) are
> tracked in **`TODO.md`** — this section is the longer-horizon work.

### 22.1 Deferred OCR backends: GLM-OCR & PaddleOCR-VL (text-SOTA; figures TBD)

GLM-OCR (#19677) and PaddleOCR-VL-1.5 (#18825) are **SOTA at text/table/equation
OCR** and would be valuable backends — `inscriber`'s `OcrBackend` abstraction (§8)
is built to accept them additively (`name`, `ocr_page`, `supports_grounding`,
prompt/parse). They are **deferred from v1 for one specific reason**: in
llama.cpp they emit **no figure bounding boxes**, and `inscriber`'s core job is
turning figures into descriptions.

- **GLM-OCR** is text-only by design (it deliberately doesn't predict
  coordinate tokens; upstream pairs it with PP-DocLayoutV3).
- **PaddleOCR-VL** _has_ layout detection, but as a **separate PaddlePaddle model
  (PP-DocLayout), not in llama.cpp** — standalone in llama.cpp it recognizes
  content without reliable figure localization.

So the blocker is **figure detection**, and shipping them means picking a
solution (all TBD; each is a tradeoff):

1. **External layout model (PP-DocLayout / PP-DocLayoutV3).** Highest fidelity,
   matches upstream usage; lets the backend set `supports_grounding = True`.
   Cost: heavy optional PaddlePaddle dependency, extra model to manage, more
   integration — keep strictly opt-in.
2. **PyMuPDF vector-aware detection.** Cluster the PDF's vector drawings
   (`page.get_drawings()` / `cluster_drawings()`) **plus** raster image rects to
   infer figure regions. No extra model/dependency, fully local. Cost: heuristic
   — risks catching tables/equations/rules or splitting composite figures; needs
   tuning and validation.
3. **`pdf-embedded` raster fallback only** (the experimental path, §8.4). Cheap
   and already specified, but **misses the vector figures common in LaTeX
   papers** — acceptable only for raster-heavy/scanned PDFs, not as the general
   answer.
4. **Prefer a grounding-capable model instead.** If the goal is "another backend
   besides DeepSeek," **Dots.OCR** (#17575) emits JSON layout _with_ boxes and
   may be a better next target than retrofitting detection onto GLM/Paddle.

**Recommendation when this is picked up:** treat GLM-OCR/PaddleOCR-VL as
**text-OCR backends** first (figure detection via option 1 or 2), pin each
model's prompt and output format on real captured output (same M1 discipline as
DeepSeek, §8.3), and decide whether `pdf-embedded` is an acceptable interim
default for them or whether figures should simply be `none` until a real detector
is wired.

### 22.2 Other future work

- **More grounding-capable OCR backends** — Dots.OCR (#17575, JSON layout _with_
  boxes; natural next backend) and HunyuanOCR (#21395).
- **DeepSeek-OCR-2** (arXiv 2601.20552, DeepEncoder V2 "Visual Causal Flow",
  +3.73% OmniDocBench over the original) — a likely upgrade once it has a
  **llama.cpp path**, which does not exist yet (no PR/GGUFs as of June 2026). Its
  grounding/coordinate convention must be re-confirmed when that lands.
- **Table reconstruction across page breaks** (§10.3) — currently a documented
  limitation.
- **Batch mode** — process a directory of PDFs reusing a single warm server.

---

## 23. Relationship to `paper2llm` (reuse map)

Logic ported (reimplemented in Python), not shared as a library:

| `paper2llm` (TypeScript)                               | `inscriber` (Python)         | Notes                                                                            |
| ------------------------------------------------------ | ---------------------------- | -------------------------------------------------------------------------------- |
| `core/templates/image-prompt-template.ts`              | `postprocess/prompt.py`      | Prompt + `<img_desc>` extractor — used verbatim                                  |
| `core/utils/markdown-splitter.ts`                      | `postprocess/splitter.py`    | Section regexes + boundary logic                                                 |
| `core/utils/bibtex-generator.ts`                       | `bibtex/semantic_scholar.py` | Semantic Scholar lookup + title validation                                       |
| `core/domain-handlers/{base,generic,index}-handler.ts` | `input/domain_handlers.py`   | One config-driven `GenericDomainHandler`; port the **7 repo regex configs** (§6) |
| `core/ocr-service.ts` (Mistral)                        | `ocr/` backends              | Replaced by local llama.cpp OCR                                                  |
| `core/image-service*.ts` (cloud VLMs)                  | `vlm/` backends              | Replaced by local llama.cpp VLM                                                  |
| API-key storage/encryption                             | —                            | Not needed; no cloud keys in core flow                                           |

---

## 24. paper2llm feature-parity checklist (with source pointers)

The dev will be given the `paper2llm` source. This table enumerates **every
paper2llm feature** and states whether `inscriber` keeps it, where it's
specified here, and which paper2llm file to read as the reference
implementation. Paths are relative to `paper2llm-web/src/`.

| #   | paper2llm feature                                                                                                                               | Keep?                  | `inscriber` § | Reference source in paper2llm                                                                                                                |
| --- | ----------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | PDF file input + validation                                                                                                                     | ✅                     | §6            | `adapters/web/file-handler.ts`                                                                                                               |
| 2   | URL input + domain handlers (**7 repos**: arXiv, OpenReview, ACL, bioRxiv, medRxiv, NeurIPS, MLRP; no generic fallback)                         | ✅                     | §6            | `core/domain-handlers/{base,generic,index}-handler.ts` (one config-driven handler; `createAllRepositoryHandlers`)                            |
| 3   | Page-count detection + **page-range selection**                                                                                                 | ✅                     | §7            | `core/utils/pdf-page-utils.ts`, `web/components/PageRangeSelector.tsx`                                                                       |
| 4   | OCR of text / tables / equations                                                                                                                | ✅ (local)             | §8            | `core/ocr-service.ts` (Mistral → DeepSeek-OCR)                                                                                               |
| 5   | Figure description via vision model                                                                                                             | ✅ (local)             | §9            | `core/image-service.ts`, `core/image-services/*` (cloud → llama.cpp VLM)                                                                     |
| 6   | Image **context = whole page text** (~2000-char cap, preamble)                                                                                  | ✅                     | §9.5          | `core/markdown-processor.ts` → `buildImageContextMap`, `extractImageContext`                                                                 |
| 7   | Figure-description **prompt template** + `<img_desc>` extraction                                                                                | ✅ (verbatim)          | §9.3–9.4      | `core/templates/image-prompt-template.ts`                                                                                                    |
| 8   | Figure as **blockquote** `> **Image description.**` (placeholder uses `> **Image.** [not displayed]`); format ported, `![]()` matching loop not | ✅                     | §10.2         | `core/markdown-processor.ts` → `enhanceImageReferences` (`:298`, `:329`)                                                                     |
| 9   | Figure modes: **describe-only (default, =paper2llm)** / describe-and-keep / placeholder                                                         | ✅                     | §10.2, §13    | `MarkdownOptions` (`keepOriginalImages` defaults **off**, `replaceImagesWithPlaceholder`) in `types/interfaces.ts` + `markdown-processor.ts` |
| 10  | Page **numbers** (`#### Page N`) and page **separators** (`---`)                                                                                | ✅                     | §10.1         | `core/markdown-processor.ts` (`addPageNumbers`, `addPageSeparators`)                                                                         |
| 11  | `normalizeLineBreaks` (collapse 3+ blank lines)                                                                                                 | ✅                     | §10.3(a)      | `core/markdown-processor.ts`                                                                                                                 |
| 11b | `ensureImageDescriptionSpacing` (blank lines around `> **Image.**` blocks & `Figure …` captions)                                                | ✅                     | §10.3(a)      | `core/markdown-processor.ts` → `ensureImageDescriptionSpacing`                                                                               |
| 12  | Split into **main / appendix / backmatter** (heading heuristics)                                                                                | ✅                     | §11           | `core/utils/markdown-splitter.ts`                                                                                                            |
| 13  | **Combined "allparts"** with `# {title} - Appendix/Backmatter` headers                                                                          | ✅                     | §11           | `web/components/markdown-preview/utils/content-utils.ts` → `getSectionContent`                                                               |
| 14  | **BibTeX** generation (Semantic Scholar)                                                                                                        | ✅ (online, opt-in)    | §12           | `core/utils/bibtex-generator.ts`                                                                                                             |
| 15  | BibTeX **title validation** + `% WARNING` mismatch comment                                                                                      | ✅                     | §12           | `bibtex-generator.ts`, `content-utils.ts`, `BibTeXTitleValidation` in `types/interfaces.ts`                                                  |
| 15b | BibTeX **mock fallback** entry (mock text in `content-utils`) + empty-string failure sentinel (`bibtex-generator`)                              | ✅                     | §12           | `content-utils.ts` (mock text), `bibtex-generator.ts:515` (`""` sentinel)                                                                    |
| 16  | **Inject BibTeX into document** — _prepended_, fenced code block, `---` separator                                                               | ✅                     | §12           | `content-utils.ts:195` → `getContentWithOptionalBibtex`                                                                                      |
| 17  | BibTeX retry on demand                                                                                                                          | ⤳ Reclassified         | §12           | UI affordance (`useCopyDownload.ts` `retryBibtexGeneration`); no CLI analog — re-run with `--bibtex`                                         |
| 18  | Output **filename** derived from source (PDF name / URL handler)                                                                                | ✅                     | §14           | `useCopyDownload.ts`, domain handlers                                                                                                        |
| 19  | **Progress reporting** per stage                                                                                                                | ✅                     | §16           | `adapters/web/progress-reporter.ts`, `web/components/ProcessingStatus.tsx`                                                                   |
| 20  | **Cancel** an in-flight operation                                                                                                               | ✅ (Ctrl-C → teardown) | §5.3, §16     | `OcrService.cancelOperation`, `ImageService.cancelOperation`                                                                                 |
| 21  | Debug mode (verbose / keep intermediates)                                                                                                       | ✅                     | §13, §16      | `MarkdownOptions.debugMode`                                                                                                                  |
| 22  | Multi-**provider** model selection (Mistral/OpenAI/Gemini/Anthropic)                                                                            | ⤳ Replaced             | §8.1, §9.2    | `core/image-services/image-service-factory.ts` → replaced by pluggable local OCR/VLM **backends**                                            |

**`MarkdownOptions` flag accounting** (all 8): `addPageNumbers`,
`addPageSeparators` (§10.1); `normalizeLineBreaks` (§10.3a); `processImages`
(→ `figure.detect = none` / `--no-figures`, with the placeholder caveat in §13.2);
`keepOriginalImages`, `replaceImagesWithPlaceholder` (→ `figure.mode`, §10.2);
`debugMode` (→ `-v`/`--keep-intermediates`). **`extractImageReferences`** only
populates a bookkeeping `imageReferences[]` list for UI use — **dropped** as
internal; inscriber tracks figures via `Region`/placeholders instead.

> **Latent-bug warning — do not replicate:** `content-utils.ts:237`'s
> `calculateImageMetrics` counts described images with the regex
> `/> \*\*Image Description:\*\*/g` (capital D, colon), which does **not** match
> the text actually emitted (`> **Image description.**`, lowercase, period). It's
> a real paper2llm bug; if any image-metrics logic is ported, fix the regex.

### Intentionally **dropped** (cloud/UI-only, no local analog)

| paper2llm feature                                                           | Why dropped                                                | Source (for reference)                                                                                          |
| --------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| API-key storage + encryption (session/local/encrypted, Web Crypto)          | No cloud keys in the local flow; nothing secret to store   | `adapters/web/api-storage/*`, `docs/security/`                                                                  |
| Cloud provider selection & per-provider key validation                      | Superseded by local backend config (model file paths)      | `web/components/ApiKeyManager.tsx`, `api-storage/internal/providers/*`                                          |
| In-browser Markdown **preview / rendering**                                 | CLI writes files instead of rendering                      | `web/components/markdown-preview/MarkdownRenderer.tsx`, `MarkdownPreview.tsx`                                   |
| Copy/Download **menus**, filename field UI, document/processing info panels | CLI output layout (§14) replaces interactive copy/download | `markdown-preview/components/*` (`CopyMenu`, `DownloadMenu`, `DocumentInfo`, `ProcessingInfo`, `FilenameField`) |
| MUI theme / React app shell                                                 | No GUI                                                     | `web/theme/theme.tsx`, `web/App.tsx`, `App.tsx`                                                                 |

> **Note on output framing:** unlike the in-browser copy/download variants
> (`full`, `main`, `appendix`, `backmatter`, `allparts`), `inscriber` writes the
> equivalent set as **files** (§14). The content-shaping logic behind those
> variants — section assembly, optional BibTeX, per-section titles — is the part
> worth porting (`content-utils.ts`); the menu/UI around it is not.

---

## 25. End-to-end worked example (one page, one figure)

A concrete trace threading §7→§12 for `paper.pdf`, page 3, which contains one
figure. (The committed M1a fixtures capture a real page of this shape —
`tests/fixtures/deepseek_paper_p1_raw.txt`.)

**1. Rasterize (§7).** Page 3 (A4, 595×842 pt) at `large` (1280px long edge):
`zoom = 1280/842 ≈ 1.52`, producing `PageImage(page_number=3, png, W=905, H=1280)`.

**2. OCR call (§8.3).** `DeepSeekOcrBackend.ocr_page` sends the page PNG (image
content-part first, §2.1) with prompt
`<|grounding|>Convert the document to markdown.`, `temperature: 0`,
`max_tokens` capped. Raw output (illustrative, in the M1a-confirmed block
format):

```
sub_title[[230, 95, 540, 120]]
## 3. Method

text[[160, 150, 840, 172]]
We train the model as shown below.

image[[300, 240, 760, 612]]
image_caption[[300, 625, 700, 645]]
<center>Figure 1: Training pipeline overview.</center>
```

**3. Parse + map coords (§8.3).** Four blocks; one figure-class block (`image`),
coords `[300, 240, 760, 612]` on the 0–999 **padded-square** grid. With
`W=905, H=1280`: `L = 1280`, `pad_x = (1280−905)/2 = 187.5`, `pad_y = 0`, so
`px = grid/999 · 1280 − 187.5` → pixels `x=[197, 786]`, `y=[307, 784]` →
`bbox_norm ≈ (0.218, 0.240, 0.869, 0.613)`. The `image` block is **replaced** by
a placeholder (not deleted); the following `image_caption` block supplies
`Region.text = "<center>Figure 1: Training pipeline overview.</center>"` while
its text also stays in the markdown. Resulting `OcrPageResult.markdown`:

```
## 3. Method

We train the model as shown below.

⟦INSCRIBER_FIG:fig_p3_1⟧

<center>Figure 1: Training pipeline overview.</center>
```

**4. Crop (§8.4).** `bbox_norm` × `(905,1280)` + 2% margin → crop saved as
`figures/fig_p3_1.png`.

**5. VLM call (§9).** Context = page 3's whole text (≤2000 chars) with the
preamble `This image appears on page 3. …`; prompt assembled per §9.3; Gemma 4
returns `<img_desc>A flow diagram showing … </img_desc>`; §9.4 extracts the inner
text.

**6. Inject (§10.2), default `describe-only`.** The placeholder is replaced by:

```
> **Image description.** A flow diagram showing the three-stage training
> pipeline: data ingestion, pretraining, and fine-tuning, connected left to
> right by arrows.
```

(With `describe-and-keep`, an `![<center>Figure 1: …</center>](figures/fig_p3_1.png)`
line precedes it, alt text = `Region.text`.) §10.3 `ensureImageDescriptionSpacing`
guarantees blank lines around the block, and the `<center>…</center>` caption line
is a protected artifact line for the header/footer stripper (§10.3b).

**7. Assemble / split / write (§10–§14).** Pages concatenated → cleanup → split →
`paper.md` (full), `paper.main.md` / `paper.appendix.md` / `paper.backmatter.md`
as detected, `figures/fig_p3_1.png`, and `paper.bib` if `--bibtex`.

---

_End of design document._
