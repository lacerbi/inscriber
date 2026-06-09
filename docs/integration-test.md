# Integration test (real llama.cpp + real GGUFs)

This checklist validates `inscriber` against **real** models on a machine with
llama.cpp and GPU/large RAM. It is **not** run in CI (CI mocks the inference layer
on GPU-less runners — see `.github/workflows/ci.yml`). Run it before a release.

## Prerequisites

- A llama.cpp build (`llama-server` + `llama-mtmd-cli`). v1 was validated on
  **build 9028 (`d6e7b033a`)**, CUDA, on an RTX 4060 Laptop (8 GB VRAM).
- Two `(model, mmproj)` GGUF pairs:
  - **OCR:** DeepSeek-OCR — `deepseek-ocr-bf16.gguf` + `mmproj-deepseek-ocr-bf16.gguf`
    (BF16 recommended over Q4_K_M; DESIGN §2.2). A Q8_0 pair also works.
  - **VLM:** Gemma 4 E4B — `gemma-4-E4B-it-Q4_K_M.gguf` + `mmproj-BF16.gguf`.
- A known sample PDF (e.g. `tests/fixtures/sample_paper.pdf`, or a real arXiv PDF).

See `docs/M1A-FINDINGS.md` for the pinned, empirically-confirmed facts (server HTTP
path works; image-before-text required; padded-square coordinate frame; the
`LABEL[[bbox]]` grounding format).

## Steps

Set common flags (adjust paths):

```
BIN=C:/Users/luigi/llms
OCR_M=$BIN/models/deepseek-ocr-bf16.gguf
OCR_P=$BIN/models/mmproj-deepseek-ocr-bf16.gguf
VLM_M=$BIN/models/gemma-4-E4B-it-Q4_K_M.gguf
VLM_P=$BIN/models/mmproj-BF16.gguf
```

1. **Version + help**

   ```
   inscriber --version
   inscriber --help
   ```

2. **End-to-end run**

   ```
   inscriber run sample_paper.pdf -o out \
     --llama-bin-dir $BIN \
     --ocr-model $OCR_M --ocr-mmproj $OCR_P --ocr-ngl 99 \
     --vlm-model $VLM_M --vlm-mmproj $VLM_P --vlm-ngl 99
   ```

   Expect `out/sample_paper.md` (+ `.main.md` and any `.appendix.md` /
   `.backmatter.md`) with figures replaced by `> **Image description.** …`
   blockquotes.

3. **Two-step (compare VLMs on the same OCR + crops)**

   ```
   inscriber ocr sample_paper.pdf -o out --llama-bin-dir $BIN \
     --ocr-model $OCR_M --ocr-mmproj $OCR_P --ocr-ngl 99
   inscriber describe out/sample_paper.inscriber-ocr -o out --llama-bin-dir $BIN \
     --vlm-model $VLM_M --vlm-mmproj $VLM_P --vlm-ngl 99
   ```

   The bundle (`out/sample_paper.inscriber-ocr/`) is inspectable (`manifest.json` +
   `figures/`). Re-running `describe` with a different `--vlm-model` re-describes
   only (OCR + crops reused via the cache).

4. **Caching** — re-run step 2; the OCR pass should report cache hits and **not**
   relaunch the OCR server.

5. **Figure modes** — try `--figure-mode describe-and-keep` (adds `figures/` +
   image refs) and `--figure-mode placeholder` (`> **Image.** [not displayed]`,
   no VLM).

6. **No figures / offline** — `inscriber run sample_paper.pdf --no-figures --offline`
   produces a clean text-only document with no VLM server launched.

7. **BibTeX (online)** — `--bibtex` writes `out/sample_paper.bib`; `--bibtex-in-doc`
   also prepends a fenced entry to `paper.md` / `paper.main.md`.

8. **URL input** — `inscriber run https://arxiv.org/abs/<id> -o out …` downloads and
   processes (requires network; blocked by `--offline`).

## Known limitations (v1)

- DeepSeek-OCR labels the title as `sub_title`/`title` (often `##`), so a doc with
  no `# ` H1 yields title `Untitled_Paper` in split headers (faithful to the ported
  `extractTitle`).
- `llama-mtmd-cli` fallback currently crashes on build 9028 (server HTTP path is the
  shipped path).
- Cross-page tables/equations and the Gundam coordinate frame are not fully handled
  (DESIGN §10.3, §2.2).
