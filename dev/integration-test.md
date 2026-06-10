# Integration test (real llama.cpp + real GGUFs)

This checklist validates `inscriber` against **real** models on a machine with
llama.cpp and GPU/large RAM. It is **not** run in CI (CI mocks the inference layer
on GPU-less runners — see `.github/workflows/ci.yml`). Run it before a release.

## Prerequisites

- A llama.cpp build (`llama-server` + `llama-mtmd-cli`), **build 9587 or
  newer** — older builds are refused for OCR (`min_server_build`; the
  grounding frame changed upstream, DESIGN §2.2). v1 was validated on
  **build 9587 (`d2e22ed97`)**, CUDA, on an RTX 4060 Laptop (8 GB VRAM)
  (originally on 9028 — `dev/notes/2026-06-10-build-9587-verification.md` records the
  re-validation).
- Two `(model, mmproj)` GGUF pairs:
  - **OCR:** DeepSeek-OCR — `deepseek-ocr-bf16.gguf` + `mmproj-deepseek-ocr-bf16.gguf`
    (BF16 recommended over Q4_K_M; DESIGN §2.2). A Q8_0 pair also works.
  - **VLM:** Gemma 4 E4B — `gemma-4-E4B-it-Q4_K_M.gguf` + `mmproj-BF16.gguf`.
- A known sample PDF (e.g. `tests/fixtures/sample_paper.pdf`, or a real arXiv PDF).

See `dev/notes/2026-06-09-m1a-findings.md` + `dev/notes/2026-06-10-build-9587-verification.md` for the
pinned, empirically-confirmed facts (server HTTP path works; image-before-text
required; per-axis coordinate frame on ≥9587; the `LABEL[[bbox]]` grounding
format).

## Steps

Set common flags from **your** `config.toml` values (machine-local, gitignored;
the same file the CLI and the `dev/scripts/` harnesses read):

```
BIN=<your [llama] bin_dir>           # must hold llama.cpp >= 9587
MODELS=/path/to/your/gguf/folder
OCR_M=$MODELS/deepseek-ocr-bf16.gguf
OCR_P=$MODELS/mmproj-deepseek-ocr-bf16.gguf
VLM_M=$MODELS/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf
VLM_P=$MODELS/gemma-4-E4B-it-mmproj-BF16.gguf
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
   produces a clean text-only document with no VLM server launched. Note:
   BibTeX defaults to `auto` — with no VLM configured the probe is skipped
   with a warning and no `.bib` appears; with `--vlm-model`/`--vlm-mmproj` set,
   the probe runs locally and a citable document yields a clearly-marked
   best-effort `out/….bib` (fully offline — verify no network traffic).

7. **BibTeX** — default `auto`:
   - an arXiv-URL run (step 8) writes `out/….bib` **without a probe call**
     (look for `BibTeX (auto): wrote entry via s2-arxiv-id` — the published
     version when one exists, else the arXiv `@misc`);
   - a local citable PDF goes through the probe (one extra small VLM call,
     cached on re-runs) then the title-search/best-effort chain;
   - a non-citable PDF (slides, an invoice) logs `document judged not
     citable; skipping` and writes no `.bib` —
     `dev/scripts/bibtex_probe_check.py` covers the probe verdicts standalone
     (record any prompt change in `dev/notes/2026-06-10-bibtex-probe-findings.md`);
   - explicit `--bibtex` (= `--bibtex-mode on`) keeps the original
     always-look-up path; `--bibtex-in-doc` also prepends a fenced entry to
     `paper.md` / `paper.main.md`.

8. **URL input** — `inscriber run https://arxiv.org/abs/<id> -o out …` downloads and
   processes (requires network; blocked by `--offline`).

## Known limitations (v1)

- DeepSeek-OCR labels the title as `sub_title`/`title` (often `##`), so a doc with
  no `# ` H1 yields title `Untitled_Paper` in split headers (faithful to the ported
  `extractTitle`).
- `llama-mtmd-cli` fallback crashed when last tested (build 9028; untested since —
  the server HTTP path is the shipped path).
- Cross-page tables/equations are not fully handled (DESIGN §10.3). (The Gundam
  coordinate-frame question is resolved: same per-axis frame at every render size,
  DESIGN §2.2.)
