# inscriber

**Convert academic PDFs into clean, LLM-friendly text-only Markdown — entirely on
your own machine.**

`inscriber` is a cross-platform command-line tool that runs OCR and figure
description locally with [llama.cpp](https://github.com/ggml-org/llama.cpp):
DeepSeek-OCR reads the text and locates the figures, and a Gemma 4 vision model
turns each figure into a textual description and restructures tables. It is the
local, offline-first reimagining of the cloud web app
[`paper2llm`](https://github.com/lacerbi/paper2llm).

## What it does

Given a PDF (local file, or a URL from a supported paper repository),
`inscriber` writes:

- **A full Markdown file** — the paper's text, equations, and tables, with each
  figure replaced by a generated textual description (or kept alongside it with
  `--figure-mode describe-and-keep`).
- **Clean tables.** DeepSeek-OCR emits tables as degenerate HTML; by default the
  VLM restructures each one into a Markdown pipe table, using the page image for
  the true layout while preserving the OCR values. On any failure the raw OCR
  table is kept. Disable with `--no-table-refine`.
- **Split files** — the document divided into `main`, `appendix`, and
  `backmatter` parts (disable with `--no-split`).
- Optionally, a **BibTeX entry** for the paper (`--bibtex`; uses Semantic
  Scholar, the one feature besides URL input that touches the network).

Results are **cached** (content-addressed, per page / figure / table), so
re-running with different output options takes seconds. Cache keys cover the
model files, prompts, and the llama.cpp build, so swapping or upgrading any of
them recomputes instead of serving stale results. A **two-step mode** runs
OCR once and lets you compare different VLMs on the identical OCR text and
figure crops (see [Usage](#usage)).

## Requirements

- Python 3.10+ on Windows, Linux, or macOS
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (the `llama-server` binary)
- Two multimodal GGUF model pairs, ~8–11 GB total depending on quant
  (download links below)
- A GPU helps a lot but is not required. Reference setup: a laptop RTX 4060
  with 8 GB VRAM.

## Install

Not yet on PyPI — install from source:

```bash
pip install git+https://github.com/lacerbi/inscriber.git
```

## Setup

### 1. llama.cpp

Download a prebuilt release from
[github.com/ggml-org/llama.cpp/releases/latest](https://github.com/ggml-org/llama.cpp/releases/latest)
— pick the variant matching your GPU backend (CUDA / Vulkan / Metal / CPU) — or
build from source. `inscriber` only needs the directory containing
`llama-server` (`llama-server.exe` on Windows).

> ⚠️ **Use build 9587 or newer.** llama.cpp's model-side preprocessing
> changes between builds — DeepSeek-OCR's grounding coordinate frame changed
> at some point after build 9028 — so `inscriber` is pinned to the frame of
> build 9587 (the verified one) and **refuses to run OCR against older
> builds** rather than silently misplace figure crops.

### 2. Models

A multimodal model in llama.cpp is **two GGUF files**: the text/decoder model
and a multimodal projector (`mmproj`). `inscriber` uses two such pairs —
DeepSeek-OCR for OCR + figure grounding, Gemma 4 for figure descriptions and
table restructuring:

| model | role | download |
| ----- | ---- | -------- |
| **DeepSeek-OCR** BF16 *(recommended)* | OCR + figure grounding | [model (5.5 GB)](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF/resolve/main/deepseek-ocr-bf16.gguf?download=true) · [mmproj (0.8 GB)](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF/resolve/main/mmproj-deepseek-ocr-bf16.gguf?download=true) |
| DeepSeek-OCR Q8_0 *(smaller, also verified)* | OCR + figure grounding | [model (2.9 GB)](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/DeepSeek-OCR-Q8_0.gguf?download=true) · [mmproj (0.4 GB)](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/mmproj-DeepSeek-OCR-Q8_0.gguf?download=true) |
| **Gemma 4 E4B** QAT Q4_K_XL | figure description + tables | [model (3.9 GB)](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf?download=true) · [mmproj (0.9 GB)](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/mmproj-BF16.gguf?download=true) |

> ⚠️ Keep DeepSeek-OCR at BF16 or Q8_0 — **Q4_K_M causes runaway repetition
> loops**. The Gemma side has no such restriction: any reasonable quant works,
> and larger Gemma 4 variants are fine if you have the hardware.

Sources: [sabafallah/DeepSeek-OCR-GGUF](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF),
[ggml-org/DeepSeek-OCR-GGUF](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF),
[unsloth/gemma-4-E4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF).
unsloth's projector file is literally named `mmproj-BF16.gguf` — consider
renaming it to something Gemma-specific if you keep models from several
families in one folder.

### 3. Configuration

Copy [`config.example.toml`](config.example.toml) to `config.toml` in the
directory you run from (or the platform config dir, e.g.
`%APPDATA%\inscriber\config.toml` on Windows, `~/.config/inscriber/config.toml`
on Linux) and fill in the llama.cpp `bin_dir` and the four model paths. Every
config field is also overridable from the CLI (precedence: CLI flag > config
file > built-in default).

## Usage

```bash
inscriber paper.pdf -o out/                          # end-to-end
inscriber https://arxiv.org/abs/2510.18234 -o out/   # paper-repository URL
```

URL input supports arXiv, OpenReview, ACL Anthology, bioRxiv, medRxiv,
NeurIPS, and PMLR; for anything else, download the PDF and pass the local path.

Outputs in `out/`:

```
paper.md                # full document
paper.main.md           # split parts (as detected)
paper.appendix.md
paper.backmatter.md
paper.bib               # with --bibtex
figures/                # with --figure-mode describe-and-keep
```

### Two-step: compare VLMs on identical OCR

```bash
inscriber ocr paper.pdf -o out/                # OCR once → portable bundle
inscriber describe out/paper.inscriber-ocr     # VLM passes + assembly + write
inscriber describe out/paper.inscriber-ocr --vlm-model other.gguf --vlm-mmproj other-mmproj.gguf
```

`ocr` writes an inspectable bundle (`manifest.json`, cropped `figures/`, and
`pages/` rasters for pages with tables); `describe` runs the VLM stages from it
with no OCR model loaded, so each VLM sees the identical input. The bundle's
per-page markdown is hand-editable — fix an OCR glitch once, then try N VLMs.

## Options

`inscriber --help` shows the full surface; every `config.example.toml` field
has a matching flag. Highlights:

| flag                                                          | meaning                                                       |
| ------------------------------------------------------------- | ------------------------------------------------------------- |
| `--ocr-resolution {tiny,small,base,large,gundam}`             | OCR render quality (default `large`; `gundam` for dense pages) |
| `--figure-mode {describe-only,describe-and-keep,placeholder}` | how figures render                                            |
| `--no-figures`                                                | skip figure detection and description entirely                |
| `--no-table-refine`                                           | keep raw OCR tables (skip VLM restructuring)                  |
| `--no-split` / `--page-numbers` / `--page-separators`         | output options                                                |
| `--pages RANGE`                                               | page selection, e.g. `"1-10"`, `"3"`, `"5-"`                  |
| `--bibtex` / `--bibtex-in-doc`                                | online Semantic Scholar BibTeX (opt-in)                       |
| `--offline`                                                   | hard-disable all network use (URL input + BibTeX)             |
| `--mode {sequential,concurrent}`                              | one model resident at a time (default) vs. both (needs VRAM)  |
| `--no-cache` / `--refresh`                                    | cache control                                                 |

GPU offload is automatic by default (`n_gpu_layers = "auto"` lets llama.cpp fit
as many layers into VRAM as it can); override per server with `--ocr-ngl` /
`--vlm-ngl` (`all`, `0` for CPU, or a layer count). `--ocr-resolution` is the
main speed/quality lever. `--ctx` (default 16384) sizes the context window that
prompt and generation share — complex tables need headroom for the VLM's
reasoning, so don't shrink it without reason.

## Privacy / offline

The OCR + figure-description core pipeline is **fully local** — your documents
and figures go only to your own llama.cpp server on `127.0.0.1`, never to any
cloud API. The only features that touch the network are (1) downloading a PDF
when the input is a URL and (2) the opt-in `--bibtex` lookup. Both are
hard-disabled by `--offline`. No telemetry, no persisted secrets.

## Development

```bash
git clone https://github.com/lacerbi/inscriber && cd inscriber
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                          # mocked inference — no GPU or models needed
ruff check
```

Contributor guidance lives in [`AGENTS.md`](AGENTS.md); the full technical
specification is [`DESIGN.md`](DESIGN.md).

## License

MIT.
