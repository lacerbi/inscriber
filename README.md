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
  VLM restructures each one into a Markdown pipe table, reading the true layout
  from a cropped image of the table (or the full page image when no table box
  was detected) while preserving the OCR values. On any failure the raw OCR
  table is kept. Disable with `--no-table-refine`.
- **Split files** — the document divided into `main`, `appendix`, and
  `backmatter` parts (disable with `--no-split`).
- A **BibTeX entry** for the paper, when the document is judged citable
  (default `auto` mode; for arXiv inputs it prefers the _published_ version of
  the preprint when one exists). Online lookups send only the extracted title
  or arXiv ID — never the document — and under `--offline` it degrades to a
  clearly-marked, fully-local best-effort entry. `--bibtex-mode off` disables
  it; `--bibtex` forces the classic always-look-up mode.

> ⚠️ **Accuracy.** The output is a best-effort machine transcription, not a
> faithful copy, and **will** contain errors. Body text and table values are
> generally reliable; errors take the form of missing/repeated sentences near
> non-standard page formatting, typos, missing pieces and structural issues in
> complex equations or tables, or image descriptions missing or misrepresenting
> parts of the figure. An LLM consuming the Markdown tolerates this noise well;
> for critical use, you **must** verify against the PDF — the [`/inscribe` skill](#convert--verify-with-agent-skills)
> automates that step. Every generated file ends with a short notice saying it
> was machine-transcribed.

Results are **cached** (content-addressed, per page / figure / table), so
re-running with different output options takes seconds. Cache keys cover the
model files, prompts, and the llama.cpp build, so swapping or upgrading any of
them recomputes instead of serving stale results. A **two-step mode** runs
OCR once and lets you compare different VLMs on the identical OCR text and
figure crops (see [Usage](#usage)).

## Requirements

- Python 3.10+ on Windows, Linux, or macOS
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (the `llama-server` binary)
- Two multimodal GGUF model pairs, ~9–12 GB total depending on quant
  (download links below)
- A GPU helps a lot but is not required. Reference setup: a laptop RTX 4060
  with 8 GB VRAM.

> **Speed.** On the reference setup (laptop RTX 4060, 8 GB VRAM), OCR takes
> ~20–25 seconds per page (a 39-page paper ≈ 15 minutes), and the VLM passes
> then take ~20–40 seconds per table and per figure — so a long paper can run
> 30–40 minutes end to end. Runtime has not been thoroughly optimized; the
> Q8_0 DeepSeek-OCR quant might speed OCR up at the risk of some quality
> loss, and `--ocr-resolution large` is a faster OCR setting for simple
> documents (see [Options](#options)). Cached re-runs take seconds.

## Install

```bash
pip install inscriber
```

Or install the latest development version from source:

```bash
pip install git+https://github.com/lacerbi/inscriber.git
```

## Setup

Quick path: install llama.cpp (step 1 below), then let `inscriber setup` do
steps 2–3 for you —

```bash
inscriber setup --llama-bin-dir /path/to/llama.cpp/bin
```

downloads the recommended models below (~12 GB; `--deepseek-quant q8_0` picks
the smaller DeepSeek pair, ~9 GB) into the platform data dir, verifies each
file against pinned checksums, and writes a ready-to-run config to the
platform config dir. Interrupted downloads resume on re-run; already-complete
files are verified and skipped. Prefer manual control? Steps 2–3 below do the
same by hand.

### 1. llama.cpp

Download a prebuilt release from
[github.com/ggml-org/llama.cpp/releases/latest](https://github.com/ggml-org/llama.cpp/releases/latest)
— pick the variant matching your GPU backend (CUDA / Vulkan / Metal / CPU) — or
build from source. `inscriber` only needs the directory containing
`llama-server` (`llama-server.exe` on Windows).

> ⚠️ **Use build 9587 or newer** — older builds handle DeepSeek-OCR images
> differently and would misplace figure crops, so `inscriber` refuses to run
> OCR against them.

### 2. Models

A multimodal model in llama.cpp is **two GGUF files**: the text/decoder model
and a multimodal projector (`mmproj`). `inscriber` uses two such pairs —
DeepSeek-OCR for OCR + figure grounding, Gemma 4 for figure descriptions and
table restructuring:

| model                                         | role                        | download                                                                                                                                                                                                                                                       |
| --------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DeepSeek-OCR** BF16 _(recommended)_         | OCR + figure grounding      | [model (5.9 GB)](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF/resolve/main/deepseek-ocr-bf16.gguf?download=true) · [mmproj (0.8 GB)](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF/resolve/main/mmproj-deepseek-ocr-bf16.gguf?download=true)      |
| DeepSeek-OCR Q8_0 _(smaller, also verified)_  | OCR + figure grounding      | [model (3.1 GB)](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/DeepSeek-OCR-Q8_0.gguf?download=true) · [mmproj (0.4 GB)](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF/resolve/main/mmproj-DeepSeek-OCR-Q8_0.gguf?download=true)          |
| **Gemma 4 E4B** QAT Q4_K_XL                   | figure description + tables | [model (4.2 GB)](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf?download=true) · [mmproj (1.0 GB)](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF/resolve/main/mmproj-BF16.gguf?download=true) |

(Sizes are decimal GB, matching what `inscriber setup` prints while downloading.)

> ⚠️ Keep DeepSeek-OCR at BF16 or Q8_0 — **Q4_K_M causes runaway repetition
> loops**. The Gemma side has no such restriction: any reasonable quant works,
> and larger Gemma 4 variants are fine if you have the hardware.

Sources: [sabafallah/DeepSeek-OCR-GGUF](https://huggingface.co/sabafallah/DeepSeek-OCR-GGUF),
[ggml-org/DeepSeek-OCR-GGUF](https://huggingface.co/ggml-org/DeepSeek-OCR-GGUF),
[unsloth/gemma-4-E4B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-qat-GGUF).
unsloth's projector file is literally named `mmproj-BF16.gguf` — consider
renaming it to something Gemma-specific if you keep models from several
families in one folder (`inscriber setup` does this automatically, saving it
as `mmproj-gemma-4-E4B-it-qat-BF16.gguf`).

### 3. Configuration

Copy [`config.example.toml`](https://github.com/lacerbi/inscriber/blob/main/config.example.toml) to `config.toml` in the
directory you run from (or the platform config dir, e.g.
`%APPDATA%\inscriber\config.toml` on Windows, `~/.config/inscriber/config.toml`
on Linux) and fill in the llama.cpp `bin_dir` and the four model paths
(`inscriber setup` writes these for you). Every config field is also
overridable from the CLI (precedence: CLI flag > config file > built-in
default).

## Usage

```bash
inscriber paper.pdf -o out/                          # end-to-end
inscriber https://arxiv.org/abs/2510.18234 -o out/   # paper-repository URL
```

URL input supports arXiv, OpenReview, ACL Anthology, bioRxiv, medRxiv,
NeurIPS, and PMLR; for anything else, download the PDF and pass the local path.

Outputs in `out/` (here for a paper whose BibTeX citation key is
`chang2025amortized`):

```
chang2025amortized_full.md        # full document
chang2025amortized_main.md        # split parts (as detected)
chang2025amortized_appendix.md
chang2025amortized_backmatter.md
chang2025amortized.bib            # when judged citable (--bibtex-mode off disables)
figures/                          # with --figure-mode describe-and-keep
```

**Naming:** when a BibTeX entry is produced (the default `auto` mode does this
for any citable paper), its citation key — `authorYEARfirstword`, e.g.
`chang2025amortized` — becomes the base name, giving you library-style
filenames for free. Otherwise the name derives from the source (PDF stem or
repository filename). `--name NAME` pins an explicit base name;
`--no-bibtex-name` disables key-derived naming; `--no-full-suffix` writes the
full document as `chang2025amortized.md` instead of `…_full.md` (handy with
`--no-split` or a one-file-per-paper library). Every run logs which name was
chosen.

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

### Fix the splits, regenerate the full document

All outputs are plain Markdown. To correct an OCR/VLM slip after the fact,
edit the **split** files and rebuild the full document from them — no models
needed:

```bash
inscriber join out/paper      # paper_main/_appendix/_backmatter.md → paper_full.md
```

`join` strips each split's footer notice (and the BibTeX block, if injected),
concatenates main → appendix → backmatter, and re-applies the framing — so each
fix is made once, not once per file. Note the regenerated `{base}_full.md` uses
the combined ordering (appendix before backmatter, under `# Title - Appendix`-
style headings), which may differ from the original document order.

### Convert + verify with agent skills

The repo ships assistant skills for the convert-and-verify workflow:

- [Claude Code](https://claude.com/claude-code): [`/inscribe`](https://github.com/lacerbi/inscriber/blob/main/.claude/skills/inscribe/SKILL.md)
- Codex: [`$inscribe`](https://github.com/lacerbi/inscriber/blob/main/.agents/skills/inscribe/SKILL.md)

When run inside this repository, the skill takes a PDF path or URL (plus any
options in plain words), runs `inscriber`, then checks the transcription against
the source PDF in ≤10-page chunks with parallel subagents briefed on the known
failure modes (table cells, subscripts, equations, truncated pages, figure
descriptions), applies the fixes that matter to the split files, and rejoins
them with `inscriber join`. Say "no verification" to stop after the conversion.

## Options

`inscriber --help` shows the full surface; every `config.example.toml` field
has a matching flag. Highlights:

| flag                                                          | meaning                                                          |
| ------------------------------------------------------------- | ---------------------------------------------------------------- |
| `--ocr-resolution {tiny,small,base,large,gundam}`             | OCR render quality (default `gundam`, 2048px; `large` is faster) |
| `--figure-mode {describe-only,describe-and-keep,placeholder}` | how figures render                                               |
| `--no-figures`                                                | skip figure detection and description entirely                   |
| `--no-table-refine`                                           | keep raw OCR tables (skip VLM restructuring)                     |
| `--name NAME` / `--no-bibtex-name`                            | explicit output base name / never name by BibTeX citation key    |
| `--no-full-suffix`                                            | full document as `{base}.md` instead of `{base}_full.md`         |
| `--no-split` / `--page-numbers` / `--page-separators`         | output options                                                   |
| `--pages RANGE`                                               | page selection, e.g. `"1-10"`, `"3"`, `"5-"`                     |
| `--bibtex-mode {off,on,auto}` / `--bibtex-in-doc`             | BibTeX mode (default `auto`; `--bibtex` ⇒ `on`)                  |
| `--offline`                                                   | no network: URL input + online BibTeX sources disabled           |
| `--mode {sequential,concurrent}`                              | one model resident at a time (default) vs. both (needs VRAM)     |
| `--no-cache` / `--refresh`                                    | cache control                                                    |

GPU offload is automatic by default (`n_gpu_layers = "auto"` lets llama.cpp fit
as many layers into VRAM as it can); override per server with `--ocr-ngl` /
`--vlm-ngl` (`all`, `0` for CPU, or a layer count). `--ocr-resolution` is the
main speed/quality lever: the `gundam` default renders pages at 2048px, which
measurably reduces OCR misreads of small subscripts and digits; `--ocr-resolution
large` is ~20% faster and fine for simple documents. `--ctx` (default 16384)
sizes the context window that prompt and generation share — complex tables need
headroom for the VLM's reasoning, so don't shrink it without reason.

## Privacy / offline

**Your documents and figures never leave your machine** — they go only to your
own llama.cpp server on `127.0.0.1`, never to any cloud model. The only
features that touch the network are (1) downloading a PDF when the input is a
URL and (2) the BibTeX citation lookups (on by default), which send **only the
extracted title or arXiv ID** to citation APIs — never the document. Both are
hard-disabled by `--offline` (BibTeX then degrades to a clearly-marked,
fully-local best-effort entry). No telemetry, no persisted secrets.

## Development

```bash
git clone https://github.com/lacerbi/inscriber && cd inscriber
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest                          # mocked inference — no GPU or models needed
ruff check
```

Contributor guidance lives in
[`AGENTS.md`](https://github.com/lacerbi/inscriber/blob/main/AGENTS.md); the
full technical specification is
[`DESIGN.md`](https://github.com/lacerbi/inscriber/blob/main/DESIGN.md).

## License

MIT.
