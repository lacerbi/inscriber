# inscriber

**Convert academic PDFs into clean, LLM-friendly text-only Markdown — entirely on
your own machine.**

`inscriber` is a cross-platform command-line tool that runs OCR + figure
description locally using [llama.cpp](https://github.com/ggml-org/llama.cpp)
(DeepSeek-OCR for text + figure grounding, Gemma 4 for figure description). It is
the local, offline-first reimagining of the cloud web app
[`paper2llm`](https://github.com/lacerbi/paper2llm).

> **Status:** under active development. See `DESIGN.md` for the authoritative
> specification and `PLAN-inscriber-v1.md` for the build roadmap.

## Privacy / offline

The OCR + figure-description **core pipeline is fully local** — your documents and
figures go only to your own llama.cpp server on `127.0.0.1`, never to any cloud
API. The **only** features that touch the network are:

1. Downloading a PDF when the input is a URL.
2. The opt-in Semantic Scholar BibTeX lookup (`--bibtex`).

Both are hard-disabled by `--offline`. There is no telemetry and no persisted
secrets.

## Install

```bash
pip install inscriber      # or: pipx install inscriber
```

You supply, at runtime: the llama.cpp `bin` directory (containing
`llama-server[.exe]`) and two `(model, mmproj)` GGUF pairs — DeepSeek-OCR for OCR
and Gemma 4 for figure description. `inscriber` bundles no model weights.

## Models & llama.cpp setup

A multimodal model in llama.cpp is **two GGUF files**: the text/decoder model
(`-m`) and the multimodal projector (`--mmproj`). You need both for each model:

| role | text model | projector (`mmproj`) | source |
| ---- | ---------- | -------------------- | ------ |
| OCR  | `deepseek-ocr-bf16.gguf` (BF16 recommended; **avoid Q4_K_M** — it loops) | `mmproj-deepseek-ocr-bf16.gguf` | `ggml-org/DeepSeek-OCR-GGUF` |
| VLM  | `gemma-4-E4B-it-*.gguf` | `mmproj-*.gguf` | unsloth `gemma-4-E4B-it-GGUF` |

Point `inscriber` at the llama.cpp `bin` dir and the four GGUF paths (via config or
flags). **GPU offload is automatic by default** — with `n_gpu_layers = "auto"`,
inscriber omits `-ngl` so llama.cpp uses its own default (modern builds auto-fit as
many layers into VRAM as they can). Override per-server with `--ocr-ngl` / `--vlm-ngl`
(`all`, `0` for CPU, or an explicit layer count for partial offload).

## Quickstart

```bash
# End-to-end (default):
inscriber paper.pdf -o out/ \
  --llama-bin-dir /path/to/llama.cpp/bin \
  --ocr-model deepseek-ocr-bf16.gguf --ocr-mmproj mmproj-deepseek-ocr-bf16.gguf \
  --vlm-model gemma-4-E4B-it-Q4_K_M.gguf --vlm-mmproj mmproj-BF16.gguf

# Two-step — run OCR once, then compare VLMs on the SAME OCR text + figure crops:
inscriber ocr paper.pdf -o out/ ...ocr flags...
inscriber describe out/paper.inscriber-ocr --vlm-model gemma-4-e4b.gguf ...
inscriber describe out/paper.inscriber-ocr --vlm-model some-other-vlm.gguf ...
```

The `ocr` step writes a portable, inspectable **bundle**
(`out/paper.inscriber-ocr/` — `manifest.json` + `figures/`). Its per-page markdown
is hand-editable before `describe`. Caching makes re-runs cheap (changing
split/figure/bibtex options re-runs in seconds).

## Configuration

Config lives in a TOML file. By default, `inscriber` checks `./config.toml` first,
then the platform config dir; `--config PATH` overrides both. **Every field is
overridable from the CLI** (precedence: CLI flag > config file > default).
Highlights (see `DESIGN.md` §13 for the full surface):

| flag | meaning |
| ---- | ------- |
| `--ocr-resolution {tiny,small,base,large,gundam}` | OCR render quality (default `large`) |
| `--figure-detect {auto,grounding,none,pdf-embedded}` | figure detection (`--no-figures` = `none`) |
| `--figure-mode {describe-only,describe-and-keep,placeholder}` | how figures render |
| `--no-split` / `--page-numbers` / `--page-separators` | output/assembly options |
| `--bibtex` / `--bibtex-in-doc` | online Semantic Scholar BibTeX (opt-in) |
| `--offline` | hard-disable all network use (URL input + BibTeX) |
| `--mode {sequential,concurrent}` | one model resident (default) vs both (VRAM caveat) |
| `--no-cache` / `--refresh` | cache control |

Status: see `PLAN-inscriber-v1.md` for the build roadmap and `docs/M1A-FINDINGS.md`
for the real-hardware findings the OCR backend is pinned to.

## Development

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
ruff check
```

## License

MIT.
