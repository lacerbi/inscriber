# Plan: Build `inscriber` v1 (full M0–M5 roadmap)

Created: 2026-06-09
Status: ✅ COMPLETE 2026-06-09 — all milestones M0–M5 implemented, 136 tests green,
ruff clean, packaged, and verified end-to-end on real hardware. Build backend
`hatchling` confirmed. (See Execution Progress tracker below.)
Revised 2026-06-09 (post external review): 8 fixes applied — local-path input pulled
to M1a (#1); canonical models home = `models.py` (#2); command/stage-aware validation
(#3); generated calibration fixture in M1a (#4); VLM tag-extraction divergence labeled
(#5); canonical BibTeX mock fixture (#6); `paper.md` = full stitched, no allparts file
(#7); `LlamaServerManager` tests moved to M1a (#8).

## Summary

Implement `inscriber` — a cross-platform, local-first CLI that converts academic
PDFs into LLM-friendly text-only Markdown using llama.cpp (DeepSeek-OCR for text +
figure grounding, Gemma 4 for figure description). The build follows the
authoritative `DESIGN.md` across seven milestones (M0–M5), porting the reusable
string/markdown/BibTeX/URL logic from the `paper2llm/` TypeScript reference
(reimplemented in Python, not shared). Hardware (llama.cpp + GGUFs) is available,
so the highest-risk milestone **M1a runs early** to pin two empirical unknowns
(image round-trip + grounding coordinate frame) before the OCR backend is locked.

`DESIGN.md` is the spec; this plan is the **execution roadmap** that sequences the
work, names the exact port sources (file:line), captures gotchas the explorers
verified in the real `paper2llm` source, and defines per-milestone tests and exit
gates.

## Execution Progress (live tracker — added during implementation)

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked / needs hardware.

- [x] **M0 — Skeleton**: pyproject, models, config, cli, logging, pipeline stub, test_config — DONE (17 tests green, ruff clean, `--version` works)
- [x] **M1a — De-risk spike**: resolver(local), llama/server, llama/client, ocr/base (Inferencer), pdf/rasterize, test_llama_server, calibration fixture — **DONE + RESOLVED ON REAL HARDWARE**. See `docs/M1A-FINDINGS.md`. Build 9028, RTX 4060 8GB.
  - **⚠️ 3 divergences from DESIGN found empirically (M1b locks to these):**
    1. **Round-trip OK via server HTTP path** (issue #21022 absent on build 9028). mtmd-cli currently **crashes** (STATUS_STACK_BUFFER_OVERRUN) — server path ships; mtmd kept as documented-broken fallback.
    2. **Image content-part MUST precede text** or grounding never activates (`chat_image(image_first=True)` — fixed).
    3. **Coordinate frame = PADDED-SQUARE** (side=long edge, short axis centered), NOT the DESIGN-default per-axis/original-image mapping. And the **grounding format is `LABEL[[x1,y1,x2,y2]]` blocks**, NOT `<|ref|>…<|det|>…`. Labels: title/sub_title/text/image/image_caption.
  - Fixtures captured: `deepseek_calibration_raw.txt`, `sample_paper.pdf` + `deepseek_paper_p1_raw.txt`. mmproj files now present (both DeepSeek pairs + Gemma pair).
- [x] **M1b — OCR vertical slice**: ocr/registry, ocr/deepseek (REAL `LABEL[[bbox]]` parser + padded-square mapping), cache (OcrCache), serialize, test_deepseek_parser, test_cache — **DONE + verified on real hardware** (`scripts/m1b_check.py`: sample_paper.pdf → clean markdown w/ placeholder; cache hit → no server). pipeline.run_ocr_pass built (shared by M2 run/ocr).
- [x] **M2 — Figures + two-step**: pdf/figures, pdf/crop, vlm/* (Gemma), postprocess/prompt+inject, bundle, VLM cache, output(M2 subset), errors.py, full run/ocr/describe orchestration, test_prompt/test_inject/test_crop/test_bundle_roundtrip — **DONE + verified end-to-end on real hardware** (`inscriber ocr` → bundle; `inscriber describe` → Gemma described the figure accurately; blockquote format exact). VLM cache keyed on model identity ⇒ swapping `--vlm-model` re-describes while OCR/crops reuse.
- [x] **M3 — Assembly & splitting**: postprocess/stitch (normalize/ensure-spacing ports + header-footer/dehyphen), postprocess/splitter (full port), output(splits), pipeline assembly wired, test_splitter, test_stitch — **DONE + verified on hardware**. ⚠️ Splitter A-pattern guard made **stricter than paper2llm** per DESIGN §11 intent (bare "A " heading requires an ack anchor) — caught the "A Calibration Study" title false-positive.
- [x] **M4 — Inputs & BibTeX**: resolver(URL)+domain_handlers (7 configs verbatim), bibtex/semantic_scholar (lookup/key/validation/mock + 429 degrade), pipeline wiring (.bib + fenced prepend), test_domain_handlers, test_bibtex — **DONE**.
- [x] **M5 — Hardening**: test_pipeline_mocked (full run mocked), test_pdf_embedded_figures, concurrent mode (pre-launch VLM overlapping OCR), CI matrix (`.github/workflows/ci.yml`, 3 OS × py3.10–3.12), packaging (wheel builds + installs fresh, console script resolves), README + docs/integration-test.md — **DONE**.

### ✅ v1 COMPLETE — all milestones M0–M5 done (2026-06-09)

**Post-completion `/doublecheck`** (3 parallel Opus reviewers: orchestration correctness,
port fidelity, cross-cutting/tests). Headline concerns came back clean (figure-id↔placeholder
alignment, 6/6 port pairs faithful, bundle gate, resilience). **8 fixes applied:**
(1) `--no-cache` no longer writes the `hashes.json` sidecar; (2) reject `concurrent` mode with a
fixed `--port` (would collide); (3) `max_tokens` added to the OCR cache key (§8.6); (4) `normalize_title`
whitespace order matched to the TS; (5) mtmd-cli/Gemma raise `InferenceError` (an `InscriberError`);
(6) `OcrBackend.prompt` ABC signature fixed; (7) backend registries de-duplicated (single source of truth);
(8) `describe-and-keep` + `detect=none` no longer copies orphan crops. **+12 tests added** for the flagged
gaps (concurrent mode, `--no-clobber`, describe-and-keep copy, page-numbers e2e, `--no-cache` no-write, output writer).

- **157 tests pass; ruff clean.** Wheel builds + installs; `inscriber --version` works.
  (Count grew across post-completion review rounds — see refinement/review notes below.)

**Post-v1 refinement (2026-06-09):** changed the default `ocr/vlm.n_gpu_layers` from `0`
(forced CPU) to **`"auto"`** — llama.cpp build 9028 documents `-ngl` accepting
`auto`/`all` with its own default being `auto`, so the old `0` was strictly worse (CPU
unless the user passed `--*-ngl`). `--ocr-ngl`/`--vlm-ngl` now accept `auto`|`all`|integer.
**Portability:** for `"auto"` the launcher **omits `-ngl` entirely** (rather than passing
the literal `-ngl auto`), so llama.cpp uses its own default — GPU auto-fit on modern
builds — and no symbolic token can break arg-parsing on any build. Verified on real
hardware (`inscriber run`/`ocr` with no `--ngl` flags → launch omits `-ngl`, server
healthy, GPU offload, figure described). DESIGN §13.1/§13.2 + README updated to match.

**Second external review (2026-06-09) — 6 real findings fixed (+5 regression tests):**
(P1) **split-output swap** — pipeline unpacked `prepare_formatted_sections()` as
`(main, appendix, backmatter)` but it returns `(main, backmatter, appendix)`, so
`.appendix.md`/`.backmatter.md` were swapped (shipped in `1cdd0a6`; only undetected
because the sample has neither section); (P1) **failed OCR pages were cached** as empty
→ now only successful pages are cached; (P2) endpoint cache identity now includes the
endpoint URL (no cross-endpoint collision); (P2) malformed-type TOML values now raise a
clean `ConfigError` instead of a `TypeError`; (P3) `copy_figures` honors `--no-clobber`;
(P3) URL `ensure-.pdf` applies to the path, not after a `?query`. (A separate `-ngl auto`
"finding" from the same reviewer was incorrect for build 9028 — verified — and was already
neutralized by the omit-on-auto change above.)
- **Verified end-to-end on real hardware** (build 9028, RTX 4060 8GB): `inscriber ocr` → bundle; `inscriber describe` (Gemma) → accurate figure descriptions; cache hits skip the server.
- **3 design divergences caught empirically in M1a and locked in** (see `docs/M1A-FINDINGS.md`): server HTTP path (not mtmd-cli), image-before-text ordering, **padded-square** coord frame + `LABEL[[bbox]]` grounding format (not `<|ref|>`/`<|det|>`).
- **1 splitter correctness fix** beyond paper2llm: bare "A " appendix heading requires an ack anchor (DESIGN §11 intent) — prevents title false-positives.
- Deferred (per DESIGN §22): GLM-OCR/PaddleOCR-VL/Dots.OCR backends, DeepSeek-OCR-2, Gundam coord-frame confirmation, cross-page table reconstruction, batch mode. mtmd-cli fallback present but currently crashes on build 9028 (server path ships).

> **Hardware-gated note (M1a/M1b):** The two empirical unknowns (DeepSeek-OCR image
> round-trip over `/v1/chat/completions`; grounding coordinate frame) require real
> llama.cpp + GGUFs and cannot be executed by the implementing agent. Code is written
> to the DESIGN defaults (reference per-axis mapping, server HTTP path) and golden
> tests use **synthetic illustrative** DeepSeek output; `docs/M1A-FINDINGS.md` records
> what the user must confirm on real hardware and where to lock the result.

## Scope

- **In scope**: Full v1 per `DESIGN.md` — M0 skeleton, M1a de-risk spike, M1b OCR
  vertical slice, M2 figures + two-step `ocr`/`describe` bundle, M3 assembly/split,
  M4 URL inputs + BibTeX, M5 cross-platform hardening + packaging. One OCR backend
  (DeepSeek-OCR) and one VLM backend (Gemma 4).
- **Out of scope (v1, per DESIGN §1.3 / §22)**: GUI/web; weight download/bundling;
  deferred OCR backends (GLM-OCR, PaddleOCR-VL, Dots.OCR, HunyuanOCR); DeepSeek-OCR-2;
  cross-page table/equation reconstruction; batch mode; alternate BibTeX sources.
  These remain additive behind the existing abstractions.

## Prerequisites & reference layout (read first)

- **Python** ≥ 3.10 (3.12.6 is installed on the dev machine; `tomllib` is stdlib on
  3.11+, so the `tomli` dep only applies to < 3.11 users).
- **User-supplied at runtime** (the tool bundles no weights — DESIGN §1.3): the
  llama.cpp **bin dir** (containing `llama-server[.exe]`) plus two `(model, mmproj)`
  GGUF pairs — DeepSeek-OCR for OCR and Gemma 4 for figure description. Configured in
  `config.toml` / overridable per-flag (DESIGN §13).
- **Reference source root**: every `*.ts` `file:line` citation in this plan is
  relative to `paper2llm/paper2llm-web/src/` (the in-repo TypeScript reference).
- **Spec**: `DESIGN.md` is authoritative; every `§N` reference points there. This
  plan sequences and grounds that spec; on any conflict, prefer `DESIGN.md` (and flag
  it).

## Tooling decision (resolved 2026-06-09 — hatchling confirmed)

- **End-user install** (the "anybody can get started" path): standard PEP 621
  `pyproject.toml`, console entry point `inscriber = "inscriber.cli:main"`,
  installable via `pip install inscriber` or `pipx install inscriber`. No dev tools
  required by users.
- **Build backend**: `hatchling`.
- **Dev workflow (zero extra tools)**: `python -m venv .venv` + `pip install -e ".[dev]"`.
  `uv` is documented as an optional faster alternative (it also creates a
  project-local `.venv` — never a global install).
- **Dev `[dev]` extras**: `pytest`, `ruff` (lint+format), `mypy` (optional, light).
  `ruff` is dev-only; end users never see it.
- **Runtime deps (DESIGN §18.1)**: `pymupdf`, `pillow`, `httpx`, `platformdirs`,
  `tomli; python_version < "3.11"`, `rich` (optional). No heavy ML libs.

## Corrections / clarifications to DESIGN.md (verified against paper2llm source)

These are confirmed facts from reading the real reference code. **Decision (2026-06-09):
the substantive ones (#1 registry, #3 thresholds, #4 skip-words, #5 API/429) have been
folded inline into `DESIGN.md` §6/§12.** The full list is retained here for the
implementer; #7/#8 were already in `DESIGN.md` (§9.5/§24).

1. **`domain-handler-registry.ts` DOES exist** at `core/domain-handler-registry.ts`
   (one level above `core/domain-handlers/`). DESIGN §6 says it doesn't. The
   `domain-handlers/` *directory* does only contain `base/generic/index` (design is
   right about that), but `DefaultDomainHandlerRegistry` (a singleton wrapping the
   handler list, `getHandler()` = find-first-`canHandle`) lives one level up. Python
   port: a plain list + first-match is sufficient; a tiny registry is optional.
2. **Exact mock-BibTeX text** is richer than DESIGN §12's abbreviation. Real text in
   `content-utils.ts` (port verbatim):
   ```
   % WARNING: This is a fallback mock citation.
   {titleWarning}% BibTeX generation failed to find this paper in academic databases.
   % Please replace with the correct citation if available.
   %
   % Generated: {YYYY-MM-DD}
   @article{unknownYear,
     title={{...}},
     author={Unknown Author},
     journal={Unknown Journal},
     year={{YYYY}},
     note={This is an automatically generated fallback citation}
   }
   ```
   where `{titleWarning}` is the mock-path 4-line block
   (`% WARNING: The paper title does not match the citation title.` …) when titles
   mismatch, else empty. DESIGN §12 says to **standardize** both paths on the single
   4-line `% WARNING: The retrieved citation title may not match the paper title.`
   form — keep that design decision; just be aware the source has two wordings.
   *(There are actually two mock variants in `content-utils.ts`: the no-result path
   above, and an error path `~:180-191` with `% BibTeX generation failed with error: …`
   and `note={… due to an error}`. v1 can collapse both to the single standardized
   mock per DESIGN §12; just don't be surprised by the second block.)*
3. **Title-validation thresholds** (`bibtex-generator.ts`): normalize =
   `lower → strip all but [a-z ] → collapse spaces → trim`; titles `<10` normalized
   chars require **exact** match; longer titles match if word-overlap ratio
   `commonWords / max(origWords, bibtexWords) > 0.75`.
4. **Citation-key skip-words** (verbatim): `["a","an","the","on","in","of","for","and","or"]`;
   first title word that is `>2` chars and not a skip-word (non-alphanumerics
   stripped); fallback to first word. Author part = last whitespace-token of first
   author, lowercased; year = paper year or current year.
5. **Semantic Scholar call**: `GET https://api.semanticscholar.org/graph/v1/paper/search`
   `?query={url-encoded title}&limit=3&fields=title,authors,venue,year,abstract,externalIds,url`;
   take `data.data[0]`. Source has **no explicit 429 handling** (any HTTP error →
   `[]`). DESIGN §12 wants graceful 429 degradation — *add* a clean message + skip.
6. **7 domain configs** — exact regexes captured (see M4). Note NeurIPS filename is
   `neurips-{year}-{id}.pdf` (fallback `neurips-{id}.pdf`); bioRxiv/medRxiv share the
   identical `10.1101` rule; MLRP filename `mlrp-v{n}-{id}.pdf`.
7. **`extractImageContext` page-number bug** confirmed: `image.id.split("-")[0]`
   yields `"img"`/`"unknown"`. DESIGN §9.5 already says to fix — use real page `N`.
8. **Image-metrics latent bug** confirmed (`content-utils.ts` counts
   `/> \*\*Image Description:\*\*/g`, capital-D + colon, never matches emitted
   `> **Image description.**`). DESIGN §24 already flags; we don't port image-metrics.

## Cross-cutting conventions (apply in every milestone — DESIGN §15)

- `pathlib.Path` everywhere; `Path.expanduser()`; never string-concat paths.
- Write text `encoding="utf-8", newline="\n"` (no Windows `\r\n`).
- `platformdirs` for config/cache/data dirs; never hardcode `~/.config`.
- Subprocess = list-args only, never `shell=True`; `Popen.terminate()` for teardown;
  avoid POSIX-only `os.killpg`/`preexec_fn` unless guarded by `os.name`.
- Binary discovery appends `.exe` on `os.name == "nt"`; PATH fallback via
  `shutil.which` (honors `PATHEXT`).
- TOML: `import tomllib` (3.11+) with `tomli` fallback (`python_version < "3.11"`).
- Progress/logs → **stderr**; final written file paths → **stdout** (one per line).
- Resilience: one failed figure / BibTeX / looping OCR page never kills the run —
  log, insert placeholder / skip, continue.

---

## Phases

### Phase M0 — Skeleton

**Goal**: Installable package; `inscriber --version` works; config loads/merges/
validates with full CLI override precedence; logging set up. No model code yet.

**Files to create** (DESIGN §4 layout):
- `pyproject.toml` (PEP 621, hatchling, entry point, runtime + `[dev]` deps).
- `README.md` (stub: what it is, install, privacy/offline statement — DESIGN §6/§20).
- `inscriber/__init__.py`, `inscriber/__main__.py` (`python -m inscriber`).
- `inscriber/models.py` — dataclasses: `Region`, `OcrPageResult`, `Figure`,
  `PageImage`, `ResolvedInput`, `RunConfig`, `ResolutionMode`, enums (DESIGN §7/§8.2).
- `inscriber/config.py` — TOML load + 3-layer merge (**CLI > config file > default**),
  **two-layer validation** (review Fix 3): (a) *structural* — enum membership, numeric
  ranges, types — runs always after merge, raises `ConfigError`; (b) *path-existence* —
  llama binary, OCR/VLM `model`+`mmproj`, input PDF readable — is **command/stage-aware**
  and runs only just before a server actually launches: **skipped for `--help`/`--version`**,
  **bypassed when `--ocr-endpoint`/`--vlm-endpoint` is set**, and `describe` never
  validates `[ocr].*`. (DESIGN §16's "validate before any model loads" = layer (b).)
- `inscriber/cli.py` — argparse with 3 subparsers (`run` default, `ocr`, `describe`),
  the full flag surface (DESIGN §13.2), maps to `RunConfig`. Bare `inscriber INPUT`
  ≡ `inscriber run INPUT`.
- `inscriber/logging.py` — `-v`→DEBUG / default INFO / `-q`→WARNING; stderr logs.
- `inscriber/pipeline.py` — orchestrator skeleton (entry funcs `run/ocr/describe`,
  empty bodies wired to config).
- `tests/test_config.py` — TOML load, every CLI override (the §13.3 mapping table),
  validation errors. Assert the §1.2 "every field overridable" contract literally.

**Key notes**: Lock the §13.3 config↔CLI map exactly (incl. `--server-timeout`,
`--no-normalize-breaks`, `--no-figures`⇒`figure.detect=none`,
`--no-split/--no-clean/--no-clobber` set-false flags, inscriber-only `--pages`).
Default config values from DESIGN §13.1 verbatim.

**Verification**:
- [ ] `inscriber --version` prints version (stdout).
- [ ] `pip install -e ".[dev]"` succeeds in a fresh venv on Windows.
- [ ] `pytest tests/test_config.py` green; precedence + validation covered.
- [ ] `ruff check` clean.

**Exit gate**: Config/CLI is the contract every later milestone depends on — it must
be complete and tested here, not retrofitted.

---

### Phase M1a — De-risk spike (RUN FIRST after M0; hardware is ready)

**Goal**: Resolve the two highest-risk unknowns on the pinned llama.cpp build before
committing the OCR backend. DESIGN §2.1, §2.2, §8.3 — *"Nothing else can be trusted
until this lands."*

**Files to create**:
- `inscriber/input/resolver.py` (**local-path only for now**, review Fix 1): validate a
  local PDF path (exists / readable / `%PDF` magic) → `ResolvedInput`. URL download + the
  7 domain handlers are deferred to **M4**; this minimal local-input contract exists now
  so M1a/M1b/M2 don't grow ad-hoc path glue that M4 must unwind. (`ResolvedInput` is
  defined in M0 `models.py`.)
- `inscriber/llama/server.py` — `LlamaServerManager`: ephemeral free-port probe,
  `Popen` launch (list-args), `/health` poll (503=loading→wait, 200=ready, timeout→
  error with server-log tail), `contextmanager serve()`, atexit/signal teardown, and the
  `--ocr-endpoint`/`--vlm-endpoint` **no-spawn** branch (DESIGN §5, §5.1).
- `inscriber/llama/client.py` — httpx OpenAI-compatible `/v1/chat/completions` client
  (base64 data-URL image content-part).
- `inscriber/ocr/base.py` — `Inferencer` Protocol + `HttpInferencer` (server) and
  `MtmdCliInferencer` (one-shot `llama-mtmd-cli` subprocess fallback). Path-aware
  `chat_template(path)`. **`Region`/`OcrPageResult` are imported from `models.py` (their
  canonical home, M0 — review Fix 2); the `OcrBackend` ABC is added in this file in M1b.**
  M1a needs only the inference layer.
- `inscriber/pdf/rasterize.py` — PyMuPDF: page count, 1-indexed page range,
  `zoom = target_px / max(page_pt_w, page_pt_h)` (**no `*72`**), per-mode long-edge
  targets (DESIGN §7 table). Returns `PageImage(page, png, W, H)`.
- `tests/test_llama_server.py` (review Fix 8 — **here, not M5**): `LlamaServerManager`
  launch-arg construction + `.exe`-suffix logic with `Popen` **mocked** (no real spawn),
  so Windows/process-teardown issues surface before real-hardware work.
- `tests/fixtures/` — (1) a **generated calibration PDF** (review Fix 4): a filled
  rectangle at exact known PDF coordinates, with recorded render metadata + the expected
  pixel box, so the coordinate-frame check below is *computable*, not eyeballed; (2) the
  recorded real DeepSeek-OCR grounding outputs produced below.

**The two empirical questions (must answer on real hardware)**:
1. **Image round-trip**: does a base64 image submit successfully to **DeepSeek-OCR**
   via `llama-server` `/v1/chat/completions` on the pinned build? (Open upstream issue
   #21022 may still break this.) If broken → fall back to `MtmdCliInferencer`
   (one-shot, accept reload cost). Decide and record which path v1 ships.
   - Separately confirm the **VLM** (Gemma 4) base64 round-trip (lower risk, own check).
2. **Grounding coordinate frame**: render the **calibration fixture** (figure at a known
   location), send `<|grounding|>Convert the document to markdown.` (`temperature:0`), read the
   emitted `<|ref|>…<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` coords (0–999 grid), and
   determine empirically whether `coord/999` maps against the **original image**
   (reference default) or the **padded 1024² square**. Pick whichever reproduces the
   box. Also confirm the **Gundam** frame (global 1024 view vs tiles).
   - Pin the exact prompt string, token strings, and the chosen mapping.

**Outputs of M1a (feed M1b)**:
- 2–3 representative pages of **real DeepSeek-OCR grounding output** committed to
  `tests/fixtures/`.
- A short `docs/M1A-FINDINGS.md` recording: build/version pinned, server-vs-mtmd-cli
  decision, the coordinate-frame answer (with the reproduced-box evidence), Gundam
  note, and the exact prompt/token strings.

**Verification**:
- [ ] A real image round-trips (server path) OR mtmd-cli fallback proven, documented.
- [ ] Coordinate frame determined with reproduced-box evidence; locked for M1b.
- [ ] Fixtures committed; `docs/M1A-FINDINGS.md` written.

**Exit gate**: M1b's parser + coordinate mapping are pinned to these fixtures/findings.
Do not start M1b's mapping logic until this is locked.

---

### Phase M1b — OCR vertical slice

**Goal**: `DeepSeekOcrBackend.ocr_page` (single grounding call → clean markdown +
figure boxes in original-page `[0,1]` frame), the OCR cache, per-page markdown with
`⟦INSCRIBER_FIG:{id}⟧` placeholders, for a real PDF. DESIGN §8.2–8.3, §8.6.

**Files to create**:
- `inscriber/ocr/registry.py` — name→backend class.
- `inscriber/ocr/deepseek.py` — `DeepSeekOcrBackend` (`name="deepseek-ocr"`,
  `supports_grounding=True`, `sampling()` temp 0 + fixed seed + `max_tokens` cap,
  path-aware `chat_template()`, `server_flags()` DRY/repeat-penalty partial mitigation).
  `ocr_page` algorithm (DESIGN §8.3 steps 1–4): grounding regex
  `<\|ref\|>(?P<label>.*?)<\|/ref\|><\|det\|>\[\[(?P<coords>[\d,\s]+)\]\]<\|/det\|>`,
  coord→`[0,1]` mapping **from M1a**, replace figure-class spans
  (`{figure,image,picture,chart,diagram,plot}`) with `⟦INSCRIBER_FIG:fig_p{page}_{i}⟧`
  placeholders (NOT delete), strip markup but keep text for non-figure regions, set
  `Region.text`. Malformed/absent grounding → whole output as plain markdown,
  `regions=[]`, warn.
- `inscriber/cache.py` — `OcrCache`: key = hash of `(pdf_content_hash, page,
  backend_name, model_identity, mmproj_identity, resolution_mode, render_long_edge_px,
  prompt, sampling_params)`; `*_identity` = path+size+content-hash (hash itself cached
  by path+size+mtime); value = pre-crop `OcrPageResult` + raw output + `value_schema`;
  `platformdirs.user_cache_dir("inscriber")/ocr/`; **written per-page** (resumable);
  `--refresh` (recompute+overwrite) vs `--no-cache` (no read/write) distinct.
- `tests/test_deepseek_parser.py` — **highest-value test**: golden-string parse +
  coordinate mapping pinned to M1a fixtures.

**Key notes**: Design the `OcrPageResult` (de)serialization **once here** — it's
reused by the cache (now) and the bundle (M2). Don't pick a format the bundle must
migrate. When figures disabled, use plain `Convert the document to markdown.` prompt.

**Verification**:
- [ ] `test_deepseek_parser.py` green against real fixtures (tokens + locked frame).
- [ ] Real PDF → per-page markdown with placeholders; cache hit on re-run (OCR server
      not relaunched); per-page write resumes after interrupt.

---

### Phase M2 — Figures + two-step (`ocr`/`describe`) split

**Goal**: Figure detection→crop, VLM server + `GemmaVlmBackend`, prompt+extraction,
whole-page context, blockquote injection, VLM cache, and the portable OCR **bundle**
with `ocr`/`describe` subcommands. DESIGN §3.1, §8.4–8.6, §9, §10.2.

**Files to create**:
- `inscriber/pdf/figures.py` — `figure.detect` strategies: `auto` (grounding when
  `supports_grounding`), `grounding` (error if backend can't), `none`
  (`--no-figures` alias), `pdf-embedded` (experimental: `page.get_images()` +
  `page.get_image_rects()`; raster-only, misses vector figures; never auto-selected
  while DeepSeek grounds). DESIGN §8.4.
- `inscriber/pdf/crop.py` — bbox `[0,1]`×(W,H) + `crop_padding` (default 0.02), clamp,
  skip near-zero area, Pillow crop → `figures/fig_p{page}_{i}.png`.
- `inscriber/vlm/base.py`, `inscriber/vlm/registry.py`,
  `inscriber/vlm/gemma.py` — `GemmaVlmBackend.describe(image_png, context_text)`:
  build prompt, base64 data-URL image, extract `<img_desc>…</img_desc>`.
- `inscriber/postprocess/prompt.py` — **port verbatim** the figure-description prompt
  template (DESIGN §9.3, already exact in the doc; source
  `core/templates/image-prompt-template.ts:12-33`) + `{contextText}` formatter
  (`:41-51`) + tag extraction (`:60-89`). ⚠️ **Tag-extraction divergence — mark as
  intentional, NOT a verbatim port (review Fix 5):** the source returns `null` on a
  missing opening tag (and the base image service then throws/retries); inscriber instead
  follows **DESIGN §9.4** — treat the whole trimmed response as the description + log a
  warning. The "opening-but-no-closing → everything after the opening tag" half *is* a
  faithful port of `:60-89`.
- Context builder (whole-page text, preamble `This image appears on page {N}. The
  surrounding page content follows.`, cap `figure.context_chars` default 2000 →
  `substring(0,1997)+"..."` only when >2000; **use real page N**, not `.split("-")[0]`).
  Source `markdown-processor.ts:360-408`.
- Figure injection (DESIGN §10.2): replace `⟦INSCRIBER_FIG:{id}⟧` with blockquote;
  every line prefixed `> ` (blank lines → `>`); exact headers
  `> **Image description.**` (real) / `> **Image.** [not displayed]` (placeholder);
  modes `describe-only` (default), `describe-and-keep` (`![{caption}](figures/{id}.png)`
  + desc), `placeholder`; single trailing `\n` per block. Port the *format* from
  `enhanceImageReferences` (`:298,:329`), **not** the `![]()`-matching loop.
- `inscriber/bundle.py` — bundle read/write: `OUT/paper.inscriber-ocr/`
  (`manifest.json` + `figures/` + optional `pages/`); `bundle_schema=1` gate (refuse
  higher); `describe` honors `[vlm].*`/`[figure].mode`/`[figure].context_chars`/
  `[output].*`/`[bibtex].*`/`[net].offline`/`[llama]`+`[inference]`, ignores
  `[ocr].*`/`[figure].detect`/`[figure].crop_padding`; base name from
  `manifest.source.name`; validate every `crop_path` exists. DESIGN §8.5.
- VLM cache (`cache.py` extension) — key
  `(figure_crop_hash, vlm_backend_name, vlm_model_identity, vlm_mmproj_identity,
  full_assembled_prompt, sampling_params)`; **full assembled prompt incl. context**;
  per-figure write. DESIGN §9.6.
- `tests/test_bundle_roundtrip.py` — `ocr` writes bundle; `describe` loads → output
  consistent with `run`; hand-edited page markdown survives; higher `bundle_schema`
  rejected.

**Key notes**: `run` = `ocr`-then-`describe` sharing the **same `OcrPageResult`
serialization** (M1b) and threading in-memory (consults cache, skips bundle I/O).
Sequential single-model-resident default (OCR server up → torn down → VLM server up).
**Sequencing caveat**: the full multi-file document writer is `output.py` (**M3**), so
at the end of M2 `describe` produces the *enhanced in-memory markdown* (figures
injected) and may emit a minimal `paper.md`; the polished `main/appendix/backmatter`
split file-set is delivered in M3. Don't assert the full DESIGN §14 file-set at M2.

**Verification**:
- [ ] `inscriber ocr paper.pdf -o out/` → inspectable bundle; `inscriber describe
      out/paper.inscriber-ocr` reuses crops; swapping `--vlm-model` re-describes only.
- [ ] `test_bundle_roundtrip.py` green; schema gate + hand-edit survive.
- [ ] Figure blockquotes match exact headers/spacing; `describe-and-keep` adds image ref.

---

### Phase M3 — Assembly & splitting

**Goal**: Page stitching + cleanup, the main/appendix/backmatter splitter with
standalone-file headers + allparts assembly, full output writer. DESIGN §10–§11, §14.

**Files to create**:
- `inscriber/postprocess/stitch.py` — concat per-page markdown; optional
  `#### Page {n}` numbers + `---` separators (both default off; keep `#### Page N`
  shape the splitter recognizes). Ported `normalizeLineBreaks` (`\n{3,}`→`\n\n`) and
  `ensureImageDescriptionSpacing` (regex
  `^> \*\*(?:Image description|Image Description|Image)\.\*\*` + `^Figure ` — keep all
  three spellings) — sources `markdown-processor.ts` (`:55`, `:94-185`, `:112`).
  **New for inscriber** (toggle `--no-clean`): running header/footer + page-number
  stripping (recurring same-position short lines, threshold-based, log removals);
  conservative de-hyphenation (`word-\nword`→`word`, lowercase cross-page merge).
- `inscriber/postprocess/splitter.py` — port `markdown-splitter.ts`: `extractTitle`
  (`# Title` → BibTeX `title={…}` → `"Untitled_Paper"`); backmatter regexes
  (ack/contrib/funding/impact/ethics + references family) and appendix regexes
  (Appendix/Appendices, Supplementary/Supporting (Material…), Supplemental, `SI `,
  `S\d+\.`, `A `/`A. ` **only after ack match** — the guard); page-marker
  `^#{3,4}\s+Page\s+\d+\s*$` boundary-shift (move boundary before a dangling marker,
  search last 5 lines); order validation (if ack>appendix, search backmatter only in
  `content[:appendixStart]`). Standalone framing: main's first H1 → canonical title;
  appendix/backmatter prefixed `# {title} - Appendix|Backmatter\n\n---\n\n`. allparts
  ordering is deliberately **main→appendix→backmatter** (`content-utils.ts:43-66`) —
  faithful, don't "fix". This is an **internal function only** (not written as a file in
  v1 — review Fix 7; see `output.py`). Sources `markdown-splitter.ts`, `content-utils.ts`.
  - **Port note (TS `.source` guard)**: the TS `A `-guard compares regex `.source`
    strings; in Python track the pattern by identity/index or a named flag instead.
- `inscriber/output.py` — write `paper.md` (always), `paper.main/appendix/backmatter.md`
  (when split + detected), `figures/` (only `describe-and-keep` or
  `--keep-intermediates`), all UTF-8 `\n`; sanitize base name so `paper.main.pdf`
  can't collide with `paper.main.md`; `clobber` default true, `--no-clobber`=hard error;
  log each write; print written paths to stdout.
  - **`full` vs `allparts` (decision, review Fix 7):** `paper.md` is the **enhanced,
    stitched full document in source order** (DESIGN §14 literal). The `allparts`
    reassembly (main→appendix→backmatter) stays an **internal splitter function** used
    only as the `--bibtex-in-doc` injection target; **v1 writes no separate
    `paper.allparts.md`**. (Reconciles DESIGN §11's "basis for the full file" wording
    with §14's file list, which lists no allparts file.)
- `tests/test_splitter.py`, `tests/test_stitch.py` — synthetic docs incl. the `A `
  edge case, page markers, with/without appendix/backmatter; header-strip + de-hyphen.

**Verification**:
- [ ] Splitter passes the battery (esp. `A `-after-ack guard, out-of-order ack/appendix,
      page-marker boundary shift).
- [ ] Stitch strips recurring headers/footers conservatively; de-hyphenation joins.
- [ ] Output files written with correct names/encoding; no `paper.main` collision.

---

### Phase M4 — Inputs & BibTeX (the network features)

**Goal**: URL input + 7 domain configs + `--offline`; Semantic Scholar BibTeX with
title validation, mock fallback, prepend/fenced injection. DESIGN §6, §12.

**Files to create**:
- `inscriber/input/resolver.py` (**extends the M1a local-path resolver**, review Fix 1)
  — add http(s) URL handling → `ResolvedInput(pdf_bytes, source, original_url,
  suggested_name)`; httpx download (follow redirects, timeout, descriptive UA, validate
  PDF bytes); route URLs through the domain handlers below; `--offline` hard-errors on
  URL input. (Local-path validation already landed in M1a.)
- `inscriber/input/domain_handlers.py` — config-driven `GenericDomainHandler`
  (`can_handle`/`normalize_pdf_url`/`file_name`) + the **7 regex configs ported
  verbatim** from `generic-handler.ts` (pin each as a fixture):
  - **arXiv**: host `arxiv.org`; url `/(abs|pdf|html)/(\d+\.\d+|[\w-]+\/\d+)`;
    transform `/(abs|html)/`→`/pdf/`; file `arxiv-$2.pdf`.
  - **OpenReview**: host `openreview.net`; **host-level branch in `normalize_pdf_url`
    BEFORE generic rules** — require `?id=`, set path `/pdf`, **preserve query**
    (don't drop `?id=`); file `openreview-{id}.pdf` (fallback `openreview-paper.pdf`).
  - **ACL**: host `aclanthology.org`; append `.pdf` to last segment; file
    `acl-{seg}.pdf`.
  - **bioRxiv**/**medRxiv** (identical rule): host `biorxiv.org`/`medrxiv.org`; url
    `/content/10\.1101/`; transform `/content/(10\.1101/[\d.]+)(v\d+)?(?:\.full\.pdf|\.full|$)`
    →`/content/$1$2.full.pdf`; file `biorxiv-$1.pdf`/`medrxiv-$1.pdf`.
  - **NeurIPS**: hosts `papers.nips.cc`,`papers.neurips.cc`; transform
    `/hash/{x}-Abstract.html`→`/file/{x}-Paper.pdf`; file `neurips-{year}-{id}.pdf`
    (fallback `neurips-{id}.pdf`).
  - **MLRP**: host `proceedings.mlr.press`; url `/v\d+/[a-z0-9]+`; transform
    `/(v\d+)/([a-z0-9]+)(?:\.html)?$`→`/$1/$2/$2.pdf`; file `mlrp-v$1-$2.pdf`.
  - Unmatched URLs are **not handled** (no catch-all) — error clearly.
- `inscriber/bibtex/semantic_scholar.py` — title→entry (API call per Correction #5),
  citation key (Correction #4), title validation (Correction #3), **empty-string
  sentinel** on failure, mock-text assembly (Correction #2). Define **one canonical
  inscriber mock fixture** `tests/fixtures/bibtex_mock.txt` (review Fix 6) — the
  standardized 4-line warning per DESIGN §12 + the collapsed `@article{unknownYear, …}`
  block — and test against *that*, not paper2llm byte-parity. `bibtex.append_to_document`
  injects prepend+fenced-code-block+`---` into the **full document** (`paper.md`) and the
  **main** split (DESIGN's `allparts` injection target maps to `paper.md`; v1 emits no
  separate allparts file — see M3 Fix 7). Standalone `paper.bib`; degrade on
  429/offline/network-fail (never fail the run).
- `tests/test_domain_handlers.py` (the 7 fixtures incl. OpenReview query preservation),
  `tests/test_bibtex.py` (key gen, title validation thresholds, mock fallback text,
  injection format) — mock httpx; no live network in CI.

**Verification**:
- [ ] All 7 transforms produce expected PDF URLs + filenames (fixtures); OpenReview
      keeps `?id=`.
- [ ] BibTeX: real-match path, mismatch warning, and fenced prepend correct; mock
      fallback matches the **canonical inscriber mock fixture** (not paper2llm byte-parity).
- [ ] `--offline` blocks URL input + BibTeX with clear messages; local servers
      **not** blocked (loopback ≠ network).

---

### Phase M5 — Hardening, packaging, docs

**Goal**: Cross-platform CI, mocked end-to-end, `concurrent` mode, README/docs,
PyPI packaging. DESIGN §5.4, §15, §17, §18.

**Work**:
- `tests/test_pipeline_mocked.py` — end-to-end on a tiny fixture PDF, OCR+VLM clients
  mocked (canned responses), asserts full output-file set + figure injection.
- `tests/test_pdf_embedded_figures.py` — `figure.detect=pdf-embedded` on a fixture PDF
  with an embedded raster → crop + appended placeholder.
- (`LlamaServerManager` launch-arg + `.exe`-suffix unit tests now live in **M1a**, per
  review Fix 8 — not repeated here.)
- `concurrent` mode (DESIGN §5.4): both servers up, independent per-server `-ngl`;
  even here consult OCR cache before launching OCR server; document VRAM caveat.
- CI matrix: Windows + Linux + macOS; no GPU → servers mocked; `ruff` + `pytest`.
- Smoke: `inscriber --version`; `inscriber sample.pdf --no-figures --offline` (mocked)
  passes.
- Packaging: finalize `pyproject.toml`, verify `pip install` + `pipx install` of the
  built wheel; `inscriber` console command resolves.

**Verification**:
- [ ] CI green on all 3 OSes (mocked).
- [ ] Mocked end-to-end produces the full DESIGN §14 file set.
- [ ] Built wheel installs and runs `inscriber --version` via pip and pipx.
- [ ] Manual/integration doc validates one real PDF against real GGUFs (not in CI).

---

## Documentation (deliverables, not afterthoughts)

- `README.md` (M0 stub → M5 complete): what it is, install (`pip`/`pipx`), quickstart,
  the **privacy/offline statement** (DESIGN §6/§20 — only URL input + BibTeX touch the
  network; `--offline` hard-disables), config reference, the two-step `ocr`/`describe`
  workflow, llama.cpp + GGUF setup pointers.
- `docs/M1A-FINDINGS.md` (M1a): pinned build, server-vs-mtmd-cli decision, coordinate
  frame answer + evidence, Gundam note, exact prompt/token strings.
- `docs/integration-test.md` (M5): how to validate against real llama.cpp + real GGUFs
  with a known sample PDF (release checklist).
- Done: the substantive corrections are folded into `DESIGN.md` §6/§12; the full
  8-item list is tracked in the "Corrections" section of this plan.
- Inline: keep comment density matched to ported code; cite the paper2llm source
  file:line next to each ported transform for future maintainers.

## Risks

- **M1a empirical answers differ from design defaults** (coord frame = padded square,
  or server image path broken → mtmd-cli). *Mitigation*: M1a runs first and encapsulates
  the answer inside the backend; everything downstream is original-page-frame agnostic.
- **DeepSeek-OCR f16 repetition loops** (Q4_K_M banned; no n-gram penalty in llama.cpp).
  *Mitigation*: f16 + hard `max_tokens` cap + per-request wall-clock timeout + soft-fail
  on looping/truncated page; DRY/repeat-penalty flags as partial mitigation (DESIGN §2.2).
- **Port-fidelity drift** (regexes/strings subtly wrong). *Mitigation*: golden tests
  pin exact strings; each transform cites its paper2llm source; fixtures per domain.
- **Windows-specific breakage** (paths, `.exe`, `\r\n`, process teardown).
  *Mitigation*: cross-cutting conventions enforced from M0; CI matrix includes Windows.

## Decisions (resolved 2026-06-09)

- **#1 Build backend / tooling** → **`hatchling`** confirmed (zero-config, invisible
  to end users — they still `pip install`/`pipx install`). Dev path: `venv + pip`;
  `ruff` + `pytest` + optional `mypy` in `[dev]`; `uv` as an optional speed-up.
- **#2 DESIGN.md corrections** → folded inline into `DESIGN.md` §6/§12.
- **#3 M1a scheduling** → run **M1a immediately after M0** (hardware is ready). The
  hardware-free logic milestones are NOT pulled earlier; strict M0→M1a→M1b→M2→M3→M4→M5.
- **#4 PyPI** → package and verify install in M5, but **defer the public PyPI upload**
  until after a real-hardware integration pass.

## Open Questions

None outstanding — all resolved above. (Add here if anything new arises.)

---
**Plan is ready. Awaiting your explicit go-ahead before any code is written —
execution starts at M0.**
