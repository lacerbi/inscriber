# 2026-06-11 — Pre-release codebase review: remaining findings (handoff)

**Status: open work list.** This note is the self-contained record of the
findings from the 2026-06-11 full-codebase pre-release review that were **not**
fixed before release. It is written so a developer with no prior context can
pick up any item independently. Severities are from the review
(HIGH/MEDIUM/LOW); none blocks a release — the blocking items were fixed the
same day (see "Already fixed" at the end).

How the review was run: one reviewer pass over security/privacy/packaging/repo
hygiene, plus seven parallel in-depth review agents (pipeline core; postprocess
& output; caching & VLM; CLI/config/docs; tests & CI; runtime robustness &
cross-platform; BibTeX & URL input). Full suite at the time: 369 passed, ruff
clean.

Before touching anything here, read `AGENTS.md` ("Where truth lives",
"Invariants and gotchas") — in particular:

* **Pinned model-facing behavior** (prompts, template kwargs, server flags)
  cannot be validated by mocked tests; changes need a real-hardware spike and a
  dated note in `dev/notes/` (DESIGN §9.7 discipline).
* **Anything that changes model output must become cache-key material**
  (AGENTS.md / DESIGN §8.6); conversely, changing key construction orphans warm
  caches — say so loudly in the change.
* Several behaviors are **deliberate verbatim paper2llm ports** (DESIGN §23–24)
  — "fixing" them breaks parity on purpose only.
* Docs are first-class: every behavior change updates DESIGN.md (+ README /
  `config.example.toml` when user-facing) in the same change.

---

## A. Robustness / UX

### A1. [HIGH] Windows "file in use" on setup promote and config write → raw traceback

* **Where:** `inscriber/setup.py` — `part.replace(dest)` (promotion, in
  `download_model`) and the two `open(target, "w", ...)` config writes in
  `write_setup_config`.
* **What:** on Windows, replacing/overwriting a file another process holds open
  raises `PermissionError`/`WinError 32`. Re-running `setup` while a previous
  `inscriber run` still has the GGUF mmap'd by llama-server, or with
  `config.toml` open in an editor that locks it, gives an uncaught traceback.
* **Fix:** wrap promotion and config write in `try/except OSError` → re-raise
  as `SetupError` with a "the file may be open in another program (a running
  llama-server / an editor); close it and re-run setup" hint. The verified
  `.part` survives promotion failure, so the retry is cheap — say so in the
  message.
* **Test:** hard to simulate a lock portably; at minimum unit-test that an
  `OSError` from `Path.replace` is converted (monkeypatch `Path.replace`).

### A2. [HIGH] Server-log tail can be empty exactly when the user needs it

* **Where:** `inscriber/llama/server.py` — `_log_tail` (read path),
  `serve`/`_wait_healthy` (raise sites).
* **What:** llama-server's stdout/stderr go to a log file (correct — no pipe
  deadlock). But when the server dies *during model load* (wrong mmproj
  pairing, VRAM OOM — the most common first-run failures), its own stdio
  buffers may not be flushed at the moment `_log_tail` reads, so the
  `ServerError` shows an empty/partial tail or `(no server log captured)`,
  defeating DESIGN §16's "surface the server log tail".
* **Fix:** after detecting early exit / timeout, give the file a brief settle
  (e.g. re-read once after ~0.5 s if empty), and append the **path** of the
  persistent log file to the error message so the user can always look
  themselves.

### A3. [MEDIUM] `UnicodeEncodeError` from log lines on a cp1252 stderr

* **Where:** `inscriber/logging.py` (the `StreamHandler` setup). Context: the
  same failure mode was already fixed for argparse *help* text (see the ASCII
  note in `cli.py`), but log **messages** interpolate user-controlled strings.
* **What:** on a redirected/piped Windows console (cp1252), a non-ASCII model
  path, URL, or paper title inside an INFO/WARNING line raises
  `UnicodeEncodeError` inside the logging module (printed as a handler error,
  message lost).
* **Fix:** construct the handler over a stream wrapped with
  `errors="backslashreplace"` (or `io.TextIOWrapper(sys.stderr.buffer,
  encoding=..., errors="backslashreplace")`). One-liner; add a test that logs a
  `→` to a `StringIO` with a cp1252-like codec if practical.

### A4. [MEDIUM] Workdir cleanup failures are silent (`ignore_errors=True`)

* **Where:** `inscriber/pipeline.py` — the workdir contextmanager
  (`shutil.rmtree(..., ignore_errors=True)` on success).
* **What:** on Windows, a still-open handle (AV scan, slow log close) makes
  rmtree leave residue; with `ignore_errors=True` the user is never told, so
  GBs of page rasters can accumulate in the OS temp dir over many runs —
  contradicting "deleted on success".
* **Fix:** use `rmtree(..., onerror=...)` (or check `path.exists()` after) and
  log one WARNING with the leftover path.

### A5. [LOW] `_terminate` failures only visible at DEBUG

* **Where:** `inscriber/llama/server.py` — `_terminate` (the
  `except Exception: logger.debug(...)` swallow).
* **What:** if `terminate→wait→kill` fails (wedged GPU process), the server is
  `_untrack`ed anyway and the failure is logged at DEBUG — a truly stuck server
  is silently forgotten (and the run blocks up to 2×10 s grace with no
  progress message).
* **Fix:** log at WARNING when `kill()` is reached and when it fails, including
  the PID so the user can clean up manually.

### A6. [LOW] Spurious "pre-table-pass bundle?" warning in `describe`

* **Where:** `inscriber/pipeline.py`, `_refine_tables` (~line 501) vs the
  raster-save condition in `run_ocr` (~line 996).
* **What:** `run_ocr` saves a page raster only when the page has a
  **refinable** blob; `_refine_tables` warns about a missing raster for any
  page with *any* `<table>` span, before checking `blob_is_refinable`. A bundle
  page whose only blobs are non-refinable (empty / nested / placeholder-
  carrying) warns misleadingly even though nothing is lost.
* **Fix:** compute the refinable set first; warn only when it is non-empty.
  Pure reorder, no model contact; extend the existing old-bundle degradation
  test in `tests/test_tables.py`.

---

## B. Output-content correctness (model-free, but changes output text)

### B1. [MEDIUM] `dehyphenate` is broader than DESIGN §10.3b promises

* **Where:** `inscriber/postprocess/stitch.py` (~line 166):
  `re.sub(r"(\w)-\n\s*(\w)", r"\1\2", markdown)`, applied document-wide.
* **What:** two gaps vs the spec. (a) DESIGN §10.3b's "lowercase continuation"
  qualifier is attached to its page-break sentence-merge clause, and the whole
  section is capped "Conservative rules only" — the code's unconditional
  `\w-\n\w` join violates that conservative intent: `well-\nknown` →
  `wellknown` (hyphen lost), `Anti-\nPattern` → `AntiPattern`, and
  line-wrapped math `x-\ny` → `xy` — the likeliest silent character-level
  corruption in the cleanup pass. (b) The spec's mid-sentence page-break merge
  is not implemented at all.
* **Fix options (pick one, update DESIGN §10.3b either way):** restrict the
  join to a lowercase right-hand side (`-\n\s*([a-z])`) and/or skip lines
  inside pipe tables / math spans; or amend DESIGN to describe the actual
  behavior and drop the unimplemented merge clause. `--no-clean` already opts
  out wholesale.
* **Caveat:** changes output text → goldens in `tests/test_stitch.py` need
  updating; cheap to validate (no model involved).

### B2. [LOW] `sanitize_table_output` discards valid GFM tables without leading pipes

* **Where:** `inscriber/postprocess/tables.py` (~line 261).
* **What:** a VLM output whose rows use the `a | b` form (legal GFM, no leading
  `|`) is rejected wholesale and the degenerate `<table>` blob kept. Safe
  (fallback keeps every value) but throws away a good restructure.
* **Fix:** accept rows matching a slightly looser row test (contains an
  unescaped `|` + a valid separator line), keeping the strict "nothing but a
  table" stance. **Caveat:** the digit-coverage guard and never-cache-failure
  flow must be re-checked; the prompt itself is pinned — do not touch it.

### B3. [LOW] `--no-full-suffix` can collide with another paper's split; Windows reserved names

* **Where:** `inscriber/output.py` — `sanitize_base_name`,
  `write_full_document`; DESIGN §14's "cannot collide" claim.
* **What:** (a) with `full_suffix=false`, the full doc is bare `{base}.md`, so
  a source sanitizing to `X_main` writes `X_main.md` — colliding with paper
  `X`'s split in the same out dir (the `_part`-suffix argument only covers the
  suffixed outputs). (b) `--name CON` + `--no-full-suffix` yields `CON.md`, a
  reserved device stem on Windows.
* **Fix:** reject/adjust reserved device stems in `sanitize_base_name`
  (`CON, PRN, AUX, NUL, COM1-9, LPT1-9`), and soften DESIGN §14's claim (or
  warn when a `--no-full-suffix` base ends in `_main`/`_appendix`/
  `_backmatter`/`_full`).

### B4. [INFO] `describe-and-keep` alt text not escaped

* **Where:** `inscriber/postprocess/inject.py` (~line 85):
  `f"![{alt}]({fig.crop_path})"`.
* **What:** a caption containing `]` or `)` breaks the Markdown image link.
  Cosmetic; non-default mode.
* **Fix:** escape `]` in alt text (and spaces in path are already controlled).

### B5. [INFO — decision needed, parity-bound] Splitter single-search appendix miss

* **Where:** `inscriber/postprocess/splitter.py` (~line 98),
  `find_section_boundaries`.
* **What:** each appendix pattern is searched once; if the first `^#+\s+A\s+`
  match is pre-acknowledgments (correctly rejected by the guard), a *real*
  later `## A …` appendix heading is never found. This is a **faithful port of
  the identical paper2llm limitation** (markdown-splitter.ts) — per repo policy
  it is not a bug, but if it ever bites, the fix is `finditer` + first match
  positioned after the ack boundary. Update DESIGN §11 if changed (deliberate
  parity break).

---

## C. Cache subsystem (all failure modes here are over-busting / non-sharing — never stale serving)

### C1. [MEDIUM] `hashes.json` sidecar write is non-atomic

* **Where:** `inscriber/cache.py`, `file_identity` (~line 84).
* **What:** cache *entries* use tmp+`Path.replace` (atomic); the shared
  model-hash sidecar uses bare `write_text`. Concurrent processes or a crash
  mid-write can tear it. Self-healing (corrupt JSON → recompute hashes) but
  inconsistent with the module's own discipline, and a multi-GB GGUF re-hash
  is the price of a torn file.
* **Fix:** same tmp+replace pattern; optionally merge-on-write (re-read before
  writing) so two processes don't drop each other's entries.

### C2. [MEDIUM] Figure-description cache key hashes re-encoded crop PNG bytes

* **Where:** `inscriber/pipeline.py` (~line 672, `sha256_bytes(crop_bytes)`)
  vs the table-key scheme in `cache.py::make_table_key`.
* **What:** the table key was deliberately moved to
  `(verbatim raster hash, bbox, padding)` to be immune to PNG-encoder churn
  (DESIGN §9.7). The figure key still hashes the freshly encoded crop bytes:
  `run` re-crops from page bytes, `describe` reads the bundle's stored crop —
  identical only while Pillow's PNG output stays byte-stable. A Pillow upgrade
  spuriously busts every figure description and breaks run↔describe sharing.
* **Fix:** adopt the same `(raster_hash, bbox, padding)` scheme for figures.
  **Caveats:** this orphans existing figure entries (acceptable; note it), and
  the bundle path must key off the same verbatim inputs the `ocr` stage used —
  mirror how `make_table_key` threads conditional fields, and pin the
  run↔describe key equality with a test like
  `test_run_then_describe_share_table_cache`.

### C3. [LOW] `make_vlm_key` lacks a `kind` discriminator

* **Where:** `inscriber/cache.py::make_vlm_key`.
* **What:** table and probe keys carry `"kind": ...`; the figure key's
  disjointness is incidental (different field sets). Add
  `"kind": "figure-description"` for structural disjointness.
* **Caveat:** changes every figure key (orphans warm entries) — fold into the
  same change as C2 so caches are only orphaned once.

### C4. [LOW] `GemmaVlmBackend.image_first` changes output but is not key material

* **Where:** `inscriber/vlm/gemma.py` (constructor knob, default True).
* **What:** currently unreachable (never configured ≠ True), so no live
  hazard — but if it is ever exposed, it silently becomes a stale-cache bug.
* **Fix:** either remove the knob or fold it into the key payloads when it
  becomes configurable. A comment at the knob pointing at this note is enough
  for now.

---

## D. Input hardening (URL path = the main untrusted-data entry point)

### D1. [MEDIUM] stdlib `xml.etree` parses remote arXiv Atom

* **Where:** `inscriber/bibtex/arxiv.py::arxiv_bibtex`
  (`ElementTree.fromstring(resp.text)`).
* **What:** CPython documents ElementTree as not secure against maliciously
  constructed data (billion-laughs / quadratic blowup). Threat model is narrow
  (hardcoded HTTPS `export.arxiv.org`; a parse error already degrades to the
  next source), but this is the only place remote XML is parsed.
* **Fix:** use `defusedxml.ElementTree` (new optional dep) **or** keep stdlib
  and document the trust assumption here + in DESIGN §12. Failure must remain
  a degrade-not-raise (`ParseError` → `None`).

### D2. [MEDIUM] Plain `http://` input URLs accepted silently

* **Where:** `inscriber/input/resolver.py::is_url` / `_download_pdf`.
* **What:** a plaintext fetch (or downgrade redirect) exposes the request to
  MITM, which then feeds attacker bytes to PyMuPDF. All seven supported
  repositories are HTTPS in practice.
* **Fix:** upgrade `http://` → `https://` for the seven known hosts (safe), or
  warn loudly on any plaintext fetch. Keep `--offline` semantics untouched.

### D3. [LOW] Substring host matching

* **Where:** `inscriber/input/domain_handlers.py::GenericDomainHandler.can_handle`
  (`any(p in host ...)`) and `inscriber/bibtex/arxiv.py::arxiv_id_from_url`
  (`"arxiv.org" not in hostname`).
* **What:** `https://arxiv.org.evil.com/abs/1234.5678` matches both — the
  download then goes to evil.com, and the BibTeX chain treats the URL as arXiv
  *provenance* (citable by construction; sends the extracted ID to S2/arXiv).
  User-supplied URLs only, so impact is low, but suffix matching is strictly
  better.
* **Fix:** match `host == p or host.endswith("." + p)`. The TS source uses
  `hostname.includes(...)` — this is a deliberate, documented parity break;
  pin with fixtures (`tests/test_domain_handlers.py`) including the evil-host
  negative case.

---

## E. Tests / maintenance

### E1. [LOW] `hermetic_cache` fixture copy-pasted across 6 test files

* **Where:** `tests/test_pipeline_mocked.py`, `test_bundle_roundtrip.py`,
  `test_bibtex_chain.py`, `test_ocr_truncation.py`, `test_tables.py`,
  `test_bibtex_probe.py` (plus near-duplicated `_dummy_models`
  / `_mock_inference` helpers).
* **What:** all copies are correct today; the risk is drift — a future change
  to the hermeticity boundary must be applied in 6 places or one file silently
  touches the real platformdirs cache.
* **Fix:** move to a shared `tests/conftest.py`; keep AGENTS.md's "Testing
  conventions" section accurate.

### E2. [LOW] Duplicated `8192` OCR token-cap literal

* **Where:** `inscriber/ocr/base.py` (`OcrBackend.max_tokens`),
  `inscriber/ocr/deepseek.py` (`__init__` default), and
  `inscriber/ocr/glm.py` (`__init__` default — dormant, the GLM backend is
  experimental, but the consolidation should cover it).
* **What:** the anti-repetition-loop cap exists as three independent literals;
  desyncing them would silently split the cache key (sampling includes
  `max_tokens`) from the base default.
* **Fix:** one module constant, both readers. **Keep the value** — the cap is
  load-bearing (AGENTS.md invariants).

### E3. [INFO] Watch items (no action unless they bite)

* `tests/test_rasterize.py::test_calibration_box_maps_to_expected_pixels` is
  the one pixel-exact (±4 px) assertion — first suspect if macOS/Linux CI ever
  flakes on a PyMuPDF wheel bump.
* `ChatClient` uses a single scalar 600 s timeout; a wedged server stalls a
  page for the full 10 min before the per-page soft-fail. If users report
  hangs, consider a tighter read timeout (per-request wall-clock, DESIGN §2.2
  intent).
* `find_handler` rebuilds + recompiles all 7 handler regex configs per call
  (also called again for BibTeX provenance). Negligible at one-PDF scale.

---

## Already fixed pre-release (2026-06-11, same day as the review — do not re-do)

1. **POSIX SIGTERM/SIGHUP orphaning llama-server** — `_register_cleanup` /
   `_on_terminate_signal` in `llama/server.py` (DESIGN §5.3).
2. **Unbounded in-memory URL download** — streamed + 512 MiB cap + first-bytes
   `%PDF` check in `input/resolver.py`; `tests/test_resolver.py` (DESIGN §6).
3. **Concurrent-mode fixed-port rejection over-applying** — gated to `run` in
   `config.py::validate_structural` (DESIGN §5.4).
4. **README model sizes vs `setup` output units** — README switched to decimal
   GB.
