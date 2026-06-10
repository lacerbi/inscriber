# Plan: BibTeX `auto` mode (classify → source chain → best-effort)

Created: 2026-06-10
Status: 📋 DESIGNED — awaiting go-ahead; no code written.

## Summary

Not every document `inscriber` processes is a citable paper, and today BibTeX is
a blunt opt-in bool (`bibtex.enabled`) that always goes online. This plan adds a
**`bibtex.mode = "auto"`**: the pipeline decides whether the document is
*citable* (provenance heuristics first, a single cached LLM probe as the
tiebreaker), then produces an entry through an internal **ordered source chain**
— arXiv-by-ID (authoritative, when provenance gives an ID) → Semantic Scholar
(today's path) → **local best-effort** (an entry assembled from LLM-extracted
front matter, clearly marked, fully offline). `auto` becomes the **default**
mode at the end of the plan, gated on a real-hardware validation pass.

The existing `on`/`off` behaviors are preserved **exactly** (paper2llm-parity
mock fallback included); all new machinery lives behind `auto`. The LLM probe
follows the table-restructuring pass's proven pattern: pinned prompt, one
backend instance, prompt-assembled-once as cache-key material, shared
`_VlmSession`, never cache a failure.

`DESIGN.md` stays the authoritative spec — each phase below names the DESIGN
sections it must update in the same change (this repo treats docs as
first-class).

## Decisions (resolved 2026-06-10, maintainer discussion)

1. **`auto` becomes the default** (end state; flipped only in the final phase,
   after hardware validation). Consequence: the classifier errs toward
   *abstaining* — with a default-on feature, a false positive (unwanted
   `.bib` / unwanted network call) is worse than a false negative. Every
   "judged not citable" decision is a visible log line, never silent.
2. **Privacy taint rule** (resolves the conflict with DESIGN §20 / README's
   "only URL input and opt-in `--bibtex` touch the network"): in `auto`, online
   sources are attempted **only when the run is already network-tainted** — the
   input came through a URL (`ResolvedInput.source == "url"`, or the bundle's
   `source.source == "url"` for `describe`). A local-PDF run in `auto` does
   classification + local best-effort only; **nothing leaves the machine with
   zero flags**. Explicit `mode = "on"` (or `--bibtex`) always unlocks the
   online chain, as today. `--offline` disables online sources in every mode
   (the probe and best-effort are loopback-local and stay available, per
   DESIGN §15's "`--offline` does not gate the local servers").
3. **No `--bibtex-source` flag yet.** The source chain is an internal,
   code-level seam (so a source-selection axis is purely additive later);
   `auto + --offline` already covers "local only".
4. **Humble entry types.** Best-effort emits `@misc` only (no venue-type
   guessing); the arXiv source emits the standard `@misc` + `eprint` shape; the
   Semantic Scholar path keeps its current `@article`. Type inference is a
   future refinement.
5. **Transcription, not recall.** The probe prompt instructs the model to
   extract only fields *visible in the supplied text* and omit the rest —
   same philosophy as the table pass ("structuring, not re-OCR"). Best-effort
   entries never get `Unknown Journal`-style filler; absent fields are absent.
6. **`on` mode is frozen.** Explicit `--bibtex` keeps today's exact behavior
   (extract `# Title` → Semantic Scholar → mock fallback, no LLM involved), so
   the documented paper2llm ports (DESIGN §24 rows 14–16) stay byte-stable and
   `on` keeps working with no VLM configured.

## Scope

- **In**: `bibtex.mode` tri-state config/CLI; the LLM citability+metadata probe
  (text-only VLM call, cached, pinned prompt); local best-effort entry
  assembly; arXiv-by-ID source; the `auto` orchestration (taint rule, degrade
  paths); default flip + full docs ripple; mocked tests throughout.
- **Out (future refinements — add to `TODO.md` when this lands)**: a
  `--bibtex-source` CLI axis; Crossref as an additional source; structure-based
  citability heuristics (references-section detection) beyond provenance;
  entry-type inference; extraction from the paper's own reference list.

## Architecture (where this lives)

```
inscriber/bibtex/
├── semantic_scholar.py   # UNCHANGED API — becomes one source in the chain
├── probe.py              # NEW: probe prompt template + JSON extraction + ProbeResult
├── arxiv.py              # NEW: arXiv ID from URL; export.arxiv.org → @misc entry
├── local.py              # NEW: best-effort @misc from ProbeResult metadata
└── chain.py              # NEW: auto orchestration (citability → source chain)
```

Pipeline wiring (both `_run_body` and `describe`): a new `_bibtex_probe(...)`
step runs **inside the open `_VlmSession` block** (after `_vlm_describe`), so
the probe shares the already-resident VLM server instead of forcing a relaunch
at step 9; its `ProbeResult` is threaded into `_bibtex_outputs(...)`, which
grows a `mode`/`probe`/`tainted` view but keeps its signature shape (never
fails the run). Cache-first like every VLM pass: a fully-cached document still
never launches a server. (Known wart, accepted: a run with figures and tables
fully cached but a probe miss launches the VLM server for one small call —
once, then it's cached.)

The probe is **text-only**: `ChatClient.chat()` already exists
(`llama/client.py:38`) with `last_finish_reason` tracking; the new
`VlmBackend.probe_metadata(prompt)` mirrors `restructure_table` (returns
`None` on truncation/refusal; caller treats it as "unknown").

## Cross-cutting conventions (every phase)

- The probe prompt is **pinned, model-facing behavior**: assembled exactly once
  per document via the backend (`build_bibtex_probe_prompt`), used verbatim as
  cache-key material AND as the request (DESIGN §9.2 discipline); changes
  require re-validation on real hardware recorded in `dev/docs/`.
- Cache: shared `VlmCache`, new key fn with `"kind": "bibtex-probe"`
  discriminator (disjoint from figure/table payloads by construction, like
  `make_table_key`). Key material: full assembled prompt (page-1 text
  embedded), backend name, model/mmproj/server identities, sampling,
  `chat_template_kwargs`. **Never cache a failed/truncated/unparseable probe.**
- Tests mock at the chat-client boundary; the probe prompt carries the pinned
  discriminator phrase **"bibliographic metadata"** (the mock dispatch joins
  `"<|grounding|>"` / `"reconstructing ONE table"` / figure-default). Use the
  `hermetic_cache` fixture pattern; no live network in CI (mock `httpx`).
- Resilience (DESIGN §16): BibTeX never fails the run — any probe/source/chain
  error degrades to the next step or to a logged skip.
- Text files `encoding="utf-8", newline="\n"`; `pathlib`; no new heavy deps
  (arXiv's Atom response is parsed with stdlib `xml.etree`).

---

## Phases

### Phase B0 — `bibtex.mode` tri-state (no behavior change)

**Goal**: Replace the `enabled` bool with `mode: "off" | "on" | "auto"` end to
end, with `auto` temporarily behaving like `off` (wired in B4). `on`/`off`
semantics byte-identical to today.

**Changes**:
- `models.py` — `BibtexConfig.mode: str = "off"` (replaces `enabled`;
  `append_to_document` unchanged).
- `config.py` — structural validation: enum membership. **Legacy alias**: a
  TOML `[bibtex] enabled = true/false` maps to `on`/`off` with a one-line
  deprecation warning (machine-local configs exist in the wild; don't break
  them). `mode` wins if both are present.
- `cli.py` — keep `--bibtex` as a back-compat alias for `mode="on"`
  (unchanged flag, unchanged help + "(requires network)" note); add
  `--bibtex-mode {off,on,auto}` as the full knob (a separate flag, NOT
  `--bibtex [MODE]` — an optional-argument value would swallow a following
  positional `INPUT`). Precedence: `--bibtex-mode` > `--bibtex` > config.
- `pipeline.py` — `_bibtex_outputs` gates on `mode == "on"` for now (`auto`
  → log "auto mode not yet wired" at DEBUG, skip).
- `config.example.toml` — `mode = "off"` + comment; keep a commented legacy
  note.

**Verification**:
- [ ] `test_config.py`: tri-state precedence (CLI > config > default), legacy
      `enabled` alias + deprecation warning, invalid value → `ConfigError`.
- [ ] `test_bibtex.py` / `test_pipeline_mocked.py`: existing `--bibtex` runs
      produce byte-identical outputs (mock fixture untouched).
- [ ] `ruff check` clean; full suite green.

**Exit gate**: the §13.3 "every field overridable" contract holds for
`bibtex.mode`; no output of any existing invocation changed.

---

### Phase B1 — The LLM probe (classify + extract, cached)

**Goal**: One text-only VLM call per document answering "is this citable?" and
extracting front-matter metadata, with the table pass's full key/session
discipline. Not yet consumed by any output path.

**Changes**:
- `inscriber/bibtex/probe.py` —
  - `ProbeResult` dataclass: `citable: bool`, `title: str | None`,
    `authors: list[str]`, `year: str | None`, `venue: str | None`,
    `raw: str` (the model's JSON, for debugging).
  - `format_probe_prompt(page_text: str) -> str`: the pinned template
    (mirrors `postprocess/tables.py::format_table_prompt` placement). Draft
    intent (⚠️ exact wording is settled in B4's hardware validation, then
    frozen): instruct that the task is **extracting bibliographic metadata**
    from the first page of a document; define *citable* as a self-contained
    scholarly work (paper, preprint, thesis, technical report) whose
    title/authors are identifiable in the text; require a single fenced JSON
    object `{"citable": bool, "title", "authors": [...], "year", "venue"}`
    omitting any field not visible in the text; **when unsure, answer
    `"citable": false`** (decision 1: abstain-by-default); no commentary.
    Input: page-1 text truncated to `figure.context_chars`-style cap
    (its own constant, ~3000 chars — front matter fits; do not reuse the
    figure knob).
  - `parse_probe_response(raw) -> ProbeResult | None`: tolerate a wrapping
    code fence (like `sanitize_table_output`); strict `json.loads`; type-check
    fields; anything malformed → `None`.
- `vlm/base.py` — `build_bibtex_probe_prompt(page_text) -> str` (delegates to
  `probe.format_probe_prompt`; cache-key material) and
  `probe_metadata(prompt) -> str | None` (text-only `client.chat()`, backend
  sampling + `chat_template_kwargs`; `None` on
  `finish_reason != "stop"`). `vlm/gemma.py` implements (thinking kwargs as
  for the other passes — confirmed or adjusted in B4 validation).
- `cache.py` — `make_bibtex_probe_key(...)` with `"kind": "bibtex-probe"`;
  value = the validated raw JSON string (existing `VlmCache` text store,
  `VLM_VALUE_SCHEMA` unchanged — same value shape).
- `pipeline.py` — `_bibtex_probe(cfg, pages, session) -> ProbeResult | None`:
  no-op unless `mode == "auto"`; cache-first; on miss `session.backend()`
  (lazy launch) → `probe_metadata` → parse → cache **only on success**; called
  inside the session block after `_vlm_describe` in both `_run_body` and
  `describe` (page-1 text = `pages[0].page_text`, post-table-refine). If no
  VLM is configured, return `None` with the table-pass-style warning.

**Verification**:
- [ ] `test_bibtex_probe.py` (new): prompt assembly (discriminator phrase
      present, truncation cap), JSON parse round-trip, fence tolerance,
      malformed/truncated → `None` and **not cached**, cache hit skips the
      server (hermetic cache), key disjointness vs figure/table keys.
- [ ] `test_pipeline_mocked.py`: mocked probe response dispatched on
      "bibliographic metadata"; probe runs inside the session (no second
      server launch).

**Exit gate**: probe result is deterministic, cached, and inert (nothing
consumes it yet) — safe to land independently.

---

### Phase B2 — Local best-effort entry

**Goal**: Assemble a clearly-marked `@misc` entry from `ProbeResult` metadata.
Pure function, fully offline.

**Changes**:
- `inscriber/bibtex/local.py` — `best_effort_bibtex(probe: ProbeResult) -> str | None`:
  - `None` when there's no usable `title` (an entry without a title is noise).
  - Citation key via the existing `generate_citation_key` (reuse, don't fork).
  - Fields: `title` (sanitized via `sanitize_bibtex_text`), `author = "A and B"`
    when authors were extracted, `year` when extracted, `eprint`/`howpublished`
    never guessed; venue (if extracted) goes in `note`, not `journal`
    (decision 4). **No `Unknown …` filler** (decision 5).
  - Header block (canonical, pin as `tests/fixtures/bibtex_best_effort.txt`):
    ```
    % NOTE: Best-effort entry generated from the document's own front matter
    % by inscriber (no citation database was consulted). Verify before use.
    %
    ```
  - Entry type: `@misc`.
- The paper2llm mock fallback (`mock_bibtex`) is untouched and remains the
  `on`-mode failure path (decision 6).

**Verification**:
- [ ] `test_bibtex.py` additions: full-metadata entry matches the canonical
      fixture; partial metadata omits fields; no-title → `None`; sanitization
      applied (special chars, curly quotes).

---

### Phase B3 — Source chain + arXiv-by-ID

**Goal**: The internal ordered chain `auto` walks, plus the authoritative arXiv
source (kills the Semantic Scholar 429 problem for the most common provenance
and partially delivers TODO's "alternate BibTeX sources").

**Changes**:
- `inscriber/bibtex/arxiv.py` —
  - `arxiv_id_from_url(url) -> str | None` (reuse the domain-handler arXiv
    regex's capture shape; handles `/abs/`, `/pdf/`, versioned IDs).
  - `arxiv_bibtex(arxiv_id, *, timeout) -> str | None`: GET
    `https://export.arxiv.org/api/query?id_list={id}` (Atom; stdlib
    `xml.etree`), format the standard arXiv `@misc` (`title`, `author`,
    `year`, `eprint={id}`, `archivePrefix={arXiv}`, `primaryClass`, `url`);
    `None` on any HTTP/parse failure (log + fall through, mirroring
    `search_semantic_scholar`'s degrade style).
- `inscriber/bibtex/chain.py` — `generate_bibtex_auto(probe, *, original_url,
  online_allowed, fallback_title) -> tuple[str | None, str]` returning
  `(bibtex, source_label)`:
  1. `probe.citable is False` (or probe `None` **and** no citable
     provenance) → `(None, "not-citable"/"unknown")`.
  2. A matching arXiv `original_url` counts as citable provenance even when
     the probe failed (provenance beats an absent probe, never a negative
     one — an explicit `"citable": false` wins, log the disagreement).
  3. `online_allowed` + arXiv ID → `arxiv_bibtex`.
  4. `online_allowed` → Semantic Scholar with query = `probe.title` or
     `fallback_title` (= `extract_title(full_md)`); **title validation
     compares against the same string used as the query** (the extracted
     title becomes the comparison base — avoids spurious `% WARNING` from a
     mangled OCR `# Title` heading). No mock on failure — fall through.
  5. `best_effort_bibtex(probe)`.
  6. Else `(None, reason)`.
- `pipeline.py` — `_bibtex_outputs` grows the `auto` branch: compute
  `online_allowed = tainted and not cfg.net.offline` (decision 2; `tainted`
  from `ResolvedInput.source`/bundle `source.source`), call the chain, write
  `{base}.bib` + the optional fenced prepend exactly as today. Every outcome
  is one INFO line: `BibTeX (auto): <wrote entry via arxiv | wrote best-effort
  entry | document judged not citable; skipping | skipped: <reason>>`.
  `mode == "on"` path untouched.

**Verification**:
- [ ] `test_bibtex_chain.py` (new, httpx mocked): chain order and every
      fall-through (arXiv ok / arXiv 500 → S2 / S2 empty → best-effort /
      nothing → skip); taint rule (local PDF never calls httpx in `auto`;
      URL-sourced does; `--offline` blocks both but still best-efforts);
      probe-says-no short-circuits even with an arXiv URL; describe-mode
      taint read from the bundle.
- [ ] `test_pipeline_mocked.py`: `run` and `ocr`→`describe` in `auto` produce
      the same `.bib` (probe cache keys shared via verbatim bundle text).

**Exit gate**: with `mode = "auto"` set explicitly, the whole feature works
end-to-end (mocked); default is still `off`.

---

### Phase B4 — Hardware validation, default flip, docs

**Goal**: Validate the probe prompt on real hardware, freeze it, flip the
default to `auto`, and land the full documentation ripple as one change.

**Work**:
- `dev/scripts/bibtex_probe_check.py` (dev-only, follows `m1b_check.py`
  patterns): run the probe against real Gemma on (a) the sample paper,
  (b) a known arXiv paper, (c) at least two non-citable PDFs (e.g. slides,
  an invoice-like document). Tune wording only as needed; record prompt,
  build, model, outcomes, and the abstain behavior in
  **`dev/docs/bibtex-probe-findings.md`**; then treat the prompt as pinned
  (table-pass discipline).
- Flip `BibtexConfig.mode` default → `"auto"`.
- Docs (same change — AGENTS.md rule):
  - `DESIGN.md` §12 rewritten around mode/chain/probe/taint (+ "Last
    updated" header note); §13.1/§13.2/§13.3 (`bibtex.mode` ↔
    `--bibtex-mode`, `--bibtex` alias); §20 privacy (the taint rule wording);
    §24 rows 14–16 note that parity applies to `on` mode; §22 future-work
    cross-reference for Crossref/source-axis.
  - `README.md`: options table, privacy section ("a local-PDF run sends
    nothing over the network, even with BibTeX auto on"), usage examples.
  - `config.example.toml`: `[bibtex] mode = "auto"` + comments.
  - `AGENTS.md`: add the probe to the mock-discrimination list.
  - `TODO.md`: mark the alternate-sources item partially done (arXiv ✓,
    Crossref remains); add the deferred refinements from **Scope/Out**.
- `dev/docs/integration-test.md`: add the auto-mode checks to the release
  checklist.

**Verification**:
- [ ] Findings recorded in `dev/docs/bibtex-probe-findings.md`; prompt frozen.
- [ ] Default-`auto` + local PDF: zero network calls (assert no httpx in
      mocked e2e); default-`auto` + URL input: online chain attempted.
- [ ] Full suite green on the flipped default (existing tests that assumed
      "no `.bib` unless `--bibtex`" updated deliberately, not incidentally).

**Exit gate**: README/DESIGN privacy promises and actual default behavior
agree; only then does the default flip ship.

---

## Risks

- **Probe quality on a small VLM** (false positives once `auto` is default).
  *Mitigation*: abstain-biased prompt; provenance short-circuit covers the
  high-value cases even when the probe is wrong; B4 explicitly tests
  non-citable documents before the flip; worst case is an unwanted-but-marked
  `.bib` file, never a wrong pipeline output.
- **Privacy regression by default-flip** (the headline promise). *Mitigation*:
  the taint rule is implemented and tested (B3) before the flip (B4); the
  no-network assertion is a CI test, not a doc claim.
- **JSON adherence from a thinking model** (fences, commentary). *Mitigation*:
  fence-tolerant strict parsing, `None`-on-anything-else, never cached, chain
  degrades to provenance/skip — same failure philosophy as the table pass.
- **arXiv API drift / flakiness**. *Mitigation*: it's one link in the chain;
  any failure falls through to Semantic Scholar, then best-effort.
- **Config migration breakage** (machine-local `config.toml` with `enabled`).
  *Mitigation*: legacy alias + deprecation warning in B0, tested.

## Open questions

None blocking. Two deliberately deferred to B4 evidence: the exact probe
prompt wording, and whether Gemma thinking should be enabled for the probe
(both are cache-key material either way, so changing them later busts only
probe entries).

---
**Plan is ready. Awaiting explicit go-ahead — execution starts at B0.**
