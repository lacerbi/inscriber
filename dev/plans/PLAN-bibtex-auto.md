# Plan: BibTeX `auto` mode (classify → source chain → best-effort)

Created: 2026-06-10
Status: ✅ IMPLEMENTED — all phases B0–B4 landed 2026-06-10 (see Execution
checklist below); probe validated on real hardware
(`dev/notes/2026-06-10-bibtex-probe-findings.md`), default flipped to `auto`.
Revised: 2026-06-10 — pre-implementation review pass (codebase claims verified
against source; amendments folded in: provenance-first chain with probe skip
(decision 7), published-version-first by-ID lookup (decision 8), text-only
mock surface, `_bibtex_outputs` signature growth + `Bundle.original_url`
wiring, doc-ripple completeness, regex/locator corrections).

## Summary

Not every document `inscriber` processes is a citable paper, and today BibTeX is
a blunt opt-in bool (`bibtex.enabled`) that always goes online. This plan adds a
**`bibtex.mode = "auto"`**: the pipeline decides whether the document is
*citable* (provenance heuristics first, a single cached LLM probe as the
tiebreaker), then produces an entry through an internal **ordered source chain**
— Semantic Scholar **by arXiv ID** (authoritative match; surfaces the published
version when one exists — a preprint's final BibTeX is often not arXiv) →
arXiv export API (`@misc` availability fallback) → Semantic Scholar title
search (today's path) → **local best-effort** (an entry assembled from
LLM-extracted
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
   *abstaining* — with a default-on feature, a false positive (an unwanted
   `.bib` file) is worse than a false negative. Every
   "judged not citable" decision is a visible log line, never silent.
2. **Network intent comes from the existing knob, nothing new.** The project's
   privacy stance is that the *models and documents* are local — documents
   never go to a cloud model; ordinary network use (downloading a PDF, sending
   a title string to a citation API) is not the sensitive part. `net.offline`
   (`--offline`, network **on** by default) already expresses the user's
   intent, so `auto` reads exactly that: network on → the full source chain,
   online lookups included; `--offline` → probe + local best-effort only (the
   probe and best-effort are loopback-local and stay available, per DESIGN
   §15's "`--offline` does not gate the local servers"). If the network is
   "on" but actually unreachable, each online source simply fails and the
   chain falls through to best-effort — no separate detection. Docs (DESIGN
   §20 / README) are updated in B4 to state that default-`auto` BibTeX may
   consult citation APIs with the **extracted title only** — the document
   itself never leaves the machine.
3. **No `--bibtex-source` flag yet.** The source chain is an internal,
   code-level seam (so a source-selection axis is purely additive later);
   `auto + --offline` already covers "local only".
4. **Humble entry types.** Best-effort emits `@misc` only (no venue-type
   guessing); preprint-shaped entries use the standard `@misc` + `eprint`
   shape; both Semantic Scholar paths (title search, and by-ID when a
   publication venue is known — decision 8) keep the current `@article`. Type
   inference is a future refinement.
5. **Transcription, not recall.** The probe prompt instructs the model to
   extract only fields *visible in the supplied text* and omit the rest —
   same philosophy as the table pass ("structuring, not re-OCR"). Best-effort
   entries never get `Unknown Journal`-style filler; absent fields are absent.
6. **`on` mode is frozen.** Explicit `--bibtex` keeps today's exact behavior
   (extract `# Title` → Semantic Scholar → mock fallback, no LLM involved), so
   the documented paper2llm ports (DESIGN §24 rows 14–16) stay byte-stable and
   `on` keeps working with no VLM configured.
7. **Provenance first; the probe never vetoes it** (added in the review pass).
   A URL matching **any of the seven** recognized paper repositories (not just
   arXiv) counts as citable provenance and settles citability outright — an
   explicit `"citable": false` from the probe is logged as a disagreement,
   nothing more. Rationale: the probe is a small quantized VLM reading
   possibly-garbled OCR, while an ID-based lookup is authoritative — under
   provenance
   the abstain-bias cost asymmetry of decision 1 inverts (a skipped entry for
   a paper the user fetched from arXiv is a real loss; a false positive is
   impossible). Moreover, when the network is on and provenance exists the
   probe is **skipped entirely** (no VLM call): the by-ID/S2 sources don't
   need its output. The probe governs citability only for provenance-less
   documents, and supplies metadata for offline best-effort entries.
8. **Preprint provenance ≠ preprint citation** (added in the review pass). An
   arXiv URL proves the document is citable, but the right entry is often NOT
   the arXiv `@misc` — many preprints are later published at a venue. So the
   by-ID step queries **Semantic Scholar by arXiv ID** (exact match, no title
   fuzziness; its record carries the publication venue when one exists):
   venue known → the published entry (today's `@article` shape, decision 4);
   no venue → the `@misc` + `eprint` preprint shape. The arXiv export API
   becomes the **availability fallback** (S2 down/429/no record) — it stays
   authoritative for identification but can never know about later
   publication. bioRxiv/medRxiv get the same benefit through the title-search
   step, which naturally surfaces the published version; S2 by-DOI lookup for
   them (their URLs embed the `10.1101` DOI) is a future refinement
   (Scope/Out).

## Scope

- **In**: `bibtex.mode` tri-state config/CLI; the LLM citability+metadata probe
  (text-only VLM call, cached, pinned prompt); local best-effort entry
  assembly; the by-ID sources (S2-by-arXiv-ID preferring the published
  version, arXiv export API fallback); the `auto` orchestration (offline-knob
  intent, degrade paths); default flip + full docs ripple; mocked tests
  throughout.
- **Out (future refinements — add to `TODO.md` when this lands)**: a
  `--bibtex-source` CLI axis; Crossref as an additional source; structure-based
  citability heuristics (references-section detection) beyond provenance;
  entry-type inference; extraction from the paper's own reference list;
  S2 by-DOI lookup for bioRxiv/medRxiv provenance (decision 8); adding
  `eprint` fields to published entries.

## Architecture (where this lives)

```
inscriber/bibtex/
├── semantic_scholar.py   # + lookup_arxiv(id) by-ID GET; title search unchanged
├── probe.py              # NEW: probe prompt template + JSON extraction + ProbeResult
├── arxiv.py              # NEW: arXiv ID from URL; export.arxiv.org → @misc (fallback)
├── local.py              # NEW: best-effort @misc from ProbeResult metadata
└── chain.py              # NEW: auto orchestration (citability → source chain)
```

Pipeline wiring (both `_run_body` and `describe`): a new `_bibtex_probe(...)`
step runs **inside the open `_VlmSession` block** (after `_vlm_describe`) — the
session is closed (`finally: session.close()`) before `_bibtex_outputs` runs,
which is why the probe is eager. The session block is entered even in a
no-figures, no-table-refine run (the `_VlmSession` object is always
constructed; the server launches lazily on first `session.backend()`), so the
probe call site always executes and shares the already-resident VLM server
instead of forcing a relaunch at step 9. The `ProbeResult` is threaded into
`_bibtex_outputs(...)`, whose signature **grows two parameters** —
`probe: ProbeResult | None` and `original_url: str | None` (`_run_body`
threads `resolved.original_url`; `describe` threads the manifest's
`source.original_url` via a new `Bundle.original_url` accessor — the manifest
already stores it, but nothing reads it today). It still never fails the run.
Cache-first like every VLM pass: a fully-cached document still
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
  require re-validation on real hardware recorded in `dev/notes/`.
- Cache: shared `VlmCache`, new key fn with `"kind": "bibtex-probe"`
  discriminator (disjoint from figure/table payloads by construction, like
  `make_table_key`). Key material: full assembled prompt (page-1 text
  embedded), backend name, model/mmproj/server identities, sampling,
  `chat_template_kwargs`. **Never cache a failed/truncated/unparseable probe.**
  Embedding the post-refine page text couples the probe key to table-pass
  settings (toggling `[table].refine` or changing its model busts probe
  entries) — intended, not a bug: the key is the exact model input.
- Tests mock at the chat-client boundary — but the probe is the project's
  first **text-only** inference call, and today's harnesses fake
  `ChatClient.chat_image` only: an unmocked probe would hit the fake server
  URL and raise. So the mock surface gains a `ChatClient.chat` fake alongside
  `chat_image`, and every `_mock_inference` helper gains a probe branch keyed
  on the pinned discriminator phrase **"bibliographic metadata"** (existing
  dispatch: `"<|grounding|>"` / `"Convert the document to markdown"` for OCR,
  `"reconstructing ONE table"` for tables, figure as the catch-all — without
  the new branch, probe prompts would silently get the figure response).
  There is no `conftest.py`: new test files copy the per-file `hermetic_cache`
  fixture. No live network in CI (mock `httpx`).
- Resilience (DESIGN §16): BibTeX never fails the run — any probe/source/chain
  error degrades to the next step or to a logged skip.
- Text files `encoding="utf-8", newline="\n"`; `pathlib`; no new heavy deps
  (arXiv's Atom response is parsed with stdlib `xml.etree`).

---

## Execution checklist (implementation tracker, 2026-06-10)

All phases land as ONE change, so the per-phase DESIGN/README edits are folded
into the final B4 docs ripple (the same-change docs rule is satisfied by the
combined landing). Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked.

- [x] **B0** — `bibtex.mode` tri-state: models/config/cli/pipeline gate + tests
      (suite green; legacy alias + precedence covered)
- [x] **B1** — probe: `bibtex/probe.py`, backend methods, cache key, pipeline
      wiring, `ChatClient.chat` mock surface, `test_bibtex_probe.py` (suite
      green; AGENTS.md mock list updated)
- [x] **B2** — best-effort entry: `bibtex/local.py` + canonical fixture + tests
- [x] **B3** — chain + by-ID sources: `arxiv.py`, `lookup_arxiv`, `chain.py`,
      provenance skip, `Bundle.original_url`, `_bibtex_outputs` growth,
      `test_bibtex_chain.py` (suite green, 282 tests)
- [x] **B4a** — `dev/scripts/bibtex_probe_check.py` + real-hardware validation
      → `dev/notes/2026-06-10-bibtex-probe-findings.md`; prompt frozen (4/4 PASS on build
      9587 + Gemma E4B QAT, zero tuning; fence tolerance proved necessary)
- [x] **B4b** — default flip to `auto` + deliberate test updates (suite green,
      284 tests; default-auto e2e tests added)
- [x] **B4c** — docs ripple: DESIGN (header, §1.1, §3 diagram, §4 tree, §6
      privacy note, §8.5, §12 full rewrite, §13.1–13.3, §14, §15, §16, §17,
      §18.1, §20, §22.2, §24 rows 14–15b, §25), README ("What it does",
      outputs, options table, privacy), config.example.toml ([bibtex]+[net]),
      AGENTS.md (pipeline line + mock list), TODO.md (auto item closed;
      alternate-sources arXiv half ✓), dev/integration-test.md (steps
      6–7 + auto checks)
- [x] Full suite green (286 tests) + `ruff check` clean; live CLI smoke
      confirms the legacy-config deprecation alias on the maintainer's
      machine-local config.toml
- [x] /doublecheck verification pass — two Opus reviewers (code + docs): plan
      conformance (decisions 1–8) and the full docs ripple verified; the one
      should-fix (an unwrapped chain call could violate "BibTeX never fails
      the run" on a malformed HTTP-200 body) fixed with a §16 guard + two new
      tests (auto + `--bibtex-in-doc` injection; never-fails guard)

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
- `DESIGN.md` — same change (AGENTS.md docs rule, not deferred to B4):
  §13.1 config block, §13.2 CLI surface, and the §13.3 mapping table updated
  for `mode` / `--bibtex-mode` / the legacy alias; §12 gains a one-line
  tri-state note (`auto`: designed, not yet wired — the full §12 rewrite
  lands in B4).

**Verification**:
- [x] `test_config.py`: tri-state precedence (CLI > config > default), legacy
      `enabled` alias + deprecation warning, invalid value → `ConfigError`.
- [x] `test_cli.py`: `--bibtex-mode` wiring and its precedence over the
      `--bibtex` alias.
- [x] `test_bibtex.py` / `test_pipeline_mocked.py`: existing `--bibtex` runs
      produce byte-identical outputs (mock fixture untouched).
- [x] `ruff check` clean; full suite green.

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
    frozen; the discriminator phrase "bibliographic metadata" must survive
    that tuning — tests dispatch on it): instruct that the task is
    **extracting bibliographic metadata**
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
  `probe_metadata(prompt) -> str | None` (text-only `client.chat()` with a
  hand-built single-user-message list — note `chat()` pins a `temperature: 0`
  baseline before applying `sampling` overrides; backend
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
  `describe`. Input is `pages[0].page_text` — the placeholder-stripped
  `_Page.page_text` (not `.markdown`), post-table-refine, populated
  identically in both paths. `pages[0]` is the first **processed** page: under
  a `--pages` range that excludes page 1 the probe sees body text and will
  typically abstain — acceptable; log which page number fed the probe. If no
  VLM is configured, return `None` with the table-pass-style warning. (B3
  adds the decision-7 provenance skip; until then the probe runs for every
  `auto` document.)
- Test harness — add the text-only mock surface (see cross-cutting
  conventions): a `ChatClient.chat` fake plus the "bibliographic metadata"
  dispatch branch in the `_mock_inference` helpers of
  `test_pipeline_mocked.py` and `test_tables.py`.
- `AGENTS.md` — add the probe phrase to the mock-discrimination list **in
  this phase** (the discriminator ships here, not in B4).

**Verification**:
- [x] `test_bibtex_probe.py` (new): prompt assembly (discriminator phrase
      present, truncation cap), JSON parse round-trip, fence tolerance,
      malformed/truncated → `None` and **not cached**, cache hit skips the
      server (hermetic cache), key disjointness vs figure/table keys.
- [x] `test_pipeline_mocked.py`: mocked probe response dispatched on
      "bibliographic metadata" via the new `ChatClient.chat` fake; probe runs
      inside the session (no second server launch).
- [x] Existing mocked suites stay green with the added `chat` fake (nothing
      reaches it outside `auto` mode).

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
- [x] `test_bibtex.py` additions: full-metadata entry matches the canonical
      fixture; partial metadata omits fields; no-title → `None`; sanitization
      applied (special chars, curly quotes).

---

### Phase B3 — Source chain + the by-ID sources

**Goal**: The internal ordered chain `auto` walks, plus the by-ID sources for
arXiv provenance: Semantic Scholar by arXiv ID first (exact match; prefers the
published version — decision 8), arXiv export API as the availability
fallback (so an S2 429 no longer kills the lookup for the most common
provenance; partially delivers TODO's "alternate BibTeX sources").

**Changes**:
- `inscriber/bibtex/arxiv.py` —
  - `arxiv_id_from_url(url) -> str | None`: apply the arXiv domain handler's
    **filename-rule** pattern shape (`([\w.-]+/?\d+|\d+\.\d+)` — it preserves
    `v2`-style version suffixes and old-style `cs.AI/0301001` IDs) to
    `urlparse(url).path`. The handler's `url_patterns` detection regex is NOT
    the one to reuse (its `\d+\.\d+` stops before the version). Handles
    `/abs/`, `/pdf/`, `/html/`.
  - `arxiv_bibtex(arxiv_id, *, timeout) -> str | None`: GET
    `https://export.arxiv.org/api/query?id_list={id}` (Atom; stdlib
    `xml.etree`), format the standard arXiv `@misc` (`title`, `author`,
    `year`, `eprint={id}`, `archivePrefix={arXiv}`, `primaryClass`, `url`);
    `None` on any HTTP/parse failure (log + fall through, mirroring
    `search_semantic_scholar`'s degrade style). Per decision 8 this is the
    **fallback** source — the export API can never know about venue
    publication.
- `inscriber/bibtex/semantic_scholar.py` — new `lookup_arxiv(arxiv_id, *,
  timeout) -> dict | None`: GET
  `https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}` (same `fields`
  as the title search; strip any `vN` version suffix — S2 indexes the base
  ID); same never-raise degrade style as `search_semantic_scholar`. The
  title-search function and the frozen `on`-mode path are untouched.
- `inscriber/bibtex/chain.py` — `generate_bibtex_auto(probe, *, original_url,
  online_allowed, fallback_title) -> tuple[str | None, str]` returning
  `(bibtex, source_label)`:
  1. **Citable provenance** := `original_url` matches **any of the seven**
     domain-handler configs (reuse the handler list's find-first
     `can_handle` — no new regexes); an arXiv match additionally yields the
     ID. Provenance settles citability (decision 7): the probe never vetoes
     it — an explicit `"citable": false` against provenance is logged as a
     disagreement, nothing more.
  2. No provenance **and** (probe `None` or `citable: false`) →
     `(None, "not-citable"/"unknown")` (abstain, decision 1).
  3. `online_allowed` + arXiv ID (decision 8) → `lookup_arxiv` on Semantic
     Scholar: a record with a real publication venue → the published entry
     (the existing `@article` shape); no venue (or an "arXiv.org"-style
     venue) → the `@misc` + `eprint` preprint shape formatted from the S2
     data. No title validation on this path — the ID match is exact. S2
     unavailable or no record → `arxiv_bibtex` (export API `@misc`) as the
     availability fallback.
  4. `online_allowed` → Semantic Scholar with query = `probe.title` or
     `fallback_title` (= `extract_title(full_md)` — it lives in
     `postprocess/splitter.py`, already imported by the pipeline); **title
     validation compares against the same string used as the query** (the
     extracted title becomes the comparison base — avoids spurious
     `% WARNING` from a mangled OCR `# Title` heading). No mock on failure —
     fall through.
  5. `best_effort_bibtex(probe)` — `None` when the probe was skipped under
     decision 7 or failed, so an online provenance run where both arXiv and
     S2 fail ends with a logged skip (accepted trade-off).
  6. Else `(None, reason)`.
- `pipeline.py` —
  - `_bibtex_probe` gains an `original_url` parameter and the decision-7
    skip: when `online_allowed` and citable provenance exists, skip the
    probe entirely (no VLM call — the by-ID/S2 sources don't need it; log
    `"provenance recognized; probe skipped"`). Offline runs still probe
    even with provenance (best-effort needs the metadata).
  - `_bibtex_outputs` grows the `auto` branch and the two new parameters
    (`probe`, `original_url` — see Architecture): `_run_body` passes
    `resolved.original_url`; `describe` passes the new
    `Bundle.original_url` accessor added to `bundle.py` (the manifest
    already stores `source.original_url`; today nothing reads it).
    `online_allowed = not cfg.net.offline` (decision 2 — the existing knob
    is the intent signal), call the chain, write `{base}.bib` + the
    optional fenced prepend exactly as today. Every outcome is one INFO
    line: `BibTeX (auto): <wrote entry via {s2-arxiv-id | arxiv-export |
    s2-title | best-effort} | document judged not citable; skipping |
    skipped: <reason>>`.
    `mode == "on"` path untouched.

**Verification**:
- [x] `test_bibtex_chain.py` (new, httpx mocked): chain order and every
      fall-through (S2-by-ID with venue → published entry / S2-by-ID without
      venue → `@misc` + `eprint` / S2-by-ID 429 or no record → arXiv export
      fallback / no arXiv ID → S2 title search / S2 empty → best-effort /
      nothing → skip); network-unreachable mid-chain (httpx raises) degrades
      gracefully, never fails the run; `--offline` makes no httpx call but
      still best-efforts; **online + arXiv URL never calls the probe** (VLM
      untouched, decision 7); **offline + provenance + probe
      `citable: false` still writes the best-effort entry and logs the
      disagreement**; non-arXiv provenance (a bioRxiv URL) + probe `None` +
      online → S2 with `fallback_title`; no provenance + probe-says-no →
      skip; describe-mode provenance read from the bundle's `original_url`
      (new accessor).
- [x] `run` and `ocr`→`describe` in `auto` produce the same `.bib` (probe
      cache keys shared via verbatim bundle text) — landed in
      `test_bibtex_chain.py`.

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
  an invoice-like document). Tune wording only as needed — the pinned
  discriminator phrase "bibliographic metadata" must survive tuning (or the
  test dispatch moves with it in the same change); record prompt,
  build, model, outcomes, and the abstain behavior in
  **`dev/notes/2026-06-10-bibtex-probe-findings.md`**; then treat the prompt as pinned
  (table-pass discipline).
- Flip `BibtexConfig.mode` default → `"auto"`.
- Docs (same change — AGENTS.md rule). The "only URL input and BibTeX touch
  the network" statement lives in MANY places, not just §20 — the flip must
  catch every one. The reframing everywhere: the local guarantee is about
  **documents and models** (nothing ever goes to a cloud model);
  default-`auto` BibTeX may consult citation APIs with the extracted title
  only, and `--offline` disables that.
  - `DESIGN.md`: §12 rewritten around mode/chain/probe (+ "Last updated"
    header note); §13.1/§13.2/§13.3 finalized (B0 already made them
    mode-aware); privacy/network statements reworded everywhere they
    appear — §20, **§5.3** (the canonical "URL input and BibTeX are the
    _only_ features that touch the network" privacy note), the §1.1 intro
    ("this single feature requires network access"), the §3
    pipeline-diagram label ("BibTeX (optional, online)"), §15
    ("`--offline` only disables URL input and BibTeX" — extend: the probe
    and best-effort entry stay available under `--offline`), §16 (BibTeX
    resilience now means chain fall-through), §8.5 ("what `describe`
    honors": `[bibtex].*` now implies the probe may use the VLM at
    describe time); §24 rows 14–16 note that parity applies to `on` mode;
    §22 future-work cross-reference for Crossref/source-axis.
  - `README.md`: options table, privacy section, usage examples, **and the
    "What it does" BibTeX bullet** ("the one feature besides URL input that
    touches the network" is no longer accurate) — same reframing.
  - `config.example.toml`: `[bibtex] mode = "auto"` + comments, and the
    `[net] offline` comment (it says it disables "BibTeX lookup").
  - `AGENTS.md`: the pipeline-description line ("optional BibTeX") gets the
    auto-mode phrasing. (The mock-discrimination entry already landed in
    B1.)
  - `TODO.md`: close the **"BibTeX `auto` mode"** planned-feature item
    itself; mark the alternate-sources item partially done (arXiv ✓,
    Crossref + reference-list extraction remain); add the deferred
    refinements from **Scope/Out**.
- `dev/integration-test.md`: add the auto-mode checks to the release
  checklist; update step 6 (`--no-figures --offline` may now produce a
  best-effort `.bib` under the flipped default) and step 7 (`--bibtex`
  wording).

**Verification**:
- [x] Findings recorded in `dev/notes/2026-06-10-bibtex-probe-findings.md`; prompt frozen
      (4/4 PASS on build 9587 + Gemma E4B QAT, zero tuning needed).
- [x] Default-`auto` e2e (mocked): citable doc → `.bib` written via the
      chain; non-citable probe → skip with the INFO line; `--offline` →
      best-effort only, no httpx call.
- [x] Full suite green on the flipped default (existing tests that assumed
      "no `.bib` unless `--bibtex`" updated deliberately, not incidentally).

**Exit gate**: README/DESIGN describe the actual default behavior (documents
and models stay local; citation lookups are title-only and `--offline`-gated);
only then does the default flip ship.

---

## Risks

- **Probe quality on a small VLM** (false positives once `auto` is default).
  *Mitigation*: abstain-biased prompt; provenance settles the high-value
  cases outright — online + provenance never even consults the probe
  (decision 7); B4 explicitly tests
  non-citable documents before the flip; worst case is an unwanted-but-marked
  `.bib` file, never a wrong pipeline output.
- **Doc drift on the default flip** (the "only URL input and opt-in
  `--bibtex` touch the network" statement appears in DESIGN §5.3, §20, §1.1,
  §3, §15 and the README intro bullet). *Mitigation*: the B4
  docs reframing (documents/models local; title-only lookups, `--offline`
  gates them) ships in the same change as the flip, against the full list
  enumerated in B4 — the exit gate.
- **JSON adherence from a thinking model** (fences, commentary). *Mitigation*:
  fence-tolerant strict parsing, `None`-on-anything-else, never cached, chain
  degrades to provenance/skip — same failure philosophy as the table pass.
- **arXiv / S2 API drift or flakiness**. *Mitigation*: the by-ID lookup and
  the export API back each other up (an S2 429 falls to arXiv; an arXiv
  outage only matters when S2 already failed); anything else falls through
  the chain, never failing the run.
- **Config migration breakage** (machine-local `config.toml` with `enabled`).
  *Mitigation*: legacy alias + deprecation warning in B0, tested.

## Open questions

None blocking. Two deliberately deferred to B4 evidence: the exact probe
prompt wording, and whether Gemma thinking should be enabled for the probe
(both are cache-key material either way, so changing them later busts only
probe entries).

---
**Plan executed in full (2026-06-10).** All phases B0–B4 landed as one change:
suite green (284 tests), `ruff check` clean, the probe validated on real
hardware (4/4, `dev/notes/2026-06-10-bibtex-probe-findings.md`) and frozen, the default
flipped to `auto`, and the full docs ripple applied.
