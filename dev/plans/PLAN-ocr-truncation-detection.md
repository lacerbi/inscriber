# PLAN: OCR loop/truncation detection (cache-with-flag)

> **Status: EXECUTED** (2026-06-10). Archived design record of the implementation
> session; the authoritative spec is DESIGN §2.2/§8.5/§8.6/§9.6/§16, the decision
> record is the addendum in `dev/notes/2026-06-10-equation-fidelity-findings.md`,
> and the follow-up (loop-breaking retry) is tracked in `TODO.md`.

Implements the `TODO.md` item "OCR loop/truncation detection", with the cache
policy agreed in discussion (2026-06-10), which **deviates from the TODO's
"don't cache"**:

- **Cache-with-flag**: the OCR key already contains every output-determining
  knob (model/mmproj/build/resolution/render-px/prompt/sampling incl. the
  pinned `max_tokens` cap), so not caching has zero recovery value — recompute
  reproduces the same loop. Cache the best-effort page marked `truncated: true`
  and **re-warn on every cache hit** (never *silently* persist damage).
- The table pass's don't-cache is NOT mirrored: its key deliberately excludes
  `ctx_size`, so a bigger `--ctx` can fix a truncated table under the same key.
  No such escape hatch exists for OCR. Figure-pass precedent: truncated
  descriptions are already cached, marked `[...]`.
- Truncation = `finish_reason != "stop"` (string only; `None`/missing = unknown
  = fine — keeps mocks and mtmd-cli unaffected).
- Bundle manifest records per-page `truncated` (additive, `bundle_schema` 1);
  no `describe`-time warning.
- Known limitation (document, don't chase): a loop that self-terminates under
  the cap yields `finish_reason: "stop"` and is undetectable here.
- Deferred follow-up to TODO.md: loop-breaking retry (rung A: per-request
  stronger-DRY/seed retry; rung B: prefix-prefill + deflection at loop point).

## Checklist

- [x] `models.py`: `OcrPageResult.truncated: bool = False`
- [x] `serialize.py`: `truncated` written only when True; read defaults False
- [x] `ocr/base.py`: `last_finish_reason` (+`last_completion_tokens`) mirrored on `HttpInferencer`; `None` on `MtmdCliInferencer`; `inference_truncated` helper
- [x] `ocr/deepseek.py`: `ocr_page` sets `truncated` from the inferencer
- [x] `pipeline.py`: `_warn_truncated_page` on compute AND on cache hit; `run_ocr` carries the flag into bundle page results
- [x] `bundle.py`: no change needed — `write_bundle` spreads `ocr_page_result_to_dict` (flag flows via serialize.py); read already tolerant
- [x] tests: new `test_ocr_truncation.py` (15 tests — inferencer mirror, backend flag incl. unknown→False, serialize roundtrip, pipeline compute+hit warnings, cache value flagged, bundle manifest flag); full suite + ruff clean
- [x] docs: DESIGN §2.2 known-gap → implemented; §8.5 manifest field; §8.6 cache-policy nuance; §16; header changelog; AGENTS.md invariant nuance; status addendum in `dev/notes/2026-06-10-equation-fidelity-findings.md`
- [x] TODO.md: done item removed (its section was then empty — removed); deferred loop-breaking retry item added under Planned features (rungs A/B, repaired-page key question, self-terminating-loop limitation)
- [x] `pytest` + `ruff check` clean
- [x] `/doublecheck` (2 Opus reviewers: code+tests, docs) — verdict: correct/complete, no bugs; cache-key claim verified against `make_ocr_key`; no interaction with the concurrent table-matcher change

## Success criteria

- Truncated page: parsed result still used in output, cached with flag, warning
  on compute and on every later cache hit; non-truncated paths byte-identical.
- All existing mocks (no `finish_reason`) stay green without modification.
- Manifest gains `"truncated": true` only on affected pages; old bundles read fine.

## Completion summary

All criteria met. Post-review fixes applied: `ocr/glm.py` also sets `truncated`
(latent gap — the experimental backend never flagged); the cache-hit test now
asserts the flagged page's text reaches the output on run 2; DESIGN §9.6 gained
the per-operation truncation policy (figure description cached with `[...]`
marker vs table/probe never cached — completes the three-way picture next to
the cached-flagged OCR page). Full suite + ruff re-run clean after fixes.
