# PLAN: Cropped table input for the VLM restructuring pass

> **Status: EXECUTED** (2026-06-10). Archived design record of the implementation
> session; the authoritative spec is DESIGN §9.7. The real-hardware validation
> completed the same day — `dev/notes/2026-06-10-cropped-table-validation.md`
> (cropped prompt frozen; caption-carried matcher fix; digit-coverage guard).

Implements the unblocked TODO item ([TODO.md](../../TODO.md) §Table-restructuring):
crop the `table[[bbox]]` region from the page raster and send the crop to the
VLM instead of the whole page. Agreed design (discussion 2026-06-10):

- Content-based blob↔region matching; **page-fallback** for unmatched blobs,
  announced at **INFO** (not warning).
- Cache key = **(raster hash + bbox + padding)**, conditional payload fields so
  existing page-path keys are preserved.
- No new config knob; `TABLE_CROP_PADDING` module constant (0.02).
- New cropped prompt variant keeps the `"reconstructing ONE table"` mock
  discriminator; pending real-hardware validation (§9.7 pinned-prompt rule).
- **Reusable** validation script in `dev/scripts/`.

## Checklist

- [x] `models.py`: `TABLE_LABELS` frozenset
- [x] `tables.py`: cropped prompt template + `format_table_prompt(cropped=)`, `match_table_regions`, `TABLE_CROP_PADDING` + `MIN_TABLE_REGION_SPAN`
- [x] `pdf/crop.py`: shared `padded_pixel_box` helper + `crop_region_bytes`
- [x] `cache.py`: `make_table_key(crop_bbox=, crop_padding=)` — conditional payload (old keys preserved)
- [x] `vlm/base.py` + `vlm/gemma.py`: `build_table_prompt(..., cropped=)`; `restructure_table` param renamed `page_png`→`image_png`
- [x] `pipeline.py`: regions on `_Page` (run + describe); `_refine_tables` crop path, INFO fallback, workdir crop dump (`table_crops/`)
- [x] tests: unit (matching, crop, prompt variant, key pin) + integration updates (crop path, fallback, mixed page, describe-from-bundle, run↔describe key sharing) — full suite + ruff green
- [x] `dev/scripts/table_crop_check.py` — reusable validation harness (crops + page-vs-crop VLM comparison)
- [x] docs: DESIGN §9.7 + §3 diagram + header changelog, TODO.md item → 4-step validation checklist, README tables bullet, config.example.toml [table] comment
- [x] `pytest` (304 passed) + `ruff check` clean
- [x] `/doublecheck` verification — two Opus reviews, no CRITICAL/IMPORTANT code findings

## Success criteria

- All mocked tests green; no model-facing behavior asserted beyond the pinned discriminator.
- Page-path cache keys byte-identical to pre-change (pinned by test).
- Unmatched/degenerate regions fall back to the whole-page path (current behavior) with an INFO line.
- Validation runbook ready for the maintainer's hardware pass.

## Completion summary

All criteria met. Post-review fixes applied: exact-match preference in
`match_table_regions` (closes a latent substring-steal footgun; pinned by
test), DESIGN §8.5 key-wording precision, DESIGN §17 test-list refresh, a
probe-divergence comment in the harness. The cropped **prompt** ships
deliberately unvalidated behind the always-safe fallback; the prompt pin
happens after the `table_crop_check.py` hardware run (TODO.md checklist:
crop completeness → page-vs-crop diff → `table[[bbox]]` fixture capture →
dated note + DESIGN status flip).
