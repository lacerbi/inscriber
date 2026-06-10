# dev/ — developer-only material (not user-facing)

- **`scripts/`** — one-off development and verification scripts: the M1a/M1b
  real-hardware spikes the design was pinned on, and feature-verification probes
  (e.g. `verify_thinking_spike.py` for Gemma 4 thinking activation). These are
  not part of the installed package and may bit-rot; the durable conclusions
  live in `docs/` here and in the root `DESIGN.md`.
- **`docs/`** — development notes and findings: `M1A-FINDINGS.md` (the
  empirically-confirmed llama.cpp facts the OCR backend is pinned to),
  `table-reconstruction-findings.md` (the experiment behind the table
  -restructuring pass, DESIGN §9.7), `integration-test.md` (manual release
  validation against real models).
- **`plans/`** — archived feature plans, kept after execution as design
  records; the durable spec for shipped work lives in the root `DESIGN.md`.

User-facing documentation is the root `README.md`; the authoritative
specification is the root `DESIGN.md` (with `PLAN-inscriber-v1.md` as the build
roadmap).
