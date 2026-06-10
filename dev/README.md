# dev/ — developer-only material (not user-facing)

- **`scripts/`** — one-off development and verification scripts: the
  real-hardware spikes the design was pinned on and feature-verification
  probes. These are not part of the installed package and may bit-rot; the
  durable conclusions live in `notes/` here and in the root `DESIGN.md`.
- **`notes/`** — dated lab notes (`YYYY-MM-DD-name.md`): the empirical
  evidence records behind DESIGN's pinned behavior. Each opens with a
  date/status line; they are point-in-time records — when later work
  supersedes one, update its status line or add an addendum rather than
  rewriting it.
- **`plans/`** — archived feature plans, kept after execution as design
  records; the durable spec for shipped work lives in the root `DESIGN.md`.
- **`integration-test.md`** — the manual release-validation checklist against
  real models (a living procedure, not a dated note).

User-facing documentation is the root `README.md`; the authoritative
specification is the root `DESIGN.md` (with `PLAN-inscriber-v1.md` as the build
roadmap).
