"""Shared fixtures (AGENTS.md "Testing conventions").

``hermetic_cache`` was copy-pasted across six test files before 2026-06-11
(pre-release review E1): a change to the hermeticity boundary had to be applied
in six places or one file would silently touch the real platformdirs cache.
It lives here now — pytest resolves it in every test file automatically.

The near-duplicated per-file helpers (``_dummy_models``, ``_mock_inference``,
cfg builders) deliberately stay file-local: the ``_mock_inference`` variants
differ by design (each file discriminates/records different call kinds), and
``_dummy_models`` drift is harmless (dummy GGUF bytes) — only the hermeticity
boundary is load-bearing enough to centralize.
"""

from __future__ import annotations

import pytest

from inscriber import cache as cache_mod
from inscriber import pipeline


@pytest.fixture
def hermetic_cache(tmp_path, monkeypatch):
    """Keep cache + model-hash side effects inside tmp — never touch the real
    platformdirs cache (AGENTS.md). Cache keys also probe the llama.cpp build
    identity; no real binary exists in tests, so that probe is pinned too."""
    monkeypatch.setattr(cache_mod, "default_cache_dir", lambda: tmp_path / "ocrcache")
    monkeypatch.setattr(cache_mod, "default_vlm_cache_dir", lambda: tmp_path / "vlmcache")
    monkeypatch.setattr(pipeline, "llama_build_identity", lambda *a, **k: "version: 9587 (test)")
