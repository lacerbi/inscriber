"""OCR cache: content-addressed per-page memoization (DESIGN §8.6).

Per-page OCR is the expensive step. The cache key includes everything that can
change the output (model + projector identities, the llama.cpp build identity,
resolution + render size, prompt, sampling). Entries are written **per page** so an
interrupted run resumes from the last completed page.

``--refresh`` (``refresh=True``) ignores + recomputes + overwrites; ``--no-cache``
(``enabled=False``) neither reads nor writes (DESIGN §8.6, §13).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import platformdirs

from inscriber.logging import get_logger
from inscriber.models import OcrPageResult
from inscriber.serialize import ocr_page_result_from_dict, ocr_page_result_to_dict

logger = get_logger()

# Bump if the stored value shape changes, so a future backend's richer result can't
# collide with a v1 entry (DESIGN §8.6).
OCR_VALUE_SCHEMA = 1
# 2: value field renamed "description" → "text" (the store holds restructured tables
# too). Rode the server-identity key change, which orphaned all v1 entries anyway.
VLM_VALUE_SCHEMA = 2

# In-process memoization of expensive file content hashes (keyed by path+size+mtime).
_HASH_MEM: dict[tuple[str, int, int], str] = {}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def default_cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("inscriber")) / "ocr"


def default_vlm_cache_dir() -> Path:
    return Path(platformdirs.user_cache_dir("inscriber")) / "vlm"


def file_identity(path: str, *, hash_disk_cache: Path | None = None) -> str:
    """``name:size:content-hash`` for a model/mmproj file (DESIGN §8.6).

    The content hash (expensive for multi-GB GGUFs) is cached by path+size+mtime —
    in-process always, and on disk when ``hash_disk_cache`` is given — so it is
    computed once. Keying on content (not bare mtime) means a re-download/copy that
    preserves content but changes mtime does NOT bust the OCR cache spuriously.
    """
    p = Path(path).expanduser()
    st = p.stat()
    mem_key = (str(p), st.st_size, st.st_mtime_ns)
    if mem_key in _HASH_MEM:
        return f"{p.name}:{st.st_size}:{_HASH_MEM[mem_key]}"

    disk_key = f"{p}|{st.st_size}|{st.st_mtime_ns}"
    disk_map: dict[str, str] = {}
    if hash_disk_cache and hash_disk_cache.exists():
        try:
            disk_map = json.loads(hash_disk_cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            disk_map = {}
    if disk_key in disk_map:
        _HASH_MEM[mem_key] = disk_map[disk_key]
        return f"{p.name}:{st.st_size}:{disk_map[disk_key]}"

    content_hash = _sha256_file(p)
    _HASH_MEM[mem_key] = content_hash
    if hash_disk_cache is not None:
        disk_map[disk_key] = content_hash
        hash_disk_cache.parent.mkdir(parents=True, exist_ok=True)
        hash_disk_cache.write_text(json.dumps(disk_map), encoding="utf-8")
    return f"{p.name}:{st.st_size}:{content_hash}"


def make_ocr_key(
    *,
    pdf_hash: str,
    page_number: int,
    backend_name: str,
    model_identity: str,
    mmproj_identity: str,
    server_identity: str,
    resolution_mode: str,
    render_long_edge_px: int,
    prompt: str,
    sampling: dict,
) -> str:
    """Stable content-addressed key for one page's OCR (DESIGN §8.6).

    ``server_identity`` is the llama.cpp build serving inference
    (:func:`inscriber.llama.server.llama_build_identity`) — upstream
    preprocessing changes alter outputs at identical model/prompt/sampling.
    """
    payload = json.dumps(
        {
            "pdf": pdf_hash,
            "page": page_number,
            "backend": backend_name,
            "model": model_identity,
            "mmproj": mmproj_identity,
            "server": server_identity,
            "resolution": resolution_mode,
            "long_edge": render_long_edge_px,
            "prompt": prompt,
            "sampling": sampling,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class OcrCache:
    """Per-page content-addressed store of pre-crop ``OcrPageResult`` (DESIGN §8.6)."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        refresh: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.refresh = refresh
        self.dir = Path(cache_dir) if cache_dir else default_cache_dir()

    @property
    def hash_disk_cache(self) -> Path:
        return self.dir / "hashes.json"

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> OcrPageResult | None:
        # --no-cache: no read. --refresh: ignore existing (force recompute).
        if not self.enabled or self.refresh:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("value_schema") != OCR_VALUE_SCHEMA:
            logger.debug("cache entry %s has incompatible value_schema; ignoring", key[:12])
            return None
        return ocr_page_result_from_dict(data["result"])

    def put(self, key: str, result: OcrPageResult, raw_output: str) -> None:
        # --no-cache: no write. (--refresh still writes, overwriting.)
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "value_schema": OCR_VALUE_SCHEMA,
            "result": ocr_page_result_to_dict(result),
            "raw_output": raw_output,  # for debugging (DESIGN §8.6)
        }
        target = self._path(key)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)  # atomic per-page write (resumable)


def make_vlm_key(
    *,
    figure_crop_hash: str,
    vlm_backend_name: str,
    vlm_model_identity: str,
    vlm_mmproj_identity: str,
    server_identity: str,
    full_assembled_prompt: str,
    sampling: dict,
    chat_template_kwargs: dict | None = None,
) -> str:
    """VLM cache key (DESIGN §9.6).

    Keyed on the **fully assembled prompt — context text included** — so changing
    ``context_chars`` or the page text doesn't serve a stale description.
    ``chat_template_kwargs`` (e.g. Gemma thinking activation) changes outputs, so
    it is key material too, as is ``server_identity`` (the llama.cpp build — see
    :func:`make_ocr_key`). (No ``max_tokens``: generation is bounded only by
    ``ctx_size``, which doesn't change a non-truncated output.)
    """
    payload = json.dumps(
        {
            "crop": figure_crop_hash,
            "backend": vlm_backend_name,
            "model": vlm_model_identity,
            "mmproj": vlm_mmproj_identity,
            "server": server_identity,
            "prompt": full_assembled_prompt,
            "sampling": sampling,
            "chat_template_kwargs": chat_template_kwargs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_table_key(
    *,
    page_image_hash: str,
    vlm_backend_name: str,
    vlm_model_identity: str,
    vlm_mmproj_identity: str,
    server_identity: str,
    full_assembled_prompt: str,
    sampling: dict,
    chat_template_kwargs: dict | None = None,
) -> str:
    """Cache key for one table restructure.

    Same scheme as :func:`make_vlm_key` but content-addressed on the **whole page
    image** (the table pass sends the full page, not a crop) and the assembled
    table prompt (locator + page text + OCR blob included). The ``kind`` field
    keeps table entries from ever colliding with figure-description entries in
    the shared store.
    """
    payload = json.dumps(
        {
            "kind": "table-restructure",
            "page_image": page_image_hash,
            "backend": vlm_backend_name,
            "model": vlm_model_identity,
            "mmproj": vlm_mmproj_identity,
            "server": server_identity,
            "prompt": full_assembled_prompt,
            "sampling": sampling,
            "chat_template_kwargs": chat_template_kwargs,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VlmCache:
    """Per-item cache of VLM text outputs (DESIGN §9.6).

    Stores figure descriptions (:func:`make_vlm_key`) and restructured tables
    (:func:`make_table_key`) — the key payloads are disjoint by construction.
    Lets a document be re-run (re-split, re-fetch BibTeX) without re-describing
    figures or re-restructuring tables. Same ``enabled``/``refresh`` semantics
    as :class:`OcrCache`.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        refresh: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.enabled = enabled
        self.refresh = refresh
        self.dir = Path(cache_dir) if cache_dir else default_vlm_cache_dir()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> str | None:
        if not self.enabled or self.refresh:
            return None
        p = self._path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("value_schema") != VLM_VALUE_SCHEMA:
            return None
        return data.get("text")

    def put(self, key: str, text: str) -> None:
        if not self.enabled:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = {"value_schema": VLM_VALUE_SCHEMA, "text": text}
        target = self._path(key)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(target)  # atomic per-figure write (resumable)
