"""M1b: OCR cache + the shared OcrPageResult serialization (DESIGN §8.6)."""

from __future__ import annotations

import json

from inscriber.cache import (
    OCR_VALUE_SCHEMA,
    OcrCache,
    file_identity,
    make_ocr_key,
    sha256_bytes,
)
from inscriber.models import OcrPageResult, Region
from inscriber.serialize import ocr_page_result_from_dict, ocr_page_result_to_dict


def _sample_result() -> OcrPageResult:
    return OcrPageResult(
        page_number=3,
        markdown="## Method\n\n⟦INSCRIBER_FIG:fig_p3_1⟧\n\nFigure 1: x.",
        regions=[
            Region(label="image", bbox_norm=(0.1, 0.24, 0.88, 0.61), text="Figure 1: x."),
            Region(label="text", bbox_norm=(0.0, 0.0, 1.0, 0.2), text="## Method"),
        ],
    )


def test_serialize_roundtrip():
    r = _sample_result()
    back = ocr_page_result_from_dict(ocr_page_result_to_dict(r))
    assert back.page_number == r.page_number
    assert back.markdown == r.markdown
    assert len(back.regions) == 2
    assert back.regions[0].bbox_norm == (0.1, 0.24, 0.88, 0.61)
    assert isinstance(back.regions[0].bbox_norm, tuple)
    assert back.regions[0].text == "Figure 1: x."


def test_make_ocr_key_deterministic_and_sensitive():
    base = dict(
        pdf_hash="abc", page_number=1, backend_name="deepseek-ocr",
        model_identity="m:1:h", mmproj_identity="mp:1:h", resolution_mode="large",
        render_long_edge_px=1280, prompt="P", sampling={"temperature": 0, "seed": 0},
    )
    k1 = make_ocr_key(**base)
    assert k1 == make_ocr_key(**base)  # deterministic
    # mmproj identity is part of the key (DESIGN §8.6 — hashing only the text model
    # would miss projector swaps):
    assert make_ocr_key(**{**base, "mmproj_identity": "mp:2:other"}) != k1
    assert make_ocr_key(**{**base, "render_long_edge_px": 640}) != k1
    assert make_ocr_key(**{**base, "sampling": {"temperature": 0, "seed": 7}}) != k1


def test_cache_put_get_roundtrip(tmp_path):
    cache = OcrCache(cache_dir=tmp_path)
    r = _sample_result()
    cache.put("key1", r, raw_output="RAW")
    got = cache.get("key1")
    assert got is not None
    assert got.markdown == r.markdown
    assert len(got.regions) == 2
    # raw output stored for debugging:
    data = json.loads((tmp_path / "key1.json").read_text(encoding="utf-8"))
    assert data["raw_output"] == "RAW"
    assert data["value_schema"] == OCR_VALUE_SCHEMA


def test_no_cache_neither_reads_nor_writes(tmp_path):
    cache = OcrCache(enabled=False, cache_dir=tmp_path)
    cache.put("k", _sample_result(), raw_output="x")
    assert not (tmp_path / "k.json").exists()
    assert cache.get("k") is None


def test_refresh_ignores_existing_but_still_writes(tmp_path):
    writer = OcrCache(cache_dir=tmp_path)
    writer.put("k", _sample_result(), raw_output="x")
    refresher = OcrCache(refresh=True, cache_dir=tmp_path)
    assert refresher.get("k") is None  # ignores existing entry
    refresher.put("k", _sample_result(), raw_output="y")  # but overwrites
    data = json.loads((tmp_path / "k.json").read_text(encoding="utf-8"))
    assert data["raw_output"] == "y"


def test_incompatible_value_schema_ignored(tmp_path):
    cache = OcrCache(cache_dir=tmp_path)
    (tmp_path / "k.json").write_text(
        json.dumps({"value_schema": 999, "result": {}}), encoding="utf-8"
    )
    assert cache.get("k") is None


def test_file_identity_uses_content_hash_and_mtime_cache(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"hello world" * 1000)
    disk = tmp_path / "hashes.json"
    ident1 = file_identity(str(f), hash_disk_cache=disk)
    assert ident1.startswith("model.gguf:")
    assert sha256_bytes(f.read_bytes()) in ident1
    # second call hits the cache (disk map persisted):
    assert disk.exists()
    assert file_identity(str(f), hash_disk_cache=disk) == ident1
