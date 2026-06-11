"""M1b: OCR cache + the shared OcrPageResult serialization (DESIGN §8.6)."""

from __future__ import annotations

import json

import pytest

from inscriber.cache import (
    OCR_VALUE_SCHEMA,
    OcrCache,
    file_identity,
    make_ocr_key,
    make_vlm_key,
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
        model_identity="m:1:h", mmproj_identity="mp:1:h",
        server_identity="version: 9028 (abc1234)", resolution_mode="large",
        render_long_edge_px=1280, prompt="P", sampling={"temperature": 0, "seed": 0},
    )
    k1 = make_ocr_key(**base)
    assert k1 == make_ocr_key(**base)  # deterministic
    # mmproj identity is part of the key (DESIGN §8.6 — hashing only the text model
    # would miss projector swaps):
    assert make_ocr_key(**{**base, "mmproj_identity": "mp:2:other"}) != k1
    assert make_ocr_key(**{**base, "render_long_edge_px": 640}) != k1
    assert make_ocr_key(**{**base, "sampling": {"temperature": 0, "seed": 7}}) != k1
    # llama.cpp build identity is key material (DESIGN §8.6 — upstream preprocessing
    # changes alter outputs at identical model/prompt/sampling):
    assert make_ocr_key(**{**base, "server_identity": "version: 9587 (d2e22ed)"}) != k1


def test_make_vlm_key_includes_thinking_kwargs():
    # max_tokens is no longer key material (no VLM cap is sent; ctx_size is the
    # single size knob) — but chat_template_kwargs changes outputs, so it is.
    base = dict(
        figure_crop_hash="crop",
        vlm_backend_name="gemma",
        vlm_model_identity="m:1:h",
        vlm_mmproj_identity="mp:1:h",
        server_identity="version: 9028 (abc1234)",
        full_assembled_prompt="prompt",
        sampling={"temperature": 0, "seed": 0},
    )
    k1 = make_vlm_key(**base, chat_template_kwargs={"enable_thinking": True})
    assert k1 == make_vlm_key(**base, chat_template_kwargs={"enable_thinking": True})
    assert make_vlm_key(**base, chat_template_kwargs=None) != k1
    assert make_vlm_key(
        **{**base, "server_identity": "version: 9587 (d2e22ed)"},
        chat_template_kwargs={"enable_thinking": True},
    ) != k1


def test_make_vlm_key_raster_scheme():
    # Review C2+C3: the preferred image identity is (raster, bbox, padding) —
    # the crop's deterministic inputs — with the crop-bytes hash kept only as
    # the old-bundle fallback; exactly one scheme per key.
    common = dict(
        vlm_backend_name="gemma",
        vlm_model_identity="m:1:h",
        vlm_mmproj_identity="mp:1:h",
        server_identity="version: 9587 (d2e22ed)",
        full_assembled_prompt="prompt",
        sampling={"temperature": 0, "seed": 0},
    )
    raster = dict(page_image_hash="r1", crop_bbox=(0.1, 0.2, 0.8, 0.9), crop_padding=0.02)
    k = make_vlm_key(**raster, **common)
    assert k == make_vlm_key(**raster, **common)
    # raster / bbox / padding are each key material:
    assert make_vlm_key(**{**raster, "page_image_hash": "r2"}, **common) != k
    assert make_vlm_key(**{**raster, "crop_bbox": (0.1, 0.2, 0.8, 0.91)}, **common) != k
    assert make_vlm_key(**{**raster, "crop_padding": 0.03}, **common) != k
    # the legacy crop-bytes scheme yields a different key space:
    assert make_vlm_key(figure_crop_hash="c1", **common) != k
    # exactly one scheme, fully specified:
    with pytest.raises(ValueError):
        make_vlm_key(**common)
    with pytest.raises(ValueError):
        make_vlm_key(figure_crop_hash="c1", page_image_hash="r1",
                     crop_bbox=(0, 0, 1, 1), crop_padding=0.02, **common)
    with pytest.raises(ValueError):
        make_vlm_key(page_image_hash="r1", **common)  # missing bbox/padding


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


def test_hash_sidecar_merges_and_replaces_atomically(tmp_path, monkeypatch):
    # Review C1: the sidecar is written tmp+replace (like the cache entries) and
    # merge-on-write — an entry a CONCURRENT process adds while we are hashing
    # must survive our write (the pre-fix code wrote its stale initial read).
    import inscriber.cache as cache_mod

    disk = tmp_path / "hashes.json"
    disk.write_text(json.dumps({"foreign|123|456": "deadbeef"}), encoding="utf-8")
    f = tmp_path / "other-model.gguf"
    f.write_bytes(b"some new gguf bytes")

    real_sha256_file = cache_mod._sha256_file

    def hashing_races_with_writer(path):
        # Simulate another inscriber process updating the sidecar mid-hash.
        current = json.loads(disk.read_text(encoding="utf-8"))
        current["concurrent|7|8"] = "cafef00d"
        disk.write_text(json.dumps(current), encoding="utf-8")
        return real_sha256_file(path)

    monkeypatch.setattr(cache_mod, "_sha256_file", hashing_races_with_writer)
    file_identity(str(f), hash_disk_cache=disk)
    data = json.loads(disk.read_text(encoding="utf-8"))
    assert data["foreign|123|456"] == "deadbeef"  # pre-existing entry kept
    assert data["concurrent|7|8"] == "cafef00d"  # mid-hash writer's entry kept
    assert sha256_bytes(b"some new gguf bytes") in data.values()  # ours added
    assert not disk.with_suffix(".json.tmp").exists()  # no tmp leftovers
