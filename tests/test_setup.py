"""`inscriber setup` — registry integrity, verified downloads, config writing.

HTTP is mocked at the transport boundary (httpx.MockTransport threaded through
``run_setup``/``download_model``); registry entries under test are small fakes
whose sha256 is computed in-test. Nothing touches the real platformdirs
locations — every path is an explicit tmp_path.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import httpx
import pytest

from inscriber import setup as setup_mod
from inscriber.setup import (
    DEEPSEEK_QUANTS,
    GEMMA_FILES,
    ModelFile,
    SetupError,
    download_model,
    plan_files,
    run_setup,
    write_setup_config,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_integrity():
    for quant in ("bf16", "q8_0"):
        files = plan_files(quant)
        assert [f.role for f in files] == ["ocr.model", "ocr.mmproj", "vlm.model", "vlm.mmproj"]
        for f in files:
            assert f.url.startswith("https://huggingface.co/")
            assert "/resolve/main/" in f.url
            assert f.size > 0
            assert len(f.sha256) == 64 and int(f.sha256, 16) >= 0
            assert f.local_name.endswith(".gguf")


def test_registry_quants_differ_only_on_deepseek():
    assert plan_files("bf16")[2:] == GEMMA_FILES
    assert plan_files("q8_0")[2:] == GEMMA_FILES
    assert DEEPSEEK_QUANTS["bf16"][0].sha256 != DEEPSEEK_QUANTS["q8_0"][0].sha256


def test_gemma_mmproj_renamed_family_specific():
    # unsloth ships the generic "mmproj-BF16.gguf"; on disk it must be Gemma-specific.
    mmproj = GEMMA_FILES[1]
    assert mmproj.url.endswith("mmproj-BF16.gguf?download=true")
    assert "gemma" in mmproj.local_name.lower()


def test_plan_files_unknown_quant():
    with pytest.raises(SetupError, match="unknown DeepSeek quant"):
        plan_files("q4_k_m")


# --------------------------------------------------------------------------- #
# Download — fakes + MockTransport
# --------------------------------------------------------------------------- #


def _fake_file(content: bytes, name: str = "fake-model.gguf", role: str = "ocr.model") -> ModelFile:
    return ModelFile(
        role=role,
        local_name=name,
        url=f"https://huggingface.co/fake/repo/resolve/main/{name}?download=true",
        size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _serving_transport(content: bytes) -> httpx.MockTransport:
    """Serve ``content`` honoring Range (206 partial / 416 exhausted / 200 full)."""

    def handler(request: httpx.Request) -> httpx.Response:
        rng = request.headers.get("Range")
        if rng:
            start = int(rng.removeprefix("bytes=").rstrip("-"))
            if start >= len(content):
                return httpx.Response(416)
            return httpx.Response(206, content=content[start:])
        return httpx.Response(200, content=content)

    return httpx.MockTransport(handler)


def test_download_fresh(tmp_path):
    content = b"GGUF" + b"x" * 500
    f = _fake_file(content)
    dest = download_model(f, tmp_path, transport=_serving_transport(content))
    assert dest == tmp_path / f.local_name
    assert dest.read_bytes() == content
    assert not (tmp_path / (f.local_name + ".part")).exists()


def test_download_resumes_from_part(tmp_path):
    content = b"GGUF" + b"y" * 1000
    f = _fake_file(content)
    (tmp_path / (f.local_name + ".part")).write_bytes(content[:300])

    seen_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_ranges.append(request.headers.get("Range"))
        return httpx.Response(206, content=content[300:])

    dest = download_model(f, tmp_path, transport=httpx.MockTransport(handler))
    assert seen_ranges == ["bytes=300-"]
    assert dest.read_bytes() == content


def test_download_restarts_when_server_ignores_range(tmp_path):
    content = b"GGUF" + b"z" * 800
    f = _fake_file(content)
    (tmp_path / (f.local_name + ".part")).write_bytes(b"stale-junk")

    # Always answers 200 with the full body, Range or not.
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=content))
    dest = download_model(f, tmp_path, transport=transport)
    assert dest.read_bytes() == content


def test_download_part_already_complete_promotes_via_416(tmp_path):
    content = b"GGUF" + b"w" * 600
    f = _fake_file(content)
    (tmp_path / (f.local_name + ".part")).write_bytes(content)
    dest = download_model(f, tmp_path, transport=_serving_transport(content))
    assert dest.read_bytes() == content
    assert not (tmp_path / (f.local_name + ".part")).exists()


def test_download_hash_mismatch_deletes_part(tmp_path):
    content = b"GGUF" + b"a" * 400
    f = _fake_file(content)
    tampered = b"GGUF" + b"b" * 400  # same size, different bytes
    with pytest.raises(SetupError, match="pinned\nsha256|pinned sha256"):
        download_model(f, tmp_path, transport=_serving_transport(tampered))
    assert not (tmp_path / (f.local_name + ".part")).exists()
    assert not (tmp_path / f.local_name).exists()


def test_download_short_body_keeps_part_for_resume(tmp_path):
    content = b"GGUF" + b"c" * 900
    f = _fake_file(content)
    transport = httpx.MockTransport(lambda req: httpx.Response(200, content=content[:200]))
    with pytest.raises(SetupError, match="re-run setup to resume"):
        download_model(f, tmp_path, transport=transport)
    assert (tmp_path / (f.local_name + ".part")).read_bytes() == content[:200]


def test_existing_verified_file_skips_http(tmp_path):
    content = b"GGUF" + b"d" * 300
    f = _fake_file(content)
    (tmp_path / f.local_name).write_bytes(content)

    def handler(request):  # pragma: no cover - must never run
        raise AssertionError("HTTP request issued for an already-verified file")

    dest = download_model(f, tmp_path, transport=httpx.MockTransport(handler))
    assert dest.read_bytes() == content


def test_existing_wrong_size_errors_without_overwrite(tmp_path):
    content = b"GGUF" + b"e" * 300
    f = _fake_file(content)
    (tmp_path / f.local_name).write_bytes(b"user file, wrong size")
    with pytest.raises(SetupError, match="wrong size"):
        download_model(f, tmp_path, transport=_serving_transport(content))
    assert (tmp_path / f.local_name).read_bytes() == b"user file, wrong size"


def test_existing_corrupt_content_errors(tmp_path):
    content = b"GGUF" + b"f" * 300
    f = _fake_file(content)
    (tmp_path / f.local_name).write_bytes(b"GGUF" + b"g" * 300)  # right size, wrong bytes
    with pytest.raises(SetupError, match="does not match the pinned"):
        download_model(f, tmp_path, transport=_serving_transport(content))


def test_http_error_status(tmp_path):
    content = b"GGUF" + b"h" * 100
    f = _fake_file(content)
    transport = httpx.MockTransport(lambda req: httpx.Response(404))
    with pytest.raises(SetupError, match="HTTP 404"):
        download_model(f, tmp_path, transport=transport)


def test_disk_space_check(tmp_path, monkeypatch):
    content = b"GGUF" + b"i" * 100
    f = _fake_file(content)
    monkeypatch.setattr(
        setup_mod.shutil, "disk_usage",
        lambda p: SimpleNamespace(total=10, used=10, free=5),
    )
    with pytest.raises(SetupError, match="not enough disk space"):
        setup_mod._check_disk_space([f], tmp_path)
    # Nothing missing -> no free-space requirement at all.
    (tmp_path / f.local_name).write_bytes(content)
    setup_mod._check_disk_space([f], tmp_path)


# --------------------------------------------------------------------------- #
# Config writing
# --------------------------------------------------------------------------- #


def _model_paths(tmp_path):
    return {
        "ocr.model": tmp_path / "m" / "ds.gguf",
        "ocr.mmproj": tmp_path / "m" / "ds-mmproj.gguf",
        "vlm.model": tmp_path / "m" / "gemma.gguf",
        "vlm.mmproj": tmp_path / "m" / "gemma-mmproj.gguf",
    }


def test_fresh_config_write(tmp_path):
    target = tmp_path / "cfg" / "config.toml"
    write_setup_config(target, model_paths=_model_paths(tmp_path), llama_bin_dir=None)
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["ocr"]["model"] == (tmp_path / "m" / "ds.gguf").as_posix()
    assert data["vlm"]["mmproj"] == (tmp_path / "m" / "gemma-mmproj.gguf").as_posix()
    assert "bin_dir" not in data.get("llama", {})  # only the commented placeholder
    assert "# bin_dir" in target.read_text(encoding="utf-8")


def test_fresh_config_write_with_bin_dir(tmp_path):
    target = tmp_path / "config.toml"
    write_setup_config(
        target, model_paths=_model_paths(tmp_path), llama_bin_dir=str(tmp_path / "bin")
    )
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["llama"]["bin_dir"] == (tmp_path / "bin").as_posix()


def test_config_update_preserves_existing_keys(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text(
        '# my comment\n[llama]\nctx_size = 8192\n\n[ocr]\nmodel = "old.gguf"\n\n'
        '[output]\nsplit = false\n\n[custom]\nunknown_key = "kept"\n',
        encoding="utf-8",
    )
    write_setup_config(target, model_paths=_model_paths(tmp_path), llama_bin_dir=None)
    data = tomllib.loads(target.read_text(encoding="utf-8"))
    assert data["ocr"]["model"] == (tmp_path / "m" / "ds.gguf").as_posix()  # updated
    assert data["llama"]["ctx_size"] == 8192  # preserved
    assert data["output"]["split"] is False  # preserved (bool emit)
    assert data["custom"]["unknown_key"] == "kept"  # unknown section preserved
    assert "my comment" not in target.read_text(encoding="utf-8")  # lossy, by design


def test_config_update_invalid_toml_errors(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("not = valid = toml", encoding="utf-8")
    with pytest.raises(SetupError, match="not valid TOML"):
        write_setup_config(target, model_paths=_model_paths(tmp_path), llama_bin_dir=None)


def test_config_written_lf_utf8(tmp_path):
    target = tmp_path / "config.toml"
    write_setup_config(target, model_paths=_model_paths(tmp_path), llama_bin_dir=None)
    raw = target.read_bytes()
    assert b"\r\n" not in raw


# --------------------------------------------------------------------------- #
# Windows file-in-use conversions (review A1): OSError → SetupError with a hint
# --------------------------------------------------------------------------- #


def test_promotion_oserror_converted_to_setup_error(tmp_path, monkeypatch):
    # A locked destination (e.g. a running llama-server with the GGUF mmap'd)
    # must surface a hint, and the verified .part must survive for the retry.
    content = b"GGUF" + b"d" * 300
    f = _fake_file(content)

    def locked_replace(self, target):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(setup_mod.Path, "replace", locked_replace)
    with pytest.raises(SetupError, match="open in another program"):
        download_model(f, tmp_path, transport=_serving_transport(content))
    assert (tmp_path / (f.local_name + ".part")).read_bytes() == content


def test_config_write_oserror_converted_to_setup_error(tmp_path, monkeypatch):
    # An editor holding config.toml locked must not yield a raw traceback.
    target = tmp_path / "config.toml"

    def locked_open(*args, **kwargs):
        raise PermissionError(32, "being used by another process")

    monkeypatch.setattr(setup_mod, "open", locked_open, raising=False)
    with pytest.raises(SetupError, match="could not write config"):
        write_setup_config(target, model_paths=_model_paths(tmp_path), llama_bin_dir=None)


# --------------------------------------------------------------------------- #
# run_setup orchestration + CLI wiring
# --------------------------------------------------------------------------- #


def _fake_registry(monkeypatch, contents: dict[str, bytes]):
    """Swap the pinned registry for four tiny fakes served by one transport."""
    ds = (
        _fake_file(contents["ocr.model"], "ds.gguf", "ocr.model"),
        _fake_file(contents["ocr.mmproj"], "ds-mmproj.gguf", "ocr.mmproj"),
    )
    gm = (
        _fake_file(contents["vlm.model"], "gemma.gguf", "vlm.model"),
        _fake_file(contents["vlm.mmproj"], "gemma-mmproj.gguf", "vlm.mmproj"),
    )
    monkeypatch.setattr(setup_mod, "DEEPSEEK_QUANTS", {"bf16": ds, "q8_0": ds})
    monkeypatch.setattr(setup_mod, "GEMMA_FILES", gm)
    by_name = {f.local_name: contents[f.role] for f in (*ds, *gm)}

    def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.rsplit("/", 1)[-1]
        return httpx.Response(200, content=by_name[name])

    return httpx.MockTransport(handler)


def test_run_setup_end_to_end(tmp_path, monkeypatch):
    contents = {
        "ocr.model": b"GGUF-ds-model",
        "ocr.mmproj": b"GGUF-ds-mmproj",
        "vlm.model": b"GGUF-gemma-model",
        "vlm.mmproj": b"GGUF-gemma-mmproj",
    }
    transport = _fake_registry(monkeypatch, contents)
    models_dir = tmp_path / "models"
    cfg_path = tmp_path / "config.toml"

    written = run_setup(
        config_path=str(cfg_path),
        models_dir=str(models_dir),
        llama_bin_dir=None,
        deepseek_quant="bf16",
        transport=transport,
    )
    assert written[-1] == str(cfg_path)
    assert (models_dir / "ds.gguf").read_bytes() == contents["ocr.model"]
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    assert data["ocr"]["model"] == (models_dir / "ds.gguf").as_posix()
    assert data["vlm"]["model"] == (models_dir / "gemma.gguf").as_posix()

    # Idempotent re-run: everything verified + skipped, config updated in place.
    def no_http(request):  # pragma: no cover - must never run
        raise AssertionError("re-run should not hit the network")

    rerun = run_setup(
        config_path=str(cfg_path),
        models_dir=str(models_dir),
        llama_bin_dir=None,
        deepseek_quant="bf16",
        transport=httpx.MockTransport(no_http),
    )
    assert rerun == written


def test_cli_setup_dispatch(tmp_path, monkeypatch, capsys):
    from inscriber import cli

    calls = {}

    def fake_run_setup(**kwargs):
        calls.update(kwargs)
        return [str(tmp_path / "a.gguf"), str(tmp_path / "config.toml")]

    monkeypatch.setattr("inscriber.setup.run_setup", fake_run_setup)
    rc = cli.main(["setup", "--deepseek-quant", "q8_0",
                   "--models-dir", str(tmp_path), "-q"])
    assert rc == 0
    assert calls["deepseek_quant"] == "q8_0"
    assert calls["models_dir"] == str(tmp_path)
    out = capsys.readouterr().out.splitlines()
    assert out == [str(tmp_path / "a.gguf"), str(tmp_path / "config.toml")]


def test_cli_setup_error_exit_code(monkeypatch):
    from inscriber import cli

    def fail(**kwargs):
        raise SetupError("boom")

    monkeypatch.setattr("inscriber.setup.run_setup", fail)
    assert cli.main(["setup", "-q"]) == 1


def test_normalizer_leaves_setup():
    from inscriber.cli import _normalize_argv

    assert _normalize_argv(["setup", "--deepseek-quant", "q8_0"]) == [
        "setup", "--deepseek-quant", "q8_0",
    ]
