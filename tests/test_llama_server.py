"""M1a (review Fix 8): LlamaServerManager launch-arg construction, the .exe
suffix logic, the no-spawn endpoint branch, and the serve() lifecycle — all with
``Popen``/``httpx`` mocked so no real server is spawned and Windows/process
teardown issues surface before real-hardware work.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inscriber import config
from inscriber.llama import server as server_mod
from inscriber.llama.server import (
    LlamaServerManager,
    ServerError,
    ServerSpec,
    build_launch_args,
    endpoint_or_serve,
)
from inscriber.ocr.base import MtmdCliInferencer

# --------------------------------------------------------------------------- #
# build_launch_args (pure)
# --------------------------------------------------------------------------- #


def _subseq(seq, sub):
    """True if `sub` appears as a contiguous subsequence of `seq`."""
    n = len(sub)
    return any(seq[i : i + n] == sub for i in range(len(seq) - n + 1))


def test_build_launch_args_full():
    spec = ServerSpec(
        model="/m/ocr.gguf",
        mmproj="/m/ocr-mmproj.gguf",
        host="127.0.0.1",
        ctx_size=4096,
        n_gpu_layers=20,
        extra_flags=["--dry-multiplier", "0.8"],
        chat_template=None,
    )
    exe = Path("/bin/llama-server")
    args = build_launch_args(exe, spec, 55555)
    assert args[0] == str(exe)  # platform-native stringification
    assert _subseq(args, ["-m", "/m/ocr.gguf"])
    assert _subseq(args, ["--host", "127.0.0.1"])
    assert _subseq(args, ["--port", "55555"])
    assert _subseq(args, ["-c", "4096"])
    assert _subseq(args, ["-ngl", "20"])
    assert _subseq(args, ["--mmproj", "/m/ocr-mmproj.gguf"])
    assert _subseq(args, ["--dry-multiplier", "0.8"])
    # DeepSeek-OCR server path: no --chat-template (DESIGN §2.2).
    assert "--chat-template" not in args


def test_build_launch_args_with_chat_template():
    spec = ServerSpec(model="/m.gguf", chat_template="deepseek-ocr")
    args = build_launch_args(Path("/bin/llama-server"), spec, 1)
    assert _subseq(args, ["--chat-template", "deepseek-ocr"])


def test_build_launch_args_omits_mmproj_when_none():
    spec = ServerSpec(model="/m.gguf", mmproj=None)
    args = build_launch_args(Path("/bin/llama-server"), spec, 1)
    assert "--mmproj" not in args


# --------------------------------------------------------------------------- #
# find_binary .exe suffix (DESIGN §5.2)
# --------------------------------------------------------------------------- #


def test_find_binary_appends_exe_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    (tmp_path / "llama-server.exe").write_text("x")
    found = config.find_binary(str(tmp_path), "llama-server")
    assert found is not None and found.name == "llama-server.exe"


def test_binary_filename_suffix_logic(monkeypatch):
    # Test the name-construction branch directly (instantiating PosixPath on
    # Windows is impossible, so we can't exercise the full posix find_binary here).
    monkeypatch.setattr(os, "name", "nt")
    assert config.binary_filename("llama-server") == "llama-server.exe"
    monkeypatch.setattr(os, "name", "posix")
    assert config.binary_filename("llama-server") == "llama-server"


def test_find_binary_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(config.shutil, "which", lambda *_: None)
    assert config.find_binary(str(tmp_path), "llama-server") is None


def test_find_binary_path_fallback(monkeypatch):
    monkeypatch.setattr(config.shutil, "which", lambda name: "/usr/bin/" + name)
    found = config.find_binary("", "llama-server")
    assert found == Path("/usr/bin/llama-server")


# --------------------------------------------------------------------------- #
# Fakes for the serve() lifecycle
# --------------------------------------------------------------------------- #


class FakePopen:
    def __init__(self, args, stdout=None, stderr=None, alive_polls=None, exit_code=0):
        self.args = args
        self._exit_code = exit_code
        self._alive_polls = alive_polls if alive_polls is not None else 10_000
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        if self.terminated or self.killed:
            return self.returncode
        if self._alive_polls <= 0:
            self.returncode = self._exit_code
            return self.returncode
        self._alive_polls -= 1
        return None  # still "loading"

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def kill(self):  # pragma: no cover - only on grace timeout
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


class FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


def _patch_popen(monkeypatch, factory):
    monkeypatch.setattr(server_mod.subprocess, "Popen", factory)


def test_serve_yields_url_and_tears_down(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "find_binary", lambda *_: Path("/bin/llama-server"))
    created = {}

    def factory(args, stdout=None, stderr=None):
        p = FakePopen(args, stdout, stderr, alive_polls=10_000)
        created["proc"] = p
        return p

    _patch_popen(monkeypatch, factory)
    # First /health 503 (loading), then 200 (ready).
    seq = iter([FakeResp(503), FakeResp(200)])
    monkeypatch.setattr(server_mod.httpx, "get", lambda *a, **k: next(seq))

    mgr = LlamaServerManager("/bin", server_start_timeout=5, log_dir=tmp_path)
    spec = ServerSpec(model="/m.gguf", label="ocr")
    with mgr.serve(spec) as url:
        assert url.startswith("http://127.0.0.1:")
        assert created["proc"].terminated is False
    # After the context, the process is torn down.
    assert created["proc"].terminated is True


def test_serve_raises_on_early_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "find_binary", lambda *_: Path("/bin/llama-server"))

    def factory(args, stdout=None, stderr=None):
        return FakePopen(args, stdout, stderr, alive_polls=0, exit_code=1)

    _patch_popen(monkeypatch, factory)
    monkeypatch.setattr(server_mod.httpx, "get", lambda *a, **k: FakeResp(503))

    mgr = LlamaServerManager("/bin", server_start_timeout=2, log_dir=tmp_path, port_retries=1)
    with pytest.raises(ServerError, match="exited early"):
        with mgr.serve(ServerSpec(model="/m.gguf")):
            pass


def test_serve_times_out(tmp_path, monkeypatch):
    import httpx

    monkeypatch.setattr(server_mod, "find_binary", lambda *_: Path("/bin/llama-server"))
    _patch_popen(monkeypatch, lambda *a, **k: FakePopen(a[0] if a else [], alive_polls=10_000))

    def raise_conn(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(server_mod.httpx, "get", raise_conn)
    mgr = LlamaServerManager("/bin", server_start_timeout=0.3, log_dir=tmp_path, port_retries=1)
    with pytest.raises(ServerError, match="timed out"):
        with mgr.serve(ServerSpec(model="/m.gguf")):
            pass


def test_missing_binary_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(server_mod, "find_binary", lambda *_: None)
    mgr = LlamaServerManager("/nope", log_dir=tmp_path)
    with pytest.raises(ServerError, match="not found"):
        with mgr.serve(ServerSpec(model="/m.gguf")):
            pass


# --------------------------------------------------------------------------- #
# No-spawn endpoint branch (DESIGN §5.1)
# --------------------------------------------------------------------------- #


def test_endpoint_or_serve_no_spawn(monkeypatch):
    # Popen must NOT be called when an endpoint is provided.
    def boom(*a, **k):  # pragma: no cover - asserts it's never reached
        raise AssertionError("should not spawn when endpoint is set")

    _patch_popen(monkeypatch, boom)
    mgr = LlamaServerManager("/bin")
    with endpoint_or_serve(mgr, "http://gpu-box:9000/", ServerSpec(model="/m.gguf")) as url:
        assert url == "http://gpu-box:9000"


# --------------------------------------------------------------------------- #
# mtmd-cli fallback arg construction (DESIGN §2.1, §2.2)
# --------------------------------------------------------------------------- #


def test_mtmd_build_args():
    inf = MtmdCliInferencer("/bin", "/m/ocr.gguf", "/m/mmproj.gguf", n_gpu_layers=10, ctx_size=4096)
    img = Path("/tmp/p.png")
    args = inf.build_args(
        Path("/bin/llama-mtmd-cli"), img, "<|grounding|>Convert.",
        chat_template="deepseek-ocr", max_tokens=2048,
    )
    assert _subseq(args, ["-m", "/m/ocr.gguf"])
    assert _subseq(args, ["--mmproj", "/m/mmproj.gguf"])
    assert _subseq(args, ["--image", str(img)])  # platform-native path string
    assert _subseq(args, ["-p", "<|grounding|>Convert."])
    assert _subseq(args, ["--temp", "0"])
    assert _subseq(args, ["-n", "2048"])
    assert _subseq(args, ["--chat-template", "deepseek-ocr"])  # mtmd path DOES pass it
    assert _subseq(args, ["-ngl", "10"])
