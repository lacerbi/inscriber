"""llama.cpp server lifecycle: spawn / health / teardown (DESIGN §5).

``inscriber`` owns the ``llama-server`` process by default: launch it with the
right model / projector / flags, wait for ``/health``, run a pass, terminate it.
A power-user escape hatch (``--ocr-endpoint`` / ``--vlm-endpoint``) talks to an
already-running server instead — see :func:`endpoint_or_serve`.

Cross-platform notes (DESIGN §5.3, §15):
* list-args only, never ``shell=True``;
* binary discovery appends ``.exe`` on Windows (via :func:`config.find_binary`);
* ``Popen.terminate()`` maps to ``TerminateProcess`` (Windows) / ``SIGTERM``
  (POSIX) — both fine; we avoid POSIX-only ``os.killpg`` / ``preexec_fn``;
* orphan backstop = ``atexit`` **plus** ``SIGTERM``/``SIGHUP`` handlers — the
  POSIX signals bypass ``atexit``, so without the handlers a ``kill`` / logout /
  supervisor stop would leave the spawned llama-server running (``SIGHUP`` is
  absent on Windows and skipped there).
"""

from __future__ import annotations

import atexit
import os
import re
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from inscriber.config import find_binary
from inscriber.errors import InscriberError
from inscriber.logging import get_logger

logger = get_logger()

# Module-level backstop: any server we spawn is tracked here and terminated on
# interpreter exit OR a terminating signal, so a crash / hard exit / `kill`
# never orphans a (GPU-resident, multi-GB) llama-server.
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()
_CLEANUP_REGISTERED = False


class ServerError(InscriberError):
    """Raised when a llama-server fails to launch or become healthy."""


@dataclass
class ServerSpec:
    """Everything needed to launch one ``llama-server`` (DESIGN §5.3)."""

    model: str
    mmproj: str | None = None
    host: str = "127.0.0.1"
    port: int = 0  # 0 = auto-select a free ephemeral port
    ctx_size: int = 8192
    n_gpu_layers: int | str = "auto"  # "auto" | "all" | int (0 = CPU); see DESIGN §13.1
    # Backend-supplied extra flags, e.g. DRY / repeat-penalty (DESIGN §2.2).
    extra_flags: list[str] = field(default_factory=list)
    # Per-path chat template (DESIGN §2.2): None on the server path for DeepSeek-OCR
    # (the server applies the model's built-in template — do NOT pass it).
    chat_template: str | None = None
    label: str = "llama"  # for log file naming / messages


def _on_terminate_signal(signum, frame):  # pragma: no cover - kills the process
    """Terminate tracked servers, then re-deliver the signal with the default
    disposition so the process exits with the standard killed-by-signal status."""
    _terminate_all()
    signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)


def _register_cleanup() -> None:
    """Register the orphan backstops once (DESIGN §5.3).

    ``atexit`` covers normal exit and Ctrl-C (KeyboardInterrupt unwinds, then
    atexit runs). POSIX ``SIGTERM``/``SIGHUP`` **bypass** atexit — their default
    disposition kills the interpreter without cleanup — so handlers are
    installed for them too (``SIGHUP`` does not exist on Windows; ``getattr``
    skips it). A handler someone else installed is left alone, and installation
    is skipped quietly off the main thread (``signal.signal`` raises there).
    """
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    _CLEANUP_REGISTERED = True
    atexit.register(_terminate_all)
    for sig_name in ("SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        try:
            if signal.getsignal(sig) == signal.SIG_DFL:
                signal.signal(sig, _on_terminate_signal)
        except (ValueError, OSError):  # pragma: no cover - non-main thread
            pass


def _terminate_all() -> None:
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for proc in procs:
        _terminate(proc)


# In-process memoization of the build-identity probe (keyed by exe path+size+mtime,
# like the model-hash memo in cache.py — one subprocess spawn per run, not per page).
_BUILD_ID_MEM: dict[tuple[str, int, int], str] = {}


def llama_build_identity(bin_dir: str, endpoint: str = "") -> str:
    """Identity of the llama.cpp build that will serve inference — OCR/VLM
    cache-key material (DESIGN §8.6).

    Upstream preprocessing/sampling changes (e.g. llama.cpp PR #23345) alter
    model outputs across builds with identical model/prompt/sampling, so the
    build must bust the caches. Spawn mode probes ``llama-server --version``
    (no model load; memoized per binary). Endpoint mode asks the running
    server's ``/props`` for its ``build_info``; if that is unavailable it
    degrades to ``"unknown"`` with a warning rather than failing the run.
    """
    if endpoint:
        url = endpoint.rstrip("/") + "/props"
        try:
            data = httpx.get(url, timeout=10.0).json()
            info = data.get("build_info") if isinstance(data, dict) else None
            if info:
                return str(info)
        except (httpx.HTTPError, ValueError):
            pass
        logger.warning(
            "could not read build_info from %s; cache keys will not reflect "
            "the endpoint's llama.cpp build (stale entries possible after a "
            "server upgrade — use --refresh then)",
            url,
        )
        return "unknown"

    exe = find_binary(bin_dir, "llama-server")
    if exe is None:
        raise ServerError(
            "llama-server binary not found "
            f"(llama.bin_dir={bin_dir!r}; not on PATH either) — needed to key "
            "caches on the llama.cpp build"
        )
    st = exe.stat()
    mem_key = (str(exe), st.st_size, st.st_mtime_ns)
    if mem_key not in _BUILD_ID_MEM:
        try:
            proc = subprocess.run(
                [str(exe), "--version"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ServerError(f"failed to probe `{exe} --version`: {e}") from e
        # The probe prints e.g. "version: 9587 (d2e22ed97)" (on stderr) plus a
        # compiler/OS line; keep just the version line — toolchain churn doesn't
        # change model output.
        text = (proc.stdout or "") + (proc.stderr or "")
        match = re.search(r"^version:.*$", text, re.MULTILINE)
        if match:
            identity = match.group(0).strip()
        elif proc.returncode == 0 and text.strip():
            identity = text.strip()
        else:
            raise ServerError(
                f"`{exe} --version` failed (exit {proc.returncode}): {text.strip()[:500]}"
            )
        _BUILD_ID_MEM[mem_key] = identity
    return _BUILD_ID_MEM[mem_key]


def build_number(identity: str) -> int | None:
    """Numeric llama.cpp build from a :func:`llama_build_identity` string.

    Handles both forms: ``"version: 9587 (d2e22ed97)"`` (the ``--version``
    probe) and ``"b9587-d2e22ed97"`` (an endpoint's ``/props`` ``build_info``).
    Returns ``None`` when unparseable (e.g. the ``"unknown"`` endpoint
    fallback, or a nonstandard ``build_info`` shape) — callers decide whether
    that warns or blocks; the pipeline's min-build gate deliberately degrades
    to a warning for servers it cannot date.
    """
    m = re.search(r"version:\s*(\d+)", identity) or re.match(r"b(\d+)\b", identity)
    return int(m.group(1)) if m else None


def _free_port(host: str) -> int:
    """Probe a free ephemeral port (DESIGN §5.3).

    Small TOCTOU race: another process could grab the port between this bind and
    the server's bind; callers retry with a fresh port on a /health timeout.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


def build_launch_args(exe: Path, spec: ServerSpec, port: int) -> list[str]:
    """Construct the llama-server argv (list-args only; never shell). Pure, testable."""
    args: list[str] = [
        str(exe),
        "-m",
        spec.model,
        "--host",
        spec.host,
        "--port",
        str(port),
        "-c",
        str(spec.ctx_size),
    ]
    # n_gpu_layers == "auto" (the default) → omit -ngl entirely and let llama.cpp
    # use its own default (which is GPU auto-fit on modern builds, e.g. 9587). This
    # never passes a symbolic token, so it can't break arg-parsing on any build.
    # "all" or an explicit integer are passed through verbatim.
    if str(spec.n_gpu_layers).strip().lower() != "auto":
        args += ["-ngl", str(spec.n_gpu_layers)]
    if spec.mmproj:
        args += ["--mmproj", spec.mmproj]
    if spec.chat_template:
        args += ["--chat-template", spec.chat_template]
    args += spec.extra_flags
    return args


def _terminate(proc: subprocess.Popen, grace: float = 10.0) -> None:
    if proc.poll() is not None:
        _untrack(proc)
        return
    try:
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=grace)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("error terminating server: %s", e)
    finally:
        _untrack(proc)


def _track(proc: subprocess.Popen) -> None:
    _register_cleanup()
    with _ACTIVE_LOCK:
        _ACTIVE.add(proc)


def _untrack(proc: subprocess.Popen) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(proc)


def _log_tail(log_path: Path, n: int = 40) -> str:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(no server log captured)"
    lines = text.splitlines()
    return "\n".join(lines[-n:])


class LlamaServerManager:
    """Spawns and supervises ``llama-server`` processes (DESIGN §5)."""

    def __init__(
        self,
        bin_dir: str,
        *,
        server_start_timeout: float = 120.0,
        log_dir: str | Path | None = None,
        port_retries: int = 3,
    ) -> None:
        self.bin_dir = bin_dir
        self.timeout = server_start_timeout
        self.log_dir = Path(log_dir) if log_dir else Path(tempfile.gettempdir())
        self.port_retries = port_retries

    def _resolve_exe(self) -> Path:
        exe = find_binary(self.bin_dir, "llama-server")
        if exe is None:
            raise ServerError(
                "llama-server binary not found "
                f"(llama.bin_dir={self.bin_dir!r}; not on PATH either)"
            )
        return exe

    @contextmanager
    def serve(self, spec: ServerSpec) -> Iterator[str]:
        """Launch a server for ``spec``, yield its base URL, guarantee teardown.

        ::

            with manager.serve(ocr_spec) as endpoint:
                ... run OCR pass against `endpoint` ...
            # server guaranteed down here
        """
        exe = self._resolve_exe()
        last_err: Exception | None = None
        # Retry on /health timeout with a fresh port (covers the bind TOCTOU race).
        attempts = self.port_retries if spec.port == 0 else 1
        for attempt in range(attempts):
            port = spec.port or _free_port(spec.host)
            base_url = f"http://{spec.host}:{port}"
            log_path = self.log_dir / f"inscriber-{spec.label}-{port}.log"
            args = build_launch_args(exe, spec, port)
            logger.debug("launching: %s", " ".join(args))
            log_file = open(log_path, "w", encoding="utf-8")  # noqa: SIM115
            proc = subprocess.Popen(args, stdout=log_file, stderr=subprocess.STDOUT)
            _track(proc)
            try:
                self._wait_healthy(base_url, proc, log_path)
            except ServerError as e:
                last_err = e
                _terminate(proc)
                log_file.close()
                logger.warning(
                    "server attempt %d/%d failed: %s", attempt + 1, attempts, e
                )
                continue
            # Healthy.
            try:
                logger.info("llama-server (%s) ready at %s", spec.label, base_url)
                yield base_url
            finally:
                _terminate(proc)
                log_file.close()
            return
        raise ServerError(
            f"llama-server ({spec.label}) failed to become healthy after "
            f"{attempts} attempt(s): {last_err}"
        )

    def _wait_healthy(
        self, base_url: str, proc: subprocess.Popen, log_path: Path
    ) -> None:
        """Poll GET /health until 200 (DESIGN §5.3).

        503 = model still loading → keep waiting. Process death or timeout → raise
        with the tail of the server log so the user can diagnose.
        """
        deadline = time.monotonic() + self.timeout
        health_url = base_url + "/health"
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise ServerError(
                    f"llama-server exited early (code {proc.returncode}).\n"
                    f"--- server log tail ---\n{_log_tail(log_path)}"
                )
            try:
                resp = httpx.get(health_url, timeout=2.0)
                if resp.status_code == 200:
                    return
                # 503 while loading — keep waiting (DESIGN §5.3).
            except httpx.HTTPError:
                pass  # connection refused while the socket comes up; keep waiting
            time.sleep(0.5)
        raise ServerError(
            f"timed out after {self.timeout:.0f}s waiting for {health_url}.\n"
            f"--- server log tail ---\n{_log_tail(log_path)}"
        )


@contextmanager
def endpoint_or_serve(
    manager: LlamaServerManager, endpoint: str, spec: ServerSpec
) -> Iterator[str]:
    """No-spawn branch (DESIGN §5.1): if ``endpoint`` is set, yield it unchanged
    (talk to a user-managed server); otherwise spawn via ``manager.serve``.
    """
    if endpoint:
        logger.info("using existing %s endpoint: %s", spec.label, endpoint)
        yield endpoint.rstrip("/")
    else:
        with manager.serve(spec) as base_url:
            yield base_url
