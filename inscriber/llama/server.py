"""llama.cpp server lifecycle: spawn / health / teardown (DESIGN §5).

``inscriber`` owns the ``llama-server`` process by default: launch it with the
right model / projector / flags, wait for ``/health``, run a pass, terminate it.
A power-user escape hatch (``--ocr-endpoint`` / ``--vlm-endpoint``) talks to an
already-running server instead — see :func:`endpoint_or_serve`.

Cross-platform notes (DESIGN §5.3, §15):
* list-args only, never ``shell=True``;
* binary discovery appends ``.exe`` on Windows (via :func:`config.find_binary`);
* ``Popen.terminate()`` maps to ``TerminateProcess`` (Windows) / ``SIGTERM``
  (POSIX) — both fine; we avoid POSIX-only ``os.killpg`` / ``preexec_fn``.
"""

from __future__ import annotations

import atexit
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
# interpreter exit, so a crash / hard exit never orphans a llama-server.
_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


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


def _register_atexit() -> None:
    global _ATEXIT_REGISTERED
    if not _ATEXIT_REGISTERED:
        atexit.register(_terminate_all)
        _ATEXIT_REGISTERED = True


def _terminate_all() -> None:  # pragma: no cover - exercised only on process exit
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE)
    for proc in procs:
        _terminate(proc)


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
    # use its own default (which is GPU auto-fit on modern builds, e.g. 9028). This
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
    _register_atexit()
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
