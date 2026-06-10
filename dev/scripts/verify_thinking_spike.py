"""Real-hardware spike: verify Gemma 4 thinking toggles via chat_template_kwargs.

Starts the configured VLM server (config.toml), sends the same tiny image +
question with ``enable_thinking`` true / false / omitted, and prints
``finish_reason`` + ``completion_tokens`` + the answer head for each. Thinking
ON should spend noticeably more completion tokens than the visible answer.

Usage:  python dev/scripts/verify_thinking_spike.py
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from inscriber.cli import build_run_config
from inscriber.llama.client import ChatClient
from inscriber.llama.server import LlamaServerManager, ServerSpec
from inscriber.logging import setup_logging


def main() -> None:
    setup_logging(1, False)
    cfg = build_run_config(["run", "dummy.pdf", "--config", "config.toml"])

    log_dir = Path("tmp-verify")
    log_dir.mkdir(exist_ok=True)
    mgr = LlamaServerManager(
        cfg.llama.bin_dir,
        server_start_timeout=cfg.llama.server_start_timeout,
        log_dir=log_dir,
    )
    spec = ServerSpec(
        model=cfg.vlm.model,
        mmproj=cfg.vlm.mmproj,
        host=cfg.llama.host,
        port=cfg.llama.port,
        ctx_size=cfg.llama.ctx_size,
        n_gpu_layers=cfg.vlm.n_gpu_layers,
        extra_flags=[],
        chat_template=None,
        label="vlm-thinking-spike",
    )

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), "navy").save(buf, "PNG")
    png = buf.getvalue()
    prompt = "What color is this image? Answer with one word only."

    cases = [
        ("enable_thinking=True ", {"enable_thinking": True}),
        ("enable_thinking=False", {"enable_thinking": False}),
        ("kwarg omitted        ", None),
    ]
    with mgr.serve(spec) as url:
        client = ChatClient(url)
        for label, kwargs in cases:
            out = client.chat_image(
                image_png=png,
                prompt=prompt,
                sampling={"temperature": 0, "seed": 0},
                chat_template_kwargs=kwargs,
                timeout_s=300,
            )
            print(
                f"{label}: finish_reason={client.last_finish_reason!r} "
                f"completion_tokens={client.last_completion_tokens} "
                f"answer={out.strip()[:80]!r}"
            )


if __name__ == "__main__":
    main()
