"""M1a grounding probe: find what activates DeepSeek-OCR grounding via llama-server.

The first spike showed the image round-trips but NO <|ref|>/<|det|> grounding spans
appear (clean markdown, figure skipped). This probe launches one server session and
tries several message shapes / prompts to find which triggers grounding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from inscriber.llama.client import image_data_url  # noqa: E402
from inscriber.llama.server import LlamaServerManager, ServerSpec  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bin-dir", required=True)
    p.add_argument("--ocr-model", required=True)
    p.add_argument("--ocr-mmproj", required=True)
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--png", default=str(FIXTURES / "calibration_large.png"))
    args = p.parse_args()

    png = Path(args.png).read_bytes()
    data_url = image_data_url(png)
    text_part = lambda t: {"type": "text", "text": t}  # noqa: E731
    img_part = {"type": "image_url", "image_url": {"url": data_url}}

    G = "<|grounding|>Convert the document to markdown."
    variants = [
        ("1 text-first  grounding", [text_part(G), img_part]),
        ("2 image-first grounding", [img_part, text_part(G)]),
        ("3 image-first <|grounding|>OCR", [img_part, text_part("<|grounding|>OCR")]),
        ("4 image-first plain Convert", [img_part, text_part("Convert the document to markdown.")]),
        ("5 image-first plain OCR", [img_part, text_part("OCR")]),
    ]

    mgr = LlamaServerManager(args.bin_dir, server_start_timeout=180)
    spec = ServerSpec(model=args.ocr_model, mmproj=args.ocr_mmproj,
                      n_gpu_layers=args.ngl, ctx_size=8192, label="ocr")
    with mgr.serve(spec) as url:
        for name, content in variants:
            body = {
                "model": "local",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4096,
                "temperature": 0,
                "seed": 0,
                "stream": False,
            }
            try:
                r = httpx.post(f"{url}/v1/chat/completions", json=body, timeout=600)
                out = r.json()["choices"][0]["message"]["content"]
            except Exception as e:  # noqa: BLE001
                out = f"<ERROR: {e}>"
            has_ground = "<|ref|>" in out or "<|det|>" in out
            print(f"\n##### {name}  | grounding={'YES' if has_ground else 'no'} "
                  f"| {len(out)} chars")
            print(out[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
