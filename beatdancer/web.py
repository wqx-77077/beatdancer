"""Build the self-contained web demo (works from file://, no server needed)."""
from __future__ import annotations

import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(os.path.dirname(HERE), "web", "index.html")


def build(choreo_dict: dict, wav_path: str, out_path: str,
          template: str = TEMPLATE) -> str:
    with open(template, "r", encoding="utf-8") as f:
        html = f.read()
    with open(wav_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("ascii")

    data = json.dumps(choreo_dict, ensure_ascii=False, separators=(",", ":"))
    html = html.replace("__CHOREO__", data).replace("__AUDIO__", audio_b64)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
