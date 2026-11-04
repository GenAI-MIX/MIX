import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def compute_scale(width, cfg):
    s = cfg["upscale_target_width"] / width
    return min(cfg["upscale_max"], max(cfg["upscale_min"], s))


def make_strips(height, max_strip_h, min_overlap):
    if height <= max_strip_h:
        return [(0, height)]
    n = math.ceil((height - min_overlap) / (max_strip_h - min_overlap))
    strip_h = math.ceil((height + (n - 1) * min_overlap) / n)
    stride = (height - strip_h) / (n - 1)
    return [(round(i * stride), round(i * stride) + strip_h) for i in range(n)]


def build_ocr_inputs(img, out_dir, cfg, save=True):
    scale = compute_scale(img.width, cfg)
    up_w, up_h = round(img.width * scale), round(img.height * scale)
    up = img.resize((up_w, up_h), Image.Resampling.LANCZOS) if scale != 1.0 else img
    up_np = np.asarray(up)
    strips = []
    out_dir = Path(out_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
    for i, (y0, y1) in enumerate(make_strips(up_h, cfg["strip_height"], cfg["strip_overlap"])):
        path = out_dir / f"strip_{i:02d}.png"
        if save:
            Image.fromarray(up_np[y0:y1]).save(path)
        strips.append({"index": i, "y0": y0, "y1": y1, "path": str(path)})
    meta = {
        "scale": scale,
        "sx": up_w / img.width,
        "sy": up_h / img.height,
        "upscaled_size": [up_w, up_h],
        "method": "lanczos",
        "strips": strips,
    }
    if save:
        (out_dir / "strips.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return up_np, meta
