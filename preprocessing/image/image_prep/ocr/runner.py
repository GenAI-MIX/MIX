import re
import numpy as np
from PIL import Image
from ..geometry import box_area, overlap_min, quad_bbox, quad_height

HAS_TEXT = re.compile(r"[0-9A-Za-z가-힣]")


def _as_dict(block, source):
    return {"text": block.text, "quad": block.quad, "confidence": block.confidence,
            "bbox": quad_bbox(block.quad), "source": source}


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _is_duplicate(c, accepted, contain):
    covered = 0
    for a in accepted:
        if _iou(c["bbox"], a["bbox"]) >= 0.5:
            return True
        if overlap_min(c["bbox"], a["bbox"]) < contain:
            continue
        if box_area(c["bbox"]) <= box_area(a["bbox"]):
            return True                                
        if max(c["h"], a["h"]) / max(1e-6, min(c["h"], a["h"])) <= 1.5:
            return True                                  
        covered += 1
    return covered >= 2                                 


def merge_candidates(cands, cfg):
    for c in cands:
        c["h"] = quad_height(c["quad"])
        c["_prefer"] = 0 if (c["source"] == "upscaled" and c["h"] < cfg["small_text_px"]) else 1
        c["_score"] = len(re.sub(r"\s+", "", c["text"])) * c["confidence"]
    accepted = []
    for c in sorted(cands, key=lambda c: (c["_prefer"], -c["_score"])):
        if not _is_duplicate(c, accepted, cfg["merge_overlap"]):
            accepted.append(c)
    return [{k: v for k, v in c.items() if not k.startswith("_") and k != "h"} for c in accepted]


def run_multi_pass(engine, img, img_np, up_np, input_meta, input_cfg, ocr_cfg):
    stats = {}
    cands = []

    s = ocr_cfg["large_text_width"] / img.width
    if s < 0.8:                                          
        small = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
        sx, sy = small.width / img.width, small.height / img.height
        for b in engine.recognize(np.asarray(small)):
            b.quad = [[x / sx, y / sy] for x, y in b.quad]
            cands.append(_as_dict(b, "small"))
    stats["small_raw"] = len(cands)

    full = [_as_dict(b, "full") for b in engine.recognize(img_np)]
    stats["full_raw"] = len(full)
    cands += full

    sx, sy = input_meta["sx"], input_meta["sy"]
    strips = input_meta["strips"]
    edge = input_cfg["edge_margin"]
    n_up, dropped_edge = 0, 0
    for st in strips:
        h = st["y1"] - st["y0"]
        for b in engine.recognize(up_np[st["y0"]:st["y1"]]):
            ys = [p[1] for p in b.quad]
            if (st["index"] > 0 and min(ys) <= edge) or (st["index"] < len(strips) - 1 and max(ys) >= h - edge):
                dropped_edge += 1
                continue
            b.quad = [[x / sx, (y + st["y0"]) / sy] for x, y in b.quad]
            cands.append(_as_dict(b, "upscaled"))
            n_up += 1
    stats["upscaled_raw"] = n_up
    stats["upscaled_dropped_at_strip_edge"] = dropped_edge

    kept = [c for c in cands if c["confidence"] >= ocr_cfg["drop_below_conf"]
            and quad_height(c["quad"]) >= ocr_cfg["min_text_h_px"] and HAS_TEXT.search(c["text"])]
    stats["dropped_noise"] = len(cands) - len(kept)
    merged = merge_candidates(kept, ocr_cfg)
    stats["merged"] = len(merged)
    return merged, stats
