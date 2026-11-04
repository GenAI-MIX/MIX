"""노출 환경 계산 (이미지를 축소하지 않고 계산으로).

- 표시 크기(pt) = 글자 높이(px) × (표시 폭 pt / 원본 폭 px)
- 위험고지/헤드라인 글자 높이 비율
- bbox 내부 글자색·배경색의 WCAG 2.x 명암비 (원본 해상도에서 측정: 업스케일 링잉 영향 배제)
"""
import numpy as np

from .geometry import clip_box_int, ink_mask, to_gray

TYPES = ("risk_notice", "headline", "return_claim", "other", "unknown")


def rel_luminance(rgb):
    c = np.asarray(rgb, dtype=float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def contrast_ratio(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _glyph_height(row_counts, peak_frac):
    n = len(row_counts)
    c0, c1 = n // 4, max(n // 4 + 1, 3 * n // 4)
    peak = row_counts[c0:c1].max()
    if peak == 0:
        return None
    on = row_counts >= peak_frac * peak
    top = bottom = c0 + int(np.argmax(row_counts[c0:c1]))
    while top > 0 and on[top - 1]:
        top -= 1
    while bottom < n - 1 and on[bottom + 1]:
        bottom += 1
    h = bottom - top + 1
    return int(h) if h < 0.9 * n else None


def analyze_block(img_np, bbox, cfg):
    h_img, w_img = img_np.shape[:2]
    x0, y0, x1, y1 = clip_box_int(bbox, w_img, h_img)
    empty = {"ink_h_px": None, "fg_rgb": None, "bg_rgb": None, "contrast_ratio": None, "contrast_conf": "low"}
    if x1 - x0 < 3 or y1 - y0 < 3:
        return empty
    reg = img_np[y0:y1, x0:x1].astype(float)
    ink = ink_mask(to_gray(reg))
    n_fg = int(ink.sum())
    if n_fg == 0 or n_fg == ink.size:
        return empty
    lum = rel_luminance(reg)
    pad = np.pad(ink, 1)
    near = np.zeros_like(ink)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            near |= pad[dy:dy + ink.shape[0], dx:dx + ink.shape[1]]
    bg = ~near if (~near).sum() >= cfg["min_fg_pixels"] else ~ink
    bg_rgb = np.median(reg[bg], axis=0)
    bg_l = float(rel_luminance(bg_rgb))
    d = np.abs(lum[ink] - bg_l)
    k = max(1, int(len(d) * cfg["fg_extreme_quantile"]))
    fg_rgb = np.median(reg[ink][np.argsort(d)[-k:]], axis=0)
    cr = contrast_ratio(float(rel_luminance(fg_rgb)), bg_l)
    unreliable = (float(lum[bg].std()) > cfg["low_conf_bg_std"] or n_fg < cfg["min_fg_pixels"]
                  or ink.mean() > 0.6)
    ink_h = _glyph_height(ink.sum(axis=1).astype(float), cfg["ink_peak_frac"])
    return {
        "ink_h_px": ink_h,
        "fg_rgb": [int(round(v)) for v in fg_rgb],
        "bg_rgb": [int(round(v)) for v in bg_rgb],
        "contrast_ratio": round(cr, 2),
        "contrast_conf": "low" if unreliable else "high",
    }


def add_block_metrics(blocks, img_np, profiles, cfg):
    width = img_np.shape[1]
    for b in blocks:
        b.update(analyze_block(img_np, b["bbox"], cfg))
    paras = {}
    for b in blocks:
        if b["ink_h_px"]:
            paras.setdefault(b["paragraph_id"], []).append(b)
    para_med = {pid: (float(np.median([b["ink_h_px"] for b in bs])), float(np.median([b["h_px"] for b in bs])), len(bs))
                for pid, bs in paras.items()}
    for b in blocks:
        med_ink, med_box, n = para_med.get(b["paragraph_id"], (None, None, 0))
        outlier = (b["ink_h_px"] and n >= 3 and b["ink_h_px"] > cfg["ink_outlier_ratio"] * med_ink
                   and b["h_px"] <= 1.2 * med_box)
        if b["ink_h_px"] and not outlier:
            b["text_h"], b["text_h_src"] = float(b["ink_h_px"]), "ink"
        elif med_ink:
            b["text_h"], b["text_h_src"] = med_ink, "paragraph"
        else:
            b["text_h"], b["text_h_src"] = b["h_px"], "box"
        b["display_pt"] = {p: round(b["text_h"] * pt / width, 2) for p, pt in profiles.items()}


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"min": round(float(np.min(vals)), 2), "median": round(float(np.median(vals)), 2),
            "max": round(float(np.max(vals)), 2)}


def _scaled(stats, f):
    return {k: round(v * f, 2) for k, v in stats.items()} if stats else None


def ad_metrics(blocks, width, profiles, content_box, low_conf):
    ppp = {p: pt / width for p, pt in profiles.items()}
    out = {
        "display_profiles": {p: {"display_width_pt": pt, "pt_per_px": round(pt / width, 5)} for p, pt in profiles.items()},
        "content_bbox": content_box,
        "type_counts": {t: sum(b["type"] == t for b in blocks) for t in TYPES},
        "risk_notice": None,
        "headline": None,
        "risk_to_headline_ratio": None,
    }
    risk = [b for b in blocks if b["type"] == "risk_notice"]
    if risk:
        h = _stats([b["text_h"] for b in risk])
        out["risk_notice"] = {
            "block_ids": [b["id"] for b in risk],
            "text_h_px": h,
            "display_pt": {p: _scaled(h, f) for p, f in ppp.items()},
            "contrast_ratio": _stats([b["contrast_ratio"] for b in risk]),
            "contrast_low_conf_count": sum(b["contrast_conf"] == "low" for b in risk),
        }
    heads = [b for b in blocks if b["size_tier"] == "headline" and b["confidence"] >= low_conf]
    if heads:
        ref = max(heads, key=lambda b: b["text_h"])
        out["headline"] = {
            "block_ids": [b["id"] for b in heads],
            "ref_block_id": ref["id"],
            "text_h_px_max": round(ref["text_h"], 2),
            "display_pt_max": {p: round(ref["text_h"] * f, 2) for p, f in ppp.items()},
        }
    if risk and heads:
        rh, hh = out["risk_notice"]["text_h_px"], ref["text_h"]
        out["risk_to_headline_ratio"] = {"median": round(rh["median"] / hh, 4), "min": round(rh["min"] / hh, 4)}
    return out
