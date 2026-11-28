"""수정본을 전처리와 같은 식으로 다시 재서 A2·A3·A4를 재판정."""
import numpy as np
from .diagnose import ad_checks


def remeasure(ad, img, changed, drawn, mcfg, th, analyze_block):
    """changed: 원본에서 수정한 블록 id → 'recolor'/'retypeset'. drawn: [(bbox, kind)] 수정본 좌표."""
    W = ad["size"][0]
    s = th["display_width_pt"] / W
    risk_h, risk_c = [], []
    for b in ad["ocr_blocks"]:
        if b["type"] != "risk_notice" or b["id"] in changed:
            continue
        if b.get("h_px"):
            risk_h.append(b["h_px"])
        if b.get("contrast") is not None:
            risk_c.append(b["contrast"])
    measured = []
    for bbox, kind in drawn:
        r = analyze_block(img, bbox, mcfg)
        measured.append({"bbox": bbox, "kind": kind, **r})
        h = r["ink_h_px"]
        if h:
            risk_h.append(h)
        if r["contrast_ratio"] is not None:
            risk_c.append(r["contrast_ratio"])
    head = [b["h_px"] for b in ad["ocr_blocks"] if b["type"] == "headline" and b.get("h_px")]
    m = {"risk_contrast_min": round(min(risk_c), 2) if risk_c else None,
         "mobile_feed": {"risk_median_pt": round(float(np.median(risk_h)) * s, 2) if risk_h else None},
         "risk_headline_ratio": round(float(np.median(risk_h)) / max(head), 3) if risk_h and head else None}
    return ad_checks(m, th), measured
