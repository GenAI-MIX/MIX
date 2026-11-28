"""진단: 05 노트북의 A2·A3·A4 코드 판정과 같은 식으로 광고를 보고, 어느 문단을 어떻게 고칠지 정한다.

- A2 위험고지 최소 명암비 < 3.0        → 해당 블록만 글자색 밝기 조정 (L1)
- A3 위험고지 표시 크기 중앙값 < 5.9pt   → 작은 고지 문단만 다시 조판 (L2/L3)
- A4 위험고지/헤드라인 높이비 < 0.075   → 위와 같음 (목표 높이 = 두 기준 중 큰 쪽)
- 셋 다 통과하면 '원본 유지' — 수정안을 만들지 않는다.
"""
import numpy as np


def ad_checks(m, th):
    mf = m.get("mobile_feed", {})
    rows = [("A2", "명암비", m.get("risk_contrast_min"), th["contrast"], ""),
            ("A3", "글자 크기(pt)", mf.get("risk_median_pt"), th["image_pt"], "pt"),
            ("A4", "헤드라인 대비", m.get("risk_headline_ratio"), th["ratio_image"], "")]
    return [{"id": i, "name": n, "value": v, "threshold": t,
             "ok": None if v is None else bool(v >= t)} for i, n, v, t, _ in rows]


def is_logo(b, ad):
    """'NH-Amundi자산운용' 같은 로고·브랜드 표기는 글자로 다시 쓰면 안 되므로 제외."""
    t = b["text"].replace(" ", "")
    brand = (ad.get("brand") or "").replace(" ", "")
    return len(t) <= 14 and bool(brand) and brand[:2] in t and "※" not in t


def diagnose(ad, th, ladder):
    W = ad["size"][0]
    px_per_pt = W / th["display_width_pt"]
    blocks = {b["id"]: b for b in ad["ocr_blocks"]}
    checks = ad_checks(ad["metrics"], th)
    bad = {c["id"] for c in checks if c["ok"] is False}
    risk_paras = [p for p in ad["paragraphs"] if p["type"] == "risk_notice"]
    if not risk_paras:
        return {"decision": "고지 없음", "checks": checks, "targets": [],
                "note": "화면에서 위험고지 문단을 찾지 못함 — 문구 추가는 B항목(LLM 판정)·디자이너 검토"}
    if not bad:
        return {"decision": "원본 유지", "checks": checks, "targets": [],
                "note": "A2·A3·A4 모두 기준 통과 — 원본 디자인 그대로 진행"}

    head = [b["h_px"] for b in ad["ocr_blocks"] if b["type"] == "headline" and b.get("h_px")]
    need_h = 0.0
    if "A3" in bad:
        need_h = th["image_pt"] * px_per_pt
    if "A4" in bad and head:
        need_h = max(need_h, th["ratio_image"] * max(head))
    need_h *= ladder["margin"]

    targets = []
    for p in risk_paras:
        bs = [blocks[i] for i in p["block_ids"] if i in blocks and not is_logo(blocks[i], ad)]
        hs_all = [b["h_px"] for b in bs if b.get("h_px")]
        if len(hs_all) >= 3:   # 문단 안에서 혼자 큰 짧은 글자(로고)는 제외
            med = float(np.median(hs_all))
            bs = [b for b in bs if not (b.get("h_px") and b["h_px"] > 1.6 * med and len(b["text"]) <= 10)]
        hs = [b["h_px"] for b in bs if b.get("h_px")]
        if not bs:
            continue
        h = float(np.median(hs)) if hs else None
        low = [b["id"] for b in bs if b.get("contrast") is not None and b["contrast"] < th["contrast"]]
        resize = bool(need_h and h and h < need_h)
        if resize or ("A2" in bad and low):
            targets.append({"paragraph": p["id"], "bbox": p["bbox"], "blocks": [b["id"] for b in bs],
                            "action": "retypeset" if resize else "recolor",
                            "h_px": h, "target_h_px": round(need_h, 1) if resize else h,
                            "low_contrast_blocks": low})
    return {"decision": "수정안", "checks": checks, "bad": sorted(bad), "targets": targets, "need_h_px": need_h}


def host_target(ad, dg):
    """추가 문구를 붙일 문단 = 글자가 가장 많은 위험고지 문단(본 고지문). 이미 고칠 대상이면 그것."""
    paras = [p for p in ad["paragraphs"] if p["type"] == "risk_notice"]
    if not paras:
        return None
    host = max(paras, key=lambda p: len(p["text"]))     # 표 칸·짧은 줄이 아닌 본 고지 문단
    for t in dg.get("targets", []):
        if t["paragraph"] == host["id"]:
            t["action"] = "retypeset"
            return t
    blocks = {b["id"]: b for b in ad["ocr_blocks"]}
    bs = [blocks[i] for i in host["block_ids"] if i in blocks and not is_logo(blocks[i], ad)]
    hs = [b["h_px"] for b in bs if b.get("h_px")]
    if not bs or not hs:
        return None
    h = float(np.median(hs))
    t = {"paragraph": host["id"], "bbox": host["bbox"], "blocks": [b["id"] for b in bs], "action": "retypeset",
         "h_px": h, "target_h_px": max(h, dg.get("need_h_px") or 0), "low_contrast_blocks": []}
    dg.setdefault("targets", []).append(t)
    return t
