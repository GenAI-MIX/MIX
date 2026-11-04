"""VLM 입력 패키지 (위험고지 crop + 광고별 JSON) 및 전체 요약 CSV."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .geometry import clip_box_int, otsu_threshold, overlap_min, to_gray, union_box


def _ink_fn(gray, box):
    x0, y0, x1, y1 = box
    core = gray[y0:y1, x0:x1]
    t = otsu_threshold(core)
    hi = core > t
    ring = np.concatenate([hi[:2].ravel(), hi[-2:].ravel(), hi[:, :2].ravel(), hi[:, -2:].ravel()])
    bg_is_hi = ring.mean() > 0.5
    return (lambda a: a <= t) if bg_is_hi else (lambda a: a > t)


def _min_ink(x0, y0, x1, y1):
    return max(3, int(0.005 * (x1 - x0))), max(3, int(0.05 * (y1 - y0)))


def _ink_guard(gray, box, max_expand, is_ink):
    h, w = gray.shape
    x0, y0, x1, y1 = box
    min_row, min_col = _min_ink(x0, y0, x1, y1)
    moved = [0, 0, 0, 0]
    for _ in range(max_expand):
        if y0 > 0 and is_ink(gray[y0, x0:x1]).sum() >= min_row and is_ink(gray[y0 - 1, x0:x1]).sum() >= min_row:
            y0 -= 1
            moved[1] += 1
        else:
            break
    for _ in range(max_expand):
        if y1 < h and is_ink(gray[y1 - 1, x0:x1]).sum() >= min_row and is_ink(gray[y1, x0:x1]).sum() >= min_row:
            y1 += 1
            moved[3] += 1
        else:
            break
    for _ in range(max_expand):
        if x0 > 0 and is_ink(gray[y0:y1, x0]).sum() >= min_col and is_ink(gray[y0:y1, x0 - 1]).sum() >= min_col:
            x0 -= 1
            moved[0] += 1
        else:
            break
    for _ in range(max_expand):
        if x1 < w and is_ink(gray[y0:y1, x1 - 1]).sum() >= min_col and is_ink(gray[y0:y1, x1]).sum() >= min_col:
            x1 += 1
            moved[2] += 1
        else:
            break
    return [x0, y0, x1, y1], moved


def _cut_blocks(blocks, box):
    x0, y0, x1, y1 = box
    cut = []
    for b in blocks:
        bx0, by0, bx1, by1 = b["bbox"]
        cy, qh = (by0 + by1) / 2, (by1 - by0) / 4
        cx0, cy0, cx1, cy1 = bx0 + 1, cy - qh, bx1 - 1, cy + qh
        overlaps = cx0 < x1 and cx1 > x0 and cy0 < y1 and cy1 > y0
        inside = cx0 >= x0 and cx1 <= x1 and cy0 >= y0 and cy1 <= y1
        if overlaps and not inside:
            cut.append(b["id"])
    return cut


def _same_line_closure(blocks, ids, gap_ratio):
    by_id = {b["id"]: b for b in blocks}
    out, frontier = set(ids), list(ids)
    while frontier:
        a = by_id[frontier.pop()]
        ay = (a["bbox"][1] + a["bbox"][3]) / 2
        for b in blocks:
            if b["id"] in out:
                continue
            h = min(a["h_px"], b["h_px"])
            by = (b["bbox"][1] + b["bbox"][3]) / 2
            gap = max(b["bbox"][0] - a["bbox"][2], a["bbox"][0] - b["bbox"][2])
            if abs(ay - by) < 0.5 * h and gap <= gap_ratio * h:
                out.add(b["id"])
                frontier.append(b["id"])
    return out


def _clear_margin(gray, box, margin, is_ink):
    h, w = gray.shape
    x0, y0, x1, y1 = box
    min_row, min_col = _min_ink(x0, y0, x1, y1)

    def clear_run(is_clear):
        k = 0
        while k < margin and is_clear(k):
            k += 1
        return k

    top = clear_run(lambda k: y0 - 1 - k >= 0 and is_ink(gray[y0 - 1 - k, x0:x1]).sum() < min_row)
    bottom = clear_run(lambda k: y1 + k < h and is_ink(gray[y1 + k, x0:x1]).sum() < min_row)
    left = clear_run(lambda k: x0 - 1 - k >= 0 and is_ink(gray[y0:y1, x0 - 1 - k]).sum() < min_col)
    right = clear_run(lambda k: x1 + k < w and is_ink(gray[y0:y1, x1 + k]).sum() < min_col)
    return [x0 - left, y0 - top, x1 + right, y1 + bottom]


def _joins(region_box, line_h, b, cfg):
    h = b["h_px"]
    if max(h, line_h) / max(1e-6, min(h, line_h)) > cfg["join_height_ratio"]:
        return False
    rb, bb = region_box, b["bbox"]
    if max(bb[1] - rb[3], rb[1] - bb[3]) > cfg["join_gap_ratio"] * line_h:
        return False
    ox = min(rb[2], bb[2]) - max(rb[0], bb[0])
    if ox <= 0:
        return False
    similar_span = ox / max(rb[2] - rb[0], bb[2] - bb[0]) >= cfg["join_span_ratio"]
    left_aligned = abs(bb[0] - rb[0]) <= line_h
    return similar_span or left_aligned


def risk_regions(blocks, cfg):
    groups = {}
    for b in blocks:
        if b.get("disclosure_id") is not None:
            groups.setdefault(b["disclosure_id"], []).append(b)
    regions = [{"ids": {b["id"] for b in bs}, "bbox": union_box([b["bbox"] for b in bs]),
                "line_h": float(np.median([b["h_px"] for b in bs]))} for bs in groups.values()]
    seeds = (b for b in blocks if b["type"] == "risk_notice" and b.get("disclosure_id") is None)
    for b in sorted(seeds, key=lambda b: b["bbox"][1]):
        for r in regions:
            if _joins(r["bbox"], r["line_h"], b, cfg):
                r["ids"].add(b["id"])
                r["bbox"] = union_box([r["bbox"], b["bbox"]])
                break
        else:
            regions.append({"ids": {b["id"]}, "bbox": list(b["bbox"]), "line_h": b["h_px"]})
    for r in regions:                         # 기준 줄높이는 위험고지 줄로만 고정 (확장 중 흔들리지 않게)
        r["line_h"] = float(np.median([b["h_px"] for b in blocks if b["id"] in r["ids"]]))
    grown = True
    while grown:
        grown = False
        for r in regions:
            for b in blocks:
                if b["id"] not in r["ids"] and _joins(r["bbox"], r["line_h"], b, cfg):
                    r["ids"].add(b["id"])
                    r["bbox"] = union_box([r["bbox"], b["bbox"]])
                    grown = True
    merged = True
    while merged:
        merged = False
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if overlap_min(regions[i]["bbox"], regions[j]["bbox"]) > 0:
                    a, b = regions[i], regions.pop(j)
                    a["ids"] |= b["ids"]
                    a["bbox"] = union_box([a["bbox"], b["bbox"]])
                    merged = True
                    break
            if merged:
                break
    return sorted(regions, key=lambda r: r["bbox"][1])


def make_risk_crops(blocks, img, ad_id, out_dir, cfg, rel_to):
    img_np = np.asarray(img)
    gray = to_gray(img_np.astype(float))
    h, w = gray.shape
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    crops = []
    by_id = {b["id"]: b for b in blocks}
    for k, r in enumerate(risk_regions(blocks, cfg)):
        line_h = r["line_h"]
        margin = int(min(cfg["margin_max_px"], max(cfg["margin_min_px"], round(cfg["margin_line_ratio"] * line_h))))
        for _ in range(cfg["max_expand_rounds"]):
            text_box = clip_box_int(r["bbox"], w, h)
            is_ink = _ink_fn(gray, text_box)
            guarded, moved = _ink_guard(gray, text_box, max(1, int(cfg["ink_guard_max_ratio"] * line_h)), is_ink)
            final = _clear_margin(gray, guarded, margin, is_ink)
            cut = _cut_blocks(blocks, final)
            if not cut:
                break
            # 경계에 걸친 텍스트 블록이 있으면 그 블록과, 같은 줄에 바로 이어지는 블록(표의 값 칸 등)까지
            # 포함하도록 넓혀 다시 계산 (글자도, 문장·표의 한 행도 잘리지 않게)
            cut = _same_line_closure(blocks, set(cut), cfg["same_line_gap_ratio"]) - r["ids"]
            r["ids"] |= cut
            r["bbox"] = union_box([r["bbox"]] + [by_id[i]["bbox"] for i in cut])
        path = out_dir / f"{ad_id}_risk_{k:02d}.png"
        img.crop(tuple(final)).save(path)
        crops.append({
            "path": str(path.relative_to(rel_to)),
            "bbox": final,
            "block_ids": sorted(r["ids"]),
            "ink_guard_expand_px": moved,      # [좌, 상, 우, 하] 글자 잘림 방지로 넓힌 양
            "cut_block_ids": (cut := _cut_blocks(blocks, final)),
            "edge_clear": not cut,
        })
    return crops


TYPES = ("risk_notice", "headline", "return_claim", "other", "unknown")


def slim_block(b):
    reliable = b["contrast_conf"] == "high" and b["contrast_ratio"] is not None
    return {"id": b["id"], "text": b["text"], "bbox": [round(v) for v in b["bbox"]],
            "conf": round(b["confidence"], 2), "type": b["type"], "h_px": round(b["text_h"], 1),
            "contrast": round(b["contrast_ratio"], 1) if reliable else None}


def slim_metrics(m, blocks):
    rn, hd = m["risk_notice"] or {}, m["headline"] or {}
    cr = [b["contrast_ratio"] for b in blocks if b["type"] == "risk_notice"
          and b["contrast_conf"] == "high" and b["contrast_ratio"] is not None]
    out = {"risk_headline_ratio": round(m["risk_to_headline_ratio"]["median"], 3) if m["risk_to_headline_ratio"] else None,
           "risk_contrast_min": round(min(cr), 2) if cr else None}
    for p, prof in m["display_profiles"].items():
        dp = (rn.get("display_pt") or {}).get(p) or {}
        out[p] = {"width_pt": prof["display_width_pt"], "risk_min_pt": dp.get("min"),
                  "risk_median_pt": dp.get("median"), "headline_max_pt": (hd.get("display_pt_max") or {}).get(p)}
    return out


def slim_paragraph(k, idx, blocks):
    bs = [blocks[i] for i in idx]
    weight = {}
    for b in bs:
        weight[b["type"]] = weight.get(b["type"], 0) + len(b["text"])
    return {"id": k, "type": max(weight, key=weight.get), "text": " ".join(b["text"].strip() for b in bs),
            "bbox": [round(v) for v in union_box([b["bbox"] for b in bs])], "block_ids": [b["id"] for b in bs]}


def build_ad_json(ad_id, meta_row, size, fmt, image_path, blocks, text_paras, metrics, crops):
    return {
        "ad_id": ad_id,
        "media": meta_row.get("media") or None,
        "product": meta_row.get("product") or None,
        "brand": meta_row.get("brand") or None,
        "size": list(size),
        "format": fmt,
        "image": image_path,
        "ocr_blocks": [slim_block(b) for b in blocks],
        "paragraphs": [slim_paragraph(k, idx, blocks) for k, idx in enumerate(text_paras)],
        "metrics": slim_metrics(metrics, blocks),
        "crops": [{"path": c["path"], "bbox": c["bbox"]} for c in crops],
        "vlm_judgment": None,
        "label": None,
    }


def dumps_ad_json(doc):
    c = lambda v: json.dumps(v, ensure_ascii=False)
    items = list(doc.items())
    lines = ["{"]
    for i, (k, v) in enumerate(items):
        end = "," if i < len(items) - 1 else ""
        if k in ("ocr_blocks", "paragraphs") and v:
            lines.append(f"  {c(k)}: [")
            lines += [f"    {c(b)}{',' if j < len(v) - 1 else ''}" for j, b in enumerate(v)]
            lines.append(f"  ]{end}")
        else:
            lines.append(f"  {c(k)}: {c(v)}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_summary_csv(processed_dir, out_path, profiles):
    cols = ["ad_id", "format", "width", "height", "media", "product", "brand", "n_blocks", "mean_conf"]
    cols += [f"n_{t}" for t in TYPES]
    for p in profiles:
        cols += [f"risk_min_pt@{p}", f"risk_median_pt@{p}", f"headline_max_pt@{p}"]
    cols += ["risk_headline_ratio", "risk_contrast_min", "n_crops"]
    rows = []
    for jp in sorted(Path(processed_dir).glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        m, blocks = d["metrics"], d["ocr_blocks"]
        row = {"ad_id": d["ad_id"], "format": d["format"], "width": d["size"][0], "height": d["size"][1],
               "media": d["media"] or "", "product": d["product"] or "", "brand": d.get("brand") or "",
               "n_blocks": len(blocks),
               "mean_conf": round(float(np.mean([b["conf"] for b in blocks])), 3) if blocks else "",
               "risk_headline_ratio": m["risk_headline_ratio"], "risk_contrast_min": m["risk_contrast_min"],
               "n_crops": len(d["crops"])}
        for t in TYPES:
            row[f"n_{t}"] = sum(b["type"] == t for b in blocks)
        for p in profiles:
            pm = m.get(p) or {}
            row[f"risk_min_pt@{p}"] = pm.get("risk_min_pt")
            row[f"risk_median_pt@{p}"] = pm.get("risk_median_pt")
            row[f"headline_max_pt@{p}"] = pm.get("headline_max_pt")
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:   # 엑셀에서 한글이 깨지지 않도록 BOM
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return len(rows)
