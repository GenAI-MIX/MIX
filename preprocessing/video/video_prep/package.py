import csv
import difflib
import json
from pathlib import Path
from image_prep.classify import strip_text
from image_prep.geometry import overlap_min, union_box
from .cards import union_length


def _merge(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def intersect_length(a, b):
    return sum(max(0.0, min(x[1], y[1]) - max(x[0], y[0])) for x in _merge(a) for y in _merge(b))


def video_metrics(cards, csv_intervals):
    spans = [(c["t_start"], c["t_end"]) for c in cards]
    csv_s = union_length(csv_intervals)
    return {"n_cards": len(cards), "n_static": sum(c["kind"] == "static" for c in cards),
            "n_crawl": sum(c["kind"] == "crawl" for c in cards),
            "candidate_exposure_s": round(union_length(spans), 2), "csv_exposure_s": round(csv_s, 2),
            "csv_coverage": round(intersect_length(csv_intervals, spans) / csv_s, 3) if csv_s else None,
            "min_line_h_px": min((c["line_h_px"] for c in cards), default=None)}


def _same_text(a, b, min_ratio):
    a, b = strip_text(a or ""), strip_text(b or "")
    if not a or not b:
        return False
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sm.find_longest_match(0, len(a), 0, len(b)).size >= min_ratio * min(len(a), len(b)) or sm.ratio() >= min_ratio


def risk_exposures(cards, cfg):
    out = []
    for c in sorted((c for c in cards if c.get("type") == "risk_notice"), key=lambda c: c["t_start"]):
        for e in out:
            if (c["t_start"] - e["t_end"] <= cfg["exposure_gap_s"] and overlap_min(c["bbox"], e["bbox"]) >= 0.5
                    and _same_text(c["text"], e["_text"], cfg["exposure_text_ratio"])):
                e["t_end"] = max(e["t_end"], c["t_end"])
                e["bbox"] = union_box([e["bbox"], c["bbox"]])
                e["card_ids"].append(c["id"])
                if c["risk_chars"] > e["risk_chars"]:                  # 같은 문구가 반복되므로 합치지 않고 가장 온전한 것
                    e["risk_chars"], e["_text"], e["text_card_id"] = c["risk_chars"], c["text"], c["id"]
                break
        else:
            out.append({"t_start": c["t_start"], "t_end": c["t_end"], "bbox": list(c["bbox"]), "card_ids": [c["id"]],
                        "kind": c["kind"], "risk_chars": c["risk_chars"], "_text": c["text"], "text_card_id": c["id"]})
    for i, e in enumerate(out):
        e["text"] = e.pop("_text")
        e["duration"] = round(e["t_end"] - e["t_start"], 2)
        e["risk_cps"] = round(e["risk_chars"] / e["duration"], 1) if e["duration"] > 0 and e["risk_chars"] else None
        e["id"] = i
    return [{k: e[k] for k in ("id", "t_start", "t_end", "duration", "kind", "card_ids", "risk_chars", "risk_cps",
                                "bbox", "text_card_id", "text")} for e in out]


def risk_metrics(cards, exposures, csv_intervals, profiles, readable_cps):
    risk = [c for c in cards if c.get("type") == "risk_notice"]
    spans = [(c["t_start"], c["t_end"]) for c in risk]
    csv_s = union_length(csv_intervals)
    hs = [c["risk_text_h_px"]["min"] for c in risk if c.get("risk_text_h_px")]
    cps = [e["risk_cps"] for e in exposures if e["risk_cps"]]
    cr = [c["risk_contrast_min"] for c in risk if c.get("risk_contrast_min") is not None]
    ratio = [c["risk_headline_ratio"] for c in risk if c.get("risk_headline_ratio")]
    return {"n_risk_cards": len(risk), "n_risk_crawl": sum(c["kind"] == "crawl" for c in risk),
            "n_risk_exposures": len(exposures),
            "risk_short_exposures": sum(e["duration"] < 1.0 for e in exposures),   # 1초 미만: 흐린 고지가 끊겨 잡혔거나 장면 속 글자일 수 있어 확인 필요
            "risk_exposure_s": round(union_length(spans), 2),
            "risk_csv_coverage": round(intersect_length(csv_intervals, spans) / csv_s, 3) if csv_s else None,
            "risk_chars": sum(c["risk_chars"] for c in risk),
            "risk_cps_max": max(cps, default=None),
            "risk_unreadable_exposures": sum(v > readable_cps for v in cps),
            "risk_text_h_min_px": min(hs, default=None),
            "risk_display_min_pt": {p: c_min for p, c_min in (
                (p, min((c["risk_display_pt"][p]["min"] for c in risk if c.get("risk_display_pt")), default=None))
                for p in profiles)},
            "risk_contrast_min": min(cr, default=None),
            "risk_headline_ratio_min": min(ratio, default=None)}


def build_video_json(src, meta_row, content_bbox, cards, screens, metrics, video_path, review_no=None,
                     ocr_blocks=None, paragraphs=None, exposures=None):
    return {
        "ad_id": src.ad_id,
        "media": meta_row.get("media") or None,
        "product": meta_row.get("product") or None,
        "brand": meta_row.get("brand") or None,
        "title": meta_row.get("title") or None,
        "video": video_path,
        "size": [src.info["width"], src.info["height"]],
        "fps": src.info["fps"],
        "duration": src.info["duration"],
        "has_audio": src.info["has_audio"],
        "content_bbox": content_bbox,
        "review_no": review_no,
        "csv": src.csv_label,
        "screens": screens,
        "cards": cards,
        "risk_exposures": exposures,
        "ocr_blocks": ocr_blocks,
        "paragraphs": paragraphs,
        "metrics": metrics,
        "label": None,
    }


def dumps_video_json(doc):
    c = lambda v: json.dumps(v, ensure_ascii=False)
    items = list(doc.items())
    lines = ["{"]
    for i, (k, v) in enumerate(items):
        end = "," if i < len(items) - 1 else ""
        if k in ("screens", "cards", "risk_exposures", "ocr_blocks", "paragraphs") and v:
            lines.append(f"  {c(k)}: [")
            lines += [f"    {c(b)}{',' if j < len(v) - 1 else ''}" for j, b in enumerate(v)]
            lines.append(f"  ]{end}")
        else:
            lines.append(f"  {c(k)}: {c(v)}{end}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_summary_csv(json_dir, out_path, profiles):
    risk_cols = ["n_risk_cards", "n_risk_crawl", "n_risk_exposures", "risk_short_exposures", "risk_exposure_s",
                 "risk_csv_coverage",
                 "risk_chars", "risk_cps_max", "risk_unreadable_exposures", "risk_text_h_min_px"]
    cols = ["ad_id", "media", "brand", "width", "height", "duration", "csv_risk", "n_screens", "n_cards", "n_static",
            "n_crawl", "candidate_exposure_s", "csv_exposure_s", "csv_coverage", "min_line_h_px", "letterboxed"]
    cols += risk_cols + [f"risk_display_min_pt@{p}" for p in profiles]
    cols += ["risk_contrast_min", "risk_headline_ratio_min", "review_no"]
    rows = []
    for jp in sorted(Path(json_dir).glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        m = d["metrics"]
        row = {"ad_id": d["ad_id"], "media": d["media"] or "", "brand": d["brand"] or "", "width": d["size"][0],
               "height": d["size"][1], "duration": d["duration"], "csv_risk": d["csv"]["risk"],
               "n_screens": len(d["screens"]), "letterboxed": d["content_bbox"] != [0, 0, *d["size"]],
               "review_no": " / ".join(d.get("review_no") or []),
               **{k: m[k] for k in cols if k in m}}
        for p in profiles:
            row[f"risk_display_min_pt@{p}"] = (m.get("risk_display_min_pt") or {}).get(p)
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:   # 엑셀에서 한글이 깨지지 않도록 BOM
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    return rows
