"""리포트 요약(서비스 결과 화면)용 통합 결과 파일 만들기.

remediate.py / remediate_video.py 결과(JSON)를 광고 하나당 한 항목으로 모아
data/processed/remediation/report_input.json 으로 저장한다. 형식은 README "리포트 연동" 참고.

  python export_summary.py
"""
import datetime
import json
from pathlib import Path

import yaml

CODE_DIR = Path(__file__).resolve().parent


def slim_checks(cs):
    return [{"id": c["id"], "name": c["name"], "value": c.get("value"), "threshold": c.get("threshold"),
             "ok": c.get("ok")} for c in (cs or [])]


def image_entry(rec, panel, fixed):
    best = next((c for c in rec.get("candidates", []) if c.get("recommended")), None)
    out = {
        "decision": rec["decision"],
        "checks_before": slim_checks(rec.get("checks_before")),
        "checks_after": slim_checks(rec.get("checks_after")),
        "recommended": None,
        "candidates": [{"name": c["name"], "label": c["label"], "passed": c.get("passed", False),
                        "why_not": c.get("why", []), "changed_pct": c.get("changed_pct")}
                       for c in rec.get("candidates", [])],
        "designer_brief": rec.get("brief"),
        "panel_image": panel,
        "fixed_image": fixed,
    }
    if best:
        tr = next((s.get("contrast_treatment") for s in best.get("steps", []) if s.get("contrast_treatment")), None)
        out["recommended"] = {
            "name": best["name"], "label": best["label"], "changed_pct": best.get("changed_pct"),
            "size_factor": best.get("size_factor"), "ink_delta_e": best.get("ink_delta_e"),
            "rows_removed": best.get("rows_removed"), "rows_trimmed": best.get("rows_trimmed"),
            "visual_scale": best.get("visual_scale"), "height_added": best.get("height_added"),
            "contrast_treatment": tr,
        }
    return out


def main():
    cfg = yaml.safe_load(open(CODE_DIR / "config.yaml", encoding="utf-8"))
    root = (CODE_DIR / cfg["paths"]["repo_root"]).resolve()
    out = root / cfg["paths"]["out"]
    rel = lambda p: str(p.relative_to(root)) if p.exists() else None

    images = {}
    for key, prof in cfg["profiles"].items():
        jd = out / key / "json"
        for p in sorted(jd.glob("*.json")) if jd.exists() else []:
            rec = json.loads(p.read_text(encoding="utf-8"))
            ad = images.setdefault(rec["ad_id"], {"ad_id": rec["ad_id"], "kind": "image",
                                                  "situation": rec.get("situation"), "profiles": {},
                                                  "content_add": rec.get("content_add", []),
                                                  "content_flag": rec.get("content_flag", [])})
            ad["profiles"][key] = {"label": prof["label"],
                                   **image_entry(rec, rel(out / key / "viz" / f"{rec['ad_id']}.jpg"),
                                                 rel(out / key / "fixed" / f"{rec['ad_id']}.png"))}

    videos = []
    vd = out / "video" / "json"
    meta = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in (root / cfg["paths"]["video_json"]).glob("*.json")}
    for p in sorted(vd.glob("*.json")) if vd.exists() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        m = meta.get(rec["ad_id"], {})
        videos.append({
            "ad_id": rec["ad_id"], "kind": "video", "brand": m.get("brand"), "media": m.get("media"),
            "duration": m.get("duration"), "size": m.get("size"),
            "decision": rec["decision"],
            "checks_before": slim_checks(rec.get("checks_before")),
            "checks_after": slim_checks(rec.get("checks_after")),
            "timeline": {"before": rec.get("windows_before"), "after": rec.get("windows_after"),
                         "end_card_hold": rec.get("hold")},
            "frames": rec.get("frames", []),
            "edit_sheet": rec.get("edit_sheet", []),
            "why_not": rec.get("fails", []) or ([rec["note"]] if rec.get("note") else []),
            "panel_image": rel(out / "video" / "viz" / f"{rec['ad_id']}.jpg"),
        })

    th = cfg["thresholds"]
    doc = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "thresholds": {k: th[k] for k in ("contrast", "ratio_image", "ratio_video", "exposure_frac", "cps", "video_pt")},
        "profiles": {k: v["label"] for k, v in cfg["profiles"].items()},
        "decisions": ["원본 유지", "수정안", "수정안 + 표현 검토", "표현 검토", "디자이너 검토", "고지 없음"],
        "images": list(images.values()),
        "videos": videos,
    }
    dst = out / "report_input.json"
    dst.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(dst, f"이미지 {len(images)} · 영상 {len(videos)}")


if __name__ == "__main__":
    main()
