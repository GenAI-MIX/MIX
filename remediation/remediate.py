"""06 수정안 생성 실행 진입점.

  python remediate.py --ids P021,AD42     # 지정 광고
  python remediate.py --all               # 전체 (data/processed/image/json/*.json)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from remedy.color import contrast, fit_ink, delta_e
from remedy.content import load_judge, plan
from remedy.diagnose import diagnose, host_target
from remedy.measure import analyze_block
from remedy.render import Fonts, new_band, protected_rows, recolor, redistribute, retypeset, scale_visual
from remedy.verify import remeasure
from remedy.viz import panel, panel_candidates

CODE_DIR = Path(__file__).resolve().parent


def find_metric_cfg(d):
    if isinstance(d, dict):
        if "fg_extreme_quantile" in d:
            return d
        for v in d.values():
            r = find_metric_cfg(v)
            if r:
                return r
    return None


def fix_ad(ad, root, cfg, fonts, mcfg, out, phrases, judge_dir):
    th, ladder = cfg["thresholds"], cfg["ladder"]
    orig = Image.open(root / ad["image"]).convert("RGB")
    dg = diagnose(ad, th, ladder)
    ad["_checks"] = dg["checks"]
    blocks = {b["id"]: b for b in ad["ocr_blocks"]}
    judge = load_judge(judge_dir, ad["ad_id"]) if judge_dir else None
    cp = plan(ad, judge, phrases)                       # ① 내용: 추가할 문구·표시할 표현
    rec = {"ad_id": ad["ad_id"], "decision": dg["decision"], "checks_before": dg["checks"], "note": dg.get("note"),
           "judge_model": cp["model"], "content_add": cp["add"], "content_flag": cp["flag"]}
    viz = out / "viz" / f"{ad['ad_id']}.jpg"
    flag_boxes = [b for f in cp["flag"] for b in f["boxes"]]
    content_notes = content_lines(cp)
    texts = [a["text"] for a in cp["add"]]
    host = host_target(ad, dg) if texts else None
    if not dg["targets"] and not (texts and dg["decision"] == "고지 없음"):
        decision = dg["decision"] if not cp["flag"] else "표현 검토"
        rec["decision"] = decision
        panel(viz, fonts.paths[0], ad, decision, orig, flag_boxes=flag_boxes,
              notes=[dg["note"]] + content_notes)
        return rec

    before_boxes = [blocks[i]["bbox"] for t in dg["targets"] for i in t["blocks"]]

    def build(mode):
        """모든 고칠 문단을 아래에서 위로 처리. mode: tight(A) / keep(B·C, 일단 늘린 뒤 규격 복원)."""
        img = np.asarray(orig).copy()
        changed, drawn, steps, seams = {}, [], [], []
        for t in sorted(dg["targets"], key=lambda t: -t["bbox"][1]):
            bs = [blocks[i] for i in t["blocks"]]
            if t["action"] == "recolor":
                for b in bs:
                    if b["id"] not in t["low_contrast_blocks"]:
                        continue
                    r = analyze_block(img, b["bbox"], mcfg)
                    if not r["fg_rgb"]:
                        continue
                    new = fit_ink(r["fg_rgb"], r["bg_rgb"], th["contrast"] * ladder["contrast_margin"])
                    recolor(img, b["bbox"], r["fg_rgb"], r["bg_rgb"], new)
                    changed[b["id"]] = "recolor"
                    drawn.append((b["bbox"], "recolor"))
                    steps.append({"paragraph": t["paragraph"], "level": "L1", "block": b["id"],
                                  "fg_before": r["fg_rgb"], "fg_after": new,
                                  "ink_delta_e": round(delta_e(r["fg_rgb"], new), 1)})
            else:
                obstacles = [b["bbox"] for b in ad["ocr_blocks"] if b["id"] not in t["blocks"] and b["id"] not in changed]
                obstacles += [b for b, _ in drawn]
                extra = texts if host is t else ()
                img, lines, info, (grow, seam) = retypeset(img, t, bs, obstacles, fonts, th, ladder, mcfg, extra, mode)
                if grow:
                    drawn = [([b[0], b[1] + grow * (b[1] > seam), b[2], b[3] + grow * (b[1] > seam)], k) for b, k in drawn]
                    seams.append((seam, grow))
                for b in bs:
                    changed[b["id"]] = "retypeset"
                drawn += [(b, "added" if new else "retypeset") for b, new in zip(lines, info.pop("added_flags"))]
                steps.append({"paragraph": t["paragraph"], **info})
        if texts and host is None:          # 고지 문단이 없음 → 맨 아래에 띠
            hp = (dg.get("need_h_px") or th["image_pt"] * ad["size"][0] / th["display_width_pt"] * ladder["margin"])
            margin = int(min([b["bbox"][0] for b in ad["ocr_blocks"]] or [ad["size"][0] * 0.05]))
            H = img.shape[0]
            img, lines, info = new_band(img, texts, fonts, th, ladder, hp, max(margin, 8))
            drawn += [(b, "added") for b in lines]
            seams.append((H - 1, info["grow_px"]))
            steps.append({"paragraph": None, **info})
        return img, drawn, steps, changed, seams

    def map_y(y, seams):            # 원본 y → 줄 삽입 후 y
        for sy, g in seams:
            y = y + g * (y > sy)
        return y

    def finish(name, label, img, drawn, steps, changed, extra_info):
        ok_shape = img.shape == np.asarray(orig).shape
        checks_after, measured = remeasure(ad, img, changed, drawn, mcfg, th, analyze_block)
        big = any(s.get("size_factor", 1) > ladder["max_size_factor"] for s in steps)
        short = sum(s.get("short_px", 0) for s in steps)
        passed = ok_shape and not big and not short and all(c["ok"] is not False for c in checks_after)
        diff = (np.abs(img.astype(int) - np.asarray(orig).astype(int)).max(axis=2) > 16).mean() if ok_shape else 1.0
        dEs = [s["ink_delta_e"] for s in steps if "ink_delta_e" in s]
        sf = [s["size_factor"] for s in steps if "size_factor" in s]
        why = []
        if short:
            why.append(f"고지 영역에 {short}px 부족")
        if big:
            why.append(f"글자 ×{max(sf):.1f} 확대 필요")
        why += [f"{c['id']} 미달" for c in checks_after if c["ok"] is False]
        why += extra_info.pop("fail", [])
        return {"name": name, "label": label, "img": img, "drawn": drawn, "steps": steps, "passed": passed,
                "checks_after": checks_after, "measured": measured, "why": why,
                "changed_pct": round(float(diff) * 100, 1), "ink_delta_e": max(dEs) if dEs else 0.0,
                "size_factor": max(sf) if sf else 1.0, **extra_info}

    cands = []
    img, drawn, steps, changed, seams = build("keep")
    grow_total = sum(g for _, g in seams)
    if not grow_total:
        cands.append(finish("A", "제자리 수정 (규격·배치 그대로)", img, drawn, steps, changed, {}))
    else:
        # A안: 고지 영역 안에서만 (줄 간격 정리)
        iA, dA, sA, cA, _ = build("tight")
        cands.append(finish("A", "A안 · 고지 영역 안에서 줄 간격 정리", iA, dA, sA, cA, {}))
        # B·C는 A안처럼 줄 간격을 먼저 정리하고, 그래도 모자란 만큼만 다른 곳에서 확보
        img, drawn, steps, changed, seams = build("tight+")
        grow_total = sum(g for _, g in seams)
    if grow_total:
        # 보호할 줄: 모든 글자(원본 블록은 삽입 후 좌표로) + 새로 쓴 줄
        H2 = img.shape[0]
        prot_boxes = [[0, map_y(b["bbox"][1], seams), 0, map_y(b["bbox"][3], seams)] for b in ad["ocr_blocks"]]
        prot_boxes += [b for b, _ in drawn]
        protect = protected_rows(H2, prot_boxes, int(0.5 * (dg.get("need_h_px") or 10)))
        # B안: 평평한 배경 줄에서 같은 높이만큼 회수
        imgB, rows = redistribute(img, grow_total, protect, ladder["redistribute_max_diff"])
        if imgB is not None:
            rows_a = np.array(rows)
            sh = lambda y: y - int((rows_a < y).sum())
            dB = [([b[0], sh(b[1]), b[2], sh(b[3])], k) for b, k in drawn]
            cands.append(finish("B", "B안 · 배경 여백을 나눠 고지 공간 확보", imgB, dB, steps, changed,
                                {"rows_removed": len(rows)}))
        else:
            cands.append({"name": "B", "label": "B안 · 배경 여백을 나눠 고지 공간 확보", "img": None, "passed": False,
                          "why": [f"평평한 배경 줄이 {rows}px 모자람"]})
        # C안: 고지 위 비주얼 전체를 같은 비율로 살짝 축소
        y_top = min(b[1] for b, _ in drawn) - int(0.6 * (dg.get("need_h_px") or 10))
        xs = [b["bbox"][0] for b in ad["ocr_blocks"]] + [b[0] for b, _ in drawn]
        xe = [b["bbox"][2] for b in ad["ocr_blocks"]] + [b[2] for b, _ in drawn]
        pad = int(0.01 * orig.width)
        imgC, sc = scale_visual(img, grow_total, y_top, ladder["min_visual_scale"],
                                max(0, min(xs) - pad), min(orig.width, max(xe) + pad))
        if imgC is not None:
            dC = [([b[0], b[1] - grow_total, b[2], b[3] - grow_total], k) for b, k in drawn]
            cands.append(finish("C", "C안 · 위쪽 비주얼을 살짝 줄여 공간 확보", imgC, dC, steps, changed,
                                {"visual_scale": sc}))
        else:
            cands.append({"name": "C", "label": "C안 · 위쪽 비주얼을 살짝 줄여 공간 확보", "img": None, "passed": False,
                          "why": [f"비주얼을 {sc:.0%}까지 줄여야 함 (한도 {ladder['min_visual_scale']:.0%})"]})

        # B+C안: 평평한 배경 줄을 뺄 수 있는 만큼 빼고, 나머지만 비주얼 축소
        if not any(c["passed"] for c in cands):
            imgP, rowsP = redistribute(img, grow_total, protect, ladder["redistribute_max_diff"], partial=True)
            rest = grow_total - len(rowsP)
            ra = np.array(rowsP or [10**9])
            shP = lambda y: y - int((ra < y).sum())
            dP = [([b[0], shP(b[1]), b[2], shP(b[3])], k) for b, k in drawn]
            y_top2 = min(b[1] for b, _ in dP) - int(0.6 * (dg.get("need_h_px") or 10))
            imgD, sc2 = scale_visual(imgP, rest, y_top2, ladder["min_visual_scale"],
                                     max(0, min(xs) - pad), min(orig.width, max(xe) + pad))
            lab = "B+C안 · 배경 여백 회수 + 비주얼 살짝 축소"
            if imgD is not None and rowsP:
                dD = [([b[0], b[1] - rest, b[2], b[3] - rest], k) for b, k in dP]
                cands.append(finish("D", lab, imgD, dD, steps, changed,
                                    {"rows_removed": len(rowsP), "visual_scale": sc2}))
            else:
                cands.append({"name": "D", "label": lab, "img": None, "passed": False,
                              "why": [f"배경 줄 {len(rowsP)}px 회수 후에도 비주얼을 {sc2:.0%}까지 줄여야 함"]})

    ok = [c for c in cands if c["passed"]]
    # 추천: 덜 건드리는 순서 A(고지 영역 안) → B(배경 여백만) → C(비주얼 축소)
    best = min(ok, key=lambda c: ("ABCD".index(c["name"]), c["changed_pct"])) if ok else None
    for c in cands:
        c["recommended"] = c is best
        if c.get("img") is not None:
            Image.fromarray(c["img"]).save(out / "fixed" / f"{ad['ad_id']}_{c['name']}.png")
    if best is not None:
        decision = "수정안 + 표현 검토" if cp["flag"] else "수정안"
        Image.fromarray(best["img"]).save(out / "fixed" / f"{ad['ad_id']}.png")
    else:
        decision = "디자이너 검토"
    brief = None
    if best is None and grow_total:
        brief = (f"디자이너 브리프: 규격 유지 시 고지 영역에 약 {grow_total}px(원본 높이의 {grow_total / orig.height:.0%}) 추가 공간 필요. "
                 + " / ".join(f"{c['name']}: {', '.join(c['why'])}" for c in cands if c.get("why")))
    fonts_used = sorted({s["font"] for c in cands for s in c.get("steps", []) if "font" in s})
    notes = ([f"재조판 폰트: {', '.join(fonts_used)} (원본 획 굵기에 가장 가까운 것) · 정렬·글자색 원본 기준"] if fonts_used else [])
    notes += ([brief] if brief else []) + content_notes
    panel_candidates(viz, fonts.paths[0], ad, decision, orig, before_boxes, cands, notes, flag_boxes)
    pub = lambda c: {k: v for k, v in c.items() if k not in ("img", "drawn")}
    rec.update(decision=decision, recommended=best["name"] if best else None, brief=brief,
               candidates=[pub(c) for c in cands],
               checks_after=best["checks_after"] if best else (cands[0].get("checks_after") or []))
    return rec


def content_lines(cp):
    out = []
    for a in cp["add"]:
        tag = "채울 값 있음 " if a["placeholder"] else ""
        out.append(f"[추가 {a['check_id']} {a['verdict']}] {tag}{a['text']}")
    for f in cp["flag"]:
        q = f"'{f['ad_quote'][:24]}' " if f["ad_quote"] else ""
        out.append(f"[표현 {f['check_id']} {f['verdict']}] {q}→ {f['suggest']}")
    return out


def main():
    ap = argparse.ArgumentParser(description="위험고지 표시 수정안 생성 (이미지 광고)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ids", help="쉼표로 구분한 ad_id 목록")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", help="출력 폴더 (기본: config paths.out)")
    ap.add_argument("--judge-dir", help="05 판정 결과 폴더 (기본: config paths.judge_dir, 'none'이면 내용 수정 생략)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CODE_DIR / args.config, encoding="utf-8"))
    root = (CODE_DIR / cfg["paths"]["repo_root"]).resolve()
    mcfg = find_metric_cfg(yaml.safe_load(open(root / cfg["paths"]["prep_config"], encoding="utf-8")))
    out = Path(args.out).resolve() if args.out else root / cfg["paths"]["out"]
    for sub in ("json", "fixed", "viz"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fonts = Fonts(cfg["fonts"])
    phrases = yaml.safe_load(open(CODE_DIR / "rules" / "phrases.yaml", encoding="utf-8"))
    jd = args.judge_dir or cfg["paths"].get("judge_dir")
    judge_dir = None if jd in (None, "none") else (CODE_DIR / jd if not Path(jd).is_absolute() else Path(jd))
    if args.judge_dir and judge_dir is not None and not Path(args.judge_dir).is_absolute():
        judge_dir = Path.cwd() / args.judge_dir
    if judge_dir is not None and not judge_dir.exists():
        print(f"  05 판정 결과 폴더 없음({judge_dir}) → 표시 방법만 수정")
        judge_dir = None
    files = sorted((root / cfg["paths"]["ad_json"]).glob("*.json"))
    if args.ids:
        want = set(args.ids.split(","))
        files = [f for f in files if f.stem in want]
    elif not args.all:
        ap.error("--ids 또는 --all")
    rows = []
    for f in files:
        ad = json.loads(f.read_text(encoding="utf-8"))
        rec = fix_ad(ad, root, cfg, fonts, mcfg, out, phrases, judge_dir)
        (out / "json" / f.name).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
        cb = {c["id"]: c["ok"] for c in rec["checks_before"]}
        ca = {c["id"]: c["ok"] for c in rec.get("checks_after", [])}
        rows.append({"ad_id": ad["ad_id"], "decision": rec["decision"],
                     **{f"{k}_before": cb.get(k) for k in ("A2", "A3", "A4")},
                     **{f"{k}_after": ca.get(k) for k in ("A2", "A3", "A4")},
                     "recommended": rec.get("recommended") or "",
                     "passing": "+".join(c["name"] for c in rec.get("candidates", []) if c["passed"]),
                     "added": "+".join(a["check_id"] for a in rec["content_add"]),
                     "flagged": "+".join(f["check_id"] for f in rec["content_flag"])})
        print(f"{ad['ad_id']:6s} {rec['decision']:10s} 추천 {rows[-1]['recommended'] or '-'}  통과안 {rows[-1]['passing'] or '-'}", flush=True)
    if args.all:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
