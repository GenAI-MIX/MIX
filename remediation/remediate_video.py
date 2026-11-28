"""06 수정안 생성 — 영상 광고. 영상 길이·화면 크기는 그대로, 위험고지 자막만 고친 수정 영상(mp4)과 비교 패널을 만든다.

  python remediate_video.py --ids V110,V105
  python remediate_video.py --all
"""
import argparse
import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from remediate import find_metric_cfg
from remedy.diagnose import is_logo
from remedy.measure import analyze_block
from remedy.render import Fonts
from remedy.video import CardFix, checks, free_windows, grab, hold_plan, plan_timeline, render_video, union_len

CODE_DIR = Path(__file__).resolve().parent
INK, SUB, BG = (24, 24, 27), (100, 104, 112), (246, 247, 249)
GREEN, RED, BLUE, PURPLE, ORANGE = (22, 163, 74), (220, 38, 38), (37, 99, 235), (109, 40, 217), (217, 119, 6)
BADGE = {"원본 유지": GREEN, "수정안": BLUE, "디자이너 검토": PURPLE, "고지 없음": ORANGE}


def fix_video(ad, root, cfg, fonts, mcfg, out, tmp, render=False):
    th, ladder = cfg["thresholds"], cfg["ladder"]
    margin = ladder["margin"]
    W, H = ad["size"]
    D = ad["duration"]
    before = checks(ad["metrics"], D, th)
    rec = {"ad_id": ad["ad_id"], "checks_before": before}
    src = root / ad["video"]
    if not ad["metrics"].get("n_risk_cards"):
        rec.update(decision="고지 없음", note="화면 위험고지 자막 없음 — 문구 추가는 B항목(05 판정)·음성 고지 확인")
        return rec, None
    bad = {c["id"] for c in before if c["ok"] is False}
    if not bad:
        rec.update(decision="원본 유지", note="A2·A4·A5·A6·A7 모두 기준 통과 — 원본 영상 그대로 진행")
        return rec, None

    plan, windows, tl_notes = plan_timeline(ad, th, margin)
    screens = {s["id"]: s for s in ad["screens"]}
    exps = {e["id"]: e for e in ad.get("risk_exposures", [])}
    fixes, overlays, fails, steps, blurs = [], [], [], [], []
    hold = None
    for c in ad["cards"]:
        if c["type"] != "risk_notice":
            continue
        crawl = c["kind"] != "static"
        disp = (c.get("risk_display_pt") or {}).get("mobile_landscape") or {}
        hl = screens[c["screen_id"]].get("headline_h_px")
        small = disp.get("min") is not None and disp["min"] < th["video_pt"]
        low_ratio = c.get("risk_headline_ratio") is not None and c["risk_headline_ratio"] < th["ratio_video"]
        low_c = c.get("risk_contrast_min") is not None and c["risk_contrast_min"] < th["contrast"]
        ext = [e for e in exps.values() if e.get("text_card_id") == c["id"] and e["id"] in plan]
        if not (small or low_ratio or low_c or ext or (crawl and ("A5" in bad or "A6" in bad))):
            continue
        blocks = [b for b in ad["ocr_blocks"] if b.get("card_id") == c["id"] and not is_logo(b, ad)]
        if not blocks and c.get("text"):
            # 흐르는 자막: 한 줄 블록으로 보고 같은 띠 자리에 고정 자막으로 다시 씀
            lh = c.get("line_h_px") or 20
            bx = c["bbox"]
            blocks = [{"id": -1, "text": c["text"], "bbox": [bx[0], bx[3] - lh, bx[2], bx[3]], "h_px": lh * 0.7}]
        if not blocks:
            continue
        h0 = float(np.median([b["h_px"] for b in blocks if b.get("h_px")] or [10]))
        need = h0
        if small:
            # 가장 작은 줄이 기준을 넘도록: 문단 전체를 같은 배율로 (최소 줄 pt → 기준 × margin)
            need = max(need, h0 * th["video_pt"] * margin / disp["min"])
        if low_ratio and hl:
            need = max(need, th["ratio_video"] * hl * margin)
        frame = grab(src, c["rep_t"], tmp / f"{ad['ad_id']}_c{c['id']}.png")
        obstacles = [b["bbox"] for b in ad["ocr_blocks"] if b["screen_id"] == c["screen_id"] and b.get("card_id") != c["id"]]
        obstacles += [b["bbox"] for b in ad["ocr_blocks"] if b.get("card_id") == c["id"] and is_logo(b, ad)]
        # 고지가 떠 있는 동안 배경이 그대로인지 (처음·끝 프레임 비교)
        x0_, y0_, x1_, y1_ = [int(v) for v in c["bbox"]]
        ring = (slice(max(0, y0_ - 40), min(H, y1_ + 40)), slice(max(0, x0_ - 40), min(W, x1_ + 40)))
        ts = np.linspace(c["t_start"] + 0.1, max(c["t_start"] + 0.1, c["t_end"] - 0.15), 4)
        shots_ = [np.asarray(grab(src, t, tmp / f"s{i}.png")).astype(int)[ring] for i, t in enumerate(ts)]
        diff = max(float(np.abs(a_ - shots_[0]).mean()) for a_ in shots_[1:])
        static = diff < cfg["ladder"]["video_static_diff"] and not crawl
        f = CardFix(frame, c, blocks, obstacles, need, fonts, mcfg, th, ladder, static)
        if f.info["size_factor"] > ladder["max_size_factor"]:
            f.fail.append(f"글자를 ×{f.info['size_factor']:.1f} 키워야 함")
        fails += [f"자막 {c['id']}: {x}" for x in f.fail]
        orig_win = (c["t_start"], c["t_end"])
        new_win = next((plan[e["id"]] for e in ext), orig_win)
        frame_fix = small or low_ratio or low_c or crawl
        main_png, ext_png = tmp / f"{ad['ad_id']}_c{c['id']}_main.png", tmp / f"{ad['ad_id']}_c{c['id']}_ext.png"
        f.main.save(main_png)
        extra = (new_win[1] - new_win[0]) - (orig_win[1] - orig_win[0])
        ext_wins = []
        if extra > 0.05:
            busy = [(e["t_start"], e["t_end"]) for e in exps.values()]
            ext_wins, got = free_windows(ad, f.text_rect, busy, orig_win[0], orig_win[1], extra)
            if got < extra - 0.1:
                hp = hold_plan(ad, orig_win, round(extra, 2)) if static else None
                if hp:
                    # V3: 빈 장면 대신 엔드 카드를 일찍 시작 (앞 컷 끝을 다듬음, 음성·길이 그대로)
                    ext_wins, got, hold = [], extra, hp
                    orig_win = (hp[0], orig_win[1])
                    tl_notes.append(f"자막 {c['id']}: 앞 컷 끝을 {hp[1]:.1f}초 다듬고 고지 화면을 {hp[0]:.1f}초부터 시작 (음성·전체 길이 그대로)")
                else:
                    fails.append(f"자막 {c['id']}: 고지 자리에 다른 글자가 없는 장면이 {got:.1f}초뿐 (필요 {extra:.1f}초)")
            where = {(0, 0): "원래 자리"}
            segs = ", ".join(f"{a:.1f}~{b:.1f}초" + ("" if (dx, dy) == (0, 0) else "(자리 옮김)") for a, b, dx, dy in ext_wins)
            if ext_wins:
                tl_notes.append(f"자막 {c['id']}: 고지 자리가 비어 있는 장면({segs})에 같은 자막을 이어서 표시 → +{got:.1f}초")
        if frame_fix:
            overlays.append((main_png, [orig_win]))
        ext_notes = []
        for off in sorted({(dx, dy) for *_, dx, dy in ext_wins}):
            segs_ = [(a, b) for a, b, dx, dy in ext_wins if (dx, dy) == off]
            tr = f.text_rect
            r_ = (slice(max(0, tr[1] + off[1]), min(H, tr[3] + off[1])), slice(max(0, tr[0] + off[0]), min(W, tr[2] + off[0])))
            bgp = np.concatenate([np.asarray(grab(src, (a + b) / 2, tmp / "e.png"))[r_].reshape(-1, 3) for a, b in segs_])
            lay, ink_e, how_e = f.ext_at(off[0], off[1], bgp)
            ext_notes.append(how_e)
            png = tmp / f"{ad['ad_id']}_c{c['id']}_ext_{off[0]}_{off[1]}.png"
            lay.save(png)
            overlays.append((png, segs_))
        if frame_fix and not static:
            blurs.append((f.orig_rect, [orig_win]))
        f.info["ext_ink_how"] = sorted(set(ext_notes))
        fixes.append({"card": c, "fix": f, "orig": orig_win, "new": new_win, "frame_fix": frame_fix,
                      "hold": (hold[0], c["t_start"]) if hold and orig_win[0] == hold[0] else None,
                      "ext": ext_wins, "frame": frame, "hl": hl})
        steps.append({**f.info, "window_before": orig_win, "windows_added": ext_wins, "frame_fix": frame_fix,
                      "fail": f.fail})
    if not overlays and not hold:
        rec.update(decision="디자이너 검토", fails=fails,
                   note="자동으로 넣을 자리가 없음 — " + ("; ".join(fails) if fails else "고칠 수 있는 고지 자막 없음"), steps=steps,
                   edit_sheet=list(tl_notes))
        return rec, None

    dst = out / "fixed" / f"{ad['ad_id']}.mp4"
    if render:   # 선택: 실제 수정 영상(mp4)
        render_video(src, dst, overlays, blurs, hold)
    ov_imgs = [(Image.open(p_).convert("RGBA"), w_) for p_, w_ in overlays]
    on = lambda t, wins: any(a - 1e-3 <= t <= b + 1e-3 for a, b in wins)

    def compose(t):
        """수정 후 t초 화면: 원본 프레임 + (엔드카드 당김) + 블러 + 오버레이. 영상을 만들지 않고 같은 결과를 이미지로."""
        st = t
        if hold and hold[0] <= t < hold[0] + hold[1]:
            st = hold[0] + hold[1] + 0.02
        fr_ = grab(src, st, tmp / "c.png").convert("RGBA")
        for rect, wins in blurs:
            if on(t, wins):
                x0_, y0_, x1_, y1_ = [int(max(0, v)) for v in rect]
                reg_ = fr_.crop((x0_, y0_, x1_, y1_)).filter(ImageFilter.GaussianBlur(max(2, (y1_ - y0_) // 6)))
                fr_.paste(reg_, (x0_, y0_))
        for ov, wins in ov_imgs:
            if on(t, wins):
                fr_.alpha_composite(ov)
        return fr_.convert("RGB")

    # 재측정: 수정 영상에서 프레임을 뽑아 같은 식으로 잰다
    m = ad["metrics"]
    cards_c, cards_pt, cards_r, measured = {}, {}, {}, []
    for c in ad["cards"]:
        if c["type"] == "risk_notice":
            cards_c[c["id"]] = c.get("risk_contrast_min")
            cards_pt[c["id"]] = ((c.get("risk_display_pt") or {}).get("mobile_landscape") or {}).get("min")
            cards_r[c["id"]] = c.get("risk_headline_ratio")
    shots, frames_out = [], []
    for fx in fixes:
        cid = fx["card"]["id"]
        times = []
        if fx["frame_fix"]:
            times.append(("원래 구간", fx["card"]["rep_t"] if fx["orig"][0] <= fx["card"]["rep_t"] <= fx["orig"][1]
                          else sum(fx["orig"]) / 2, 0, 0))
        if fx["ext"]:
            a, b, dx_, dy_ = fx["ext"][0]
            times.append(("늘린 구간", (a + b) / 2, dx_, dy_))
        cs, hs = [], []
        for label, t, dx_, dy_ in times:
            fr = compose(t)
            org = grab(src, t, tmp / f"{ad['ad_id']}_o{cid}_{t:.1f}.png")
            tag = f"{ad['ad_id']}_c{cid}_{'orig' if label == '원래 구간' else 'ext'}_{t:.1f}s"
            org.save(out / "frames" / f"{tag}_before.jpg", quality=90)
            fr.save(out / "frames" / f"{tag}_after.jpg", quality=90)
            frames_out.append({"card": cid, "where": label, "t": round(t, 2),
                               "before": str((out / "frames" / f"{tag}_before.jpg").relative_to(root)),
                               "after": str((out / "frames" / f"{tag}_after.jpg").relative_to(root))})
            arr = np.asarray(fr)
            for bb0 in fx["fix"].boxes:
                bb = [bb0[0] + dx_, bb0[1] + dy_, bb0[2] + dx_, bb0[3] + dy_]
                r = analyze_block(arr, bb, mcfg)
                measured.append({"card": cid, "t": round(t, 2), "where": label, "bbox": bb, **r})
                if r["contrast_ratio"] is not None:
                    cs.append(r["contrast_ratio"])
                if r["ink_h_px"]:
                    hs.append(r["ink_h_px"])
            shots.append((label, t, org, fr, fx, (dx_, dy_)))
        if cs:
            cards_c[cid] = min(cs)
        if hs:
            cards_pt[cid] = round(min(hs) * th["video_width_pt"] / W, 2)
            if fx["hl"]:
                cards_r[cid] = round(float(np.median(hs)) / fx["hl"], 3)
    added = {fx["card"]["id"]: [(a, b) for a, b, *_ in fx["ext"]] + ([fx["hold"]] if fx.get("hold") else []) for fx in fixes}
    new_wins = [[e["t_start"], e["t_end"]] for e in exps.values()] + [list(w) for ws in added.values() for w in ws]
    exp_new = union_len(new_wins)
    cps_new = max([e["risk_chars"] / max((e["t_end"] - e["t_start"]) + sum(b - a for a, b in added.get(e.get("text_card_id"), [])), 0.1)
                   for e in exps.values() if e.get("risk_chars")] or [0])
    m2 = {"risk_contrast_min": min([v for v in cards_c.values() if v is not None], default=None),
          "risk_headline_ratio_min": min([v for v in cards_r.values() if v is not None], default=None),
          "risk_exposure_s": round(exp_new, 2), "risk_cps_max": round(cps_new, 1),
          "risk_display_min_pt": {"mobile_landscape": min([v for v in cards_pt.values() if v is not None], default=None)}}
    after = checks(m2, D, th)
    passed = not fails and all(c["ok"] is not False for c in after)
    decision = "수정안" if passed else "디자이너 검토"
    edit = [n for n in tl_notes if not n.startswith("A5:") and not n.startswith("A6:")]
    for s in steps:
        line = f"자막 {s['card']}: "
        if s["frame_fix"]:
            line += (f"글자 {s['h_before_px']:.0f}→{s['h_after_px']:.0f}px(×{s['size_factor']}) {s['font']}, {s['ink_how']}"
                     + ("" if s["static_bg"] else ", 움직이는 배경이라 원래 작은 글자 자리는 블러로 지움"))
        else:
            line += "원래 크기·색 그대로"
        if s["windows_added"]:
            line += f" · 원래 {s['window_before'][0]:.1f}~{s['window_before'][1]:.1f}초 + 추가 " + ", ".join(f"{w[0]:.1f}~{w[1]:.1f}초" for w in s["windows_added"])
        edit.append(line)
    rec.update(decision=decision, checks_after=after, steps=steps, edit_sheet=edit, fails=fails, measured=measured,
               windows_before={e["id"]: [e["t_start"], e["t_end"]] for e in exps.values()},
               windows_after=new_wins,
               frames=frames_out, hold=hold,
               fixed_video=str(dst.relative_to(root)) if render else None)
    return rec, shots


def fmt(c):
    v = c["value"]
    if v is None:
        return "-"
    return f"{v:.0%}" if c["id"] == "A5" else (f"{v:.3f}" if c["id"] == "A4" else f"{v:.2f}")


def panel(path, font_path, ad, rec, shots):
    F = lambda s: ImageFont.truetype(font_path, s)
    cw = 900
    rows = []
    for label, t, org, fr, fx, off in (shots or [])[:2]:
        rows.append((label, t, org.resize((cw, int(org.height * cw / org.width))),
                     fr.resize((cw, int(fr.height * cw / fr.width))), fx, off))
    Wd = 2 * cw + 72
    tl_h = 220
    body = sum(r[2].height + 50 for r in rows)
    lines = 6 + len(rec.get("edit_sheet", [])) + len(rec.get("fails", []))
    Ht = 110 + body + tl_h + 40 * lines + 60
    sh = Image.new("RGB", (Wd, Ht), BG)
    d = ImageDraw.Draw(sh)
    d.text((24, 24), f"{ad['ad_id']}  {ad.get('brand') or ''} · {ad.get('media')} · {ad['duration']:.1f}초 · {ad['size'][0]}×{ad['size'][1]}",
           font=F(30), fill=INK)
    lab = rec["decision"]
    tw = d.textlength(lab, font=F(22)) + 32
    d.rounded_rectangle((Wd - tw - 24, 24, Wd - 24, 64), 12, fill=BADGE.get(lab, SUB))
    d.text((Wd - tw - 8, 31), lab, font=F(22), fill="white")
    d.text((24, 72), "영상 길이·화면 크기 그대로 · 같은 시점의 원본 프레임 / 수정 영상 프레임", font=F(20), fill=SUB)
    y = 110
    for label, t, o, f, fx, off in rows:
        d.text((24, y), f"원본 · {label} {t:.1f}초", font=F(22), fill=INK)
        d.text((48 + cw, y), f"수정 영상 · {label} {t:.1f}초", font=F(22), fill=BLUE)
        sh.paste(o, (24, y + 34)); sh.paste(f, (48 + cw, y + 34))
        s = cw / ad["size"][0]
        cb = fx["card"]["bbox"]
        if label == "원래 구간":
            d.rectangle([24 + cb[0] * s, y + 34 + cb[1] * s, 24 + cb[2] * s, y + 34 + cb[3] * s], outline=RED, width=3)
        y += o.height + 50
    # 타임라인
    D = ad["duration"]
    x0, x1 = 24, Wd - 24
    sx = lambda t: x0 + (x1 - x0) * t / D
    d.text((24, y), "위험고지 노출 타임라인", font=F(24), fill=INK)
    for k, (name, wins, col) in enumerate([("원본", rec.get("windows_before", {}), (150, 150, 160)),
                                            ("수정", rec.get("windows_after", {}), BLUE)]):
        yy = y + 46 + k * 64
        d.text((24, yy + 6), name, font=F(20), fill=SUB)
        d.rounded_rectangle((90, yy, x1, yy + 40), 8, fill=(228, 230, 235))
        sx2 = lambda t: 90 + (x1 - 90) * t / D
        for a, b in (wins.values() if isinstance(wins, dict) else wins):
            d.rounded_rectangle((sx2(a), yy, sx2(b), yy + 40), 8, fill=col)
        d.line((sx2(D / 3), y + 40, sx2(D / 3), y + 180), fill=ORANGE, width=3)
    d.text((90 + (x1 - 90) / 3 + 8, y + 176), "1/3 (A5 기준)", font=F(18), fill=ORANGE)
    for t in range(0, int(D) + 1, 5):
        d.text((90 + (x1 - 90) * t / D, y + 150), f"{t}s", font=F(16), fill=SUB)
    y += tl_h
    after = {c["id"]: c for c in rec.get("checks_after", [])}
    mark = lambda ok: "통과" if ok else ("미달" if ok is False else "-")
    for c in rec["checks_before"]:
        a = after.get(c["id"])
        op = "≥" if c["op"] == ">=" else "≤"
        thv = f"{c['threshold']:.0%}" if c["id"] == "A5" else c["threshold"]
        t = f"{c['id']} {c['name']}: {fmt(c)} ({mark(c['ok'])})" + (f" → {fmt(a)} ({mark(a['ok'])})" if a else "") + f"   기준 {op} {thv}"
        okv = (a or c)["ok"]
        d.text((24, y), t, font=F(22), fill=GREEN if okv else (RED if okv is False else SUB))
        y += 36
    y += 8
    for e in rec.get("edit_sheet", []):
        d.text((24, y), "· " + e[:120], font=F(20), fill=BLUE)
        y += 34
    for e in rec.get("fails", []):
        d.text((24, y), "· 불가: " + e, font=F(20), fill=RED)
        y += 34
    if rec.get("note"):
        d.text((24, y), "· " + rec["note"], font=F(20), fill=SUB)
    sh.save(path, quality=90)


def main():
    ap = argparse.ArgumentParser(description="위험고지 자막 수정안 생성 (영상 광고)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ids")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--render", action="store_true", help="수정 영상(mp4)도 만듦 (기본: 고지 화면 수정 이미지 + 타임라인만)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(CODE_DIR / args.config, encoding="utf-8"))
    root = (CODE_DIR / cfg["paths"]["repo_root"]).resolve()
    mcfg = find_metric_cfg(yaml.safe_load(open(root / cfg["paths"]["prep_config"], encoding="utf-8")))
    out = root / cfg["paths"]["video_out"]
    for sub in ("json", "fixed", "viz", "frames"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    fonts = Fonts(cfg["fonts"])
    files = sorted((root / cfg["paths"]["video_json"]).glob("*.json"))
    if args.ids:
        files = [f for f in files if f.stem in set(args.ids.split(","))]
    elif not args.all:
        ap.error("--ids 또는 --all")
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            ad = json.loads(f.read_text(encoding="utf-8"))
            try:
                rec, shots = fix_video(ad, root, cfg, fonts, mcfg, out, Path(td), args.render)
            except Exception as e:  # 한 영상 실패가 전체를 멈추지 않게
                rec, shots = {"ad_id": ad["ad_id"], "decision": "디자이너 검토", "note": f"처리 오류: {e}",
                              "checks_before": checks(ad["metrics"], ad["duration"], cfg["thresholds"])}, None
            panel(out / "viz" / f"{ad['ad_id']}.jpg", fonts.paths[0], ad, rec, shots)
            (out / "json" / f.name).write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            cb = {c["id"]: c["ok"] for c in rec["checks_before"]}
            ca = {c["id"]: c["ok"] for c in rec.get("checks_after", [])}
            rows.append({"ad_id": ad["ad_id"], "decision": rec["decision"],
                         **{f"{k}_before": cb.get(k) for k in ("A2", "A4", "A5", "A6", "A7")},
                         **{f"{k}_after": ca.get(k) for k in ("A2", "A4", "A5", "A6", "A7")}})
            print(f"{ad['ad_id']:6s} {rec['decision']}  {'; '.join(rec.get('fails', []))[:80]}", flush=True)
    if args.all:
        with open(out / "summary.csv", "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
