"""06 수정안 결과 보고서(PDF) 생성. remediate.py를 먼저 실행해 viz·summary가 있어야 함.

  python report/make_report.py
"""
import csv
import collections
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "data" / "processed" / "remediation"
FD = HERE.parent / "fonts"            # 레포에 포함된 Pretendard
F = lambda w, s: ImageFont.truetype(str(FD / f"Pretendard-{w}.otf"), s)
W, H = 2339, 1654                      # A4 가로, 200dpi
INK, SUB, LINE, BG = (24, 24, 27), (95, 99, 110), (220, 223, 230), (255, 255, 255)
BLUE, GREEN, RED, ORANGE, PURPLE = (37, 99, 235), (22, 163, 74), (220, 38, 38), (234, 88, 12), (109, 40, 217)

VOUT = OUT / "video"
def video_page(im, d, ad_id, color, lines):
    rec = json.loads((VOUT / "json" / f"{ad_id}.json").read_text(encoding="utf-8"))
    pan = Image.open(VOUT / "viz" / f"{ad_id}.jpg").convert("RGB")
    s = min((W - 110 - 700) / pan.width, (H - 230) / pan.height)
    pan = pan.resize((int(pan.width * s), int(pan.height * s)), Image.LANCZOS)
    x = W - 110 - pan.width
    im.paste(pan, (x, 190))
    left_w = x - 140 - 50
    d.rectangle((110, 200, 122, 200 + 52 * 9), fill=color)
    y = bullets(d, 140, 200, lines, size=28, width=left_w)
    y += 20
    d.text((140, y), "편집 지시", font=F("SemiBold", 30), fill=INK)
    bullets(d, 140, y + 50, rec.get("edit_sheet", [])[:4] + ["불가: " + x_ for x_ in rec.get("fails", [])][:2],
            size=24, width=left_w, color=SUB)
    d.rectangle((x - 2, 188, x + pan.width + 1, 190 + pan.height + 1), outline=LINE, width=2)


def page():
    im = Image.new("RGB", (W, H), BG)
    return im, ImageDraw.Draw(im)


def header(d, title, n):
    d.rectangle((0, 0, W, 10), fill=BLUE)
    d.text((110, 70), title, font=F("Bold", 54), fill=INK)
    d.text((W - 110, 82), f"06 수정안 생성 · {n}", font=F("Regular", 26), fill=SUB, anchor="ra")
    d.line((110, 160, W - 110, 160), fill=LINE, width=3)


def bullets(d, x, y, lines, size=34, gap=1.65, color=INK, width=None):
    f = F("Regular", size)
    for ln in lines:
        words, cur, rows = ln.split(" "), "", []
        for w in words:
            t = (cur + " " + w).strip()
            if width and d.textlength(t, font=f) > width - 40 and cur:
                rows.append(cur); cur = w
            else:
                cur = t
        rows.append(cur)
        d.text((x, y), "•", font=f, fill=BLUE)
        for r in rows:
            d.text((x + 40, y), r, font=f, fill=color)
            y += int(size * gap)
        y += int(size * 0.35)
    return y


def table(d, x, y, rows, widths, size=30, head=True):
    f, fb = F("Regular", size), F("SemiBold", size)
    rh = int(size * 2.1)
    for i, r in enumerate(rows):
        if i == 0 and head:
            d.rectangle((x, y, x + sum(widths), y + rh), fill=(241, 245, 255))
        cx = x
        for c, w in zip(r, widths):
            col = INK
            if isinstance(c, tuple):
                c, col = c
            d.text((cx + 20, y + rh // 2), str(c), font=fb if (i == 0 and head) else f, fill=col, anchor="lm")
            cx += w
        d.line((x, y + rh, x + sum(widths), y + rh), fill=LINE, width=2)
        y += rh
    return y


def box(d, xy, title, body, color):
    d.rounded_rectangle(xy, 22, outline=color, width=4, fill=(250, 251, 253))
    d.text((xy[0] + 28, xy[1] + 24), title, font=F("Bold", 34), fill=color)
    yy = xy[1] + 80
    for b in body:
        d.text((xy[0] + 28, yy), b, font=F("Regular", 27), fill=INK)
        yy += 42


def fit(im, w, h):
    s = min(w / im.width, h / im.height)
    return im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)


def case_page(im, d, base, color, lines):
    out, ad_id = base.parent, base.name
    rec = json.loads((out / "json" / f"{ad_id}.json").read_text(encoding="utf-8"))
    src = json.loads((ROOT / "data" / "processed" / "image" / "json" / f"{ad_id}.json").read_text(encoding="utf-8"))
    orig = Image.open(ROOT / src["image"]).convert("RGB")
    d.rectangle((110, 190, 122, 190 + 46 * len(lines)), fill=color)
    y = 190
    for ln in lines:
        d.text((150, y), ln, font=F("Regular", 30), fill=INK)
        y += 46
    cands = rec.get("candidates", [])
    tiles = [("원본", orig, None)]
    for c in cands:
        p = out / "fixed" / f"{ad_id}_{c['name']}.png"
        tiles.append((c["label"].split(" · ")[0] + (" ★추천" if c.get("recommended") else ""),
                      Image.open(p).convert("RGB") if p.exists() else None, c))
    best = next((c for c in cands if c.get("recommended")), None)
    zoom = best is not None
    # 세로로 아주 긴 광고는 고친 부분 주변만 잘라서 보여줌 (같은 위치)
    win = None
    if orig.height / orig.width > 2.0:
        ys = [m["bbox"][1] for m in (best or {}).get("measured", [])] or [orig.height * 0.8]
        cy = (min(ys) + max(ys)) / 2
        hh = int(orig.width * 1.3)
        y0w = int(max(0, min(orig.height - hh, cy - hh / 2)))
        win = (y0w, y0w + hh)
        tiles = [(n + " (고지 부분)", im_.crop((0, win[0], im_.width, min(im_.height, win[1]))) if im_ is not None else None, c)
                 for n, im_, c in tiles]
    top = y + 40
    th_ = H - top - 400
    n = len(tiles) + (1 if zoom else 0)
    tw = min(520, (W - 220 - (n - 1) * 30) // n)
    x = 110
    f_l, f_s = F("SemiBold", 28), F("Regular", 24)
    for name, img, c in tiles:
        rec_ = c is not None and c.get("recommended")
        d.text((x, top), name, font=f_l, fill=BLUE if rec_ else INK)
        if img is None:
            d.rounded_rectangle((x, top + 50, x + tw, top + 50 + th_ * 0.6), 16, fill=(240, 241, 245))
            d.text((x + tw // 2, top + 50 + th_ * 0.3), "만들 수 없음", font=f_l, fill=SUB, anchor="mm")
            ih = int(th_ * 0.6)
        else:
            t = fit(img, tw, th_)
            im.paste(t, (x, top + 50))
            d.rectangle((x - 1, top + 49, x + t.width, top + 50 + t.height), outline=BLUE if rec_ else LINE, width=5 if rec_ else 2)
            ih = t.height
        yy = top + 70 + ih
        if c is None:
            d.text((x, yy), f"{orig.width}×{orig.height}px", font=f_s, fill=SUB)
        elif c.get("passed"):
            d.text((x, yy), "기준 통과", font=f_l, fill=GREEN)
            info = [f"달라진 면적 {c['changed_pct']}%"]
            if c.get("rows_removed"):
                info.append(f"여백 {c['rows_removed']}px 회수")
            if c.get("visual_scale"):
                info.append(f"비주얼 {c['visual_scale']:.0%} 축소")
            info.append(f"글자 ×{c['size_factor']:.2f}")
            for i, t_ in enumerate(info):
                d.text((x, yy + 40 + i * 32), t_, font=f_s, fill=SUB)
        else:
            d.text((x, yy), "통과 못 함", font=f_l, fill=RED)
            words, cur, rows_ = " / ".join(c.get("why", [])[:2]).split(" "), "", []
            for w_ in words:
                t_ = (cur + " " + w_).strip()
                if d.textlength(t_, font=f_s) > tw and cur:
                    rows_.append(cur); cur = w_
                else:
                    cur = t_
            rows_.append(cur)
            for i, t_ in enumerate(rows_[:4]):
                d.text((x, yy + 40 + i * 32), t_, font=f_s, fill=SUB)
        x += tw + 30
    if zoom:
        d.text((x, top), "추천안 확대 (전 → 후)", font=f_l, fill=INK)
        bx = [b for t in [] for b in t]
        boxes = [m["bbox"] for m in best.get("measured", [])]
        fixed = Image.open(out / "fixed" / f"{ad_id}_{best['name']}.png").convert("RGB")
        added = [m["bbox"] for m in best.get("measured", []) if m.get("kind") == "added"]
        zb = added or boxes
        x0 = max(0, min(b[0] for b in zb) - 30); x1 = min(fixed.width, max(b[2] for b in zb) + 30)
        x1 = min(fixed.width, max(x1, x0 + int((x1 - x0) * 1.0)))
        hz = int((x1 - x0) * 0.55)
        cy = (min(b[1] for b in zb) + max(b[3] for b in zb)) / 2
        y0 = int(max(0, min(fixed.height - hz, cy - hz / 2))); y1 = y0 + hz
        z1 = fit(fixed.crop((x0, y0, x1, y1)), tw, th_ / 2 - 30)
        z0 = fit(orig.crop((x0, y0, x1, y1)), tw, th_ / 2 - 30)
        im.paste(z0, (x, top + 50)); d.rectangle((x - 1, top + 49, x + z0.width, top + 50 + z0.height), outline=LINE, width=2)
        d.text((x + tw // 2, top + 60 + z0.height), "↓", font=F("Bold", 34), fill=SUB, anchor="ma")
        im.paste(z1, (x, top + 110 + z0.height)); d.rectangle((x - 1, top + 109 + z0.height, x + z1.width, top + 110 + z0.height + z1.height), outline=BLUE, width=3)
    # 판정표
    after = {c["id"]: c for c in (rec.get("checks_after") or [])}
    yb = H - 200
    d.line((110, yb - 20, W - 110, yb - 20), fill=LINE, width=2)
    mark = lambda ok: "통과" if ok else ("미달" if ok is False else "-")
    fmt = lambda c: "-" if c.get("value") is None else (f"{c['value']:.3f}" if c["id"] == "A4" else f"{c['value']:.2f}")
    xx = 110
    for c in rec["checks_before"]:
        a = after.get(c["id"]) if best else None
        t = f"{c['id']} {c['name']}  {fmt(c)} ({mark(c['ok'])})" + (f" → {fmt(a)} ({mark(a['ok'])})" if a else "") + f"   기준 {c['threshold']}"
        ok = (a or c)["ok"]
        d.text((xx, yb), t, font=F("Regular", 27), fill=GREEN if ok else (RED if ok is False else SUB))
        yb += 40
    extra = [f"[추가 {a_['check_id']}] {a_['text']}" for a_ in rec.get("content_add", [])][:2]
    extra += [f"[표현 {f_['check_id']}] {f_['suggest']}" for f_ in rec.get("content_flag", [])][:1]
    for i, e in enumerate(extra):
        d.text((1250, H - 200 + i * 40), e[:62], font=F("Regular", 25), fill=BLUE if e.startswith("[추가") else ORANGE)


METHOD_NAME = {"A": "고지 영역 안에서 줄 간격 정리", "B": "배경 여백 회수", "T": "사진 아래쪽 다듬기",
               "C": "비주얼 살짝 축소", "D": "여러 방법 조합", "H": "상세페이지 높이 확장"}


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def fmtv(c):
    v = c.get("value")
    return "-" if v is None else (f"{v:.3f}" if c["id"] == "A4" else f"{v:.2f}")


def image_lines(rec, prof_label):
    out = [f"상황: {rec.get('situation')}  ·  판형: {prof_label}"]
    after = {c["id"]: c for c in rec.get("checks_after") or []}
    bad = [c for c in rec["checks_before"] if c["ok"] is False]
    if bad:
        out.append("미달: " + ", ".join(f"{c['id']} {c['name']} {fmtv(c)} (기준 {c['threshold']})"
                                       + (f" → {fmtv(after[c['id']])}" if c["id"] in after else "") for c in bad))
    best = next((c for c in rec.get("candidates", []) if c.get("recommended")), None)
    if rec["decision"] == "원본 유지":
        out.append("A2·A3·A4 모두 기준 통과함 → 수정하지 않고 원본 디자인 그대로 진행.")
    elif best:
        det = []
        if best.get("rows_removed"):
            det.append(f"여백 {best['rows_removed']}px 회수")
        if best.get("rows_trimmed"):
            det.append(f"사진 아래쪽 {best['rows_trimmed']}px 다듬음")
        if best.get("visual_scale"):
            det.append(f"위쪽 비주얼 {best['visual_scale']:.0%}")
        if best.get("height_added"):
            det.append(f"높이 +{best['height_added']}px")
        tr = next((s_.get("contrast_treatment") for s_ in best.get("steps", []) if s_.get("contrast_treatment")), None)
        if tr:
            det.append(tr)
        out.append(f"추천: {best['label']}" + (f" — {', '.join(det)}" if det else ""))
        fails = [c for c in rec["candidates"] if not c.get("passed")]
        if fails:
            out.append("통과 못 한 안: " + " / ".join(f"{c['name']} {(c.get('why') or ['-'])[0]}" for c in fails[:3]))
    elif rec.get("brief"):
        out.append(rec["brief"][:150])
    return out


def pick_image_cases():
    """판형·방법별 대표 사례를 자동으로 고름."""
    cases, used = [], set()
    for prof in ["A4", "mobile"]:
        jd = OUT / prof / "json"
        if not jd.exists():
            continue
        recs = [load_json(p) for p in sorted(jd.glob("*.json"))]
        want = ["H", "A", "B", "T", "C", "D"] if prof == "A4" else ["D", "C"]
        for m in want:
            r = next((r for r in recs if r.get("recommended") == m and r["ad_id"] not in used), None)
            if r:
                used.add(r["ad_id"])
                cases.append((prof, r, f"{METHOD_NAME[m]}", BLUE))
        if prof == "A4":
            r = next((r for r in recs if r["decision"] == "원본 유지" and r["ad_id"] not in used), None)
            if r:
                used.add(r["ad_id"]); cases.insert(0, (prof, r, "원본 유지", GREEN))
        r = next((r for r in recs if r["decision"] == "디자이너 검토" and r.get("candidates") and r["ad_id"] not in used), None)
        if r:
            used.add(r["ad_id"]); cases.append((prof, r, "통과 못 함 → 디자이너 브리프", PURPLE))
    return cases


def pick_video_cases():
    jd = VOUT / "json"
    recs = [load_json(p) for p in sorted(jd.glob("*.json"))] if jd.exists() else []
    out, used = [], set()
    rules = [("엔드 카드를 일찍 시작 (앞 컷 다듬기)", lambda r: r["decision"] == "수정안" and any("앞 컷" in e for e in r.get("edit_sheet", []))),
             ("빈 장면에 같은 자막 이어서 표시", lambda r: r["decision"] == "수정안" and any("비어 있는 장면" in e for e in r.get("edit_sheet", []))),
             ("글자 크기·색만 조정", lambda r: r["decision"] == "수정안"),
             ("통과 못 함 → 디자이너 검토", lambda r: r["decision"] == "디자이너 검토" and r.get("fails"))]
    for title, f in rules:
        r = next((r for r in recs if f(r) and r["ad_id"] not in used), None)
        if r:
            used.add(r["ad_id"]); out.append((title, r))
    return out


def main():
    rows = list(csv.DictReader(open(OUT / "A4" / "summary.csv", encoding="utf-8-sig")))
    dec = collections.Counter(r["decision"] for r in rows)
    rec = collections.Counter(r["recommended"] for r in rows if r["recommended"])
    n_fix = sum(v for k, v in dec.items() if k.startswith("수정안"))
    pages = []

    # 1. 표지
    im, d = page()
    d.rectangle((0, 0, W, H), fill=(245, 248, 255))
    d.rectangle((0, 0, 30, H), fill=BLUE)
    d.text((200, 520), "금융광고 위험고지 수정안 생성", font=F("Bold", 92), fill=INK)
    d.text((200, 650), "06 수정안 생성 · 시각화 결과 보고", font=F("Medium", 52), fill=BLUE)
    d.text((200, 780), f"이미지 광고 {len(rows)}개 · 영상 광고 · 규격과 길이는 그대로 두고 위험고지만 고친 수정안", font=F("Regular", 36), fill=SUB)
    d.text((200, 1380), "MIX 팀 · 06 수정안 파트", font=F("Regular", 30), fill=SUB)
    pages.append(im)

    # 2. 한눈에 보기
    im, d = page(); header(d, "한눈에 보기", 2)
    bullets(d, 110, 220, [
        "05 판정과 같은 기준(A2 명암비 3.0 · A3 글자 크기 · A4 헤드라인 대비 0.075)으로 진단함. A3은 출력 판형별로 따로 판단함.",
        "광고마다 상황(매체·고지 컨테이너·배경)을 먼저 분류하고, 상황에 맞는 디자인 방법을 조합함.",
        "기준을 이미 통과한 광고는 손대지 않음 → 원본 유지.",
        "미달인 광고만 고침. '무조건 크게·흰 바탕 검정 글씨'가 아니라 원본의 색·정렬·굵기·배경을 그대로 살림.",
        "광고 규격(가로×세로 px)은 절대 바꾸지 않음. 필요한 공간은 광고 안에서 찾음.",
        "여러 후보안을 만들고 다시 재서, 통과한 안 중 원본을 가장 덜 건드리는 안을 추천함.",
        "어떤 안으로도 안 되면 억지로 맞추지 않고 '얼마나 공간이 더 필요한지'를 디자이너 브리프로 넘김.",
    ], width=1150)
    d.text((1330, 200), "A4 인쇄 기준 결과", font=F("SemiBold", 30), fill=INK)
    y = table(d, 1330, 250, [
        ["결정", "광고 수", "뜻"],
        [("원본 유지", GREEN), dec.get("원본 유지", 0), "기준 모두 통과"],
        [("수정안", BLUE), n_fix, "같은 규격에서 기준 통과"],
        [("디자이너 검토", PURPLE), dec.get("디자이너 검토", 0), "브리프와 함께 넘김"],
        [("고지 없음", ORANGE), dec.get("고지 없음", 0), "① 내용(문구 추가) 대상"],
    ], [330, 190, 380])
    d.text((1330, y + 40), "수정안 추천 내역", font=F("SemiBold", 32), fill=INK)
    table(d, 1330, y + 100, [["줄간격", "여백", "다듬기", "축소", "조합", "높이"],
                             [rec.get(k, 0) for k in "ABTCDH"]], [150] * 6)
    pages.append(im)

    # 3. 작동 방식
    im, d = page(); header(d, "작동 방식", 3)
    steps = [("① 진단", ["전처리 JSON의 측정값으로", "A2·A3·A4 판정", "+ 05 LLM 판정(B·C·D·E)"], INK),
             ("② 통과면 원본 유지", ["고칠 필요 없음", "디자인 그대로 진행"], GREEN),
             ("③ 상황 분류 + 후보안", ["띠·박스·사진·문서형 등", "상황별 방법 조합, 같은 규격"], BLUE),
             ("④ 다시 재서 확인", ["전처리와 같은 계산식으로", "재측정 → 통과 여부"], BLUE),
             ("⑤ 추천 / 브리프", ["통과한 안 중 최소 변경 추천", "없으면 디자이너 브리프"], PURPLE)]
    bw, gap = 380, 40
    for i, (t, b, c) in enumerate(steps):
        x = 110 + i * (bw + gap)
        box(d, (x, 220, x + bw, 470), t, b, c)
        if i < len(steps) - 1:
            d.text((x + bw + 6, 320), "→", font=F("Bold", 40), fill=SUB)
    table(d, 110, 540, [
        ["후보안", "공간을 만드는 방법", "디자인에 주는 영향"],
        ["대비 처리", "단색: 글자 밝기 / 띠·박스: 글자·띠 중 덜 바뀌는 쪽 / 사진: 아래쪽 톤 페이드 (새 박스 없음)", "원래 색 체계 유지"],
        ["A 줄 간격", "고지 영역 안에서 줄 간격만 정리 (글자 높이의 1.25배까지)", "고지 영역만 바뀜"],
        ["B 여백 회수", "A + 위아래와 거의 같은 '평평한 배경 줄'을 여러 곳에서 조금씩 회수", "글자·인물·로고 그대로"],
        ["T 사진 다듬기", "A + 고지 바로 위 사진의 아래쪽을 조금 잘라냄 (높이의 6%까지, 글자 없는 줄만)", "사진 아래 끝만 조금 줄어듦"],
        ["C 비주얼 축소", "A + 고지 위쪽 내용을 같은 비율로 축소 (90%까지), 테두리는 세로로만", "비주얼이 조금 작아짐"],
        ["D 조합", "B → T → C를 조금씩 나눠 씀 (축소는 86%까지)", "영향을 나눠 가짐"],
        ["H 높이 확장", "상세페이지는 폭만 고정이라 높이를 늘림", "폭·여백 그대로"],
        ["디자이너 검토", "모든 안이 불가할 때", "필요한 공간(px·%)과 안 된 이유 전달"],
    ], [300, 1280, 540], size=27)
    d.text((110, 1560), "재조판 시 원본 획 굵기에 가장 가까운 폰트를 고르고, 정렬·줄바꿈·글자색은 원본 기준으로 맞춤. 로고는 글자로 다시 쓰지 않음.",
           font=F("Regular", 27), fill=SUB)
    pages.append(im)

    # 4. 상황 분류 결과
    a4 = OUT / "A4" / "json"
    if a4.exists():
        sits = collections.Counter(load_json(p).get("situation") for p in a4.glob("*.json"))
        im, d = page(); header(d, "상황 분류 결과 (이미지 126개)", len(pages) + 1)
        d.text((110, 200), "고지 영역 안의 글자 아닌 픽셀로 배경을 판단하고, 띠·박스 경계를 따라가 컨테이너 종류를 정함.",
               font=F("Regular", 28), fill=SUB)
        table(d, 110, 260, [["상황 (매체 · 컨테이너 · 배경)", "광고 수"]] + [[k or "-", v] for k, v in sits.most_common(14)],
              [1500, 300], size=27)
        pages.append(im)

    # 5. 판형별 결과
    ps = OUT / "profiles_summary.csv"
    if ps.exists():
        pr = list(csv.reader(open(ps, encoding="utf-8-sig")))
        head_ = pr[0][1:]
        im, d = page(); header(d, "판형별 결과 (같은 광고 126개를 판형마다 판정·수정)", len(pages) + 1)
        rows_ = [["결정"] + head_]
        for dec_ in ["원본 유지", "수정안", "수정안 + 표현 검토", "디자이너 검토", "고지 없음"]:
            cnt = [sum(1 for r in pr[1:] if r[i + 1] == dec_) for i in range(len(head_))]
            if any(cnt):
                rows_.append([(dec_, {"원본 유지": GREEN, "디자이너 검토": PURPLE, "고지 없음": ORANGE}.get(dec_, BLUE))] + cnt)
        y = table(d, 110, 220, rows_, [330] + [270] * len(head_), size=28)
        bullets(d, 110, y + 50, [
            "같은 포스터라도 출력 판형이 클수록 실제 글자가 커져 원본 유지가 늘어남.",
            "인쇄 기준: 9pt(신문 전면 10pt)를 실제 글자 높이(mm)로 바꿔 판단. 모바일은 05 노트북 기준(390pt 화면에서 5.9pt).",
            "광고마다 판형별 결과는 profiles_summary.csv, 판형별 비교 패널은 data/processed/remediation/<판형>/viz.",
        ], width=2100)
        pages.append(im)

    # 6. 이미지 사례 (자동 선택)
    for prof, rec, what, color in pick_image_cases():
        label_ = {"A4": "A4 인쇄", "mobile": "모바일 피드"}.get(prof, prof)
        im, d = page(); header(d, f"이미지 · {label_} · {what} — {rec['ad_id']}", len(pages) + 1)
        case_page(im, d, OUT / prof / rec["ad_id"], color, image_lines(rec, label_))
        pages.append(im)

    # 영상
    if (VOUT / "summary.csv").exists():
        vrows = list(csv.DictReader(open(VOUT / "summary.csv", encoding="utf-8-sig")))
        vdec = collections.Counter(r["decision"] for r in vrows)
        im, d = page(); header(d, "영상 광고 · 작동 방식과 결과", len(pages) + 1)
        bullets(d, 110, 220, [
            "영상 길이와 화면 크기는 그대로 두고, 위험고지 자막만 고친 수정 영상(mp4)을 만듦.",
            "A5 노출·A6 초당 글자 수: ① 앞 컷 끝을 다듬어 엔드 카드를 일찍 시작(음성·길이 그대로) ② 고지 자리가 빈 장면에 같은 자막을 이어서 표시(빈 자리가 없으면 그 장면의 다른 빈 자리로).",
            "A7 글자 크기·A4 헤드라인 대비: 원래 자리·단 폭·정렬 유지, 기준을 살짝 넘길 크기로 다시 씀. 겹치면 아래 안전선 → 전폭 → 위쪽 순으로 자리 조정.",
            "A2 명암비: 박스·띠 없이 글자색으로 해결 — 원래 색 → 광고에 이미 쓰인 색 → 원래 색의 밝기 조정 순. 마지막 수단으로만 글자 바로 뒤를 국소적으로 톤다운.",
            "배경이 그대로인 엔드 카드는 원래 글자를 지운 뒤 다시 쓰고, 움직이는 장면은 원래 작은 글자 자리를 영상 블러로 지움.",
            "수정 영상에서 프레임을 다시 뽑아 같은 식으로 재측정함.",
        ], width=1150)
        table(d, 1330, 230, [
            ["결정", "영상 수"],
            [("원본 유지", GREEN), vdec.get("원본 유지", 0)],
            [("수정안", BLUE), vdec.get("수정안", 0)],
            [("디자이너 검토", PURPLE), vdec.get("디자이너 검토", 0)],
            [("고지 없음", ORANGE), vdec.get("고지 없음", 0)],
        ], [450, 450])
        pages.append(im)
        for title, rec in pick_video_cases():
            im, d = page(); header(d, f"영상 · {title} — {rec['ad_id']}", len(pages) + 1)
            after = {c["id"]: c for c in rec.get("checks_after") or []}
            lines = [f"{c['id']} {c['name']}: {fmtv(c)} → {fmtv(after[c['id']]) if c['id'] in after else '-'}"
                     for c in rec["checks_before"] if c["ok"] is False]
            video_page(im, d, rec["ad_id"], PURPLE if rec["decision"] != "수정안" else BLUE, lines or ["기준 미달 항목 없음"])
            pages.append(im)

    # 마지막. 기준 이슈와 다음 할 일
    im, d = page(); header(d, "확인된 문제와 다음 할 일", len(pages) + 1)
    d.text((110, 210), "A3 글자 크기 기준 보정 필요", font=F("Bold", 40), fill=RED)
    bullets(d, 110, 290, [
        "디자이너 검토가 많은 주된 이유는 레이아웃이 아니라 A3 기준(모바일 390pt 환산 5.9pt)임.",
        "고지문이 긴 포스터는 이 크기로 키우면 원본 높이의 10~20%가 더 필요함. 승인된 실제 광고의 약 80%도 이 기준에 미달함.",
        "인쇄물 기준(A4 9pt, 같은 환산으로 약 4.1pt)으로 바꿔 돌려 보면 원본 유지 52 · 수정안 31 · 디자이너 검토 42로 달라짐.",
        "→ A3 보정값은 05(RAG·판정)와 함께 정해야 함.",
    ], width=2100)
    d.text((110, 830), "다음 할 일", font=F("Bold", 40), fill=INK)
    bullets(d, 110, 910, [
        "05 실제 판정 결과(rag/outputs)로 ① 내용 수정 다시 확인 (지금은 샘플 판정 파일로만 시험함).",
        "표준 추가 문구(rules/phrases.yaml) 준법 검토 후 확정.",
        "OCR로 읽은 글자를 그대로 다시 쓰므로 오타·띄어쓰기 누락이 남음 → 최종본은 원문 문구로 교체 필요.",
        "추가 문구가 원본보다 한 단계 굵게 나오는 경우가 있음 → 굵기 추정 개선.",
        "영상: 흐르는 자막(crawl) 수정, 음성 고지 확인은 아직 미적용. 수정 영상은 편집 지시와 함께 검토용 시안으로 사용.",
    ], width=2100)
    pages.append(im)

    dst = HERE / "06_수정안_결과보고.pdf"
    pages[0].save(dst, save_all=True, append_images=pages[1:], resolution=200)
    print(dst, len(pages), "pages")


if __name__ == "__main__":
    main()
