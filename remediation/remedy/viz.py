"""비교 패널: 원본 | 수정안 | 고친 부분 확대(전/후) + 판정표 + 디자인 보존 지표."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BADGE = {"원본 유지": (22, 163, 74), "수정안": (37, 99, 235), "수정안 + 표현 검토": (37, 99, 235), "표현 검토": (217, 119, 6), "디자이너 검토": (109, 40, 217), "고지 없음": (217, 119, 6)}
RED, GREEN, BLUE, ORANGE = (230, 25, 75), (22, 163, 74), (37, 99, 235), (234, 88, 12)
BG, INK, SUB = (246, 247, 249), (24, 24, 27), (100, 104, 112)


def _fit(im, w):
    return im.resize((w, max(1, int(im.height * w / im.width))), Image.LANCZOS)


def _boxes(im, boxes, color, scale, width=3):
    d = ImageDraw.Draw(im)
    for b in boxes:
        d.rectangle([b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale], outline=color, width=width)
    return im


def _zoom(img, boxes, pad_ratio=0.6):
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    p = int((y1 - y0) * pad_ratio) + 10
    W, H = img.size
    return img.crop((max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)))


def _fmt(c):
    v = c["value"]
    return "-" if v is None else (f"{v:.3f}" if c["id"] == "A4" else f"{v:.2f}")


def panel(path, font_path, ad, decision, orig, fixed=None, before_boxes=(), after_boxes=(),
          checks_after=None, design=None, notes=(), added_boxes=(), flag_boxes=()):
    cw = 520
    f_t, f_b, f_s = (ImageFont.truetype(font_path, s) for s in (30, 21, 18))
    o = _boxes(_fit(orig, cw), before_boxes, RED, cw / orig.width)
    cols = [("원본", _boxes(o, flag_boxes, ORANGE, cw / orig.width, 4))]
    if fixed is not None:
        f = _boxes(_fit(fixed, cw), after_boxes, GREEN, cw / fixed.width)
        cols.append(("수정안", _boxes(f, added_boxes, BLUE, cw / fixed.width)))
        z0 = _fit(_zoom(orig, list(before_boxes) or list(flag_boxes) or [[0, int(orig.height * .8), orig.width, orig.height]]), cw)
        z1 = _fit(_zoom(fixed, list(after_boxes) + list(added_boxes)), cw)
        cols.append(("확대 · 전 / 후", (z0, z1)))
    head_h = 92
    body_h = max((c[1].height if isinstance(c[1], Image.Image) else c[1][0].height + c[1][1].height + 40) for c in cols)
    rows = 5 + len(notes) + (len(design) if design else 0)
    foot_h = 60 + 28 * rows
    W = max(len(cols) * (cw + 24) + 24, 1000)
    sheet = Image.new("RGB", (W, head_h + body_h + 44 + foot_h), BG)
    d = ImageDraw.Draw(sheet)
    title = "  ".join([ad["ad_id"], " · ".join(str(ad[k]) for k in ("brand", "product", "media") if ad.get(k))])
    d.text((24, 22), title, font=f_t, fill=INK)
    lab = decision
    tw = d.textlength(lab, font=f_b) + 32
    d.rounded_rectangle((W - tw - 24, 24, W - 24, 62), 12, fill=BADGE.get(lab, SUB))
    d.text((W - tw - 8, 30), lab, font=f_b, fill="white")
    for k, (name, im) in enumerate(cols):
        x = 24 + k * (cw + 24)
        d.text((x, head_h - 4), name, font=f_b, fill=SUB)
        y = head_h + 28
        if isinstance(im, Image.Image):
            sheet.paste(im, (x, y))
        else:
            sheet.paste(im[0], (x, y))
            d.text((x, y + im[0].height + 6), "↓", font=f_b, fill=SUB)
            sheet.paste(im[1], (x, y + im[0].height + 36))
    y = head_h + body_h + 52
    d.text((24, y), "② 표시 방법 (05 노트북 A2·A3·A4와 같은 기준)   상자: 빨강=고칠 곳 · 초록=고친 곳 · 파랑=추가 문구 · 주황=표현 검토", font=f_b, fill=INK)
    y += 36
    after = {c["id"]: c for c in (checks_after or [])}
    for c in ad["_checks"]:
        a = after.get(c["id"])
        mark = lambda ok: "통과" if ok else ("미달" if ok is False else "측정불가")
        line = f"{c['id']} {c['name']}: {_fmt(c)} ({mark(c['ok'])})"
        if a:
            line += f"  →  {_fmt(a)} ({mark(a['ok'])})"
        line += f"   기준 ≥ {c['threshold']}"
        col = GREEN if (a or c)["ok"] else (RED if (a or c)["ok"] is False else SUB)
        d.text((24, y), line, font=f_s, fill=col)
        y += 28
    for k, v in (design or {}).items():
        d.text((24, y), f"· {k}: {v}", font=f_s, fill=SUB)
        y += 28
    if any(str(n).startswith("[") for n in notes):
        d.text((24, y + 4), "① 내용 (05 LLM 판정 B·C·D·E)", font=f_b, fill=INK)
        y += 36
    for n in notes:
        col = BLUE if str(n).startswith("[추가") else (ORANGE if str(n).startswith("[표현") else SUB)
        d.text((24, y), f"· {n}"[:110], font=f_s, fill=col)
        y += 28
    sheet.save(path, quality=90)


def panel_candidates(path, font_path, ad, decision, orig, before_boxes, cands, notes, flag_boxes=()):
    """원본 | A안 | B안 | C안 (모두 원본과 같은 규격) + 추천안 확대 전/후 + 판정표."""
    cw = 440
    f_t, f_b, f_s, f_x = (ImageFont.truetype(font_path, s) for s in (30, 21, 17, 15))
    o = _boxes(_fit(orig, cw), before_boxes, RED, cw / orig.width)
    o = _boxes(o, flag_boxes, ORANGE, cw / orig.width, 4)
    tiles = [("원본", o, None)]
    for c in cands:
        if c.get("img") is None:
            ph = Image.new("RGB", o.size, (232, 234, 238))
            ImageDraw.Draw(ph).text((20, o.height // 2 - 10), "만들 수 없음", font=f_b, fill=SUB)
            tiles.append((c["label"], ph, c))
            continue
        im = Image.fromarray(c["img"])
        t = _fit(im, cw)
        t = _boxes(t, [b for b, k in c["drawn"] if k != "added"], GREEN, cw / im.width, 2)
        t = _boxes(t, [b for b, k in c["drawn"] if k == "added"], BLUE, cw / im.width, 2)
        tiles.append((c["label"], t, c))
    best = next((c for c in cands if c.get("recommended")), None)
    zoom = None
    if best is not None:
        im = Image.fromarray(best["img"])
        z0 = _fit(_zoom(orig, list(before_boxes) or [[0, int(orig.height * .8), orig.width, orig.height]]), cw)
        z1 = _fit(_zoom(im, [b for b, _ in best["drawn"]]), cw)
        zoom = (z0, z1)
    cap_h = 150
    body_h = max(t[1].height for t in tiles) + cap_h
    if zoom:
        body_h = max(body_h, zoom[0].height + zoom[1].height + 70)
    ncol = len(tiles) + (1 if zoom else 0)
    W = max(ncol * (cw + 20) + 24, 1000)
    after = {c["id"]: c for c in (best or cands[0]).get("checks_after") or []}
    foot_h = 80 + 28 * (3 + len(notes))
    sheet = Image.new("RGB", (W, 96 + body_h + foot_h), BG)
    d = ImageDraw.Draw(sheet)
    d.text((24, 22), "  ".join([ad["ad_id"], " · ".join(str(ad[k]) for k in ("brand", "product", "media") if ad.get(k))]),
           font=f_t, fill=INK)
    tw = d.textlength(decision, font=f_b) + 32
    d.rounded_rectangle((W - tw - 24, 24, W - 24, 62), 12, fill=BADGE.get(decision, SUB))
    d.text((W - tw - 8, 30), decision, font=f_b, fill="white")
    d.text((24, 70), f"모든 안은 원본과 같은 규격 {orig.width}×{orig.height}px · 초록=고친 줄 · 파랑=추가 문구 · 빨강=원본 문제 위치 · 주황=표현 검토",
           font=f_x, fill=SUB)
    for k, (name, im, c) in enumerate(tiles):
        x, y = 24 + k * (cw + 20), 100
        sheet.paste(im, (x, y + 30))
        if c is not None and c.get("recommended"):
            d.rectangle((x - 4, y + 26, x + cw + 3, y + 33 + im.height), outline=BLUE, width=4)
        d.text((x, y), ("★ 추천  " if c is not None and c.get("recommended") else "") + name, font=f_s,
               fill=BLUE if c is not None and c.get("recommended") else INK)
        if c is None:
            continue
        yy = y + 40 + im.height
        if c["passed"]:
            d.text((x, yy), "기준 통과", font=f_s, fill=GREEN)
            info = [f"원본과 달라진 면적 {c['changed_pct']}%", f"글자 ×{c['size_factor']:.2f} · 글자색 ΔE {c['ink_delta_e']:.1f}"]
            if c.get("rows_removed"):
                info.append(f"배경 줄 {c['rows_removed']}px를 여러 곳에서 나눠 회수")
            if c.get("visual_scale"):
                info.append(f"위쪽 비주얼 {c['visual_scale']:.0%}로 축소")
            pr = next((s.get("pitch_ratio") for s in c.get("steps", []) if s.get("pitch_ratio")), None)
            if pr:
                info.append(f"줄 간격 {pr[0]} → {pr[1]} (글자 높이 배수)")
        else:
            d.text((x, yy), "불가 / 미달", font=f_s, fill=RED)
            info = c.get("why", [])[:3]
        for i, line in enumerate(info):
            d.text((x, yy + 26 + i * 22), line[:34], font=f_x, fill=SUB)
    if zoom:
        x = 24 + len(tiles) * (cw + 20)
        d.text((x, 100), "추천안 확대 · 전 / 후", font=f_s, fill=INK)
        sheet.paste(zoom[0], (x, 130))
        d.text((x, 136 + zoom[0].height), "↓", font=f_b, fill=SUB)
        sheet.paste(zoom[1], (x, 166 + zoom[0].height))
    y = 96 + body_h + 10
    d.text((24, y), "② 표시 방법 (05 노트북 A2·A3·A4 기준) — 추천안 기준 전 → 후", font=f_b, fill=INK)
    y += 36
    mark = lambda ok: "통과" if ok else ("미달" if ok is False else "측정불가")
    for c in ad["_checks"]:
        a = after.get(c["id"])
        line = f"{c['id']} {c['name']}: {_fmt(c)} ({mark(c['ok'])})" + (f"  →  {_fmt(a)} ({mark(a['ok'])})" if a else "")
        line += f"   기준 ≥ {c['threshold']}"
        okv = (a or c)["ok"]
        d.text((24, y), line, font=f_s, fill=GREEN if okv else (RED if okv is False else SUB))
        y += 28
    for n in notes:
        col = BLUE if str(n).startswith("[추가") else (ORANGE if str(n).startswith("[표현") else SUB)
        d.text((24, y), f"· {n}"[:130], font=f_s, fill=col)
        y += 28
    sheet.save(path, quality=90)
