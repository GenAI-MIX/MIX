"""영상 광고 수정안. 영상 길이·화면 크기는 그대로 두고, 위험고지 자막만 고친다.

- 타임라인 (A5 노출 1/3, A6 초당 글자 수): 기존 고지 자막을 앞 장면까지 이어서 보여 줌 (영상 길이 그대로)
- 화면 (A2 명암비, A4 헤드라인 대비, A7 글자 크기): 원래 자리·색·단 폭 유지, 기준을 살짝 넘길 크기로 다시 씀.
  아래 기준선은 그대로 두고 위로 키움. 글자색은 그대로, 명암비가 모자라면 배경과 같은 색조의 띠를 깔아 해결
- ffmpeg overlay로 실제 수정 영상(mp4)을 만들고, 그 영상에서 프레임을 뽑아 다시 잰다
"""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .color import contrast, delta_e, fit_ink, luminance
from .measure import analyze_block
from .render import erase, lines_of, wrap


def grab(video, t, dst):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", str(dst)],
                   check=True)
    return Image.open(dst).convert("RGB")


def checks(m, duration, th):
    d = duration or 0
    e = m.get("risk_exposure_s") or 0
    vp = (m.get("risk_display_min_pt") or {}).get("mobile_landscape")
    rows = [("A2", "명암비", m.get("risk_contrast_min"), th["contrast"], ">="),
            ("A4", "헤드라인 대비", m.get("risk_headline_ratio_min"), th["ratio_video"], ">="),
            ("A5", "노출 비율", round(e / d, 3) if d else None, round(th["exposure_frac"], 3), ">="),
            ("A6", "초당 글자 수", m.get("risk_cps_max"), th["cps"], "<="),
            ("A7", "글자 크기(pt)", vp, th["video_pt"], ">=")]
    out = []
    for i, n, v, t, op in rows:
        ok = None if v is None else bool(v >= t if op == ">=" else v <= t)
        out.append({"id": i, "name": n, "value": v, "threshold": t, "op": op, "ok": ok})
    return out


def union_len(windows):
    tot, end = 0.0, -1e9
    for a, b in sorted(windows):
        if b <= end:
            continue
        tot += b - max(a, end)
        end = b
    return tot


def plan_timeline(ad, th, margin):
    """고지 노출 구간을 늘릴 계획. 반환: {card_id: (new_start, new_end)}, 메모."""
    D = ad["duration"]
    exps = ad.get("risk_exposures", [])
    cards = {c["id"]: c for c in ad["cards"]}
    windows = {e["id"]: [e["t_start"], e["t_end"]] for e in exps}
    plan, notes = {}, []
    # A6: 너무 빨리 지나가는 고지 → 그 자막을 읽을 수 있을 만큼 앞으로 늘림
    for e in exps:
        if e.get("risk_cps") and e["risk_cps"] > th["cps"]:
            need = e["risk_chars"] / (th["cps"] / margin) - (e["t_end"] - e["t_start"])
            a = max(0.0, e["t_start"] - need)
            windows[e["id"]][0] = a
            notes.append(f"A6: 자막 {e['risk_chars']}자를 읽을 시간 확보 → {e['t_start']:.1f}초가 아니라 {a:.1f}초부터 표시")
    # A5: 전체 노출이 영상의 1/3 미만 → 글자가 가장 많은 정지 자막을 앞 장면까지 유지
    need_total = D * th["exposure_frac"] * margin
    have = union_len(windows.values())
    statics = [e for e in exps if e.get("kind") == "static" and e.get("text_card_id") is not None]
    if have < need_total and statics:
        main = max(statics, key=lambda e: e["risk_chars"])
        w = windows[main["id"]]
        short = need_total - have
        a = max(0.0, w[0] - short)
        short -= w[0] - a
        b = min(D, w[1] + max(0.0, short))
        windows[main["id"]] = [a, b]
        notes.append(f"A5: 노출 {have:.1f}초 → 영상 {D:.1f}초의 1/3({D / 3:.1f}초) 이상이 되도록 "
                     f"고지 자막을 {a:.1f}~{b:.1f}초로 유지 (원래 {main['t_start']:.1f}~{main['t_end']:.1f}초)")
    for e in exps:
        if windows[e["id"]] != [e["t_start"], e["t_end"]]:
            plan[e["id"]] = tuple(round(v, 2) for v in windows[e["id"]])
    return plan, windows, notes


class CardFix:
    """정지 고지 자막 하나를 다시 그린 오버레이 두 장: main(원래 구간: 원래 글자를 지우고 덮음), ext(늘린 구간: 띠+글자)."""

    def __init__(self, frame, card, blocks, obstacles, h_target, fonts, mcfg, th, ladder, static=True):
        W, H = frame.size
        img = np.asarray(frame).copy()
        lines = lines_of(blocks)
        big = max(blocks, key=lambda b: len(b["text"]))
        probe = analyze_block(img, card["bbox"], mcfg)
        fg = probe["fg_rgb"] or [255, 255, 255]
        weight = fonts.closest_weight(img, big)
        x0, y0, x1, y1 = card["bbox"]
        h0 = float(np.median([b["h_px"] for b in blocks if b.get("h_px")] or [y1 - y0]))
        same = [o for o in obstacles if o[1] < y1 and o[3] > y0]
        gap = 0.8 * h0
        lw = max([o[2] for o in same if o[2] <= x0 + 2] + [0.04 * W]) + gap
        rw = min([o[0] for o in same if o[0] >= x1 - 2] + [0.96 * W]) - gap
        font = fonts.font_for(weight, h_target)
        text = " ".join(l["text"] for l in lines)          # 자막은 한 문단으로 다시 줄바꿈 (짧은 꼬리 줄 방지)
        tops = [l["bbox"][1] for l in lines]
        pr = float(np.clip(np.median(np.diff(tops)) / h0 if len(lines) > 1 else 1.5, 1.25, 1.8))
        pitch = h_target * pr
        bottom = y1
        cx = (x0 + x1) / 2
        # 원래 단 폭에서 시작, 위로 넘치거나 다른 글자와 겹치면 양옆 벽까지 조금씩 넓힘
        half = (x1 - x0) / 2
        max_half = min(cx - lw, rw - cx)
        while True:
            avail_l, avail_r = cx - half, cx + half
            text_lines = wrap(text, font, avail_r - avail_l)
            block_h = pitch * (len(text_lines) - 1) + h_target * 1.45
            top = bottom - block_h
            hit = [o for o in obstacles if o[3] > top and o[1] < bottom and o[0] < avail_r and o[2] > avail_l]
            if (not hit and top >= 0.03 * H) or half >= max_half:
                break
            half = min(max_half, half * 1.08 + 4)
        above = [o for o in obstacles if o[3] > top and o[1] < bottom and o[0] < avail_r and o[2] > avail_l]
        self.fail = []
        if top < 0.03 * H:
            self.fail.append("화면 위쪽까지 넘침")
        if above:
            self.fail.append("위쪽 다른 글자와 겹침")
        # 원래 자리 정렬: 원래 줄들이 가운데 정렬이면 가운데, 아니면 왼쪽
        cents = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in lines]
        lefts = [l["bbox"][0] for l in lines]
        center = len(lines) > 1 and np.std(cents) + 2 < np.std(lefts)
        center = center or len(lines) == 1
        # 지운 배경 (원래 구간용): 원래 글자 영역을 주변색으로 채움
        plate = img.copy()
        erase(plate, [int(x0), int(y0), int(x1), int(y1)], pad=max(3, int(0.4 * h0)))
        pad = int(0.45 * h_target)
        P = [int(max(0, min(avail_l, x0) - pad)), int(max(0, top - pad)), int(min(W, max(avail_r, x1) + pad)),
             int(min(H, bottom + pad))]
        reg = plate[P[1]:P[3], P[0]:P[2]].reshape(-1, 3).astype(float)
        bg = np.median(reg, axis=0).astype(int).tolist()
        lum = reg @ [0.299, 0.587, 0.114]
        worst = reg[lum >= np.percentile(lum, 85)].mean(0) if luminance(fg) < 0.4 else reg[lum <= np.percentile(lum, 15)].mean(0)
        target_c = th["contrast"] * ladder["contrast_margin"]
        self.scrim = None
        if min(contrast(fg, bg), contrast(fg, worst)) < target_c:
            # 글자색은 그대로 두고, 배경과 같은 색조로 밝기만 옮긴 띠를 깔아 명암비 확보
            self.scrim = fit_ink(bg, fg, target_c)
        band_col = self.scrim or bg
        self.size = (W, H)
        self.info = {"card": card["id"], "font": Path(weight).stem, "h_before_px": h0, "h_after_px": round(h_target, 1),
                     "size_factor": round(h_target / h0, 2), "lines": len(text_lines), "fg": fg, "bg": bg,
                     "band": band_col if self.scrim else None, "center": bool(center)}
        # 텍스트 배치
        boxes, rows = [], []
        from PIL import ImageFont  # noqa: F401
        d0 = ImageDraw.Draw(Image.new("RGB", (W, H)))
        asc = font.getbbox("가")[1]
        for i, t in enumerate(text_lines):
            tw = font.getlength(t)
            x = (avail_l + avail_r - tw) / 2 if center else avail_l
            y = top + i * pitch - asc
            bb = d0.textbbox((x, y), t, font=font)
            rows.append((x, y, t))
            boxes.append([int(bb[0]) - 2, int(bb[1]) - 3, int(bb[2]) + 2, int(bb[3]) + 3])
        self.boxes = boxes
        tb = [min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad, max(b[2] for b in boxes) + pad,
              max(b[3] for b in boxes) + pad]
        if not static:   # 배경이 움직이면 띠가 원래 작은 글자까지 덮어야 함
            tb = [min(tb[0], x0 - pad), min(tb[1], y0 - pad), max(tb[2], x1 + pad), max(tb[3], y1 + pad)]
        self.info["static_bg"] = bool(static)

        def band_layer(alpha):
            a = Image.new("L", (W, H), 0)
            ImageDraw.Draw(a).rounded_rectangle(tb, radius=int(pad * 0.8), fill=int(255 * alpha))
            a = a.filter(ImageFilter.GaussianBlur(pad * 0.35))
            lay = Image.new("RGBA", (W, H), tuple(band_col) + (0,))
            lay.putalpha(a)
            return lay

        def text_layer():
            lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            dd = ImageDraw.Draw(lay)
            for x, y, t in rows:
                dd.text((x, y), t, font=font, fill=tuple(fg) + (255,))
            return lay

        # main: 지운 배경 패치(가장자리 부드럽게) + (필요하면) 띠 + 글자
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle([int(x0) - pad, int(y0) - pad, int(x1) + pad, int(y1) + pad], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(2, pad * 0.3)))
        patch = Image.fromarray(plate).convert("RGBA")
        patch.putalpha(m)
        main = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if static:       # 정지 배경: 원래 글자를 지운 배경 패치로 덮음
            main.alpha_composite(patch)
            if self.scrim:
                main.alpha_composite(band_layer(0.92))
        else:            # 움직이는 배경: 패치 대신 원래 톤의 띠로 덮음
            main.alpha_composite(band_layer(0.94))
        main.alpha_composite(text_layer())
        self.main = main
        # ext: 다른 장면 위 → 띠는 항상 (원래 자막 배경 톤), 글자
        ext = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ext.alpha_composite(band_layer(0.9))
        ext.alpha_composite(text_layer())
        self.ext = ext


def render_video(src, dst, overlays):
    """overlays: [(png, [(a, b), ...])]. 영상 길이·해상도·음성 그대로."""
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    for png, _ in overlays:
        args += ["-i", str(png)]
    chain, last = [], "0:v"
    for k, (_, wins) in enumerate(overlays):
        en = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in wins)
        out = f"v{k}"
        chain.append(f"[{last}][{k + 1}:v]overlay=0:0:enable='{en}'[{out}]")
        last = out
    args += ["-filter_complex", ";".join(chain), "-map", f"[{last}]", "-map", "0:a?", "-c:v", "libx264", "-crf", "20",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)]
    subprocess.run(args, check=True)
