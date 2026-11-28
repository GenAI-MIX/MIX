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

from .color import contrast, delta_e, fit_ink, luminance, palette, pick_ink
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
        # 자리 후보 (원래 결에서 가까운 순): 원래 기준선 → 아래 안전선 → 위 안전선, 가로 중심은 원래 단 → 화면 가운데.
        # 각 자리에서 실제로 비어 있는 가로 폭을 재고 그 폭에 맞춰 줄바꿈 (넓히기만 하지 않고 좁히기도 함)
        safe_b, safe_t = H * 0.955, H * 0.05
        gap = 0.8 * h0

        def layout(width):
            tl = wrap(text, font, width)
            return tl, pitch * (len(tl) - 1) + h_target * 1.45

        def span(t_, b_, cx_):
            rowo = [o for o in obstacles if o[1] < b_ + gap / 2 and o[3] > t_ - gap / 2]
            if any(o[0] < cx_ < o[2] for o in rowo):
                return None
            l_ = max([o[2] for o in rowo if o[2] <= cx_] + [0.04 * W]) + gap
            r_ = min([o[0] for o in rowo if o[0] >= cx_] + [0.96 * W]) - gap
            return (l_, r_) if r_ - l_ > max(0.25 * W, 6 * h_target) else None

        orig_w = max(x1 - x0, 8 * h_target)
        chosen, last = None, None
        for anchor, yb in (("bottom", y1), ("bottom", max(y1, safe_b)), ("top", safe_t)):
            for cx in ((x0 + x1) / 2, W / 2):
                width = orig_w
                for _ in range(5):
                    tl, bh = layout(width)
                    top_, bot_ = (yb - bh, yb) if anchor == "bottom" else (yb, yb + bh)
                    sp = span(top_, bot_, cx)
                    if sp is None:
                        break
                    half = min(cx - sp[0], sp[1] - cx)
                    new_w = min(2 * half, 0.9 * W)
                    if abs(new_w - width) < 4:
                        break
                    width = new_w
                tl, bh = layout(width)
                top_, bot_ = (yb - bh, yb) if anchor == "bottom" else (yb, yb + bh)
                al, ar = cx - width / 2, cx + width / 2
                ok_ = (top_ >= 0.03 * H and bot_ <= H * 0.985 and
                       not any(o[3] > top_ and o[1] < bot_ and o[0] < ar and o[2] > al for o in obstacles))
                last = last or (al, ar, tl, top_, bot_)
                if ok_:
                    chosen = (al, ar, tl, top_, bot_)
                    break
            if chosen:
                break
        if chosen is None:
            chosen = last
        avail_l, avail_r, text_lines, top, bottom = chosen
        self.placement = ("원래 자리" if abs(bottom - y1) < 2 and abs((avail_l + avail_r) / 2 - (x0 + x1) / 2) < 2
                          else "자리 조정 (겹침 피함)")
        self.fail = []
        if top < 0.03 * H:
            self.fail.append("화면 위쪽까지 넘침")
        if any(o[3] > top and o[1] < bottom and o[0] < avail_r and o[2] > avail_l for o in obstacles):
            self.fail.append("어느 자리에 두어도 다른 글자와 겹침")
        # 원래 자리 정렬: 원래 줄들이 가운데 정렬이면 가운데, 아니면 왼쪽
        cents = [(l["bbox"][0] + l["bbox"][2]) / 2 for l in lines]
        lefts = [l["bbox"][0] for l in lines]
        center = len(lines) > 1 and np.std(cents) + 2 < np.std(lefts)
        center = center or len(lines) == 1
        # 지운 배경 (원래 구간용): 원래 글자 영역을 주변색으로 채움
        plate = img.copy()
        erase(plate, [int(x0), int(y0), int(x1), int(y1)], pad=max(3, int(0.4 * h0)))
        pad = int(0.45 * h_target)
        self.size = (W, H)
        self.static = static
        self.target_c = th["contrast"] * ladder["contrast_margin"]
        self.pal = palette(img)
        self.fg = fg
        self.font = font
        rows, boxes = [], []
        d0 = ImageDraw.Draw(Image.new("RGB", (W, H)))
        asc = font.getbbox("가")[1]
        for i, t in enumerate(text_lines):
            tw = font.getlength(t)
            x = (avail_l + avail_r - tw) / 2 if center else avail_l
            y = top + i * pitch - asc
            bb = d0.textbbox((x, y), t, font=font)
            rows.append((x, y, t))
            boxes.append([int(bb[0]) - 2, int(bb[1]) - 3, int(bb[2]) + 2, int(bb[3]) + 3])
        self.rows, self.boxes = rows, boxes
        tb = [min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad, max(b[2] for b in boxes) + pad,
              max(b[3] for b in boxes) + pad]
        self.text_rect = tb
        # 원래 작은 글자를 지울 영역: 다른 글자(헤드라인 등)와 겹치는 쪽은 잘라내서 그 글자는 건드리지 않음
        r0 = [int(x0) - pad, int(y0) - pad, int(x1) + pad, int(y1) + pad]
        for o in obstacles:
            if o[0] < r0[2] and o[2] > r0[0] and o[1] < r0[3] and o[3] > r0[1]:
                if (o[1] + o[3]) / 2 < (y0 + y1) / 2:
                    r0[1] = max(r0[1], int(o[3]) + 1)
                else:
                    r0[3] = min(r0[3], int(o[1]) - 1)
        self.orig_rect = r0
        # 원래 구간: 정지 배경이면 지운 배경 위에서, 움직이면 (블러로 지운) 원래 프레임 위에서 글자색 고르기
        under = plate if static else img
        reg = under[max(0, tb[1]):tb[3], max(0, tb[0]):tb[2]]
        ink, how = pick_ink(fg, reg, self.pal, self.target_c)
        self.need_local_fade = ink is None
        if ink is None:
            ink = fg
        self.ink, self.how = ink, how
        self.info = {"card": card["id"], "font": fonts.name(weight), "h_before_px": h0, "h_after_px": round(h_target, 1),
                     "size_factor": round(h_target / h0, 2), "lines": len(text_lines), "fg": fg, "ink": ink,
                     "ink_how": how, "center": bool(center), "static_bg": bool(static)}
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).rectangle(self.orig_rect, fill=255)
        m = m.filter(ImageFilter.GaussianBlur(max(2, pad * 0.3)))
        patch = Image.fromarray(plate).convert("RGBA")
        patch.putalpha(m)
        main = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if static:
            main.alpha_composite(patch)
        if self.need_local_fade:
            main.alpha_composite(self.local_fade(np.median(reg.reshape(-1, 3), axis=0)))
        main.alpha_composite(self.text_layer(ink))
        self.main = main

    def text_layer(self, ink, dx=0, dy=0):
        W, H = self.size
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        dd = ImageDraw.Draw(lay)
        for x, y, t in self.rows:
            dd.text((x + dx, y + dy), t, font=self.font, fill=tuple(int(v) for v in ink) + (255,))
        return lay

    def local_fade(self, bg, dx=0, dy=0):
        """마지막 수단: 글자 바로 뒤만 원래 톤을 부드럽게 가라앉힘 (가장자리가 보이지 않게 넓게 흐림)."""
        W, H = self.size
        col = fit_ink([int(v) for v in bg], self.fg, self.target_c * 1.35)
        tb = self.text_rect
        a = Image.new("L", (W, H), 0)
        r = int((tb[3] - tb[1]) * 0.35)
        g = max(2, int((tb[3] - tb[1]) * 0.18))
        ImageDraw.Draw(a).rounded_rectangle([tb[0] + dx - g, tb[1] + dy - g, tb[2] + dx + g, tb[3] + dy + g], radius=r, fill=242)
        a = a.filter(ImageFilter.GaussianBlur(g))
        lay = Image.new("RGBA", (W, H), tuple(int(v) for v in col) + (0,))
        lay.putalpha(a)
        return lay

    def ext_at(self, dx, dy, bg_pixels):
        """늘린 구간용: 글자를 (dx, dy)로 옮기고, 그 장면 배경에 맞춰 글자색을 다시 고름."""
        ink, how = pick_ink(self.fg, bg_pixels, self.pal, self.target_c)
        W, H = self.size
        lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        if ink is None:
            lay.alpha_composite(self.local_fade(np.median(np.asarray(bg_pixels).reshape(-1, 3), axis=0), dx, dy))
            ink, how = self.fg, "글자 바로 뒤만 원래 톤으로 부드럽게 가라앉힘"
        lay.alpha_composite(self.text_layer(ink, dx, dy))
        return lay, ink, how


def free_windows(ad, rect, busy, before_t, after_t, need, pad=20):
    """같은 자막을 더 보여줄 장면 고르기. 장면마다 원래 자리 → 왼쪽·오른쪽 아래 → 위쪽 순으로 다른 글자가 없는 자리를 찾음.
    원래 구간에 가까운 장면부터 need초만큼. 반환: [(a, b, dx, dy)], 확보한 초."""
    W, H = ad["size"]
    rw, rh = rect[2] - rect[0], rect[3] - rect[1]
    m = 0.05 * W
    top_dy = 0.06 * H - rect[1]
    cx = (W - rw) / 2 - rect[0]
    offsets = [(0, 0), (m - rect[0], 0), (cx, 0), (W - m - rect[2], 0),
               (0, top_dy), (m - rect[0], top_dy), (cx, top_dy), (W - m - rect[2], top_dy)]
    blocks_by_screen = {}
    for b in ad["ocr_blocks"]:
        blocks_by_screen.setdefault(b["screen_id"], []).append(b)

    def free(sid, dx, dy):
        x0, y0, x1, y1 = rect[0] + dx - pad, rect[1] + dy - pad, rect[2] + dx + pad, rect[3] + dy + pad
        if x0 < 0 or x1 > W or y0 < 0 or y1 > H:
            return False
        return not any(bl["bbox"][0] < x1 and bl["bbox"][2] > x0 and bl["bbox"][1] < y1 and bl["bbox"][3] > y0
                       for bl in blocks_by_screen.get(sid, []) if bl.get("type") != "risk_notice")
    cands = []
    for s_ in ad["screens"]:
        a, b = s_["t_start"], s_["t_end"]
        for ba, bb in busy:           # 이미 고지가 있는 구간은 빼고
            if a < bb and b > ba:
                if a >= ba:
                    a = max(a, bb)
                if b <= bb:
                    b = min(b, ba)
        if b - a < 0.4:
            continue
        off = next(((int(dx), int(dy)) for dx, dy in offsets if free(s_["id"], dx, dy)), None)
        if off is None:
            continue
        dist = before_t - b if b <= before_t else (a - after_t if a >= after_t else 0)
        cands.append((max(dist, 0), a, b, off))
    out, got = [], 0.0
    for _, a, b, off in sorted(cands):
        if got >= need:
            break
        take = min(b - a, need - got)
        seg = (b - take, b) if b <= before_t else (a, a + take)
        out.append((round(seg[0], 2), round(seg[1], 2), off[0], off[1]))
        got += take
    return sorted(out), got


def hold_plan(ad, card_win, need, max_frac=0.6):
    """V3: 엔드 카드(정지 고지 화면)를 need초 일찍 시작. 바로 앞 컷의 끝을 그만큼 잘라내고 엔드 카드 첫 프레임으로 채움.
    음성은 그대로 → 전체 길이 그대로. 앞 컷을 max_frac 넘게 자르면 불가. 반환 (t_cut, need) 또는 None."""
    t0 = card_win[0]
    # 고지 시작 직전 화면(같은 화면에서 고지가 중간에 뜨는 경우 포함): 그 화면이 시작된 뒤 고지 전까지의 길이
    prev = [s_ for s_ in ad["screens"] if s_["t_start"] < t0 - 0.05 <= s_["t_end"] + 0.05]
    # 화면 구간 정보가 없으면(전처리가 글자 있는 화면만 나눔) 고지 앞 전체를 한 덩어리로 보고 더 보수적으로
    avail = (t0 - min(prev, key=lambda s_: t0 - s_["t_start"])["t_start"]) * max_frac if prev else t0 * 0.35
    if need > avail:
        return None
    return (round(t0 - need, 2), round(need, 2))


def render_video(src, dst, overlays, blurs=(), hold=None):
    """overlays: [(png, [(a, b), ...])], blurs: [(rect, [(a, b)])] — 움직이는 배경의 원래 작은 글자 자리를 흐려서 지움.
    영상 길이·해상도·음성 그대로."""
    args = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    for png, _ in overlays:
        args += ["-i", str(png)]
    chain, last, n = [], "0:v", 0
    if hold:   # (t_cut, need): [0,t_cut] + 엔드카드 첫 프레임 need초 + [t_cut+need, 끝]  (음성은 원본 그대로)
        tc, nd = hold
        t0 = tc + nd
        chain.append(f"[0:v]split=3[h0][h1][h2];[h0]trim=0:{tc:.3f},setpts=PTS-STARTPTS[ha];"
                     f"[h1]trim={t0:.3f}:{t0 + 0.05:.3f},setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={nd:.3f},"
                     f"trim=0:{nd:.3f},setpts=PTS-STARTPTS[hb];[h2]trim=start={t0:.3f},setpts=PTS-STARTPTS[hc];"
                     f"[ha][hb][hc]concat=n=3:v=1:a=0,setpts=PTS-STARTPTS[vh]")
        last = "vh"
    for k, (rect, wins) in enumerate(blurs):
        x0, y0, x1, y1 = [int(v) for v in rect]
        x0, y0 = max(0, x0), max(0, y0)
        w, h = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
        en = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in wins)
        r = max(1, min(12, h // 2 - 1, w // 2 - 1))
        chain.append(f"[{last}]split[bs{k}][bb{k}];[bb{k}]crop={w}:{h}:{x0}:{y0},boxblur=luma_radius={r}:luma_power=3:chroma_radius={max(1, min(r, h // 4 - 1, w // 4 - 1))}:chroma_power=3[bl{k}];"
                     f"[bs{k}][bl{k}]overlay={x0}:{y0}:enable='{en}'[vb{k}]")
        last = f"vb{k}"
    for k, (_, wins) in enumerate(overlays):
        en = "+".join(f"between(t,{a:.2f},{b:.2f})" for a, b in wins)
        out = f"v{k}"
        chain.append(f"[{last}][{k + 1}:v]overlay=0:0:enable='{en}'[{out}]")
        last = out
    args += ["-filter_complex", ";".join(chain), "-map", f"[{last}]", "-map", "0:a?", "-c:v", "libx264", "-crf", "20",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "copy", str(dst)]
    subprocess.run(args, check=True)
