"""수정 렌더링. 원본 픽셀은 고쳐야 하는 블록 영역 밖에서는 건드리지 않는다.

L1 recolor   : 글자 모양(원본 폰트·안티앨리어싱)은 그대로, 글자 픽셀의 밝기만 옮김
L2 retypeset : 블록을 주변 배경으로 지우고, 같은 자리·같은 정렬·같은 색·가장 비슷한 굵기로 다시 조판
L3 extend    : L2로 자리가 모자라면 고지 아래 가장 평평한 배경 줄을 늘려(seam) 공간을 만듦
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .color import contrast, fit_ink, delta_e, luminance
from .measure import analyze_block, glyph_height, ink_mask, to_gray


# ---------- 공통 ----------
def erase(img, box, pad=3):
    """상자 안을 좌우·상하 바깥 픽셀의 선형 보간으로 채움 (그라데이션 배경도 자연스럽게)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, box[0] - pad), max(0, box[1] - pad), min(W, box[2] + pad), min(H, box[3] + pad)
    L = img[y0:y1, max(0, x0 - 3):max(1, x0)].astype(float).mean(1) if x0 > 0 else None
    R = img[y0:y1, x1:min(W, x1 + 3)].astype(float).mean(1) if x1 < W else None
    T = img[max(0, y0 - 3):max(1, y0), x0:x1].astype(float).mean(0) if y0 > 0 else None
    B = img[y1:min(H, y1 + 3), x0:x1].astype(float).mean(0) if y1 < H else None
    L = L if L is not None else R
    R = R if R is not None else L
    T = T if T is not None else B
    B = B if B is not None else T
    tx = np.linspace(0, 1, x1 - x0)[None, :, None]
    ty = np.linspace(0, 1, y1 - y0)[:, None, None]
    horiz = L[:, None] * (1 - tx) + R[:, None] * tx
    vert = T[None] * (1 - ty) + B[None] * ty
    img[y0:y1, x0:x1] = np.clip((horiz + vert) / 2, 0, 255).astype(np.uint8)


def recolor(img, box, fg, bg, new_fg):
    x0, y0, x1, y1 = [int(v) for v in box]
    reg = img[y0:y1, x0:x1].astype(float)
    fg, bg, new_fg = (np.array(v, float) for v in (fg, bg, new_fg))
    d = fg - bg
    a = np.clip(((reg - bg) @ d) / max(d @ d, 1e-6), 0, 1)[..., None]
    a[a < 0.08] = 0                     # 배경 픽셀은 그대로
    img[y0:y1, x0:x1] = np.clip(reg + a * (new_fg - fg), 0, 255).astype(np.uint8)


# ---------- 폰트 ----------
class Fonts:
    def __init__(self, cfg):
        d = Path(cfg["dir"]).expanduser()
        self.paths = [str(d / f) for f in cfg["weights"] if (d / f).exists()] or [cfg["fallback"]]
        self._ratio = {}

    def ink_ratio(self, path):
        """폰트 크기 1px당 측정 글자 높이 (재측정과 같은 _glyph_height 식으로 보정)."""
        if path not in self._ratio:
            f = ImageFont.truetype(path, 200)
            im = Image.new("L", (2400, 320), 255)
            ImageDraw.Draw(im).text((10, 40), "※투자원금 손실 가능 설명서 필독", font=f, fill=0)
            ink = np.asarray(im) < 128
            self._ratio[path] = glyph_height(ink.sum(1).astype(float)) / 200
        return self._ratio[path]

    def font_for(self, path, ink_h):
        return ImageFont.truetype(path, max(6, int(round(ink_h / self.ink_ratio(path)))))

    def closest_weight(self, img, block):
        """원본 블록의 획 밀도(글자 픽셀/글자 높이²)에 가장 가까운 굵기."""
        x0, y0, x1, y1 = block["bbox"]
        ink = ink_mask(to_gray(img[y0:y1, x0:x1]))
        h = block.get("h_px") or (y1 - y0)
        target = ink.sum() / max(h * h, 1) / max(len(block["text"]), 1)
        best, err = self.paths[0], 1e9
        for p in self.paths:
            f = self.font_for(p, h)
            w = int(f.getlength(block["text"])) + 20
            im = Image.new("L", (w, int(h * 3)), 255)
            ImageDraw.Draw(im).text((5, int(h * 0.6)), block["text"], font=f, fill=0)
            dens = (np.asarray(im) < 128).sum() / max(h * h, 1) / max(len(block["text"]), 1)
            if abs(dens - target) < err:
                best, err = p, abs(dens - target)
        return best


# ---------- 조판 ----------
def lines_of(blocks):
    """같은 줄의 블록을 묶어 원본 줄바꿈을 그대로 보존."""
    rows = []
    for b in sorted(blocks, key=lambda b: (b["bbox"][1] + b["bbox"][3]) / 2):
        cy = (b["bbox"][1] + b["bbox"][3]) / 2
        if rows and abs(cy - rows[-1]["cy"]) < 0.5 * (b["bbox"][3] - b["bbox"][1]):
            rows[-1]["bs"].append(b)
        else:
            rows.append({"cy": cy, "bs": [b]})
    out = []
    for r in rows:
        bs = sorted(r["bs"], key=lambda b: b["bbox"][0])
        out.append({"text": " ".join(b["text"] for b in bs),
                    "bbox": [min(b["bbox"][0] for b in bs), min(b["bbox"][1] for b in bs),
                             max(b["bbox"][2] for b in bs), max(b["bbox"][3] for b in bs)]})
    return out


def alignment(lines, W):
    if len(lines) >= 2:
        lefts = np.std([l["bbox"][0] for l in lines])
        cents = np.std([(l["bbox"][0] + l["bbox"][2]) / 2 for l in lines])
        if cents + 2 < lefts:
            return "center"
        return "left"
    b = lines[0]["bbox"]
    return "center" if abs((b[0] + b[2]) / 2 - W / 2) < 0.04 * W else "left"


def wrap(text, font, width):
    words, out, cur = text.split(" "), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if font.getlength(t) <= width or not cur:
            cur = t
        else:
            out.append(cur)
            cur = w
    return out + ([cur] if cur else [])


def seam_score(img, y, k=4):
    """y줄을 늘렸을 때 티가 나는 정도 = 위아래 k줄과의 평균 색 차이 (옆 단 글자가 걸리면 커짐)."""
    a = img[max(0, y - k):y + k + 1].astype(float)
    return float(np.abs(a - img[y].astype(float)).mean())


def flattest_row(img, y_from, y_to):
    y_from, y_to = int(max(4, y_from)), int(min(y_to, img.shape[0] - 5))
    if y_from >= y_to:
        return min(max(4, y_from), img.shape[0] - 5)
    return min(range(y_from, y_to), key=lambda y: seam_score(img, y))


def insert_rows(img, y, n):
    """y번째 줄(평평한 배경)을 n번 반복해 끼워 넣음 — 아래 내용은 그대로 아래로 밀림."""
    return np.concatenate([img[:y], np.repeat(img[y:y + 1], n, axis=0), img[y:]], axis=0)


def retypeset(img, target, blocks, obstacles, fonts, th, ladder, mcfg, extra=(), mode="keep"):
    """extra: ① 내용 수정으로 문단 끝에 덧붙일 문장들 (같은 폰트·색·정렬로)."""
    """한 문단을 다시 조판. 반환: (새 이미지, 그린 줄 bbox들, 정보)."""
    H, W = img.shape[:2]
    lines = lines_of(blocks)
    big = max(blocks, key=lambda b: len(b["text"]))
    probe = analyze_block(img, big["bbox"], mcfg)
    fg, bg = probe["fg_rgb"] or [40, 40, 40], probe["bg_rgb"] or [255, 255, 255]
    weight = fonts.closest_weight(img, big)
    align = alignment(lines, W)

    x0 = min(l["bbox"][0] for l in lines)
    x1 = max(l["bbox"][2] for l in lines)
    y0 = min(l["bbox"][1] for l in lines)
    y1 = max(l["bbox"][3] for l in lines)
    # 가로 범위: 원래 단(column)에서 시작해, 같은 높이의 다른 요소(로고 등)나 원래 여백에 닿을 때까지만 넓힘
    margin = max(8, min(x0, W - x1))
    same_row = [o for o in obstacles if o[1] < y1 and o[3] > y0]
    left_wall = max([o[2] for o in same_row if o[2] <= x0 + 2] + [margin - 1]) + 1
    right_wall = min([o[0] for o in same_row if o[0] >= x1 - 2] + [W - margin + 1]) - 1
    gap = 0.8 * target["h_px"]
    if align == "center":
        half = min((x0 + x1) / 2 - left_wall, right_wall - (x0 + x1) / 2) - gap
        avail_l, avail_r = (x0 + x1) / 2 - max(half, (x1 - x0) / 2), (x0 + x1) / 2 + max(half, (x1 - x0) / 2)
    else:
        avail_l, avail_r = x0, max(x1, right_wall - gap)
    h0 = target["h_px"]
    tops = [l["bbox"][1] for l in lines]
    pitch_ratio = float(np.median(np.diff(tops))) / h0 if len(lines) > 1 else 1.55
    pitch_ratio = float(np.clip(pitch_ratio, 1.3, 2.2))

    # 문단 전체를 한 번에 지움 (블록마다 지우면 이웃 글자가 보간에 섞여 줄무늬가 남음)
    # 문구 추가만 하고 크기는 그대로면 원본은 건드리지 않고 아래에 이어 씀 (글머리표·표 등 원본 그래픽 보존)
    append_only = bool(extra) and target["target_h_px"] <= target["h_px"] * 1.02
    if not append_only:
        erase(img, [x0, y0, x1, y1], pad=max(3, int(0.35 * h0)))

    new_h = target["target_h_px"]
    font = fonts.font_for(weight, new_h)
    text_lines = [t for l in lines for t in wrap(l["text"], font, avail_r - avail_l)]
    if append_only:
        text_lines = []
    n_orig = len(text_lines)
    text_lines += [t for e in extra for t in wrap(e, font, avail_r - avail_l)]
    pitch = new_h * pitch_ratio
    top = (lines[-1]["bbox"][1] + pitch) if append_only else y0
    block_h = pitch * (len(text_lines) - 1) + new_h * 1.6
    # 자리 확인: 아래쪽 첫 장애물(다른 텍스트 블록) 또는 원본 하단 여백
    below = [o[1] for o in obstacles if o[1] >= y1 - 2 and o[0] < avail_r and o[2] > avail_l]
    limit = min(below) - 0.6 * h0 if below else H - max(0.0, min(H - y1, 0.04 * H))
    need = top + block_h - limit
    grow = 0
    info = {"font": Path(weight).stem, "align": align, "lines": len(text_lines), "orig_lines": len(lines),
            "size_factor": round(new_h / h0, 2), "level": "L2"}
    if new_h / h0 > ladder["max_size_factor"]:
        info["level"] = "L4"   # 원래 디자인이 감당할 수 없는 확대 — 자동안은 참고용, 디자이너 검토
    if need > 0 and mode in ("tight", "tight+"):
        # A안: 캔버스·주변은 그대로, 줄 간격만 원본 범위 아래로 좁혀 고지 영역 안에서 해결
        for pr in np.arange(pitch_ratio - 0.05, 1.249, -0.05):
            pitch = new_h * pr
            top = (lines[-1]["bbox"][1] + pitch) if append_only else y0
            block_h = pitch * (len(text_lines) - 1) + new_h * 1.6
            need = top + block_h - limit
            if need <= 0:
                break
        info["pitch_ratio"] = (round(pitch_ratio, 2), round(pitch / new_h, 2))
        if need > 0:
            info["short_px"] = int(np.ceil(need))     # 이만큼 자리가 모자람 → A안만으로는 불가
    if need > 0 and mode in ("keep", "tight+"):         # 모자란 만큼만 일단 늘림 (뒤에서 B·C로 규격 복원)
        info.pop("short_px", None)
        grow = int(np.ceil(need + 0.3 * h0))
        # 고지 자신의 배경(지운 자리) 안에서 가장 평평한 줄을 늘림 → 고지 띠의 색·질감이 그대로 연장됨
        seam = flattest_row(img, y1 + 1, y1 + pitch) if append_only else flattest_row(img, y0, y1 + 1)
        info["seam_score"] = round(seam_score(img, seam), 2)
        if info["seam_score"] > ladder["seam_max_diff"]:
            # 줄 전체가 평평하지 않음(옆에 다른 단·사진) → 늘리면 옆 내용이 늘어남. 자동안은 참고용
            info["level"] = "L4"
            info["note"] = "옆 단/사진 때문에 깔끔하게 늘릴 줄이 없음"
        img = insert_rows(img, seam, grow)
        info.update(level="L4" if info["level"] == "L4" else "L3", seam_y=seam, grow_px=grow)

    # 줄마다 그 아래 배경을 보고 색을 정함: 원본 글자색 그대로 → 모자라면 밝기만 이동 → 그래도 안 되면 은은한 스크림
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    asc = font.getbbox("가")[1]
    target_c = th["contrast"] * ladder["contrast_margin"]
    placed, inks, scrims = [], [], 0
    for i, t in enumerate(text_lines):
        if not t.strip():
            continue
        y = top + i * pitch - asc
        tw = font.getlength(t)
        x = (avail_l + avail_r - tw) / 2 if align == "center" else avail_l
        bb = [int(v) for v in d.textbbox((x, y), t, font=font)]
        placed.append((x, y, t, bb, i >= n_orig))
    arr = np.asarray(pil).astype(float)
    for x, y, t, bb, _ in placed:
        reg = arr[max(0, bb[1]):bb[3], max(0, bb[0]):bb[2]].reshape(-1, 3)
        if len(reg) == 0:            # 캔버스 밖으로 넘친 줄 (A안 공간 부족)
            inks.append(fg)
            continue
        lum = reg @ [0.299, 0.587, 0.114]
        lo, hi = reg[lum <= np.percentile(lum, 10)].mean(0), reg[lum >= np.percentile(lum, 90)].mean(0)
        ink = fit_ink(fg, lo if luminance(fg) > luminance(lo) else hi, target_c)
        if min(contrast(ink, lo), contrast(ink, hi)) < th["contrast"]:
            # 배경이 얼룩져 색만으로는 안 됨 → 글자 뒤에 원래 배경 톤의 반투명 띠 (가장자리 부드럽게)
            base = lo if luminance(ink) > luminance(lo) else hi
            pad = int(0.35 * new_h)
            x0s, y0s = max(0, bb[0] - pad), max(0, bb[1] - pad)
            x1s, y1s = min(arr.shape[1], bb[2] + pad), min(arr.shape[0], bb[3] + pad)
            yy = np.minimum(np.arange(y1s - y0s), np.arange(y1s - y0s)[::-1])[:, None]
            xx = np.minimum(np.arange(x1s - x0s), np.arange(x1s - x0s)[::-1])[None, :]
            a = 0.7 * np.clip(np.minimum(yy, xx) / max(pad, 1), 0, 1)[..., None]
            arr[y0s:y1s, x0s:x1s] = arr[y0s:y1s, x0s:x1s] * (1 - a) + np.array(base) * a
            ink = fit_ink(fg, base, target_c)
            scrims += 1
        inks.append(ink)
    pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(pil)
    drawn = []
    added = []
    for (x, y, t, bb, is_new), ink in zip(placed, inks):
        d.text((x, y), t, font=font, fill=tuple(int(v) for v in ink))
        drawn.append([bb[0] - 2, bb[1] - 3, bb[2] + 2, bb[3] + 3])
        added.append(is_new)
    worst = max(inks, key=lambda c: delta_e(fg, c)) if inks else fg
    info.update(fg_before=[int(v) for v in fg], fg_after=[int(v) for v in worst],
                ink_delta_e=round(delta_e(fg, worst), 1), scrim_lines=scrims, added_flags=added,
                added_lines=len(text_lines) - n_orig)
    return np.asarray(pil).copy(), drawn, info, (grow, info.get("seam_y"))


def new_band(img, texts, fonts, th, ladder, h_px, margin):
    """고지 문단이 아예 없을 때: 광고 맨 아래 배경색을 그대로 이어 붙인 띠에 문구를 넣음."""
    H, W = img.shape[:2]
    bg = np.median(img[-max(3, H // 100):].reshape(-1, 3), axis=0).astype(int).tolist()
    font = fonts.font_for(fonts.paths[min(1, len(fonts.paths) - 1)], h_px)
    lines = [t for e in texts for t in wrap(e, font, W - 2 * margin)]
    pitch = h_px * 1.6
    band = int(pitch * len(lines) + h_px * 1.6)
    img = insert_rows(img, H - 1, band)
    ink = fit_ink([60, 60, 60] if luminance(bg) > 0.4 else [235, 235, 235], bg, th["contrast"] * ladder["contrast_margin"])
    pil = Image.fromarray(img)
    d = ImageDraw.Draw(pil)
    asc = font.getbbox("가")[1]
    drawn = []
    for i, t in enumerate(lines):
        x, y = margin, H + h_px * 0.6 + i * pitch - asc
        d.text((x, y), t, font=font, fill=tuple(ink))
        bb = d.textbbox((x, y), t, font=font)
        drawn.append([int(bb[0]) - 2, int(bb[1]) - 3, int(bb[2]) + 2, int(bb[3]) + 3])
    return np.asarray(pil).copy(), drawn, {"level": "L3", "grow_px": band, "font": Path(fonts.paths[0]).stem,
                                           "added_lines": len(lines), "new_band": True}


# ---------- 규격 복원: 늘린 만큼 다른 곳에서 확보해 캔버스 크기를 원본과 같게 ----------
def protected_rows(H, boxes, pad):
    keep = np.zeros(H, bool)
    for b in boxes:
        keep[max(0, int(b[1]) - pad):min(H, int(b[3]) + pad)] = True
    return keep


def redistribute(img, n, protect, max_score, partial=False):
    """B안: 글자·인물이 없는 평평한 배경 줄(위아래와 거의 같은 줄) n개를 빼서 원래 높이로 되돌림.
    반환: (이미지, 뺀 줄 index 목록) 또는 (None, 부족한 줄 수)."""
    H = img.shape[0]
    a = img.astype(np.int16)
    score = np.full(H, 1e9)
    d = np.abs(a[1:] - a[:-1]).mean(axis=(1, 2))      # 바로 윗줄과의 차이
    score[1:-1] = np.maximum(d[:-1], d[1:])
    score[protect] = 1e9
    ok = np.where(score <= max_score)[0]
    if len(ok) < n and not partial:
        return None, n - len(ok)
    # 같은 곳만 크게 줄지 않도록: 평평한 구간마다 최대 절반까지만, 점수 낮은 순
    rows = []
    runs = np.split(ok, np.where(np.diff(ok) > 1)[0] + 1)
    quota = {int(r[0]): max(1, int(len(r) * 0.7)) for r in runs}
    start = {int(y): int(r[0]) for r in runs for y in r}
    used = {}
    for y in sorted(ok, key=lambda y: score[y]):
        k = start[int(y)]
        if used.get(k, 0) < quota[k]:
            rows.append(int(y)); used[k] = used.get(k, 0) + 1
            if len(rows) == n:
                break
    if len(rows) < n and not partial:
        return None, n - len(rows)
    return np.delete(img, rows, axis=0), sorted(rows)


def scale_visual(img, n, y_top, min_scale, xl=0, xr=None):
    """C안: 고지 위쪽 전체(비주얼·카피)를 같은 비율로 조금 줄여 n줄을 확보. 좌우 빈 곳은 가장자리 색으로 채움.
    반환: (이미지, 배율) 또는 (None, 필요한 배율)."""
    H, W = img.shape[:2]
    y_top = int(y_top)
    s = (y_top - n) / max(y_top, 1)
    if s < min_scale or y_top <= n:
        return None, round(s, 3)
    xr = W if xr is None else int(xr)
    xl = int(xl)
    nh = y_top - n
    out = np.empty((nh, W, 3), np.uint8)
    # 바깥 여백·테두리(xl 왼쪽, xr 오른쪽)는 세로로만 눌러 선이 끊기지 않게
    if xl > 0:
        out[:, :xl] = np.asarray(Image.fromarray(img[:y_top, :xl]).resize((xl, nh), Image.LANCZOS))
    if xr < W:
        out[:, xr:] = np.asarray(Image.fromarray(img[:y_top, xr:]).resize((W - xr, nh), Image.LANCZOS))
    # 안쪽 내용(카피·비주얼)은 가로세로 같은 비율로
    iw = xr - xl
    nw = int(round(iw * s))
    small = np.asarray(Image.fromarray(img[:y_top, xl:xr]).resize((nw, nh), Image.LANCZOS))
    lpad = xl + (iw - nw) // 2
    out[:, lpad:lpad + nw] = small
    out[:, xl:lpad] = small[:, :6].mean(axis=1, keepdims=True).astype(np.uint8)
    out[:, lpad + nw:xr] = small[:, -6:].mean(axis=1, keepdims=True).astype(np.uint8)
    return np.concatenate([out, img[y_top:]], axis=0), round(s, 3)
