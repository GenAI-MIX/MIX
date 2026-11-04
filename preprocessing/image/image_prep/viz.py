"""검수용 시각화: OCR 박스를 type별 색으로 칠하고 블록 번호·위험고지 crop 범위·핵심 지표를 표시."""
from PIL import Image, ImageDraw, ImageFont

TYPE_ORDER = ["risk_notice", "headline", "return_claim", "other", "unknown"]


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def _dashed_rect(d, box, color, width, dash):
    x0, y0, x1, y1 = box
    for a, b, horizontal, fixed in [(x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)]:
        for s in range(int(a), int(b), dash * 2):
            e = min(s + dash, b)
            d.line([(s, fixed), (e, fixed)] if horizontal else [(fixed, s), (fixed, e)], fill=color, width=width)


def _fmt(v, nd=1, suffix=""):
    return "-" if v is None else f"{v:.{nd}f}{suffix}"


def draw_viz(img, ad_id, blocks, graphic, crops, metrics, ocr_info, out_path, cfg):
    colors = {k: tuple(v) for k, v in cfg["colors"].items()}
    w, h = img.size
    lw = max(2, round(w / 700))
    fs = max(14, round(w / 110))
    font, font_big = _font(cfg["font_path"], fs), _font(cfg["font_path"], round(fs * 1.3))

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for b in blocks:
        c = colors[b["type"]]
        d.polygon([tuple(p) for p in b["quad"]], fill=c + (45,), outline=c + (255,), width=lw)
    for b in graphic:                                    # 그래픽 글자로 제외된 블록 (JSON에는 없음)
        _dashed_rect(d, b["bbox"], colors["graphic"] + (255,), lw, lw * 2)
    for c in crops:
        _dashed_rect(d, c["bbox"], colors["crop"] + (255,), lw * 2, lw * 6)
    canvas = Image.alpha_composite(base, overlay)

    d = ImageDraw.Draw(canvas)
    for b in blocks:
        label = str(b["id"])
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        x0, y0 = b["bbox"][0], b["bbox"][1]
        lx = x0 - tw - 4 if x0 - tw - 4 >= 0 else x0 + 2
        ly = max(0, min(h - th, y0))
        d.rectangle([lx - 2, ly, lx + tw + 2, ly + th + 2], fill=colors[b["type"]] + (255,))
        d.text((lx, ly), label, fill=(255, 255, 255, 255), font=font)

    # 상단 정보 패널
    line_h = round(fs * 1.7)
    profiles = list(metrics["display_profiles"])
    panel_h = line_h * (3 + len(profiles)) + fs
    out = Image.new("RGB", (w, h + panel_h), (255, 255, 255))
    out.paste(canvas.convert("RGB"), (0, panel_h))
    pd = ImageDraw.Draw(out)
    y = fs // 2
    pd.text((fs, y), f"{ad_id}   {w}×{h}px   블록 {len(blocks)}개   OCR 업스케일 ×{ocr_info['scale']:.2f}, "
                     f"띠 {ocr_info['n_strips']}개", fill=(0, 0, 0), font=font_big)
    y += line_h
    x = fs
    extra = {"graphic": ("그래픽 제외", len(graphic)), "crop": ("위험고지 crop", len(crops))}
    for t in TYPE_ORDER + ["graphic", "crop"]:
        name, n = extra.get(t, (t, metrics["type_counts"].get(t, 0)))
        pd.rectangle([x, y + fs * 0.15, x + fs, y + fs * 1.15], fill=colors[t])
        label = f"{name} {n}"
        pd.text((x + fs * 1.4, y), label, fill=(0, 0, 0), font=font)
        x += pd.textbbox((0, 0), label, font=font)[2] + fs * 3
    y += line_h
    rn, hd, ratio = metrics["risk_notice"], metrics["headline"], metrics["risk_to_headline_ratio"]
    cr = (rn or {}).get("contrast_ratio") or {}
    pd.text((fs, y), f"위험고지/헤드라인 높이비: 중앙 {_fmt((ratio or {}).get('median'), 3)} · "
                     f"최소 {_fmt((ratio or {}).get('min'), 3)}    위험고지 명암비: 최소 {_fmt(cr.get('min'), 2, ':1')} · "
                     f"중앙 {_fmt(cr.get('median'), 2, ':1')}", fill=(0, 0, 0), font=font)
    for p in profiles:
        y += line_h
        pt = metrics["display_profiles"][p]["display_width_pt"]
        dp = ((rn or {}).get("display_pt") or {}).get(p) or {}
        hp = ((hd or {}).get("display_pt_max") or {}).get(p)
        pd.text((fs, y), f"[{p} · 폭 {pt}pt] 위험고지 표시 높이: 최소 {_fmt(dp.get('min'), 2, 'pt')} · "
                         f"중앙 {_fmt(dp.get('median'), 2, 'pt')}    헤드라인 최대 {_fmt(hp, 1, 'pt')}",
                fill=(0, 0, 0), font=font)
    out.save(out_path)
