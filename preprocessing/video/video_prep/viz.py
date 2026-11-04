"""검수용 타임라인: 원본 CSV 노출 구간 vs 후보 카드 구간 막대 + 화면별 대표 프레임(카드 박스 표시)."""
import math

from PIL import Image, ImageDraw, ImageFont

WIDTH, PAD, LABEL_W, ROW_H = 1520, 20, 260, 24


def _font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default(size)


def draw_timeline(doc, frame_paths, out_path, cfg):
    colors = {k: tuple(v) for k, v in cfg["colors"].items()}
    color_of = lambda c: colors["risk_notice"] if c.get("type") == "risk_notice" else colors[c["kind"]]
    f_title, f_text = _font(cfg["font_path"], 22), _font(cfg["font_path"], 15)
    dur, cards, screens = doc["duration"], doc["cards"], doc["screens"]
    tw = cfg["thumb_width"]
    per_row = max(1, (WIDTH - PAD) // (tw + PAD))
    th = round(tw * doc["size"][1] / doc["size"][0])
    n_thumb_rows = math.ceil(len(screens) / per_row)
    bar_x0, bar_x1 = PAD + LABEL_W, WIDTH - PAD
    timeline_h = (len(cards) + 2) * ROW_H + 30
    height = 70 + timeline_h + n_thumb_rows * (th + 34) + PAD
    canvas = Image.new("RGB", (WIDTH, height), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    m = doc["metrics"]
    cov = "-" if m["csv_coverage"] is None else f"{m['csv_coverage']:.0%}"
    d.text((PAD, 12), f"{doc['ad_id']}  {doc['brand'] or ''}  ·  {dur:.1f}s  ·  {doc['size'][0]}x{doc['size'][1]}",
           fill=(0, 0, 0), font=f_title)
    line = (f"카드 {m['n_cards']} (static {m['n_static']} · crawl {m['n_crawl']}) · 화면 {len(screens)} · "
            f"후보 노출 {m['candidate_exposure_s']}s · CSV 노출 {m['csv_exposure_s']}s · CSV 구간 커버 {cov}")
    if "n_risk_cards" in m:
        cps = "-" if m["risk_cps_max"] is None else f"{m['risk_cps_max']}자/초"
        line += f"  ‖  위험고지 카드 {m['n_risk_cards']} · 노출 {m['risk_exposure_s']}s · 최대 {cps}"
    d.text((PAD, 42), line, fill=(80, 80, 80), font=f_text)

    x_of = lambda t: bar_x0 + (bar_x1 - bar_x0) * t / dur
    y = 70
    for s in range(0, int(dur) + 1, 5):                               # 5초 눈금
        d.line([(x_of(s), y), (x_of(s), y + timeline_h - 30)], fill=(225, 225, 225))
        d.text((x_of(s) - 6, y + timeline_h - 26), f"{s}", fill=(120, 120, 120), font=f_text)
    rows = [("CSV 노출 구간", [(a, b) for a, b in doc["csv"]["intervals"]], colors["csv"])]
    rows += [(f"c{c['id']} s{c['screen_id']} {c['kind']} {c.get('type') or ''} {c['n_lines']}줄",
              [(c["t_start"], c["t_end"])], color_of(c)) for c in cards]
    for label, spans, color in rows:
        d.text((PAD, y + 3), label, fill=(40, 40, 40), font=f_text)
        d.rectangle([bar_x0, y + 4, bar_x1, y + ROW_H - 4], fill=(245, 245, 245))
        for a, b in spans:
            d.rectangle([x_of(a), y + 4, max(x_of(a) + 2, x_of(b)), y + ROW_H - 4], fill=color)
        y += ROW_H

    y = 70 + timeline_h
    lw = max(2, round(doc["size"][0] / 400))
    for k, s in enumerate(screens):
        frame = Image.open(frame_paths[s["id"]]).convert("RGB")
        fd = ImageDraw.Draw(frame)
        for cid in s["card_ids"]:
            c = cards[cid]
            fd.rectangle(c["bbox"], outline=color_of(c), width=lw)
        thumb = frame.resize((tw, th), Image.Resampling.LANCZOS)
        x0 = PAD + (k % per_row) * (tw + PAD)
        y0 = y + (k // per_row) * (th + 34)
        canvas.paste(thumb, (x0, y0 + 26))
        d.text((x0, y0 + 4), f"s{s['id']}  {s['t_start']:.1f}~{s['t_end']:.1f}s  · 대표 {s['rep_t']:.1f}s · "
                             f"카드 {','.join(f'c{i}' for i in s['card_ids'])}", fill=(40, 40, 40), font=f_text)
    canvas.save(out_path)
