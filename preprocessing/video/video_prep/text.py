"""카드 글자 읽기, 위험고지 분류, 측정

- static: 화면 대표 프레임 1장을 이미지 광고처럼 처리 (3패스 OCR → 문단 → 분류 → 잉크 높이·대비) 후 줄을 카드에 배정
- crawl : 흐르는 띠를 프레임마다 이동량만큼 이어 붙인 긴 띠를 겹쳐 잘라 OCR하고, 겹쳐 읽힌 글자를 합친다
"""
import difflib
import hashlib
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from image_prep.classify import classify_blocks, split_graphic_text, strip_text
from image_prep.geometry import box_area, quad_height
from image_prep.layout import build_text_paragraphs, group_paragraphs, reading_order
from image_prep.metrics import add_block_metrics, analyze_block
from image_prep.ocr.runner import HAS_TEXT, _as_dict, merge_candidates, run_multi_pass
from image_prep.ocr_input import build_ocr_inputs, compute_scale
from image_prep.package import slim_block, slim_paragraph
from .cards import edge_map, profile_shift
from .frames import iter_frames

REVIEW_NO = re.compile(r"(?:심의필|심사필)\s*(?:제\s*)?[0-9A-Za-z][0-9A-Za-z\-]*\s*호")


class OcrCache:
    """OCR 결과 캐시 (ocr_raw/{ad_id}.json). 이번 실행에서 쓴 항목만 다시 저장한다."""

    def __init__(self, path, ocr_cfg):
        self.path = Path(path)
        self.salt = json.dumps(ocr_cfg, sort_keys=True)
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.old, self.used, self.hits = old, {}, 0

    def get(self, parts, compute):
        key = hashlib.sha256(json.dumps([self.salt, parts], sort_keys=True).encode()).hexdigest()
        if key in self.old:
            self.hits += 1
            self.used[key] = self.old[key]
        else:
            self.used[key] = compute()
        return self.used[key]

    def save(self):
        self.path.write_text(json.dumps(self.used, ensure_ascii=False), encoding="utf-8")


def ocr_image(engine, img, icfg):
    img_np = np.asarray(img)
    up, meta = build_ocr_inputs(img, "unused", icfg["ocr_input"], save=False)
    return run_multi_pass(engine, img, img_np, up, meta, icfg["ocr_input"], icfg["ocr"])[0]


def card_bands(cards, W, H, margin_ratio):
    spans = sorted((max(0, int(c["bbox"][1] - margin_ratio * c["line_h_px"])),
                    min(H, int(np.ceil(c["bbox"][3] + margin_ratio * c["line_h_px"])))) for c in cards)
    out = []
    for y0, y1 in spans:
        if out and y0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], y1)
        else:
            out.append([y0, y1])
    return [[0, y0, W, y1] for y0, y1 in out]


def ocr_screen(engine, img, bands, icfg):
    ocr_cfg, in_cfg = icfg["ocr"], icfg["ocr_input"]
    cands = []
    s = ocr_cfg["large_text_width"] / img.width
    if s < 0.8:
        small = img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)
        sx, sy = small.width / img.width, small.height / img.height
        for b in engine.recognize(np.asarray(small)):
            b.quad = [[x / sx, y / sy] for x, y in b.quad]
            cands.append(_as_dict(b, "small"))
    else:                                                       # SD 영상은 원본 그대로 한 번
        cands += [_as_dict(b, "full") for b in engine.recognize(np.asarray(img))]
    scale, edge = compute_scale(img.width, in_cfg), in_cfg["edge_margin"]
    for x0, y0, x1, y1 in bands:
        band = img.crop((x0, y0, x1, y1))
        up = band.resize((round(band.width * scale), round(band.height * scale)), Image.Resampling.LANCZOS)
        for b in engine.recognize(np.asarray(up)):
            ys = [p[1] for p in b.quad]
            if (y0 > 0 and min(ys) <= edge) or (y1 < img.height and max(ys) >= up.height - edge):
                continue                                        # 띠 경계에 걸려 잘린 이웃 글자
            b.quad = [[x / scale + x0, y / scale + y0] for x, y in b.quad]
            cands.append(_as_dict(b, "upscaled"))
    kept = [c for c in cands if c["confidence"] >= ocr_cfg["drop_below_conf"]
            and quad_height(c["quad"]) >= ocr_cfg["min_text_h_px"] and HAS_TEXT.search(c["text"])]
    return merge_candidates(kept, ocr_cfg)


def read_frame(raw_blocks, img_np, icfg, rules, profiles):
    blocks = [dict(b) for b in raw_blocks]
    for b in blocks:
        b["h_px"] = quad_height(b["quad"])
    blocks, _ = split_graphic_text(blocks, icfg["graphic_filter"])
    blocks = reading_order(blocks)
    for i, b in enumerate(blocks):
        b["id"] = i
    for b, pid in zip(blocks, group_paragraphs(blocks, icfg["layout"])):
        b["paragraph_id"] = pid
    add_block_metrics(blocks, img_np, profiles, icfg["contrast"])
    paras = build_text_paragraphs(blocks, icfg["text_paragraph"])
    for k, idx in enumerate(paras):
        for i in idx:
            blocks[i]["text_para_id"] = k
    classify_blocks(blocks, rules, icfg["classify"])
    return blocks, paras


def _inside(b, box):
    a = b["bbox"]
    ix = max(0.0, min(a[2], box[2]) - max(a[0], box[0]))
    iy = max(0.0, min(a[3], box[3]) - max(a[1], box[1]))
    return ix * iy / box_area(a) if box_area(a) > 0 else 0.0


def assign_blocks(blocks, cards, cfg):
    for b in blocks:
        best, best_v = None, cfg["card_overlap"]
        for c in cards:
            m = cfg["card_margin_ratio"] * c["line_h_px"]
            v = _inside(b, [c["bbox"][0] - m, c["bbox"][1] - m, c["bbox"][2] + m, c["bbox"][3] + m])
            if v >= best_v:
                best, best_v = c["id"], v
        b["card_id"] = best


def build_crawl_strip(video, size, card, cfg):
    W, H = size
    m = cfg["crawl_margin_ratio"] * card["line_h_px"]
    x0, y0, x1, y1 = card["bbox"]
    crop = [0, max(0, int(y0 - m)) // 2 * 2, W, min(H, int(np.ceil(y1 + m))) // 2 * 2]
    fps = cfg["crawl_fps"]
    max_shift = min(W // 2, max(8, int(abs(card["crawl_px_s"]) / fps * 1.5)))
    strip, prev = None, None
    for _, f in iter_frames(video, size, fps, crop=crop, start=card["t_start"], end=card["t_end"]):
        g = f.mean(axis=2)
        if strip is None:
            strip = f.copy()
        else:
            v, corr = profile_shift(edge_map(prev).sum(axis=0), edge_map(g).sum(axis=0), max_shift)
            if v > 0 and corr >= cfg["crawl_min_corr"]:
                strip = np.concatenate([strip, f[:, W - v:]], axis=1)
        prev = g
    return strip


def merge_overlap(acc, nxt, min_chars, window=80):
    if not acc:
        return nxt
    a, b = acc[-window:], nxt[:window]
    m = difflib.SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    if m.size >= min_chars:
        return acc[:len(acc) - len(a) + m.a] + nxt[m.b:]
    return f"{acc} {nxt}"


def read_crawl(engine, strip, icfg, cfg, cache, cache_parts):
    width, step = cfg["crawl_chunk_width"], cfg["crawl_chunk_width"] - cfg["crawl_chunk_overlap"]
    xs = list(range(0, max(1, strip.shape[1] - cfg["crawl_chunk_overlap"]), step))
    text, blocks = "", []
    for x in xs:
        piece = np.ascontiguousarray(strip[:, x:x + width])
        raw = cache.get(cache_parts + [x], lambda: ocr_image(engine, Image.fromarray(piece), icfg))
        raw = sorted(raw, key=lambda b: b["bbox"][0])
        text = merge_overlap(text, " ".join(b["text"].strip() for b in raw), cfg["merge_min_chars"])
        for b in raw:
            m = analyze_block(piece, b["bbox"], icfg["contrast"])
            h = m["ink_h_px"] or quad_height(b["quad"])
            blocks.append({"text": b["text"], "confidence": b["confidence"], "text_h": float(h),
                           "contrast_ratio": m["contrast_ratio"], "contrast_conf": m["contrast_conf"],
                           "bbox": [b["bbox"][0] + x, b["bbox"][1], b["bbox"][2] + x, b["bbox"][3]]})
    return text.strip(), blocks


def _stats(vals):
    vals = [v for v in vals if v is not None]
    return {"min": round(float(min(vals)), 1), "median": round(float(np.median(vals)), 1)} if vals else None


def card_measures(blocks, frame_w, profiles, duration, headline_h, min_size_chars):
    risk = [b for b in blocks if b["type"] == "risk_notice"]
    chars = sum(len(strip_text(b["text"])) for b in risk)
    sized = [b for b in risk if len(strip_text(b["text"])) >= min_size_chars] or risk
    h = _stats([b["text_h"] for b in sized])
    cr = [b["contrast_ratio"] for b in risk if b["contrast_conf"] == "high" and b["contrast_ratio"] is not None]
    return {"risk_chars": chars,
            "risk_cps": round(chars / duration, 1) if chars and duration > 0 else None,
            "risk_text_h_px": h,
            "risk_display_pt": {p: {k: round(v * pt / frame_w, 2) for k, v in h.items()} for p, pt in profiles.items()}
            if h else None,
            "risk_contrast_min": round(min(cr), 2) if cr else None,
            "risk_headline_ratio": round(h["median"] / headline_h, 3) if h and headline_h else None}


def _card_type(blocks):
    weight = {}
    for b in blocks:
        weight[b["type"]] = weight.get(b["type"], 0) + len(strip_text(b["text"]))
    return max(weight, key=weight.get) if weight else None


def read_cards(src, doc_cards, screens, frame_paths, engine, icfg, rules, profiles, cfg, cache):
    W = src.info["width"]
    salt = [src.sha256]
    by_id = {c["id"]: c for c in doc_cards}
    all_blocks, all_paras = [], []
    for s in screens:
        img = Image.open(frame_paths[s["id"]]).convert("RGB")
        cards = [by_id[i] for i in s["card_ids"]]
        bands = card_bands(cards, img.width, img.height, cfg["band_margin_ratio"])
        raw = cache.get(salt + ["screen", s["rep_t"], bands], lambda: ocr_screen(engine, img, bands, icfg))
        blocks, paras = read_frame(raw, np.asarray(img), icfg, rules, profiles)
        assign_blocks(blocks, [c for c in cards if c["kind"] == "static"], cfg)
        heads = [b["text_h"] for b in blocks if b["size_tier"] == "headline" and b["confidence"] >= icfg["classify"]["low_conf"]]
        s["headline_h_px"] = round(max(heads), 1) if heads else None
        all_blocks += [{"screen_id": s["id"], **slim_block(b), "card_id": b["card_id"]} for b in blocks]
        all_paras += [{"screen_id": s["id"], **slim_paragraph(k, idx, blocks)} for k, idx in enumerate(paras)]
        for c in cards:
            if c["kind"] != "static":
                continue
            cb = [b for b in blocks if b["card_id"] == c["id"]]
            text = " ".join(b["text"].strip() for b in cb) or None
            ctype = _card_type(cb)
            if ctype == "risk_notice" and rules.match(strip_text(text or ""))[0] != "risk_notice":
                ctype = "unknown"
                c["type_rule"] = "no_risk_keyword"
            c.update({"type": ctype, "text": text,
                      "block_ids": [b["id"] for b in cb],
                      **card_measures(cb, W, profiles, c["duration"], s["headline_h_px"], cfg["min_size_chars"])})

    for c in doc_cards:
        if c["kind"] != "crawl":
            continue
        strip = build_crawl_strip(src.path, (W, src.info["height"]), c, cfg)
        text, blocks = read_crawl(engine, strip, icfg, cfg, cache,
                                  salt + ["crawl", c["t_start"], c["t_end"], c["bbox"], cfg["crawl_fps"]])
        t, _ = rules.match(strip_text(text)) if text else (None, None)
        for b in blocks:
            b["type"] = t or "unknown"
        headline_h = screens[c["screen_id"]].get("headline_h_px")
        c.update({"type": t or ("unknown" if text else None), "text": text or None, "strip_w_px": int(strip.shape[1]),
                  **card_measures(blocks, W, profiles, c["duration"], headline_h, cfg["min_size_chars"])})
        c["_strip"] = strip
    return all_blocks, all_paras


def review_numbers(doc_cards, paragraphs):
    """심의필·심사필 번호 (중복 제거, 등장 순). 번호가 줄바꿈으로 끊길 수 있어 문단 텍스트에서 찾는다."""
    texts = [p["text"] for p in paragraphs]
    texts += [c["text"] for c in doc_cards if c["kind"] == "crawl" and c.get("text")]
    found = []
    for t in texts:
        for m in REVIEW_NO.finditer(t):
            no = re.sub(r"\s+", "", m.group(0))                
            if no not in found:
                found.append(no)
    return found
