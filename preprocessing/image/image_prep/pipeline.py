"""광고 1건 처리 흐름: 정리 → OCR 사본 → OCR(캐시) → 분류 → 측정 → crop·JSON → 시각화."""
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .classify import Rules, classify_blocks, split_graphic_text
from .geometry import quad_height
from .layout import build_text_paragraphs, group_paragraphs, reading_order
from .metrics import add_block_metrics, ad_metrics
from .normalize import content_bbox, needs_normalized_copy, normalize_image
from .ocr import build_engine
from .ocr.runner import run_multi_pass
from .ocr_input import build_ocr_inputs
from .package import build_ad_json, dumps_ad_json, make_risk_crops
from .viz import draw_viz

META_COLS = ["ad_id", "source_file", "sha256", "dup_of", "near_dup_of", "media", "product", "brand", "title", "note"]


def sync_meta_csv(path, sources, media_types, raw_dir):
    path = Path(path)
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            existing = {r["ad_id"]: r for r in csv.DictReader(f)}
    rows, seen = [], set()
    for s in sources:
        old = existing.get(s.ad_id, {})
        row = {"ad_id": s.ad_id, "source_file": str(s.path.relative_to(raw_dir)), "sha256": s.sha256,
               "dup_of": s.dup_of or "", "near_dup_of": s.near_dup_of or "", "note": old.get("note", "")}
        for k in ("media", "product", "brand", "title"):
            row[k] = old.get(k) or s.meta.get(k, "")
        rows.append(row)
        seen.add(s.ad_id)
    rows += [r for k, r in existing.items() if k not in seen]    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=META_COLS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    for r in rows:
        if r["media"] and r["media"] not in media_types:
            print(f"  [경고] {r['ad_id']}: media '{r['media']}'가 config media_types에 없음")
    return {r["ad_id"]: r for r in rows}


class Pipeline:
    def __init__(self, cfg, root, code_dir, redo_ocr=False):
        self.cfg = cfg
        self.root = Path(root)
        self.redo_ocr = redo_ocr
        self.processed = self.root / cfg["paths"]["processed"]
        self.rules = Rules(Path(code_dir) / cfg["paths"]["rules"], cfg["classify"]["fuzzy_min_len"])
        self.profiles = cfg["display_profiles"]
        self._engine = None
        for sub in ("json", "ocr_raw", "crops", "viz"):
            (self.processed / sub).mkdir(parents=True, exist_ok=True)

    @property
    def engine(self):
        if self._engine is None:
            print("  OCR 엔진 로딩...")
            self._engine = build_engine(self.cfg["ocr"])
        return self._engine

    def _cache_key(self, sha256):
        payload = json.dumps({"sha256": sha256, "ocr": self.cfg["ocr"], "ocr_input": self.cfg["ocr_input"]},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _rel(self, p):
        return str(Path(p).relative_to(self.root))

    def run_ocr(self, src, img, img_np):
        cache_path = self.processed / "ocr_raw" / f"{src.ad_id}.json"
        input_dir = self.processed / "ocr_input" / src.ad_id
        key = self._cache_key(src.sha256)
        save_strips = self.cfg["output"]["save_ocr_strips"]
        if not self.redo_ocr and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key:
                if save_strips and not (input_dir / "strips.json").exists():
                    build_ocr_inputs(img, input_dir, self.cfg["ocr_input"])
                return cached, True
        up_np, input_meta = build_ocr_inputs(img, input_dir, self.cfg["ocr_input"], save=save_strips)
        blocks, stats = run_multi_pass(self.engine, img, img_np, up_np, input_meta, self.cfg["ocr_input"], self.cfg["ocr"])
        for s in input_meta["strips"]:
            s["path"] = self._rel(s["path"]) if save_strips else None
        cached = {"cache_key": key, "engine": self.engine.describe(), "input_meta": input_meta,
                  "stats": stats, "blocks": blocks}
        cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        return cached, False

    def process(self, src, meta_row):
        t0 = time.time()
        img, info = normalize_image(src.path, self.cfg["normalize"]["alpha_background"])
        img_np = np.asarray(img)
        # 정규화로 픽셀·좌표가 실제로 바뀐 경우에만 PNG 저장. 아니면 원본이 곧 정규화 이미지
        norm_path = self.processed / "normalized" / f"{src.ad_id}.png"
        if needs_normalized_copy(info):
            norm_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(norm_path)
            image_path = self._rel(norm_path)
        else:
            norm_path.unlink(missing_ok=True)
            image_path = self._rel(src.path)

        ocr, from_cache = self.run_ocr(src, img, img_np)
        t_ocr = time.time() - t0

        blocks = [dict(b) for b in ocr["blocks"]]
        for b in blocks:
            b["h_px"] = quad_height(b["quad"])
        blocks, graphic = split_graphic_text(blocks, self.cfg["graphic_filter"])
        blocks = reading_order(blocks)
        for i, b in enumerate(blocks):
            b["id"] = i
        for b, pid in zip(blocks, group_paragraphs(blocks, self.cfg["layout"])):
            b["paragraph_id"] = pid
        add_block_metrics(blocks, img_np, self.profiles, self.cfg["contrast"])   # 분류가 글자 높이를 쓰므로 먼저
        text_paras = build_text_paragraphs(blocks, self.cfg["text_paragraph"])
        for k, idx in enumerate(text_paras):
            for i in idx:
                blocks[i]["text_para_id"] = k
        classify_blocks(blocks, self.rules, self.cfg["classify"])
        crops = make_risk_crops(blocks, img, src.ad_id, self.processed / "crops", self.cfg["crop"], self.root)
        metrics = ad_metrics(blocks, img.width, self.profiles, content_bbox(img), self.cfg["classify"]["low_conf"])

        ocr_info = {"scale": ocr["input_meta"]["scale"], "n_strips": len(ocr["input_meta"]["strips"])}
        doc = build_ad_json(src.ad_id, meta_row, img.size, info["format"], image_path, blocks, text_paras,
                            metrics, crops)
        (self.processed / "json" / f"{src.ad_id}.json").write_text(dumps_ad_json(doc), encoding="utf-8")
        draw_viz(img, src.ad_id, blocks, graphic, crops, metrics, ocr_info,
                 self.processed / "viz" / f"{src.ad_id}.png", self.cfg["viz"])
        return {"blocks": len(blocks), "graphic": len(graphic), "counts": metrics["type_counts"], "crops": len(crops),
                "crops_cut": [c["path"] for c in crops if not c["edge_clear"]],
                "ocr_sec": round(t_ocr, 1), "total_sec": round(time.time() - t0, 1), "ocr_cached": from_cache}
