"""영상 1건 처리 흐름: 검출(캐시) → 블록 추적 → scan 프레임 → 카드·화면 → 대표 프레임
→ 카드 OCR·분류·측정(캐시) → JSON → 타임라인."""
import hashlib
import json
import time
from pathlib import Path
import yaml
from PIL import Image
from image_prep.classify import Rules
from image_prep.ocr import build_engine
from .cards import collect_crops, content_box, group_screens, split_cards, track_blocks, track_regions
from .detect import TextDetector
from .frames import grab_frame, iter_frames
from .package import build_video_json, dumps_video_json, risk_exposures, risk_metrics, video_metrics
from .text import OcrCache, read_cards, review_numbers
from .viz import draw_timeline


class _LazyEngine:

    def __init__(self, ocr_cfg):
        self.cfg, self._engine = ocr_cfg, None

    def recognize(self, image_rgb):
        if self._engine is None:
            print("  OCR 엔진 로딩...")
            self._engine = build_engine(self.cfg)
        return self._engine.recognize(image_rgb)


class Pipeline:
    def __init__(self, cfg, root, image_dir, redo_detect=False, with_ocr=True):
        self.cfg = cfg
        self.root = Path(root)
        self.redo_detect = redo_detect
        self.with_ocr = with_ocr
        self.processed = self.root / cfg["paths"]["processed"]
        self.card_cfg = {**cfg["cards"], "det_fps": cfg["detect"]["fps"]}
        self.profiles = cfg["display_profiles"]
        with open(Path(image_dir) / "config.yaml", encoding="utf-8") as f:
            self.icfg = yaml.safe_load(f)            # V3는 이미지 광고 설정(OCR·분류·측정)을 그대로 씀
        self.rules = Rules(Path(image_dir) / self.icfg["paths"]["rules"], self.icfg["classify"]["fuzzy_min_len"])
        self.engine = _LazyEngine(self.icfg["ocr"])
        self._detector = None
        for sub in ("json", "scan", "frames", "viz", "ocr_raw", "crawl"):
            (self.processed / sub).mkdir(parents=True, exist_ok=True)

    @property
    def detector(self):
        if self._detector is None:
            print("  글자 검출 모델 로딩...")
            self._detector = TextDetector(**self.cfg["detect"]["paddle"])
        return self._detector

    def _rel(self, p):
        return str(Path(p).relative_to(self.root))

    def run_detect(self, src):
        """detect.fps 간격 프레임의 글자 박스. 파일 해시·검출 설정이 같으면 scan/ 캐시를 쓴다."""
        cache_path = self.processed / "scan" / f"{src.ad_id}.json"
        key = hashlib.sha256(json.dumps({"sha256": src.sha256, "detect": self.cfg["detect"]},
                                        sort_keys=True).encode()).hexdigest()
        if not self.redo_detect and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == key:
                return cached, True
        size = (src.info["width"], src.info["height"])
        frames = [{"t": t, "boxes": self.detector.detect(f)}
                  for t, f in iter_frames(src.path, size, self.cfg["detect"]["fps"])]
        cached = {"cache_key": key, "engine": self.detector.describe(), "fps": self.cfg["detect"]["fps"],
                  "frames": frames}
        cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        return cached, False

    def read_text(self, src, cards, screens, frame_paths):
        cache = OcrCache(self.processed / "ocr_raw" / f"{src.ad_id}.json",
                         {"ocr": self.icfg["ocr"], "ocr_input": self.icfg["ocr_input"]})
        blocks, paras = read_cards(src, cards, screens, frame_paths, self.engine, self.icfg, self.rules,
                                   self.profiles, self.cfg["text"], cache)
        cache.save()
        for old in (self.processed / "crawl").glob(f"{src.ad_id}_c*.png"):
            old.unlink()
        for c in cards:
            strip = c.pop("_strip", None)
            if strip is not None:
                out = self.processed / "crawl" / f"{src.ad_id}_c{c['id']:02d}.png"
                Image.fromarray(strip).save(out)
                c["strip"] = self._rel(out)
        return blocks, paras, cache.hits

    def process(self, src, meta_row):
        t0 = time.time()
        W, H, dur = src.info["width"], src.info["height"], src.info["duration"]
        det, from_cache = self.run_detect(src)
        t_det = time.time() - t0

        cc, sc = self.card_cfg, self.cfg["scan"]
        tracks = track_blocks(det["frames"], W, H, cc)
        track_regions(tracks, W, H, cc)
        row_max, col_max = collect_crops(iter_frames(src.path, (W, H), sc["fps"], gray=True, scale=sc["scale"]),
                                         tracks, sc["scale"])
        cards = [c for tr in tracks for c in split_cards(tr, cc, sc["fps"], sc["scale"], dur, W)]
        cards.sort(key=lambda c: (c["t_start"], c["bbox"][1]))
        for i, c in enumerate(cards):
            c["id"] = i
        screens = group_screens(cards, cc["screen_iou"])
        cards = [{"id": c.pop("id"), "screen_id": c.pop("screen_id"), **c} for c in cards]

        frame_paths = {}
        for old in (self.processed / "frames").glob(f"{src.ad_id}_s*.png"):
            old.unlink()
        for s in screens:
            out = self.processed / "frames" / f"{src.ad_id}_s{s['id']:02d}.png"
            grab_frame(src.path, s["rep_t"], src.info["fps"], out)
            s["frame"] = self._rel(out)
            frame_paths[s["id"]] = out

        content = content_box(row_max, col_max, sc["scale"], W, H, sc["letterbox_thr"])
        metrics = video_metrics(cards, src.csv_label["intervals"])
        t1 = time.time()
        blocks = paras = review = exposures = None
        ocr_hits = 0
        if self.with_ocr:
            blocks, paras, ocr_hits = self.read_text(src, cards, screens, frame_paths)
            exposures = risk_exposures(cards, self.cfg["text"])
            metrics.update(risk_metrics(cards, exposures, src.csv_label["intervals"], self.profiles,
                                        self.cfg["text"]["readable_cps"]))
            review = review_numbers(cards, paras)
        t_ocr = time.time() - t1

        doc = build_video_json(src, meta_row, content, cards, screens, metrics, self._rel(src.path), review,
                               blocks, paras, exposures)
        (self.processed / "json" / f"{src.ad_id}.json").write_text(dumps_video_json(doc), encoding="utf-8")
        draw_timeline(doc, frame_paths, self.processed / "viz" / f"{src.ad_id}.png", self.cfg["viz"])
        return {"metrics": metrics, "screens": len(screens), "letterbox": content != [0, 0, W, H],
                "det_sec": round(t_det, 1), "det_cached": from_cache, "ocr_sec": round(t_ocr, 1),
                "ocr_hits": ocr_hits, "total_sec": round(time.time() - t0, 1)}
