import csv
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .frames import probe

META_COLS = ["ad_id", "source_file", "sha256", "dup_of", "exclude", "width", "height", "fps", "duration", "has_audio",
             "media", "product", "brand", "title", "url", "csv_risk", "csv_intervals", "note"]


@dataclass
class SourceVideo:
    ad_id: str
    path: Path
    sha256: str
    info: dict                                     # ffprobe: width·height·fps·duration·sar·has_audio
    meta: dict = field(default_factory=dict)       # 원본 CSV: media·brand·title·channel·url
    csv_label: dict = field(default_factory=dict)  # 원본 CSV 자동 분석값: risk·intervals·review_no (비교용)
    dup_of: str | None = None
    exclude: str | None = None


def _nfc(s):
    return unicodedata.normalize("NFC", str(s))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_intervals(text):
    out = []
    for part in (text or "").split(";"):
        if "~" in part:
            a, b = part.split("~")
            out.append([float(a), float(b)])
    return out


def scan_sources(raw_dir, csv_path, src_cfg, media_map, exclude):
    files = {_nfc(p.name): p for p in Path(raw_dir).iterdir() if p.is_file()}
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out, by_hash = [], {}
    for r in rows:
        ad_id = f"{src_cfg['id_prefix']}{int(r[src_cfg['id_column']])}"
        path = files.get(_nfc(r[src_cfg["file_column"]]))
        if path is None:
            print(f"  [경고] {ad_id}: CSV에 있는 파일을 찾을 수 없음: {r[src_cfg['file_column']]}")
            continue
        sha = file_sha256(path)
        media = r.get(src_cfg["media_column"], "")
        s = SourceVideo(
            ad_id=ad_id, path=path, sha256=sha, info=probe(path),
            meta={"media": media_map.get(media, media), "brand": r.get(src_cfg["brand_column"], ""),
                  "title": r.get(src_cfg["title_column"], ""), "channel": r.get(src_cfg["channel_column"], ""),
                  "url": r.get(src_cfg["url_column"], "")},
            csv_label={"risk": r.get(src_cfg["risk_column"], ""),
                       "intervals": parse_intervals(r.get(src_cfg["intervals_column"], "")),
                       "review_no": r.get(src_cfg["review_column"], "") or None},
            dup_of=by_hash.get(sha), exclude=exclude.get(ad_id))
        by_hash.setdefault(sha, ad_id)
        out.append(s)
    return out


def sync_meta_csv(path, sources, raw_dir):
    path = Path(path)
    existing = {}
    if path.exists():
        with open(path, encoding="utf-8-sig", newline="") as f:
            existing = {r["ad_id"]: r for r in csv.DictReader(f)}
    rows, seen = [], set()
    for s in sources:
        old = existing.get(s.ad_id, {})
        row = {"ad_id": s.ad_id, "source_file": str(s.path.relative_to(raw_dir)), "sha256": s.sha256,
               "dup_of": s.dup_of or "", "exclude": s.exclude or "",
               **{k: s.info[k] for k in ("width", "height", "fps", "duration", "has_audio")},
               "url": s.meta["url"], "csv_risk": s.csv_label["risk"],
               "csv_intervals": "; ".join(f"{a} ~ {b}" for a, b in s.csv_label["intervals"]),
               "note": old.get("note", "")}
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
    return {r["ad_id"]: r for r in rows}
