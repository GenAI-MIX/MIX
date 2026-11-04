import csv
import hashlib
import io
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

EXIF_ORIENTATION = 0x0112


@dataclass
class SourceImage:
    ad_id: str
    path: Path
    sha256: str
    dhash: str
    dup_of: str | None = None
    near_dup_of: str | None = None   # 근사 중복 의심 (경고용, 처리는 계속)
    meta: dict = field(default_factory=dict)   # 데이터셋 CSV에서 온 정보 (product, media, brand 등)


def dhash_bits(path, size=16):
    with Image.open(path) as im:
        im.draft("L", (size * 4, size * 4))   # JPEG는 축소 디코딩으로 빠르게
        g = np.asarray(ImageOps.exif_transpose(im).convert("L").resize((size + 1, size), Image.Resampling.LANCZOS),
                       dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).flatten()


def _nfc(s):
    return unicodedata.normalize("NFC", str(s))


def _dataset_entries(raw_dir, ds, media_map):
    root = Path(raw_dir) / ds["dir"]
    files = {_nfc(p.relative_to(root)): p for p in root.rglob("*") if p.is_file()}
    with open(root / ds["meta_csv"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        if r.get(ds["category_column"], "") in ds.get("skip_categories", []):
            continue
        path = files.get(_nfc(r[ds["file_column"]]))
        if path is None:
            print(f"  [경고] {ds['dir']}: CSV에 있는 파일을 찾을 수 없음: {r[ds['file_column']]}")
            continue
        usage = r.get(ds.get("media_column", ""), "")
        meta = {"product": r.get(ds["category_column"], ""), "media": media_map.get(usage, usage),
                "brand": r.get(ds.get("brand_column", ""), ""), "title": r.get(ds.get("title_column", ""), "")}
        out.append((f"{ds['id_prefix']}{int(r[ds['id_column']]):02d}", path, meta))
    return out


def scan_sources(raw_dir, extensions, near_dup_hamming, datasets=(), media_map=None):
    entries = [(p.stem, p, {}) for p in sorted(Path(raw_dir).iterdir())
               if p.is_file() and p.suffix.lower() in extensions]
    for ds in datasets:
        entries += _dataset_entries(raw_dir, ds, media_map or {})
    sources, by_sha, used_ids, bits = [], {}, set(), {}
    for base_id, p, meta in entries:
        ad_id, k = base_id, 2
        while ad_id in used_ids:          
            ad_id = f"{base_id}_{k}"
            k += 1
        used_ids.add(ad_id)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        b = dhash_bits(p)
        src = SourceImage(ad_id=ad_id, path=p, sha256=sha, dhash=np.packbits(b).tobytes().hex(), meta=meta)
        if sha in by_sha:
            src.dup_of = by_sha[sha]
        else:
            by_sha[sha] = ad_id
            for other_id, other_bits in bits.items():
                if int((b != other_bits).sum()) <= near_dup_hamming:
                    src.near_dup_of = other_id
                    break
            bits[ad_id] = b
        sources.append(src)
    return sources


def _is_srgb(profile):
    try:
        return "srgb" in ImageCms.getProfileDescription(profile).lower()
    except Exception:
        return False


def normalize_image(path, alpha_background):
    im = Image.open(path)
    info = {
        "format": im.format,
        "mode": im.mode,
        "width": im.width,
        "height": im.height,
        "exif_orientation": im.getexif().get(EXIF_ORIENTATION),
    }
    icc = im.info.get("icc_profile")
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)

    if info["exif_orientation"] not in (None, 1):
        im = ImageOps.exif_transpose(im)

    if im.mode in ("P", "PA", "LA", "L", "1", "I", "I;16", "F"):
        im = im.convert("RGBA" if has_alpha else "RGB")

    info["icc_profile"] = None
    info["icc_converted"] = False
    if icc:
        src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        info["icc_profile"] = ImageCms.getProfileDescription(src_profile).strip()
        if not _is_srgb(src_profile):
            out_mode = "RGBA" if im.mode == "RGBA" else "RGB"
            im = ImageCms.profileToProfile(im, src_profile, ImageCms.createProfile("sRGB"), outputMode=out_mode)
            info["icc_converted"] = True
    im = im.convert("RGBA" if has_alpha else "RGB")

    if has_alpha:
        bg = Image.new("RGB", im.size, tuple(alpha_background))
        bg.paste(im, mask=im.getchannel("A"))
        im = bg
    info["had_alpha"] = has_alpha
    return im, info


def needs_normalized_copy(info):
    return (info["exif_orientation"] not in (None, 1) or info["icc_converted"] or info["had_alpha"]
            or info["mode"] != "RGB" or info["format"] not in ("JPEG", "PNG"))


def content_bbox(img, tol=24, min_frac=0.01):
    small = img.copy()
    small.thumbnail((800, 800))
    a = np.asarray(small).astype(int)
    c = 5
    corners = np.concatenate([a[:c, :c].reshape(-1, 3), a[:c, -c:].reshape(-1, 3),
                              a[-c:, :c].reshape(-1, 3), a[-c:, -c:].reshape(-1, 3)])
    diff = np.abs(a - np.median(corners, axis=0)).max(axis=2) > tol
    rows = np.where(diff.mean(axis=1) > min_frac)[0]
    cols = np.where(diff.mean(axis=0) > min_frac)[0]
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, img.width, img.height]
    sx, sy = img.width / small.width, img.height / small.height
    return [int(cols[0] * sx), int(rows[0] * sy),
            min(img.width, int(np.ceil((cols[-1] + 1) * sx))), min(img.height, int(np.ceil((rows[-1] + 1) * sy)))]
