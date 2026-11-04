import argparse
import time
import traceback
from pathlib import Path
import yaml
from image_prep.dedupe_text import find_text_duplicates
from image_prep.normalize import scan_sources
from image_prep.package import write_summary_csv
from image_prep.pipeline import Pipeline, sync_meta_csv

CODE_DIR = Path(__file__).resolve().parent     


def main():
    ap = argparse.ArgumentParser(description="포스터 이미지 광고 전처리")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, help="처리할 광고 수 (파일명 순)")
    ap.add_argument("--ids", help="쉼표로 구분한 ad_id 목록")
    ap.add_argument("--redo-ocr", action="store_true", help="OCR 캐시(ocr_raw)를 무시하고 다시 실행")
    args = ap.parse_args()

    with open(CODE_DIR / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    root = (CODE_DIR / paths["repo_root"]).resolve()   # data 경로·JSON 안 경로의 기준 (레포 루트)

    print("[0] 원본 스캔 · 해시 중복 검사")
    sources = scan_sources(root / paths["raw_images"], set(cfg["image_extensions"]),
                           cfg["normalize"]["near_dup_hamming"], cfg.get("datasets", []), cfg.get("media_map", {}))
    dups = [s for s in sources if s.dup_of]
    near = [s for s in sources if s.near_dup_of]
    print(f"  원본 {len(sources)}개 · 완전 중복 {len(dups)}개 (제외) · 근사 중복 의심 {len(near)}개")
    for s in dups:
        print(f"    중복 제외: {s.ad_id} = {s.dup_of}")
    for s in near:
        print(f"    근사 중복 의심: {s.ad_id} ≈ {s.near_dup_of}")
    meta = sync_meta_csv(root / paths["meta_csv"], sources, cfg["media_types"], root / paths["raw_images"])

    targets = [s for s in sources if not s.dup_of]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        targets = [s for s in targets if s.ad_id in wanted]
    if args.limit:
        targets = targets[:args.limit]

    pipe = Pipeline(cfg, root, CODE_DIR, redo_ocr=args.redo_ocr)
    print(f"[1-5] 광고 {len(targets)}건 처리")
    t0, failed = time.time(), []
    for n, s in enumerate(targets, 1):
        try:
            r = pipe.process(s, meta.get(s.ad_id, {}))
            c = r["counts"]
            ocr_t = "OCR 캐시" if r["ocr_cached"] else "OCR {}s".format(r["ocr_sec"])
            print(f"  ({n}/{len(targets)}) {s.ad_id}: 블록 {r['blocks']} [risk {c['risk_notice']} · "
                  f"headline {c['headline']} · return {c['return_claim']} · other {c['other']} · "
                  f"unknown {c['unknown']}] 그래픽 제외 {r['graphic']} · crop {r['crops']} · {ocr_t} · "
                  f"총 {r['total_sec']}s")
            for p in r["crops_cut"]:
                print(f"    [확인 필요] crop 경계에 걸쳐 잘린 텍스트 블록이 있음: {p}")
        except Exception:
            failed.append(s.ad_id)
            print(f"  ({n}/{len(targets)}) {s.ad_id}: 실패")
            traceback.print_exc()

    processed = root / paths["processed"]
    n_rows = write_summary_csv(processed / "json", processed / "summary.csv", cfg["display_profiles"])
    dup_rows = find_text_duplicates(processed / "json", processed / "duplicates.csv", cfg["text_dedupe"])
    same_ad = [r for r in dup_rows if r["kind"] == "문구 거의 동일"]
    print(f"[중복] OCR 텍스트 기준 문구 거의 동일 {len(same_ad)}쌍 · 고지문 공유 "
          f"{len(dup_rows) - len(same_ad)}쌍 → duplicates.csv (train/test는 같은 쪽에)")
    for r in same_ad:
        print(f"    문구 거의 동일: {r['ad_a']} ≈ {r['ad_b']} (본문 유사도 {r['body_sim']})")
    print(f"[완료] {time.time() - t0:.0f}s · summary.csv {n_rows}행 · 실패 {len(failed)}건 {failed or ''}")


if __name__ == "__main__":
    main()
