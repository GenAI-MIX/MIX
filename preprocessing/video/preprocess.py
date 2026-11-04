import argparse
import sys
import time
import traceback
from pathlib import Path
import yaml

CODE_DIR = Path(__file__).resolve().parent    

with open(CODE_DIR / "config.yaml", encoding="utf-8") as _f:
    IMAGE_DIR = (CODE_DIR / yaml.safe_load(_f)["paths"]["image_code"]).resolve()
sys.path.insert(0, str(IMAGE_DIR))            

from video_prep.package import write_summary_csv  
from video_prep.pipeline import Pipeline  
from video_prep.probe import scan_sources, sync_meta_csv  


def main():
    ap = argparse.ArgumentParser(description="영상 광고 전처리")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, help="처리할 광고 수 (번호 순)")
    ap.add_argument("--ids", help="쉼표로 구분한 ad_id 목록")
    ap.add_argument("--redo-detect", action="store_true", help="검출 캐시(scan)를 무시하고 다시 실행")
    ap.add_argument("--no-ocr", action="store_true", help="V3(카드 OCR·분류·측정)를 건너뜀")
    args = ap.parse_args()

    with open(CODE_DIR / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    root = (CODE_DIR / paths["repo_root"]).resolve()

    print("[V0] 원본 확인 · 해시 중복 검사 · ffprobe")
    raw_dir = root / paths["raw_videos"]
    sources = scan_sources(raw_dir, root / paths["source_csv"], cfg["source"], cfg["media_map"], cfg.get("exclude") or {})
    dups = [s for s in sources if s.dup_of]
    excluded = [s for s in sources if s.exclude]
    print(f"  영상 {len(sources)}개 · 완전 중복 {len(dups)}개 · 분석 제외 {len(excluded)}개")
    for s in dups:
        print(f"    중복 제외: {s.ad_id} = {s.dup_of}")
    for s in excluded:
        print(f"    분석 제외: {s.ad_id} ({s.exclude})")
    meta = sync_meta_csv(root / paths["meta_csv"], sources, raw_dir)

    targets = [s for s in sources if not s.dup_of and not s.exclude]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",")}
        targets = [s for s in targets if s.ad_id in wanted]
    if args.limit:
        targets = targets[:args.limit]

    pipe = Pipeline(cfg, root, IMAGE_DIR, redo_detect=args.redo_detect, with_ocr=not args.no_ocr)
    print(f"[V1-V3] 영상 {len(targets)}개 처리")
    t0, failed = time.time(), []
    for n, s in enumerate(targets, 1):
        try:
            r = pipe.process(s, meta.get(s.ad_id, {}))
            m = r["metrics"]
            det_t = "검출 캐시" if r["det_cached"] else f"검출 {r['det_sec']}s"
            cov = "-" if m["csv_coverage"] is None else f"{m['csv_coverage']:.0%}"
            line = (f"  ({n}/{len(targets)}) {s.ad_id}: 카드 {m['n_cards']} [static {m['n_static']} · crawl "
                    f"{m['n_crawl']}] 화면 {r['screens']} · CSV 구간 커버 {cov}"
                    f"{' · letterbox' if r['letterbox'] else ''}")
            if "n_risk_cards" in m:
                cps = "-" if m["risk_cps_max"] is None else f"{m['risk_cps_max']}자/초"
                line += (f" → 위험고지 카드 {m['n_risk_cards']} · 노출 {m['risk_exposure_s']}s · 최대 {cps}"
                         f" · OCR {r['ocr_sec']}s (캐시 {r['ocr_hits']})")
            print(f"{line} · {det_t} · 총 {r['total_sec']}s")
        except Exception:
            failed.append(s.ad_id)
            print(f"  ({n}/{len(targets)}) {s.ad_id}: 실패")
            traceback.print_exc()

    processed = root / paths["processed"]
    rows = write_summary_csv(processed / "json", processed / "summary.csv", cfg["display_profiles"])
    low = [r["ad_id"] for r in rows if r["csv_coverage"] != "" and r["csv_coverage"] < 0.8]
    print(f"[확인 필요] 원본 CSV 노출 구간을 80% 미만 덮은 영상 {len(low)}개: {', '.join(low) or '-'}")
    if not args.no_ocr:
        miss = [r["ad_id"] for r in rows if r["csv_risk"] == "있음" and not r.get("n_risk_cards")]
        print(f"[확인 필요] 원본 CSV는 고지 '있음'인데 위험고지 카드를 못 찾은 영상 {len(miss)}개: {', '.join(miss) or '-'}"
              " (심의필·카드 남용 경고만 있는 광고는 keywords.yaml 기준 other라 정상)")
        short = [r["ad_id"] for r in rows if r.get("risk_short_exposures")]
        print(f"[확인 필요] 1초 미만 위험고지 노출이 있는 영상 {len(short)}개 (초당 글자 수 과대 가능): {', '.join(short) or '-'}")
    print(f"[완료] {time.time() - t0:.0f}s · summary.csv {len(rows)}행 · 실패 {len(failed)}건 {failed or ''}")


if __name__ == "__main__":
    main()
