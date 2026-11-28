"""05 LLM 판정 명령줄 실행기 (판정 로직은 judge.py = 05_llm_judge.ipynb와 같음).

  python run_judge.py --ids AD01,V101
  python run_judge.py --all-images --skip-existing
  python run_judge.py --all-videos --product 투자성 --provider claude --model claude-sonnet-5-5
  python run_judge.py --all-images --skip-existing --workers 4     # LLM 호출 4개 동시

결과: {out-dir}/{ad_id}__{model}.json · .csv (기본 rag/outputs)
"""
import argparse
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parent

# 1M 토큰당 달러 (입력, 출력). 표에 없는 모델은 --price-in/--price-out로 지정
PRICES = {
    'gpt-6-luna': (0.10, 0.50),
    'claude-haiku-5-5': (0.10, 0.50),
    'claude-sonnet-5-5': (2.00, 10.00),
    'claude-opus-5-5': (4.00, 20.00),
}
VERDICT_ORDER = ['위반', '확인필요', '적합', '해당없음']


def parse_args():
    ap = argparse.ArgumentParser(description='05 LLM 판정을 광고 여러 건에 실행합니다.')
    ap.add_argument('--ids', help='쉼표로 구분한 광고 ID (예: AD01,V101)')
    ap.add_argument('--all-images', action='store_true', help='data/processed/image/json 전체')
    ap.add_argument('--all-videos', action='store_true', help='data/processed/video/json 전체')
    ap.add_argument('--product', help='상품유형 필터 (투자성/보장성/대출성/예금성/전체/범위밖). 05의 자동 추정 기준')
    ap.add_argument('--provider', choices=['openai', 'claude'], default='openai')
    ap.add_argument('--model', help='모델 이름 (기본: openai=gpt-6-luna, claude=claude-haiku-5-5)')
    ap.add_argument('--skip-existing', action='store_true', help='{ad_id}__{model}.json이 이미 있으면 건너뜀')
    ap.add_argument('--out-dir', default=str(RAG_DIR / 'outputs'), help='결과 폴더 (기본 rag/outputs)')
    ap.add_argument('--ad-dir', help='광고 JSON 폴더 ({ad-dir}/{image,video}/json). 기본 data/processed. 예: eval/variants')
    ap.add_argument('--workers', type=int, default=1, help='동시에 판정할 광고 수 (기본 1). 검색은 순서대로, LLM 호출만 동시에')
    ap.add_argument('--price-in', type=float, help='입력 1M 토큰당 달러 (PRICES 표 대신)')
    ap.add_argument('--price-out', type=float, help='출력 1M 토큰당 달러 (PRICES 표 대신)')
    args = ap.parse_args()
    if not (args.ids or args.all_images or args.all_videos):
        ap.error('--ids, --all-images, --all-videos 중 하나 이상 지정하세요')
    if args.workers < 1:
        ap.error('--workers는 1 이상')
    return args


def main():
    args = parse_args()
    print('판정 모듈 불러오는 중 (BGE-M3 + 리랭커, 30초~1분)…', flush=True)
    import judge as J
    if args.ad_dir:
        J.set_ad_dir(args.ad_dir)

    key = J.API_KEY_ENV[args.provider]
    if not os.getenv(key):
        sys.exit(f'{key}가 없습니다. rag/.env에 설정하세요.')
    model = args.model or J.MODELS[args.provider]
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 대상 광고 모으기
    ids = []
    if args.ids:
        ids += [s.strip() for s in args.ids.split(',') if s.strip()]
    if args.all_images:
        ids += J.list_ids('image')
    if args.all_videos:
        ids += J.list_ids('video')
    ids = list(dict.fromkeys(ids))
    known = set(J.ALL_IDS)
    unknown = [i for i in ids if i not in known]
    if unknown:
        sys.exit(f'광고 JSON이 없는 ID: {", ".join(unknown)}')
    if args.product:
        ids = [i for i in ids if J.infer_product(J.load_ad(i))[0] == args.product]
    skipped = []
    if args.skip_existing:
        skipped = [i for i in ids if (out_dir / J.output_name(i, model)).exists()]
        ids = [i for i in ids if i not in set(skipped)]

    p_in, p_out = PRICES.get(model, (None, None))
    p_in = args.price_in if args.price_in is not None else p_in
    p_out = args.price_out if args.price_out is not None else p_out
    priced = p_in is not None and p_out is not None
    cost = lambda ti, to: (ti * p_in + to * p_out) / 1e6 if priced else None

    print(f'모델 {model} ({args.provider}) | 대상 {len(ids)}건'
          + (f' | 이미 있어 건너뜀 {len(skipped)}건' if skipped else '')
          + (f' | 상품유형 {args.product}' if args.product else '')
          + f' | 동시 {args.workers} | 저장 {out_dir}')
    if not priced:
        print(f'  ※ {model} 단가 미등록 — 비용은 표시하지 않습니다 (--price-in/--price-out으로 지정)')
    if not ids:
        print('판정할 광고가 없습니다.')
        return

    lock = threading.Lock()
    state = {'done': 0, 'tin': 0, 'tout': 0}
    totals, failed = Counter(), []
    t0 = time.time()

    def one(ad_id):
        _, rec = J.judge_ad(ad_id, args.provider, model, verbose=False, out_dir=out_dir, return_record=True)
        return rec

    def report(ad_id, rec=None, err=None):
        with lock:
            state['done'] += 1
            head = f"[{state['done']:>3}/{len(ids)}] {ad_id:<6}"
            if err is not None:
                failed.append((ad_id, err))
                print(f'{head} 실패: {type(err).__name__}: {err}', flush=True)
                return
            ti, to = rec['tokens']
            state['tin'] += ti; state['tout'] += to
            vc = Counter(it['verdict'] for it in rec['items'])
            totals.update(vc)
            verdicts = ' '.join(f'{v} {vc[v]}' for v in VERDICT_ORDER if vc[v])
            c_ad, c_sum = cost(ti, to), cost(state['tin'], state['tout'])
            money = f' ${c_ad:.4f} | 누적 ${c_sum:.3f}' if priced else ''
            el = time.time() - t0
            eta = el / state['done'] * (len(ids) - state['done'])
            print(f"{head} {rec['product']:<4} {rec['media']:<6} | {verdicts} | 토큰 {ti:,}/{to:,} 호출 {rec['llm_calls']}"
                  f" | 누적 {state['tin']:,}/{state['tout']:,}{money} | {el / 60:.1f}분 경과, 약 {eta / 60:.0f}분 남음",
                  flush=True)

    if args.workers == 1:
        for a in ids:
            try:
                report(a, one(a))
            except Exception as e:     # 한 건 실패해도 나머지는 계속
                report(a, err=e)
    else:
        with ThreadPoolExecutor(args.workers) as ex:
            futs = {ex.submit(one, a): a for a in ids}
            for f in as_completed(futs):
                try:
                    report(futs[f], f.result())
                except Exception as e:
                    report(futs[f], err=e)

    n_ok = len(ids) - len(failed)
    print('\n== 완료')
    print(f'성공 {n_ok}건 · 실패 {len(failed)}건 · {(time.time() - t0) / 60:.1f}분')
    print(f"토큰 입력 {state['tin']:,} / 출력 {state['tout']:,}"
          + (f" | 예상 비용 ${cost(state['tin'], state['tout']):.3f} (입력 ${p_in}/1M, 출력 ${p_out}/1M, 캐시 할인 미반영)" if priced else ''))
    import rag_search
    cs = rag_search.cache_stats
    print(f"검색 캐시: 재사용 {cs['hit']} · 새로 검색 {cs['miss']} ({rag_search.SEARCH_CACHE_PATH.name})")
    if totals:
        print('판정 분포: ' + ' · '.join(f'{v} {totals[v]}' for v in VERDICT_ORDER if totals[v]))
    if failed:
        print('실패한 광고 (다시 돌리려면 --skip-existing): ' + ','.join(a for a, _ in failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
