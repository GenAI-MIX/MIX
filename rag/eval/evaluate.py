"""위반 예시로 05 판단의 탐지율을 잰다.

  python eval/evaluate.py --provider openai --model gpt-6-luna      # rag/ 폴더에서 실행
  python eval/evaluate.py --model gpt-6-luna --report-only           # 판정 없이 보고서만 다시 만들기

먼저 eval/make_variants.py로 eval/variants/와 eval/labels.csv를 만든다.
결과: eval/runs/{model}/{variant_id}__{model}.json · .csv, results.csv
      eval/runs/{model}/repeat/  원본 10건 반복 판정 (LLM 흔들림 기준선)
      eval/report_{model}.md
"""
import argparse
import csv
import datetime
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
RAG_DIR = EVAL_DIR.parent
sys.path.insert(0, str(RAG_DIR))
from run_judge import PRICES   # noqa: E402  (판정 모듈은 판정할 때만 불러옴)

# target이 바뀌면 코드로 따라 바뀌는 항목 (05 judge_ad: C5는 A2~A7·C3에서, D2는 심사필 표시가 있을 때만 날짜 확인)
COUPLED = {'m_A2': {'C5'}, 'm_A3': {'A4', 'C5'}, 'm_A5': {'C5'}, 'm_C3': {'C5'}, 'm_D1': {'D2'}}
MUT_DESC = {
    'm_A2': '명암비 1.5', 'm_A3': '위험고지 글자·헤드라인 비율 ×0.5', 'm_A5': '위험고지 노출 = 영상 길이×0.2',
    'm_B1': '설명서·약관 권유 삭제', 'm_B3': '원금손실·투자자 귀속 삭제', 'm_B6': '예금자보호 문구 삭제',
    'm_C2': '헤드라인 "목표수익률 연 15% 달성!"', 'm_C3': '헤드라인 "국내 유일 최고 수익 ETF"', 'm_D1': '심사필 문구·번호 삭제',
}


def parse_args():
    ap = argparse.ArgumentParser(description='위반 예시 탐지율 평가')
    ap.add_argument('--provider', choices=['openai', 'claude'], default='openai')
    ap.add_argument('--model', help='모델 이름 (기본: provider 기본 모델)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rejudge', action='store_true', help='이미 있는 판정도 다시 함 (기본: 있으면 재사용)')
    ap.add_argument('--no-repeat', action='store_true', help='원본 반복 판정(흔들림 기준선) 생략')
    ap.add_argument('--report-only', action='store_true', help='판정 없이 있는 결과로 보고서만')
    return ap.parse_args()


def load_labels():
    return list(csv.DictReader(open(EVAL_DIR / 'labels.csv', encoding='utf-8-sig')))


def judge_all(J, provider, model, jobs, workers):
    """jobs: [(ad_id, out_dir)]. 진행 상황 출력."""
    t0, done = time.time(), 0
    if not jobs:
        return
    print(f'판정 {len(jobs)}건 (동시 {workers})', flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(J.judge_ad, a, provider, model, False, d, True): a for a, d in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                _, rec = f.result()
                vc = Counter(it['verdict'] for it in rec['items'])
                print(f"[{done:>3}/{len(jobs)}] {futs[f]:<14} 위반 {vc['위반']} 확인필요 {vc['확인필요']} | {(time.time() - t0) / 60:.1f}분", flush=True)
            except Exception as e:
                print(f'[{done:>3}/{len(jobs)}] {futs[f]:<14} 실패: {type(e).__name__}: {e}', flush=True)


def load_rec(d, vid, model):
    p = d / f'{vid}__{model}.json'.replace('/', '_')
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else None


def verdicts(rec):
    return {it['check_id']: it for it in rec['items']}


def main():
    args = parse_args()
    model = args.model or {'openai': 'gpt-6-luna', 'claude': 'claude-haiku-5-5'}[args.provider]
    run_dir = EVAL_DIR / 'runs' / model.replace('/', '_')
    rep_dir = run_dir / 'repeat'
    labels = load_labels()
    origs = [r for r in labels if r['mutation'] == 'original']

    if not args.report_only:
        print('판정 모듈 불러오는 중…', flush=True)
        import judge as J
        J.set_ad_dir(EVAL_DIR / 'variants')
        key = J.API_KEY_ENV[args.provider]
        if not os.getenv(key):
            sys.exit(f'{key}가 없습니다. rag/.env에 설정하세요.')
        jobs = [(r['variant_id'], run_dir) for r in labels
                if args.rejudge or load_rec(run_dir, r['variant_id'], model) is None]
        if not args.no_repeat:
            jobs += [(r['variant_id'], rep_dir) for r in origs
                     if args.rejudge or load_rec(rep_dir, r['variant_id'], model) is None]
        judge_all(J, args.provider, model, jobs, args.workers)

    recs = {r['variant_id']: load_rec(run_dir, r['variant_id'], model) for r in labels}
    missing = [v for v, x in recs.items() if x is None]
    if missing:
        sys.exit(f'판정 결과가 없는 변형: {", ".join(missing)}')
    reps = {r['base_id']: load_rec(rep_dir, r['variant_id'], model) for r in origs}
    write_outputs(labels, recs, reps, model, run_dir)


def write_outputs(labels, recs, reps, model, run_dir):
    base_v = {r['base_id']: verdicts(recs[r['variant_id']]) for r in labels if r['mutation'] == 'original'}
    rows = []
    for r in labels:
        if r['mutation'] == 'original':
            continue
        v = verdicts(recs[r['variant_id']])
        t = v.get(r['target_check'], {})
        exp = r['expected_verdicts'].split('|')
        base = base_v[r['base_id']]
        changed = sorted(c for c in v if c in base and c != r['target_check'] and v[c]['verdict'] != base[c]['verdict'])
        coupled = [c for c in changed if c in COUPLED.get(r['mutation'], set())]
        rows.append({
            'variant_id': r['variant_id'], 'base_id': r['base_id'], 'mutation': r['mutation'], 'target_check': r['target_check'],
            'expected': r['expected_verdicts'], 'got': t.get('verdict', '항목 없음'),
            'detected': t.get('verdict') in exp, 'source': t.get('source', ''),
            'original_now': base.get(r['target_check'], {}).get('verdict', ''),
            'side_coupled': '|'.join(f"{c}:{base[c]['verdict']}→{v[c]['verdict']}" for c in coupled),
            'side_other': '|'.join(f"{c}:{base[c]['verdict']}→{v[c]['verdict']}" for c in changed if c not in coupled),
            'n_side_other': len(changed) - len(coupled),
            'reason': t.get('reason', ''), 'measure': t.get('measure', ''), 'issues': t.get('issues', ''),
            'recheck': t.get('recheck', ''),
        })
    with open(run_dir / 'results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # (가) 탐지율
    by_mut = defaultdict(list)
    for x in rows:
        by_mut[x['mutation']].append(x)
    n_det = sum(x['detected'] for x in rows)

    # (나) 원본 오탐: 심의 통과 광고에서 나온 '위반'
    fp = [(r['base_id'], c, it) for r in labels if r['mutation'] == 'original'
          for c, it in verdicts(recs[r['variant_id']]).items() if it['verdict'] == '위반']
    orig_check = Counter(it['verdict'] for r in labels if r['mutation'] == 'original' for it in recs[r['variant_id']]['items'])

    # (다) 부작용 기준선: 원본을 한 번 더 판정했을 때 달라지는 항목 수
    noise = None
    if all(reps.values()):
        diffs = []
        for b, rep in reps.items():
            rv, bv = verdicts(rep), base_v[b]
            diffs.append(sorted(c for c in rv if c in bv and rv[c]['verdict'] != bv[c]['verdict']))
        noise = (sum(map(len, diffs)) / len(diffs), Counter(c for d in diffs for c in d))

    # (라) 근거 검증·재확인·비용
    all_recs = list(recs.values()) + [x for x in reps.values() if x]
    items = [it for x in all_recs for it in x['items'] if it['verdict'] != '해당없음']
    pass_rate = sum(not it['issues'] for it in items) / len(items)
    rechecks = sum(x['recheck_calls'] for x in all_recs)
    re_items = [y for x in all_recs for y in x.get('rechecks', [])]
    tin, tout = sum(x['tokens'][0] for x in all_recs), sum(x['tokens'][1] for x in all_recs)
    price = PRICES.get(model)
    cost = (tin * price[0] + tout * price[1]) / 1e6 if price else None

    L = [f'# 위반 예시 탐지율 평가 — {model}', '',
         f'- 실행일: {datetime.date.today()} · 원본 {len(base_v)}건 + 변형 {len(rows)}건'
         + (f' + 원본 반복 판정 {len(reps)}건' if noise else ''),
         f'- 입력: `rag/eval/variants/` (전처리 JSON 변형, `make_variants.py`), 정답표 `rag/eval/labels.csv`',
         '- 변형은 원본 판정에서 target 항목이 적합일 때만 적용 (건너뛴 조합: `eval/variants/skipped.csv`)',
         '- 상품유형은 원본 판정값으로 고정 (`product_override`)', '',
         '## 요약', '',
         f'- **전체 탐지율 {n_det}/{len(rows)} = {n_det / len(rows):.0%}**',
         f'- 원본 오탐(심의 통과 광고에서 위반): **{len(fp)}건** / 원본 판정 항목 {sum(orig_check.values())}개'
         f' (확인필요 {orig_check["확인필요"]})',
         f'- 부작용(target·연관 항목 외에 원본과 달라진 항목): 변형 1건당 평균 **{sum(x["n_side_other"] for x in rows) / len(rows):.2f}개**'
         + (f' — 같은 원본을 다시 판정할 때 달라지는 기준선 {noise[0]:.2f}개' if noise else ''),
         f'- 근거 검증 통과율 {pass_rate:.1%} (해당없음 제외 {len(items)}개 항목) · 위반 재확인 호출 {rechecks}회'
         f' (재확인 {len(re_items)}건 중 위반 유지 {sum(y["final"] == "위반" for y in re_items)})',
         f'- 토큰 입력 {tin:,} / 출력 {tout:,}' + (f' · 예상 비용 ${cost:.3f} (정가, 캐시 할인 미반영)' if cost is not None else ''),
         '', '## (가) 변형 종류별 탐지율', '',
         '| 변형 | 내용 | target | 기대 판정 | 판정 주체 | 탐지 | 탐지율 |', '|---|---|---|---|---|---|---|']
    for m in MUT_DESC:
        xs = by_mut.get(m)
        if not xs:
            continue
        d = sum(x['detected'] for x in xs)
        src = '/'.join(sorted({x['source'] for x in xs}))
        L.append(f"| {m} | {MUT_DESC[m]} | {xs[0]['target_check']} | {xs[0]['expected'].replace('|', '·')} | {src} | {d}/{len(xs)} | {d / len(xs):.0%} |")
    L.append(f'| **전체** | | | | | **{n_det}/{len(rows)}** | **{n_det / len(rows):.0%}** |')

    miss = [x for x in rows if not x['detected']]
    L += ['', f'## 놓친 변형 ({len(miss)}건)', '']

    def cause(x):
        if x['source'] == 'code':
            return f"코드 규칙 ({x['got']})"
        if '재확인 불일치' in x['issues']:
            return '1차 위반 → 위반 재확인에서 확인필요로 낮춤'
        return f"LLM이 {x['got']}로 판단"
    if miss:
        L += ['| 놓친 원인 | 건수 | 변형 |', '|---|---|---|']
        for c, n in Counter(cause(x) for x in miss).most_common():
            L.append(f"| {c} | {n} | {', '.join(x['variant_id'] for x in miss if cause(x) == c)} |")
        L.append('')
        L += ['| 변형 | target | 기대 | 판정 | 판정 주체 | 이유 (05 reason) | 근거 검증 |', '|---|---|---|---|---|---|---|']
        for x in miss:
            why = (x['reason'] or x['measure']).replace('|', '/').replace('\n', ' ')
            L.append(f"| {x['variant_id']} | {x['target_check']} | {x['expected'].replace('|', '·')} | {x['got']} | {x['source']} | {why} | {x['issues'] or '-'} |")
    else:
        L.append('없음')

    L += ['', f'## (나) 원본 오탐 ({len(fp)}건)', '', '심의를 통과한 원본 광고에서 `위반`으로 나온 항목입니다.', '']
    if fp:
        L += ['| 원본 | 항목 | 이유 |', '|---|---|---|']
        L += [f"| {b} | {c} {it['item']} | {it['reason'].replace('|', '/')} |" for b, c, it in fp]
    else:
        L.append('없음')

    L += ['', '## (다) 부작용: target 외 항목 변화', '',
          '변형한 항목(target) 말고 다른 항목의 판정이 원본(같은 실행에서 판정)과 달라진 수입니다. '
          '연관 항목은 05가 코드로 따라 바꾸는 항목이라 예상된 변화입니다 (C5 ← A2~A7·C3, D2 ← 심사필 표시, A4 ← m_A3에서 함께 줄인 헤드라인 비율).', '',
          '| 변형 | 건수 | 연관 항목 변화 | 그 밖의 항목 변화 (합계) | 자주 바뀐 항목 |', '|---|---|---|---|---|']
    for m in MUT_DESC:
        xs = by_mut.get(m)
        if not xs:
            continue
        cp = sum(bool(x['side_coupled']) for x in xs)
        other = Counter(s.split(':')[0] for x in xs for s in x['side_other'].split('|') if s)
        L.append(f"| {m} | {len(xs)} | {cp}건 | {sum(other.values())} | {', '.join(f'{k} {v}' for k, v in other.most_common(4)) or '-'} |")
    if noise:
        L += ['', f'기준선: 원본 {len(reps)}건을 같은 조건으로 한 번 더 판정하면 광고당 평균 **{noise[0]:.2f}개** 항목이 달라집니다'
              f" ({', '.join(f'{k} {v}' for k, v in noise[1].most_common(6)) or '없음'}). "
              '변형의 "그 밖의 항목 변화"가 이 수준이면 변형 탓이 아니라 LLM 판정의 흔들림으로 봅니다.']

    L += ['', '## (라) 근거 검증 · 재확인 · 비용', '',
          f'- 근거 검증 통과율: {pass_rate:.1%} — issues 종류: '
          + ', '.join(f'{k} {v}' for k, v in Counter(s.split('(')[0] for it in items for s in it['issues'].split('; ') if s).most_common()),
          f'- 위반 재확인: 호출 {rechecks}회, 재확인 {len(re_items)}건 중 2차도 위반 {sum(y["final"] == "위반" for y in re_items)}건',
          f'- LLM 호출 {sum(x["llm_calls"] for x in all_recs)}회 · 토큰 입력 {tin:,} / 출력 {tout:,}'
          + (f' · 예상 비용 ${cost:.3f}' if cost is not None else ''),
          '', '## 파일', '',
          f'- 변형별 결과: `rag/eval/runs/{model}/results.csv` (탐지 여부, 판정, 이유, 부작용 항목)',
          f'- 판정 원본: `rag/eval/runs/{model}/{{variant_id}}__{model}.json` · `.csv`', '']
    out = EVAL_DIR / f'report_{model}.md'
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f'\n보고서: {out}')
    print('\n'.join(L[7:14]))


if __name__ == '__main__':
    main()
