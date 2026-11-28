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
    'm_C2': '헤드라인 "목표수익률 연 15%"', 'm_C3': '헤드라인 "국내 유일 최고 수익 ETF"', 'm_D1': '심사필 문구·번호 삭제',
}


def parse_args():
    ap = argparse.ArgumentParser(description='위반 예시 탐지율 평가')
    ap.add_argument('--provider', choices=['openai', 'claude'], default='openai')
    ap.add_argument('--model', help='모델 이름 (기본: provider 기본 모델)')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--rejudge', action='store_true', help='이미 있는 판정도 다시 함 (기본: 있으면 재사용)')
    ap.add_argument('--no-repeat', action='store_true', help='원본 반복 판정(흔들림 기준선) 생략')
    ap.add_argument('--report-only', action='store_true', help='판정 없이 있는 결과로 보고서만')
    ap.add_argument('--ids', help='쉼표로 구분한 variant_id. --rejudge와 함께 쓰면 이 변형만 다시 판정')
    ap.add_argument('--baseline', help='이전 실행의 results.csv. 주면 보고서 맨 위에 이전 대비 변화 표')
    ap.add_argument('--note', action='append', default=[], help='이전 대비 바뀐 점 (보고서 맨 위에 한 줄씩, 여러 번 사용)')
    ap.add_argument('--runs', type=int, default=1, help='반복 판정 횟수. 2 이상이면 runs/{model}/x{N}/r1..rN에 판정하고 평균·최소~최대로 요약')
    ap.add_argument('--single-report', help='--runs와 함께: 보고서 아래에 붙일 1회 실행 보고서(md) 경로')
    ap.add_argument('--runs-label', default='', help='반복 평가 제목 옆에 붙일 설명 (예: B1 기준 보완 후)')
    ap.add_argument('--single-label', default='', help='붙이는 1회 실행 보고서 제목 옆 설명')
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
        futs = {ex.submit(J.judge_ad, a, provider, model, False, d, True): f'{d.name}/{a}' for a, d in jobs}
        for f in as_completed(futs):
            done += 1
            try:
                _, rec = f.result()
                vc = Counter(it['verdict'] for it in rec['items'])
                print(f"[{done:>3}/{len(jobs)}] {futs[f]:<18} 위반 {vc['위반']} 확인필요 {vc['확인필요']} | {(time.time() - t0) / 60:.1f}분", flush=True)
            except Exception as e:
                print(f'[{done:>3}/{len(jobs)}] {futs[f]:<18} 실패: {type(e).__name__}: {e}', flush=True)


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
    multi_dirs = [run_dir / f'x{args.runs}' / f'r{i}' for i in range(1, args.runs + 1)]

    if not args.report_only:
        print('판정 모듈 불러오는 중…', flush=True)
        import judge as J
        J.set_ad_dir(EVAL_DIR / 'variants')
        key = J.API_KEY_ENV[args.provider]
        if not os.getenv(key):
            sys.exit(f'{key}가 없습니다. rag/.env에 설정하세요.')
        ids = set(args.ids.split(',')) if args.ids else None
        unknown = ids - {r['variant_id'] for r in labels} if ids else set()
        if unknown:
            sys.exit(f'labels.csv에 없는 variant_id: {", ".join(sorted(unknown))}')
        redo = lambda vid: args.rejudge and (ids is None or vid in ids)
        if args.runs > 1:
            jobs = [(r['variant_id'], d) for d in multi_dirs for r in labels
                    if redo(r['variant_id']) or load_rec(d, r['variant_id'], model) is None]
        else:
            jobs = [(r['variant_id'], run_dir) for r in labels
                    if redo(r['variant_id']) or load_rec(run_dir, r['variant_id'], model) is None]
        if not args.no_repeat and args.runs == 1:
            jobs += [(r['variant_id'], rep_dir) for r in origs
                     if redo(r['variant_id']) or load_rec(rep_dir, r['variant_id'], model) is None]
        judge_all(J, args.provider, model, jobs, args.workers)

    out = EVAL_DIR / f'report_{model}.md'
    if args.runs > 1:
        all_recs = []
        for d in multi_dirs:
            recs = {r['variant_id']: load_rec(d, r['variant_id'], model) for r in labels}
            missing = [v for v, x in recs.items() if x is None]
            if missing:
                sys.exit(f'{d.name}: 판정 결과가 없는 변형 {len(missing)}건 ({", ".join(missing[:5])} …)')
            all_recs.append(recs)
        L = multi_report(labels, all_recs, model, multi_dirs, args.runs_label)
        if args.single_report:
            single = Path(args.single_report).read_text(encoding='utf-8').split('\n')
            title = '## 참고: 1회 실행 결과' + (f' ({args.single_label})' if args.single_label else '')
            L += ['', '---', '', title, '', f'`{Path(args.single_report).as_posix()}`의 내용입니다. 아래 수치는 반복 평가와 별개인 1회 실행 결과입니다.', '']
            L += [('#' + x if x.startswith('## ') else x) for x in single if not x.startswith('# ')]
        out.write_text('\n'.join(L), encoding='utf-8')
        print(f'\n보고서: {out}')
        return

    recs = {r['variant_id']: load_rec(run_dir, r['variant_id'], model) for r in labels}
    missing = [v for v, x in recs.items() if x is None]
    if missing:
        sys.exit(f'판정 결과가 없는 변형: {", ".join(missing)}')
    reps = {r['base_id']: load_rec(rep_dir, r['variant_id'], model) for r in origs}
    base_rows = list(csv.DictReader(open(args.baseline, encoding='utf-8-sig'))) if args.baseline else None
    L = write_outputs(labels, recs, reps, model, run_dir, base_rows, args.baseline, args.note)
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f'\n보고서: {out}')
    print('\n'.join(L[7:14]))


def flagged(expected, got):
    """검토 전달: 사람이 검토할 목록에 올라가면 성공 (위반·확인필요). 기대가 해당없음이면 해당없음이어야 성공."""
    exp = expected.split('|')
    return got == '해당없음' if exp == ['해당없음'] else got in ('위반', '확인필요')


def compare_block(rows, base_rows, base_path, notes):
    """이전 실행(results.csv) 대비 변화 표."""
    if not base_rows:
        return []
    def stats(xs, exp_key='expected'):
        d = sum(str(x['detected']) == 'True' for x in xs)
        f = sum(flagged(x[exp_key], x['got']) for x in xs)
        dn = sum('확인필요로 변경' in x.get('recheck', '') for x in xs)
        side = sum(int(x.get('n_side_other') or 0) for x in xs) / len(xs)
        return d, f, dn, side, len(xs)
    cur = defaultdict(list); old = defaultdict(list)
    for x in rows:
        cur[x['mutation']].append({**x, 'detected': str(x['detected'])})
    for x in base_rows:
        old[x['mutation']].append(x)
    L = ['## 이전 대비 변화', '', f'기준: `{Path(base_path).as_posix()}` (이전 실행). 정확 판정·검토 전달은 각 실행의 정답표 기준입니다.', '']
    L += [f'- {n}' for n in notes]
    if notes:
        L.append('')
    L += ['| 변형 | 정확 판정 (이전 → 지금) | 검토 전달 (이전 → 지금) | 재확인으로 낮춤 (이전 → 지금) | 부작용/건 (이전 → 지금) |', '|---|---|---|---|---|']
    def cell(a, b, n1, n2):
        return f'{a}/{n1} → **{b}/{n2}**' + (f' ({b / n2 - a / n1:+.0%})' if n1 and n2 and a / n1 != b / n2 else '')
    for m in list(MUT_DESC) + ['전체']:
        xs_o = [x for k in old for x in old[k]] if m == '전체' else old.get(m, [])
        xs_c = [x for k in cur for x in cur[k]] if m == '전체' else cur.get(m, [])
        if not xs_o or not xs_c:
            continue
        o, c = stats(xs_o), stats(xs_c)
        name = f'**{m}**' if m == '전체' else m
        L.append(f'| {name} | {cell(o[0], c[0], o[4], c[4])} | {cell(o[1], c[1], o[4], c[4])} | {o[2]} → {c[2]} | {o[3]:.2f} → {c[3]:.2f} |')
    return L + ['']


def build_rows(labels, recs):
    """변형별 결과 행. 부작용은 같은 실행의 원본 판정과 비교."""
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
            'detected': t.get('verdict') in exp, 'flagged': flagged(r['expected_verdicts'], t.get('verdict', '')),
            'downgraded': '확인필요로 변경' in t.get('recheck', ''), 'source': t.get('source', ''),
            'original_now': base.get(r['target_check'], {}).get('verdict', ''),
            'side_coupled': '|'.join(f"{c}:{base[c]['verdict']}→{v[c]['verdict']}" for c in coupled),
            'side_other': '|'.join(f"{c}:{base[c]['verdict']}→{v[c]['verdict']}" for c in changed if c not in coupled),
            'n_side_other': len(changed) - len(coupled),
            'reason': t.get('reason', ''), 'measure': t.get('measure', ''), 'issues': t.get('issues', ''),
            'recheck': t.get('recheck', ''),
        })
    return rows


def run_summary(labels, recs, model):
    """회차 하나의 지표."""
    rows = build_rows(labels, recs)
    origs = [r for r in labels if r['mutation'] == 'original']
    fp = [(r['base_id'], c, it) for r in origs for c, it in verdicts(recs[r['variant_id']]).items() if it['verdict'] == '위반']
    items = [it for x in recs.values() for it in x['items'] if it['verdict'] != '해당없음']
    tin, tout = sum(x['tokens'][0] for x in recs.values()), sum(x['tokens'][1] for x in recs.values())
    price = PRICES.get(model)
    return {
        'rows': rows, 'n': len(rows),
        'exact': sum(x['detected'] for x in rows), 'flag': sum(x['flagged'] for x in rows),
        'down': sum(x['downgraded'] for x in rows), 'fp': fp,
        'side': sum(x['n_side_other'] for x in rows) / len(rows),
        'pass': sum(not it['issues'] for it in items) / len(items),
        'cost': (tin * price[0] + tout * price[1]) / 1e6 if price else None,
    }


def multi_report(labels, all_recs, model, dirs, label=''):
    """반복 판정 요약: 회차별·평균·최소~최대, 변형 종류별, 회차마다 달라진 변형."""
    S = [run_summary(labels, recs, model) for recs in all_recs]
    N, n = len(S), S[0]['n']
    avg = lambda xs: sum(xs) / len(xs)
    def row(name, vals, fmt):
        return f"| {name} | " + ' | '.join(fmt(v) for v in vals) + f" | **{fmt(avg(vals))}** | {fmt(min(vals))} ~ {fmt(max(vals))} |"
    pct = lambda k: [s[k] / n for s in S]
    L = [f'# 위반 예시 탐지율 평가 — {model}', '',
         f'- 실행일: {datetime.date.today()} · 원본 {sum(r["mutation"] == "original" for r in labels)}건 + 변형 {n}건을 **{N}회 반복 판정**'
         + (f' ({label})' if label else ''),
         f'- 입력: `rag/eval/variants/` (전처리 JSON 변형, `make_variants.py`), 정답표 `rag/eval/labels.csv`',
         f'- 판정 결과: ' + ', '.join(f'`{d.relative_to(EVAL_DIR).as_posix()}/`' for d in dirs) + ' (회차마다 원본도 다시 판정, 부작용은 같은 회차의 원본과 비교)',
         '- 정확 판정: target 판정이 기대 판정과 정확히 일치 · 검토 전달: target이 위반·확인필요로 나와 검토 목록에 오름 (기대가 해당없음이면 해당없음일 때 성공)', '',
         f'## {N}회 평균 요약', '',
         f'- **정확 판정률 평균 {avg(pct("exact")):.1%}** (최소 {min(pct("exact")):.1%} ~ 최대 {max(pct("exact")):.1%})',
         f'- **검토 전달률 평균 {avg(pct("flag")):.1%}** (최소 {min(pct("flag")):.1%} ~ 최대 {max(pct("flag")):.1%})',
         f'- 원본 오탐(심의 통과 광고에서 위반): 회차별 {" · ".join(str(len(s["fp"])) for s in S)}건 (평균 {avg([len(s["fp"]) for s in S]):.1f}건)',
         f'- 위반 재확인에서 위반→확인필요로 낮아진 target: 회차별 {" · ".join(str(s["down"]) for s in S)}건',
         f'- 부작용(target·연관 항목 외 변화) 변형 1건당 평균 {avg([s["side"] for s in S]):.2f}개 · 근거 검증 통과율 평균 {avg([s["pass"] for s in S]):.1%}',
         (f'- 비용: 회차당 약 ${avg([s["cost"] for s in S]):.2f}, {N}회 합계 ${sum(s["cost"] for s in S):.2f} (정가, 캐시 할인 미반영)' if S[0]['cost'] is not None else ''),
         '', '## 회차별 결과', '',
         '| 지표 | ' + ' | '.join(f'{i + 1}회' for i in range(N)) + ' | 평균 | 최소 ~ 최대 |', '|---|' + '---|' * (N + 2),
         row('정확 판정률', pct('exact'), lambda v: f'{v:.1%}'),
         row('검토 전달률', pct('flag'), lambda v: f'{v:.1%}'),
         row('원본 오탐 (건)', [len(s['fp']) for s in S], lambda v: f'{v:g}' if v == int(v) else f'{v:.1f}'),
         row('재확인으로 낮춤 (건)', [s['down'] for s in S], lambda v: f'{v:g}' if v == int(v) else f'{v:.1f}'),
         row('부작용 / 변형 1건', [s['side'] for s in S], lambda v: f'{v:.2f}'),
         row('근거 검증 통과율', [s['pass'] for s in S], lambda v: f'{v:.1%}')]
    if S[0]['cost'] is not None:
        L.append(row('비용 ($)', [s['cost'] for s in S], lambda v: f'{v:.3f}'))

    L += ['', '## 변형 종류별 (회차별 성공 건수)', '',
          '| 변형 | 내용 | target | 건수 | 판정 주체 | 정확 판정 (회차별) | 정확 판정 평균 | 검토 전달 (회차별) | 검토 전달 평균 |', '|---|---|---|---|---|---|---|---|---|']
    for m in list(MUT_DESC) + ['전체']:
        per = [[x for x in s['rows'] if m == '전체' or x['mutation'] == m] for s in S]
        if not per[0]:
            continue
        k = len(per[0])
        ex = [sum(x['detected'] for x in p) for p in per]
        fl = [sum(x['flagged'] for x in p) for p in per]
        src = '/'.join(sorted({x['source'] for p in per for x in p})) if m != '전체' else ''
        name, desc = (f'**{m}**', '') if m == '전체' else (m, MUT_DESC[m])
        tgt = '' if m == '전체' else per[0][0]['target_check']
        L.append(f"| {name} | {desc} | {tgt} | {k} | {src} | {' · '.join(map(str, ex))} | {avg(ex) / k:.0%} | {' · '.join(map(str, fl))} | {avg(fl) / k:.0%} |")

    # 회차마다 target 판정이 달라진 변형 / 모든 회차에서 놓친 변형
    by_id = defaultdict(list)
    for s in S:
        for x in s['rows']:
            by_id[x['variant_id']].append(x)
    unstable = [xs for xs in by_id.values() if len({x['got'] for x in xs}) > 1]
    always = [xs for xs in by_id.values() if not any(x['detected'] for x in xs)]
    head = '| 변형 | target | 기대 | ' + ' | '.join(f'{i + 1}회' for i in range(N)) + ' | 정확 판정 | 검토 전달 |'
    sep = '|---|---|---|' + '---|' * N + '---|---|'
    def line(xs):
        return (f"| {xs[0]['variant_id']} | {xs[0]['target_check']} | {xs[0]['expected'].replace('|', '·')} | "
                + ' | '.join(x['got'] for x in xs) + f" | {sum(x['detected'] for x in xs)}/{N} | {sum(x['flagged'] for x in xs)}/{N} |")
    L += ['', f'## 회차마다 target 판정이 달라진 변형 ({len(unstable)}건)', '']
    L += ([head, sep] + [line(xs) for xs in unstable]) if unstable else ['없음']
    L += ['', f'## 모든 회차에서 정확 판정을 놓친 변형 ({len(always)}건)', '']
    if always:
        L += [head, sep] + [line(xs) for xs in always]
        L += ['', '이유 (마지막 회차의 05 reason):', '']
        L += [f"- {xs[-1]['variant_id']}: {xs[-1]['reason'] or xs[-1]['measure']}" for xs in always]
    else:
        L.append('없음')

    fps = [(i + 1, b, c, it) for i, s in enumerate(S) for b, c, it in s['fp']]
    L += ['', f'## 원본 오탐 ({len(fps)}건, {N}회 합계)', '']
    if fps:
        L += ['| 회차 | 원본 | 항목 | 이유 |', '|---|---|---|---|']
        L += [f"| {i} | {b} | {c} {it['item']} | {it['reason'].replace('|', '/')} |" for i, b, c, it in fps]
    else:
        L.append('없음')

    # 회차별 results.csv
    for d, s in zip(dirs, S):
        with open(d / 'results.csv', 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(s['rows'][0]))
            w.writeheader(); w.writerows(s['rows'])
    return [x for x in L if x is not None]


def write_outputs(labels, recs, reps, model, run_dir, base_rows=None, base_path=None, notes=()):
    """1회 실행 보고서 줄 목록을 돌려주고 results.csv를 씀."""
    base_v = {r['base_id']: verdicts(recs[r['variant_id']]) for r in labels if r['mutation'] == 'original'}
    rows = build_rows(labels, recs)
    with open(run_dir / 'results.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # (가) 탐지율
    by_mut = defaultdict(list)
    for x in rows:
        by_mut[x['mutation']].append(x)
    n_det = sum(x['detected'] for x in rows)
    n_flag = sum(x['flagged'] for x in rows)
    n_down = sum(x['downgraded'] for x in rows)

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
         *compare_block(rows, base_rows, base_path, notes),
         '## 요약', '',
         f'- **정확 판정률 {n_det}/{len(rows)} = {n_det / len(rows):.0%}** (target 판정이 기대 판정과 정확히 일치)',
         f'- **검토 전달률 {n_flag}/{len(rows)} = {n_flag / len(rows):.0%}** (target이 위반·확인필요로 나와 검토 목록에 오름. 기대가 해당없음이면 해당없음일 때 성공)',
         f'- 위반 재확인에서 위반→확인필요로 낮아진 target: **{n_down}건**',
         f'- 원본 오탐(심의 통과 광고에서 위반): **{len(fp)}건** / 원본 판정 항목 {sum(orig_check.values())}개'
         f' (확인필요 {orig_check["확인필요"]})',
         f'- 부작용(target·연관 항목 외에 원본과 달라진 항목): 변형 1건당 평균 **{sum(x["n_side_other"] for x in rows) / len(rows):.2f}개**'
         + (f' — 같은 원본을 다시 판정할 때 달라지는 기준선 {noise[0]:.2f}개' if noise else ''),
         f'- 근거 검증 통과율 {pass_rate:.1%} (해당없음 제외 {len(items)}개 항목) · 위반 재확인 호출 {rechecks}회'
         f' (재확인 {len(re_items)}건 중 위반 유지 {sum(y["final"] == "위반" for y in re_items)})',
         f'- 토큰 입력 {tin:,} / 출력 {tout:,}' + (f' · 예상 비용 ${cost:.3f} (정가, 캐시 할인 미반영)' if cost is not None else ''),
         '', '## (가) 변형 종류별 탐지율', '',
         '| 변형 | 내용 | target | 기대 판정 | 판정 주체 | 정확 판정 | 검토 전달 | 재확인으로 낮춤 |', '|---|---|---|---|---|---|---|---|']
    for m in MUT_DESC:
        xs = by_mut.get(m)
        if not xs:
            continue
        d, fl, dn = (sum(x[k] for x in xs) for k in ('detected', 'flagged', 'downgraded'))
        src = '/'.join(sorted({x['source'] for x in xs}))
        exp = ' / '.join(f"{e.replace('|', '·')} {n}" if len(set(x['expected'] for x in xs)) > 1 else e.replace('|', '·')
                         for e, n in Counter(x['expected'] for x in xs).items())
        L.append(f"| {m} | {MUT_DESC[m]} | {xs[0]['target_check']} | {exp} | {src} | {d}/{len(xs)} ({d / len(xs):.0%}) | {fl}/{len(xs)} ({fl / len(xs):.0%}) | {dn} |")
    L.append(f'| **전체** | | | | | **{n_det}/{len(rows)} ({n_det / len(rows):.0%})** | **{n_flag}/{len(rows)} ({n_flag / len(rows):.0%})** | **{n_down}** |')

    miss = [x for x in rows if not x['detected']]
    L += ['', f'## 정확 판정을 놓친 변형 ({len(miss)}건, 그중 검토 전달도 놓친 것 {sum(not x["flagged"] for x in miss)}건)', '']

    def cause(x):
        if x['source'] == 'code':
            return f"코드 규칙 ({x['got']})"
        if '재확인 불일치' in x['issues']:
            return '1차 위반 → 위반 재확인에서 확인필요로 낮춤'
        josa = {'적합': '적합으로', '위반': '위반으로', '해당없음': '해당없음으로'}.get(x['got'], f"{x['got']}로")
        return f"LLM이 {josa} 판단"
    if miss:
        L += ['| 놓친 원인 | 건수 | 변형 |', '|---|---|---|']
        for c, n in Counter(cause(x) for x in miss).most_common():
            L.append(f"| {c} | {n} | {', '.join(x['variant_id'] for x in miss if cause(x) == c)} |")
        L.append('')
        L += ['| 변형 | target | 기대 | 판정 | 검토 전달 | 판정 주체 | 이유 (05 reason) | 근거 검증 |', '|---|---|---|---|---|---|---|---|']
        for x in miss:
            why = (x['reason'] or x['measure']).replace('|', '/').replace('\n', ' ')
            L.append(f"| {x['variant_id']} | {x['target_check']} | {x['expected'].replace('|', '·')} | {x['got']} | {'O' if x['flagged'] else 'X'} | {x['source']} | {why} | {x['issues'] or '-'} |")
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
    return L


if __name__ == '__main__':
    main()
