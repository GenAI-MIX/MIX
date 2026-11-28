"""위반 예시(변형본) 만들기: 심의를 통과한 광고의 전처리 JSON을 일부러 고쳐 05 판단의 탐지율을 잰다.

  python eval/make_variants.py            # rag/ 폴더에서 실행

입력: data/processed/{image,video}/json/{원본}.json, 원본 판정 rag/outputs/{원본}__gpt-6-luna.json
출력: rag/eval/variants/{image,video}/json/{원본}__{변형}.json (원본은 {원본}__original.json)
      rag/eval/labels.csv   정답표 (variant_id, base_id, mutation, target_check, expected_verdicts, ...)
      rag/eval/variants/skipped.csv   적용하지 않은 (원본, 변형)과 이유

변형은 원본 판정에서 target 항목이 '적합'일 때만 적용한다 (이미 걸린 원본은 탐지 여부를 가릴 수 없음).
변형본의 상품유형은 원본 판정 JSON의 product를 product_override로 고정한다.
"""
import copy
import csv
import json
import re
from pathlib import Path

RAG_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = RAG_DIR.parent
EVAL_DIR = RAG_DIR / 'eval'
OUT_DIR = EVAL_DIR / 'variants'
JUDGE_MODEL = 'gpt-6-luna'          # 원본 판정(상품유형·적용 여부 확인용)
BASES = ['AD01', 'AD02', 'AD03', 'V101', 'V102', 'V103', 'V113', 'V114', 'V122', 'V126']

C2_TEXT = '목표수익률 연 15% 달성!'
C3_TEXT = '국내 유일 최고 수익 ETF'
SEP = r'※*.。|'                      # 고지문 조항 구분에 쓰이는 기호


def fx(word):
    """글자 사이에 공백이 끼어도 맞는 정규식 (OCR 띄어쓰기 대응)."""
    return r'\s*'.join(re.escape(ch) for ch in word)


def alt(*words):
    return '(?:' + '|'.join(fx(w) for w in words) + ')'


def gap(n):
    """조항 안의 최대 n글자. 구분 기호와 문장 끝('~니다')은 넘지 않음 (앞 조항까지 지우지 않게)."""
    return rf'(?:(?!니\s?다)(?!(?<=니)다)(?!(?<=니 )다)[^{SEP}]){{0,{n}}}?'


# 변형별: 지울 조항 구간 → 남은 조각 정리 → 남으면 안 되는 단어 (공백 제거 후 비교)
DELETE = {
    'm_B1': {
        'spans': [
            r'\(?\s*간\s*이\s*\)?\s*' + alt('투자설명서', '설명서') + gap(40) + alt('필독', '질독', '읽어보시기바랍니다', '읽어보시기', '읽어'),
            alt('설명서', '약관') + gap(40) + alt('필독', '질독', '읽어보시기바랍니다', '읽어보시기', '읽어'),
        ],
        'residue': r'(?:\(?\s*간\s*이\s*\)?\s*)?(?:투\s*자\s*)?' + alt('설명서', '약관', '필독', '질독', '읽어보') + rf'[^\s{SEP}]{{0,8}}',
        'forbid': ['설명서', '약관', '필독', '읽어보'],
    },
    'm_B3': {
        'spans': [
            gap(60) + alt('원금의손실', '원금손실', '손실') + gap(80) + alt('귀속됩니다', '귀속'),
        ],
        'residue': rf'[^\s{SEP}]{{0,6}}' + alt('손실', '귀속') + rf'[^\s{SEP}]{{0,8}}',
        'forbid': ['손실', '귀속'],
    },
    'm_B6': {
        'spans': [
            r'(?:이\s*)?(?:금\s*융\s*상\s*품\s*(?:<[^>]{0,12}>)?\s*[은는]?\s*)?' + alt('예금자보호법', '예금자보호') + gap(30) + alt('보호되지않습니다', '보호하지않습니다', '보호되지않', '보호되지', '보호'),
        ],
        'residue': rf'[^\s{SEP}]{{0,6}}' + alt('예금자보호', '예금보험', '보호되지않', '호되지않', '보호하지않') + rf'[^\s{SEP}]{{0,8}}',
        'forbid': ['예금자보호', '예금보험', '보호되지않', '보호하지않'],
    },
    'm_D1': {
        'spans': [
            r'(?:※\s*)?(?:한\s*국\s*)?(?:금\s*융\s*투\s*자\s*협\s*회\s*)?(?:준\s*법\s*감\s*시\s*인\s*)?'
            + alt('심사필', '심의필') + r'\s*제?\s*[\d\-\s]*호?\s*(?:\([^)]{0,40}\)?)?',
        ],
        'residue': r'[^\s※|]*' + alt('심사필', '심의필', '협회심사') + r'[^\s※|]*',
        'forbid': ['심사필', '심의필'],
    },
}

# 다른 필수 문구가 함께 지워졌는지 보는 검출식 (05 code_checks와 같은 키워드, 공백 제거 후)
PRESENT = {
    'B1': lambda t: '설명서' in t and ('필독' in t or '읽어' in t),
    'B3': lambda t: '원금손실' in t or ('손실' in t and '투자자' in t),
    'B5': lambda t: '충분한설명' in t or '설명청취' in t,
    'B6': lambda t: '예금자보호' in t,
    'D1': lambda t: bool(re.search(r'(심사필|심의필)제?[\d\-]+호', t)),
}

MUTATIONS = {   # 이름: (target 항목, 기대 판정, 매체)
    'm_A2': ('A2', ['확인필요'], 'all'),
    'm_A3': ('A3', ['확인필요'], 'image'),
    'm_A5': ('A5', ['확인필요'], 'video'),
    'm_B1': ('B1', ['위반'], 'all'),
    'm_B3': ('B3', ['위반'], 'all'),
    'm_B6': ('B6', ['위반'], 'all'),
    'm_C2': ('C2', ['위반'], 'all'),
    'm_C3': ('C3', ['확인필요', '위반'], 'all'),
    'm_D1': ('D1', ['확인필요'], 'all'),
}


def norm(s):
    return re.sub(r'\s+', '', str(s or ''))


def text_fields(ad):
    """(dict, key) 목록: 문구가 들어 있는 모든 곳."""
    out = []
    for k in ('paragraphs', 'ocr_blocks', 'cards', 'risk_exposures'):
        out += [(x, 'text') for x in ad.get(k) or [] if isinstance(x.get('text'), str)]
    return out


def all_text(ad):
    parts = [d[k] for d, k in text_fields(ad)]
    parts += list(ad.get('review_no') or [])
    parts += [str((ad.get('csv') or {}).get('review_no') or '')]
    return norm('\n'.join(parts))


def scrub(text, rule):
    for pat in rule['spans']:
        text = re.sub(pat, ' ', text)
    text = re.sub(rule['residue'], ' ', text)
    text = re.sub(r'(\s*※\s*){2,}', ' ※ ', text)          # 지운 자리에 남은 기호 정리
    return re.sub(r'[ \t]{2,}', ' ', text).strip()


def delete_text(ad, mut):
    """반환: (남은 금지어, 함께 지워진 다른 필수 문구). 부수 삭제는 필드(문단·블록·카드)마다 전후를 비교."""
    rule, target = DELETE[mut], MUTATIONS[mut][0]
    collateral = set()
    for d, k in text_fields(ad):
        before, d[k] = d[k], scrub(d[k], rule)
        collateral |= {c for c, f in PRESENT.items() if c != target and f(norm(before)) and not f(norm(d[k]))}
    # 빈 문단·블록·카드 문구는 지움 (측정값 metrics는 그대로 두어 A 항목에 영향 없게)
    ad['paragraphs'] = [p for p in ad.get('paragraphs', []) if norm(p['text'])]
    keep = {b['id'] for b in ad.get('ocr_blocks', []) if norm(b['text'])}
    ad['ocr_blocks'] = [b for b in ad.get('ocr_blocks', []) if b['id'] in keep]
    for p in ad['paragraphs']:
        if 'block_ids' in p:
            p['block_ids'] = [i for i in p['block_ids'] if i in keep]
    if mut == 'm_D1':
        ad['review_no'] = [] if 'review_no' in ad else None
        if isinstance(ad.get('csv'), dict):
            ad['csv']['review_no'] = None
    left = [w for w in rule['forbid'] if w in all_text(ad)]
    if mut == 'm_D1' and (ad.get('review_no') or PRESENT['D1'](all_text(ad))):
        left.append('review_no')
    return left, sorted(collateral)


def add_headline(ad, text):
    """맨 앞에 헤드라인 문단과 OCR 블록 추가 (LLM에 넘기는 문구 목록의 앞쪽에 들어가도록)."""
    w = (ad.get('size') or [1080, 1080])[0]
    bid = max([b['id'] for b in ad.get('ocr_blocks', []) if isinstance(b.get('id'), int)] or [0]) + 1000
    bbox = [int(w * 0.08), 20, int(w * 0.92), 90]
    block = {'id': bid, 'text': text, 'bbox': bbox, 'conf': 0.99, 'type': 'headline', 'h_px': 60, 'contrast': 7.0}
    para = {'id': f'eval_{bid}', 'type': 'headline', 'text': text, 'bbox': bbox, 'block_ids': [bid]}
    if 'duration' in ad:     # 영상: 첫 화면에 붙임
        sid = (ad.get('screens') or [{'id': 0}])[0]['id']
        block['screen_id'] = para['screen_id'] = sid
    ad.setdefault('ocr_blocks', []).insert(0, block)
    ad.setdefault('paragraphs', []).insert(0, para)
    return [] if norm(text) in all_text(ad) else ['추가 실패']


def mutate(ad, mut):
    m = ad.setdefault('metrics', {})
    if mut == 'm_A2':
        m['risk_contrast_min'] = 1.5
    elif mut == 'm_A3':
        mf = m.setdefault('mobile_feed', {})
        for k in ('risk_median_pt', 'risk_min_pt'):
            if mf.get(k) is not None:
                mf[k] = round(mf[k] * 0.5, 2)
        if m.get('risk_headline_ratio') is not None:
            m['risk_headline_ratio'] = round(m['risk_headline_ratio'] * 0.5, 4)
    elif mut == 'm_A5':
        m['risk_exposure_s'] = round(ad['duration'] * 0.2, 2)
    elif mut in DELETE:
        return delete_text(ad, mut)
    elif mut == 'm_C2':
        return add_headline(ad, C2_TEXT), []
    elif mut == 'm_C3':
        return add_headline(ad, C3_TEXT), []
    return [], []


def main():
    labels, skipped = [], []
    for kind in ('image', 'video'):
        (OUT_DIR / kind / 'json').mkdir(parents=True, exist_ok=True)
    for base in BASES:
        kind = 'video' if base.startswith('V') else 'image'
        src = json.loads((REPO_DIR / f'data/processed/{kind}/json/{base}.json').read_text(encoding='utf-8'))
        judged = json.loads((RAG_DIR / f'outputs/{base}__{JUDGE_MODEL}.json').read_text(encoding='utf-8'))
        verdict = {it['check_id']: it['verdict'] for it in judged['items']}
        before = {c: f(all_text(src)) for c, f in PRESENT.items()}

        def write(vid, ad, mutation, target='', expected=(), collateral=(), note=''):
            ad['ad_id'], ad['base_id'], ad['eval_mutation'] = vid, base, mutation
            ad['product_override'] = judged['product']
            (OUT_DIR / kind / 'json' / f'{vid}.json').write_text(json.dumps(ad, ensure_ascii=False, indent=1), encoding='utf-8')
            labels.append({'variant_id': vid, 'base_id': base, 'mutation': mutation, 'target_check': target,
                           'expected_verdicts': '|'.join(expected), 'original_verdict': verdict.get(target, ''),
                           'product': judged['product'], 'media': kind,
                           'collateral_removed': '|'.join(collateral), 'note': note})

        write(f'{base}__original', copy.deepcopy(src), 'original')
        for mut, (target, expected, media) in MUTATIONS.items():
            if media != 'all' and media != kind:
                continue                                            # 매체가 맞지 않음 (조용히 건너뜀)
            if verdict.get(target) != '적합':
                skipped.append({'base_id': base, 'mutation': mut, 'target_check': target,
                                'reason': f'원본 판정이 {verdict.get(target, "항목 없음")} (적합일 때만 적용)'})
                continue
            ad = copy.deepcopy(src)
            left, collateral = mutate(ad, mut)
            if left:
                raise SystemExit(f'{base} {mut}: 지운 뒤에도 남음 → {left}')
            after = {c: f(all_text(ad)) for c, f in PRESENT.items()}
            collateral = sorted(set(collateral) | {c for c in PRESENT if before[c] and not after[c] and c != target})
            write(f'{base}__{mut}', ad, mut, target, expected, collateral)

    with open(EVAL_DIR / 'labels.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(labels[0]))
        w.writeheader(); w.writerows(labels)
    with open(OUT_DIR / 'skipped.csv', 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['base_id', 'mutation', 'target_check', 'reason'])
        w.writeheader(); w.writerows(skipped)
    n_var = sum(r['mutation'] != 'original' for r in labels)
    print(f'원본 {len(BASES)}건 + 변형 {n_var}건 → {OUT_DIR} | 건너뜀 {len(skipped)}건')
    col = [r for r in labels if r['collateral_removed']]
    if col:
        print('다른 필수 문구까지 지워진 변형:', ', '.join(f"{r['variant_id']}({r['collateral_removed']})" for r in col))


if __name__ == '__main__':
    main()
