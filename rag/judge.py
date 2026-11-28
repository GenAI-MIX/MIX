"""05 LLM 판단 모듈.

05_llm_judge.ipynb의 판정 로직을 노트북과 명령줄 실행기(run_judge.py)가 함께 쓰도록 옮긴 파일입니다.

    from judge import judge_ad
    judge_ad('AD01', 'openai')          # 결과는 rag/outputs/{ad_id}__{model}.json · .csv

처음 import할 때 rag_search(BGE-M3 + 리랭커)를 불러오므로 30초~1분 걸립니다.
"""
import datetime
import hashlib
import json
import os
import re
import threading
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from rag_search import card_by_id, search_ids, search_ids_cached

# ---------- 1. 준비 ----------
RAG_DIR = Path(__file__).resolve().parent   # rag/ 폴더
DATA_DIR = RAG_DIR / 'data'
OUT_DIR = RAG_DIR / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
load_dotenv(RAG_DIR / '.env')


def find_repo_dir():
    # 광고 JSON(data/processed/...)이 있는 MIX 레포 폴더를 찾습니다.
    # 1) .env의 MIX_REPO_DIR  2) 이 폴더와 상위 폴더 중 data/processed가 있는 곳
    env = os.getenv('MIX_REPO_DIR')
    cands = [Path(env).expanduser()] if env else []
    cands += [RAG_DIR, *RAG_DIR.parents]
    for d in cands:
        if (d / 'data/processed/image/json').is_dir() or (d / 'data/processed/video/json').is_dir():
            return d.resolve()
    raise FileNotFoundError('광고 JSON 폴더(data/processed/{image,video}/json)를 찾지 못했습니다. '
                            'rag/.env에 MIX_REPO_DIR=/path/to/MIX 를 적어 주세요.')


REPO_DIR = find_repo_dir()

# 개발 중에는 저가 모델, 최종 비교 때 중간 모델로 바꿉니다
MODELS = {
    'openai': 'gpt-6-luna',            # 최종 비교: 'gpt-6.2-sol'
    'claude': 'claude-haiku-5-5',      # 최종 비교: 'claude-sonnet-5-5'
}
API_KEY_ENV = {'openai': 'OPENAI_API_KEY', 'claude': 'ANTHROPIC_API_KEY'}
TODAY = datetime.date.today()

# ---------- 2. 검사기준표 ----------
checks = pd.read_csv(DATA_DIR / 'checks.csv', encoding='utf-8-sig').fillna('')

TH = {
    'contrast': 3.0,          # A2 명암비
    'image_pt': 5.9,          # A3 인쇄형 글자 크기 — 확정: A4 폭 비례 환산 (A4 9pt를 폭 390 기준으로 환산)
    'ratio_image': 0.075,     # A4 이미지: 실제 광고 하위 10%
    'ratio_video': 0.10,      # A4 영상: 실제 광고 하위 10%
    'exposure_frac': 1/3,     # A5 영상 노출 비율 (법 기준)
    'cps': 197,               # A6 초당 글자 수: 실제 광고 상위 10%
    'video_pt': 5.06,         # A7 영상 글자 크기 (가로 화면): 실제 광고 하위 10%
}
VERDICTS = ['위반', '확인필요', '적합', '해당없음']

# ---------- 3. 상품유형 ----------
PRODUCT_OVERRIDE = {
    # 'V112': '투자성',
}

# 상품유형은 2단계로 정합니다.
#   0) PRODUCT_OVERRIDE → JSON의 product('ETF', '증권') 순으로 우선
#   1) 키워드: 유형별 키워드를 모두 찾아서, 한 유형만 나오고 '연금'이 없으면 그 유형으로 확정
#   2) LLM: 여러 유형이 섞이거나, '연금'이 있거나, 아무 유형도 안 나오면 광고 문구 앞부분과 브랜드로 LLM에 묻기
#      (결과는 rag/cache/product_cache.json에 저장해 재사용)
PRODUCT_TYPES = ['투자성', '예금성', '대출성', '보장성', '범위밖']
KEYWORDS = {
    '투자성': r'ETF|펀드|자산운용|ISA|주식|투자|증권|ETN|ELS|채권|IRP|퇴직연금|(?<![A-Za-z])DC(?![A-Za-z])|TDF|실적배당|금융투자상품',
    '보장성': r'보험|생명|화재|해상|손해',
    '대출성': r'대출|중금리|신용대출|카드론|담보',
    '예금성': r'적금|예금|통장|파킹|정기예',
    '범위밖': r'카드|요금제|통신|모바일\s*요금',
}
PENSION_RE = r'연금'
# 상품이 아니라 고지문·세금 안내에 늘 나오는 문구는 키워드 검색 전에 뺍니다
# (예금자보호 고지문의 '예금'·'예금보험공사', 세금 안내의 건강·연금보험료)
NOT_PRODUCT_RE = r'예금자\s*보호법?|예금보험공사|예금보험|국민건강보험료?|건강보험료?|연금보험료'
PRODUCT_MODEL = 'gpt-6-luna'
PRODUCT_TEXT_LIMIT = 1500
PRODUCT_CACHE_PATH = RAG_DIR / 'cache' / 'product_cache.json'
PRODUCT_SYSTEM = """당신은 한국 금융광고의 금융상품 유형을 분류합니다.
금융소비자보호법의 유형 중 하나를 고릅니다.
- 투자성: 펀드, ETF, 주식·증권 계좌, IRP·퇴직연금(DC 등), TDF, 실적배당형 상품, 연금저축펀드 등 원금 손실 가능성이 있는 금융투자상품
- 예금성: 예금, 적금, 통장, 파킹통장 등
- 대출성: 대출, 카드론, 담보대출 등
- 보장성: 보험 (종신·건강·연금저축보험 등)
- 범위밖: 금융상품이 아닌 광고 또는 위 유형으로 보기 어려운 광고 (신용카드, 통신 요금제 등)
광고한 회사(보험사·은행·증권사)의 업종이 아니라 광고가 파는 상품으로 판단합니다.
JSON으로만 답합니다: {"product": "투자성|예금성|대출성|보장성|범위밖 중 하나", "reason": "이유 한 줄"}"""
_product_lock = threading.Lock()


def ad_text(ad, limit=None):
    texts = [p['text'] for p in ad.get('paragraphs', [])]
    texts += [c['text'] for c in ad.get('cards', []) if c.get('kind') == 'crawl']
    s = '\n'.join(texts)
    return s[:limit] if limit else s


def keyword_types(ad):
    """1단계: 브랜드·제목·광고 문구(앞 3000자)에서 찾은 유형 목록과 '연금' 여부."""
    head = ' '.join(str(ad.get(k) or '') for k in ('brand', 'title'))
    text = re.sub(NOT_PRODUCT_RE, '', head + '\n' + ad_text(ad, 3000))
    found = [t for t in PRODUCT_TYPES if re.search(KEYWORDS[t], text)]
    return found, bool(re.search(PENSION_RE, text))


def _load_product_cache():
    try:
        return json.loads(PRODUCT_CACHE_PATH.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


_product_cache = None


def llm_product(ad):
    """2단계: LLM 분류. 같은 광고 문구·모델·프롬프트면 캐시를 씀. 반환 (유형, 이유)."""
    global _product_cache
    text = ad_text(ad, PRODUCT_TEXT_LIMIT)
    user = f"브랜드: {ad.get('brand') or '(없음)'}\n\n광고 문구 (앞부분 {PRODUCT_TEXT_LIMIT}자):\n{text}"
    key = ad['ad_id'] + ':' + hashlib.sha256((PRODUCT_MODEL + PRODUCT_SYSTEM + user).encode()).hexdigest()[:16]
    with _product_lock:
        if _product_cache is None:
            _product_cache = _load_product_cache()
        if key in _product_cache:
            c = _product_cache[key]
            return c['product'], c['reason']
    out, _ = call_llm('openai', PRODUCT_SYSTEM, user, PRODUCT_MODEL)
    ptype, reason = out.get('product'), str(out.get('reason', '')).strip()
    if ptype not in PRODUCT_TYPES:
        reason = f'LLM 답이 유형 목록에 없음({ptype}) — 전체로 처리. {reason}'
        ptype = '전체'
    with _product_lock:
        _product_cache = {**_load_product_cache(), **_product_cache}   # 다른 프로세스가 저장한 것도 합침
        _product_cache[key] = {'ad_id': ad['ad_id'], 'product': ptype, 'reason': reason, 'model': PRODUCT_MODEL}
        PRODUCT_CACHE_PATH.parent.mkdir(exist_ok=True)
        tmp = PRODUCT_CACHE_PATH.with_suffix(f'.tmp{os.getpid()}')
        tmp.write_text(json.dumps(_product_cache, ensure_ascii=False, indent=1), encoding='utf-8')
        os.replace(tmp, PRODUCT_CACHE_PATH)
    return ptype, reason


def infer_product_detail(ad):
    """{'product', 'by'(override/json/keyword/llm), 'reason', 'keywords'(1단계에서 찾은 유형), 'pension'}"""
    found, pension = keyword_types(ad)
    base = {'keywords': found, 'pension': pension}
    if ad['ad_id'] in PRODUCT_OVERRIDE:
        return {'product': PRODUCT_OVERRIDE[ad['ad_id']], 'by': 'override', 'reason': 'PRODUCT_OVERRIDE', **base}
    if ad.get('product_override'):          # 광고 JSON에 고정한 상품유형 (위반 예시는 원본 판정값)
        return {'product': ad['product_override'], 'by': 'override', 'reason': '광고 JSON product_override', **base}
    if ad.get('product') in ('ETF', '증권'):
        return {'product': '투자성', 'by': 'json', 'reason': f"전처리 JSON product={ad.get('product')}", **base}
    if len(found) == 1 and not pension:
        return {'product': found[0], 'by': 'keyword', 'reason': f'키워드: {found[0]}만 검출', **base}
    ptype, reason = llm_product(ad)
    return {'product': ptype, 'by': 'llm', 'reason': reason, **base}


def infer_product(ad):
    d = infer_product_detail(ad)
    return d['product'], d['by']


AD_DIR = REPO_DIR / 'data/processed'   # 광고 JSON 폴더: {AD_DIR}/{image,video}/json/{ad_id}.json


def set_ad_dir(path):
    """광고 JSON을 다른 폴더(예: 위반 예시 rag/eval/variants)에서 읽게 바꿉니다."""
    global AD_DIR, ALL_IDS
    AD_DIR = Path(path).expanduser().resolve()
    ALL_IDS = sorted(list_ids('image') + list_ids('video'))


def load_ad(ad_id):
    kind = 'video' if ad_id.startswith('V') else 'image'
    return json.loads((AD_DIR / f'{kind}/json/{ad_id}.json').read_text(encoding='utf-8'))


def list_ids(kind):
    """kind: 'image' 또는 'video'. 해당 폴더의 광고 ID 목록."""
    return sorted(p.stem for p in (AD_DIR / f'{kind}/json').glob('*.json'))


ALL_IDS = sorted(list_ids('image') + list_ids('video'))

# ---------- 4. 광고 정리 ----------
def media_of(ad):
    if 'duration' in ad:
        return ad.get('media') or 'video'
    return ad.get('media') or 'image'


def is_video(ad):
    return 'duration' in ad


def summarize_ad(ad, ptype):
    paras = ad.get('paragraphs', [])
    risk = [p['text'] for p in paras if p['type'] == 'risk_notice']
    risk += [c['text'] for c in ad.get('cards', []) if c.get('type') == 'risk_notice' and c.get('kind') == 'crawl']
    other = [f"[{p['type']}] {p['text']}" for p in paras if p['type'] != 'risk_notice']
    return {
        'ad_id': ad['ad_id'], '상품유형': ptype, '매체': media_of(ad), '브랜드': ad.get('brand'),
        '제목': ad.get('title'), '영상 길이(초)': ad.get('duration'),
        '위험고지 문구': list(dict.fromkeys(risk))[:20],
        '그 밖의 문구': list(dict.fromkeys(other))[:60],
        '측정값': ad.get('metrics', {}),
    }


def applicable(row, ptype, media):
    p = row['products']
    ok_p = p == '전체' or (p.startswith('!') and ptype != p[1:]) or p == ptype
    if ptype == '범위밖':
        ok_p = p == '전체'
    m = row['media']
    ok_m = (m == 'all' or (m == 'video' and media in ('video', 'tv_cf', 'online_video'))
            or (m == 'image' and media in ('image', 'poster', 'banner')) or media in m.split(','))
    return ok_p and ok_m

# ---------- 5. 코드 판정 ----------
REVIEW_RE = re.compile(r'(심사필|심의필)\s*제?\s*[\d\-]+\s*호')
DATE_RE = re.compile(r'(20\d\d)[.\-](\d{1,2})[.\-](\d{1,2})')


def code_checks(ad, ptype):
    m, media, out = ad.get('metrics', {}), media_of(ad), {}
    video = is_video(ad)
    text = ad_text(ad)
    n_risk = m.get('n_risk_cards') if video else sum(p['type'] == 'risk_notice' for p in ad.get('paragraphs', []))
    out['A1'] = ('확인필요', f'위험고지 문단 미검출 (n={n_risk})') if not n_risk else ('적합', f'위험고지 문단 {n_risk}개 검출')

    c = m.get('risk_contrast_min')
    out['A2'] = ('확인필요', '명암비 측정 불가(null) — 이미지 확인 필요') if c is None else \
                (('확인필요' if c < TH['contrast'] else '적합'), f'최소 명암비 {c:.2f} (기준 {TH["contrast"]})')
    if not video:
        pt = m.get('mobile_feed', {}).get('risk_median_pt')
        if pt is not None:
            out['A3'] = (('확인필요' if pt < TH['image_pt'] else '적합'), f'위험고지 글자 {pt:.2f}pt (기준 {TH["image_pt"]}pt, 폭 390 환산)')
        r, th = m.get('risk_headline_ratio'), TH['ratio_image']
    else:
        r, th = m.get('risk_headline_ratio_min'), TH['ratio_video']
    if r is not None:
        out['A4'] = (('확인필요' if r < th else '적합'), f'위험고지/헤드라인 크기 비율 {r:.3f} (기준 {th})')

    if video:
        d, e = ad.get('duration') or 0, m.get('risk_exposure_s') or 0
        if d:
            frac = e / d
            out['A5'] = (('적합' if frac >= TH['exposure_frac'] else '확인필요'),
                         f'위험고지 노출 {e:.1f}초 / 영상 {d:.1f}초 = {frac:.0%} (기준 1/3, 미만이면 음성 고지 확인 필요)')
        cps = m.get('risk_cps_max')
        if cps is not None:
            out['A6'] = (('확인필요' if cps > TH['cps'] else '적합'), f'최대 초당 글자 수 {cps:.0f}자 (기준 {TH["cps"]}자)')
        vp = (m.get('risk_display_min_pt') or {}).get('mobile_landscape')
        if vp is not None:
            out['A7'] = (('확인필요' if vp < TH['video_pt'] else '적합'), f'가로 화면 기준 최소 글자 {vp:.2f}pt (기준 {TH["video_pt"]}pt)')
    if media == 'banner':
        out['A8'] = ('해당없음', '정지 이미지 배너 — 위험고지가 계속 표시됨')

    review = ad.get('review_no') or [x.group(0) for x in REVIEW_RE.finditer(text)]
    if review:
        out['D1'] = ('적합', f'심사필 표시 검출: {review[0]}')
    elif media == 'tv_cf':
        out['D1'] = ('해당없음', '방송 광고 — 공간 제약 시 심사필 표시 생략 가능')
    else:
        out['D1'] = ('확인필요', '심사필 번호 미검출 (OCR 누락 가능)')
    dates = [datetime.date(int(y), int(mo), int(dd)) for y, mo, dd in DATE_RE.findall(text) if 1 <= int(mo) <= 12 and 1 <= int(dd) <= 31]
    if review and dates:
        end = max(dates)
        out['D2'] = (('적합' if end >= TODAY else '확인필요'), f'표시된 가장 늦은 날짜 {end} (판정일 {TODAY}, 게시일 정보 없음)')
    else:
        out['D2'] = ('확인필요', '유효기간 날짜 미검출')

    # 필수 문구 키워드 선검사: 광고 전체 텍스트(공백 제거)에서 찾고, 있으면 적합. 없으면 LLM이 판정
    flat = re.sub(r'\s+', '', text)
    has = lambda *ks: [k for k in ks if k in flat]
    if has('설명서') and has('필독', '읽어'):
        out['B1'] = ('적합', '키워드 검출: ' + ', '.join(has('설명서', '필독', '읽어')))
    if ptype == '투자성':
        hit = has('원금손실') or (['손실', '투자자'] if has('손실') and has('투자자') else [])   # '원금 손실'은 공백 제거로 포함
        if hit:
            out['B3'] = ('적합', '키워드 검출: ' + ', '.join(hit))
    if has('예금자보호'):
        out['B6'] = ('적합', '키워드 검출: 예금자보호')
    if has('충분한설명', '설명청취'):
        out['B5'] = ('적합', '키워드 검출: ' + ', '.join(has('충분한설명', '설명청취')))
    return out

# ---------- 6. 근거 카드 검색 ----------
_search_lock = threading.Lock()   # 임베딩·리랭커 모델은 한 번에 한 스레드만 (run_judge --workers)


def ref_ids(s):
    ids = []
    for tok in re.split(r'[,;]', s):
        tok = re.sub(r'\(.*?\)|참고:', '', tok).strip().strip("'")
        tok = tok.split(' ')[0]
        if tok in card_by_id.index:
            ids.append(tok)
    return ids


def retrieve(rows, ptype, k=8, cap=16, use_cache=True):
    # use_cache: 검색 결과를 rag/cache/search_cache.json에 저장·재사용 (결과는 search_ids와 같음)
    search = search_ids_cached if use_cache else search_ids
    score = {}
    with _search_lock:
        for _, r in rows.iterrows():
            for cid in ref_ids(r['ref_cards']):
                score[cid] = score.get(cid, 0) + 100          # 기준표 근거 카드는 항상 포함
            for cid, s in search(r['query'], ptype, k=k):
                score[cid] = score.get(cid, 0) + 1
    return sorted(score, key=score.get, reverse=True)[:cap]

# ---------- 7. 프롬프트 ----------
SYSTEM = '''당신은 한국 금융광고 사전 심의를 돕는 검토자입니다.
광고 정보, 검사 항목, 규정 카드를 받아 항목마다 판정합니다.

규칙
1. 판정은 "위반", "확인필요", "적합", "해당없음" 중 하나입니다.
   - 위반: 법 기준이 명확하고 광고가 그것을 어긴 경우
   - 확인필요: 법에 수치가 없어 우리 서비스 기준으로 걸렸거나, 생략이 허용될 수 있거나, 문구가 OCR로 일부만 읽혀 판단이 불확실한 경우
   - 적합: 기준을 충족한 경우
   - 해당없음: 이 광고에 적용되지 않는 항목 (예: 운용실적 표시가 없는데 운용실적 요건 항목)
2. "코드 판정"이 주어진 항목은 그 판정을 그대로 쓰고, 근거 조항과 이유만 작성합니다.
3. 근거는 반드시 아래 "규정 카드" 중에서만 고르고 card_ids에 카드 ID를 적습니다.
4. quote에는 근거 카드 본문에 실제로 있는 문장을 한 글자도 바꾸지 말고 그대로 옮깁니다(40~150자). 없는 문장을 만들지 마세요.
5. 광고 문구는 OCR 결과라 띄어쓰기·오타가 있을 수 있습니다. 뜻이 같으면 포함된 것으로 봅니다.
6. reason에는 광고의 어떤 문구(또는 측정값) 때문에 그렇게 판정했는지 1~3문장으로 씁니다.
7. 같은 뜻의 표현이면 포함된 것으로 봅니다. 예: "(간이)투자설명서·집합투자규약 필독"은 설명서를 읽어볼 것을 권유한 것입니다.
8. "위반"은 (가) 법 기준이 명확하고 (나) 광고 문구로 어긴 것이 확실할 때만 씁니다.
   표현의 강도가 애매하거나(예: "독보적", "대표") 화면 속 예시 이미지의 숫자처럼 해석이 갈리면 "확인필요"입니다.
9. ad_quote에는 판정의 근거가 된 광고 문구를 광고 정보에서 그대로 옮깁니다. 필요한 문구가 없어서 내린 판정이면 빈 문자열("")로 둡니다.
   코드 판정이 주어진 항목은 ad_quote를 빈 문자열("")로 둡니다.
10. 아래 JSON 형식으로만 답합니다.

{"items": [{"check_id": "...", "verdict": "...", "card_ids": ["..."], "quote": "...", "ad_quote": "...", "reason": "..."}],
 "summary": "이 묶음의 판단 요약 1~2문장"}'''


def card_block(cid, maxlen=1500):
    c = card_by_id.loc[cid]
    body = c['본문'] if len(c['본문']) <= maxlen else c['본문'][:maxlen] + '…(이하 생략)'
    return f"### {cid} | {c['문서명']} {c['조']} {c['조 제목']} | 상품유형: {c['상품유형']}\n{body}"


def build_prompt(ad_info, rows, codes, card_ids):
    items = []
    for _, r in rows.iterrows():
        it = {'check_id': r['check_id'], '검사 항목': r['item'], '법 기준': r['law'], '우리 서비스 기준': r['our_rule'],
              '가능한 판정': r['verdicts']}
        if r['check_id'] in codes:
            it['코드 판정'] = {'판정': codes[r['check_id']][0], '근거 측정값': codes[r['check_id']][1]}
        items.append(it)
    return ('# 광고 정보\n' + json.dumps(ad_info, ensure_ascii=False, indent=1) +
            '\n\n# 검사 항목\n' + json.dumps(items, ensure_ascii=False, indent=1) +
            '\n\n# 규정 카드\n' + '\n\n'.join(card_block(c) for c in card_ids) +
            '\n\n위 검사 항목 전부에 대해 JSON으로 답하세요.')

# ---------- 8. LLM 호출 ----------
def _parse_json(text):
    text = text.strip()
    text = re.sub(r'^```(json)?|```$', '', text, flags=re.M).strip()
    start, end = text.find('{'), text.rfind('}')
    return json.loads(text[start:end + 1])


def call_llm(provider, system, user, model=None):
    model = model or MODELS[provider]
    if provider == 'openai':
        from openai import OpenAI
        r = OpenAI().chat.completions.create(
            model=model, response_format={'type': 'json_object'},
            messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': user}])
        text, usage = r.choices[0].message.content, (r.usage.prompt_tokens, r.usage.completion_tokens)
    elif provider == 'claude':
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=model, max_tokens=8000, system=system, messages=[{'role': 'user', 'content': user}])
        text, usage = r.content[0].text, (r.usage.input_tokens, r.usage.output_tokens)
    else:
        raise ValueError(provider)
    return _parse_json(text), {'model': model, 'input_tokens': usage[0], 'output_tokens': usage[1]}

# ---------- 9. 근거 검증 ----------
def _norm(s):
    return re.sub(r'\s+', '', str(s))


QUOTE_SPLIT_RE = re.compile(r'…|\.\.\.|\n')
NOTE_RE = re.compile(r'<[^<>]*>')          # '<개정 2021.4.15.>', '<신설 ...>' 같은 꺾쇠 주석


def _norm_card(s):
    return _norm(NOTE_RE.sub('', str(s)))


def quote_in_cards(quote, card_ids):
    # 인용을 '…', '...', 줄바꿈으로 조각낸 뒤, 공백 제거 후 5자 이상인 조각이 전부
    # 인용한 카드 중 어느 하나의 원문에 있으면 통과
    pieces = [p for p in (_norm_card(s) for s in QUOTE_SPLIT_RE.split(str(quote))) if len(p) >= 5]
    bodies = [_norm_card(card_by_id.loc[c, '본문']) for c in card_ids if c in card_by_id.index]
    return bool(pieces) and all(any(p in b for b in bodies) for p in pieces)


def ad_text_norm(ad, info):
    """ad_quote 검증에 쓰는 광고 전체 문구 (공백 제거)."""
    return _norm(json.dumps(info, ensure_ascii=False)) + _norm(ad_text(ad))


def validate(item, given_ids, codes, ad_norm):
    issues = []
    if item.get('verdict') not in VERDICTS:
        issues.append(f"판정값 이상({item.get('verdict')})")
    cid = item.get('check_id')
    if cid in codes and item.get('verdict') != codes[cid][0]:
        issues.append(f'코드 판정({codes[cid][0]})을 바꿈')
        item['verdict'] = codes[cid][0]
    bad = [c for c in item.get('card_ids', []) if c not in given_ids]
    if bad:
        issues.append(f'주지 않은 카드 인용({",".join(bad)})')
    q = _norm(item.get('quote', ''))
    if item.get('verdict') != '해당없음':
        if not q:
            issues.append('인용 없음')
        elif not quote_in_cards(item.get('quote', ''), item.get('card_ids', [])):
            issues.append('인용 문장이 카드 원문에 없음')
    aq = [p for p in (_norm(s) for s in str(item.get('ad_quote', '')).split(' / ')) if p]   # ' / '로 이은 여러 문구는 조각마다 확인
    if any(p not in ad_norm for p in aq):
        issues.append('광고 문구 인용이 광고에 없음')
    item['issues'] = '; '.join(issues)
    return item

# ---------- 10. 광고 1건 판정 ----------
GROUP_NAME = {'AD': '표시 방법·심의 표시', 'B': '필수 내용', 'CE': '금지 표현·추천보증'}
C5_FROM = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'C3']   # 이 중 하나라도 확인필요·위반이면 C5 = 확인필요
CODE_AFTER = {'C5'}   # LLM 답 뒤에 코드가 덮어쓰는 항목 (위반 재확인 대상에서 제외)


def output_name(ad_id, model):
    return f"{ad_id}__{model}.json".replace('/', '_')


def judge_ad(ad_id, provider='openai', model=None, verbose=True, out_dir=None, return_record=False):
    """광고 1건 판정. 결과 표(DataFrame)를 돌려주고 out_dir(기본 rag/outputs)에 .json·.csv로 저장.
    return_record=True면 (표, 저장한 JSON 내용)을 돌려줍니다."""
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ad = load_ad(ad_id)
    pinfo = infer_product_detail(ad)
    ptype, by = pinfo['product'], pinfo['by']
    media = media_of(ad)
    info = summarize_ad(ad, ptype)
    ad_norm = ad_text_norm(ad, info)
    codes = code_checks(ad, ptype)
    rows_all = checks[checks.apply(lambda r: applicable(r, ptype, media), axis=1)]
    results, summaries, usage, rechecks, retrieved = [], {}, [], [], {}

    for g in ['AD', 'B', 'CE']:
        rows = rows_all[(rows_all['group'] == g) & (rows_all['mode'] != 'skip')]
        rows = rows[(rows['mode'] != 'code') | rows['check_id'].isin(codes)]
        if rows.empty:
            continue
        ids = retrieve(rows, ptype)
        retrieved[g] = ids
        prompt = build_prompt(info, rows, codes, ids)
        out, u = call_llm(provider, SYSTEM, prompt, model)
        usage.append(u); summaries[g] = out.get('summary', '')
        got = {it.get('check_id'): it for it in out.get('items', [])}
        # 위반 재확인(자기 일관성): LLM이 판정한 위반이 있으면 같은 프롬프트로 한 번 더 묻고, 2차도 위반일 때만 유지
        llm_vio = [cid for cid in rows['check_id'] if cid not in codes and cid not in CODE_AFTER
                   and (got.get(cid) or {}).get('verdict') == '위반']
        second = {}
        if llm_vio:
            out2, u2 = call_llm(provider, SYSTEM, prompt, model)
            usage.append(u2)
            got2 = {it.get('check_id'): it for it in out2.get('items', [])}
            second = {cid: (got2.get(cid) or {}).get('verdict', '응답없음') for cid in llm_vio}
        for cid in rows['check_id']:
            it = got.get(cid) or {'check_id': cid, 'verdict': codes.get(cid, ('확인필요',))[0],
                                  'card_ids': [], 'quote': '', 'ad_quote': '', 'reason': 'LLM 답변에 이 항목이 없음'}
            it = validate(it, ids, codes, ad_norm)
            it['source'] = 'code' if cid in codes else 'llm'
            it['measure'] = codes.get(cid, ('', ''))[1]
            it['recheck'] = ''
            if cid in second:
                v2 = second[cid]
                if v2 == '위반':
                    it['recheck'] = '1차 위반→2차 위반(유지)'
                else:
                    it['verdict'] = '확인필요'
                    it['recheck'] = f'1차 위반→2차 {v2}(확인필요로 변경)'
                    it['issues'] = '; '.join(x for x in [it.get('issues', ''), f'재확인 불일치(1차 위반→2차 {v2})'] if x)
                rechecks.append({'group': g, 'check_id': cid, 'first': '위반', 'second': v2, 'final': it['verdict']})
            results.append(it)

    # C5 혜택·불이익 균형: 다른 항목 판정 뒤 코드로 정함. LLM 판정은 덮어쓰고 reason은 종합 의견으로 남김
    final = {it['check_id']: it for it in results}
    if 'C5' in final:
        bad = [f"{c}({final[c]['verdict']})" for c in C5_FROM if c in final and final[c]['verdict'] in ('확인필요', '위반')]
        it = final['C5']
        llm_verdict = it.get('verdict')
        it['verdict'] = '확인필요' if bad else '적합'
        it['source'] = 'code'
        it['measure'] = ('확인필요/위반 항목: ' + ', '.join(bad)) if bad else 'A2~A7·C3에 확인필요/위반 없음'
        it['reason'] = f"[LLM 종합 의견 · LLM 판정 {llm_verdict}] {it.get('reason', '')}"

    res = pd.DataFrame(results).merge(checks[['check_id', 'item']], on='check_id')
    res = res.reindex(columns=['check_id', 'item', 'verdict', 'source', 'measure', 'ad_quote', 'reason', 'card_ids', 'quote', 'issues', 'recheck']).fillna('')
    tok = (sum(u['input_tokens'] for u in usage), sum(u['output_tokens'] for u in usage))
    record = {'ad_id': ad_id, 'product': ptype, 'product_by': by, 'product_reason': pinfo['reason'],
              'product_keywords': pinfo['keywords'], 'media': media,
              'model': usage[0]['model'] if usage else (model or MODELS[provider]), 'tokens': tok, 'llm_calls': len(usage),
              'recheck_calls': len({r['group'] for r in rechecks}),
              'rechecks': rechecks, 'summaries': summaries, 'retrieved_cards': retrieved,
              'items': res.to_dict('records'), 'judged_at': str(datetime.datetime.now())}
    name = output_name(ad_id, record['model'])
    (out_dir / name).write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding='utf-8')
    res.to_csv(out_dir / name.replace('.json', '.csv'), index=False, encoding='utf-8-sig')   # 엑셀·시트로 열어 보기용
    if verbose:
        print(f"== {ad_id} | {ad.get('brand')} | {ptype}({by}) | {media} | {record['model']} | 토큰 입력 {tok[0]:,} / 출력 {tok[1]:,} | 위반 재확인 호출 {record['recheck_calls']}회")
        print(res['verdict'].value_counts().to_string())
        for g, s in summaries.items():
            print(f'[{GROUP_NAME[g]}] {s}')
    return (res, record) if return_record else res

# ---------- 11. 여러 광고 · 여러 모델 ----------
def run_many(ad_ids, providers=('openai', 'claude')):
    rows = []
    for p in providers:
        if not os.getenv(API_KEY_ENV[p]):
            print(f'{p}: 키 없음 — 건너뜀'); continue
        for a in ad_ids:
            r = judge_ad(a, p, verbose=False)
            r['ad_id'], r['provider'] = a, p
            rows.append(r)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
