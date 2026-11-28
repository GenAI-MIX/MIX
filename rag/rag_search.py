"""04 규정 기반 RAG 검색 모듈.

04_rag_search.ipynb에서 만든 검색 장치를 다른 노트북(05 LLM 판단 등)에서 불러 쓰기 위한 파일입니다.

    from rag_search import search
    search('영상 광고에서 위험고지가 짧게만 노출됨', '투자성', k=8)

처음 import할 때 BGE-M3와 리랭커 모델을 불러오므로 몇십 초 걸립니다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAG_DIR = Path(__file__).resolve().parent
DATA_DIR = RAG_DIR / 'data'
CACHE_DIR = RAG_DIR / 'cache'
CACHE_DIR.mkdir(exist_ok=True)

# ---------- 카드 ----------
cards = pd.read_csv(DATA_DIR / 'cards.csv', encoding='utf-8-sig').fillna('')
cards['search_text'] = (cards['문서명'] + ' ' + cards['조 제목'] + '\n' + cards['본문'] + '\n' + cards['메모']).str.strip()
card_by_id = cards.set_index('id')

# ---------- BM25 ----------
from kiwipiepy import Kiwi
from rank_bm25 import BM25Okapi

_kiwi = Kiwi()
_KEEP = ('NNG', 'NNP', 'NR', 'SL', 'SN', 'VV', 'VA', 'XR')


def tokenize(text):
    return [t.form for t in _kiwi.tokenize(text) if t.tag.startswith(_KEEP)]


bm25 = BM25Okapi([tokenize(t) for t in cards['search_text']])

# ---------- BGE-M3 + FAISS ----------
import faiss
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model = SentenceTransformer('BAAI/bge-m3', device=device)

_index_path = CACHE_DIR / 'cards.faiss'
_ids_path = CACHE_DIR / 'card_ids.json'
if _index_path.exists() and _ids_path.exists() and json.loads(_ids_path.read_text()) == cards['id'].tolist():
    index = faiss.read_index(str(_index_path))
else:
    _emb = model.encode(cards['search_text'].tolist(), batch_size=8, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(_emb.shape[1])
    index.add(np.asarray(_emb, dtype='float32'))
    faiss.write_index(index, str(_index_path))
    _ids_path.write_text(json.dumps(cards['id'].tolist(), ensure_ascii=False))

# ---------- 리랭커 ----------
reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', device=device, max_length=1024)


# ---------- 검색 ----------
def _rank(scores, mask):
    s = np.where(mask, scores, -np.inf)
    return [i for i in np.argsort(-s) if mask[i]]


def _dense_scores(query):
    q = model.encode([query], normalize_embeddings=True).astype('float32')
    sims, idx = index.search(q, index.ntotal)
    scores = np.full(len(cards), -np.inf)
    scores[idx[0]] = sims[0]
    return scores


def product_mask(product_type):
    if product_type in ('전체', '', None) or product_type not in ('투자성', '보장성', '예금성', '대출성'):
        return np.ones(len(cards), bool)
    return cards['상품유형'].isin([product_type, '공통']).to_numpy()


def search_ids(query, product_type='전체', k=8, mode='rerank', rrf_k=60, n_candidates=20):
    """카드 ID와 점수 목록을 돌려줍니다: [(id, score), ...]"""
    mask = product_mask(product_type)
    ranks = []
    if mode in ('hybrid', 'rerank', 'bm25'):
        ranks.append(_rank(bm25.get_scores(tokenize(query)), mask))
    if mode in ('hybrid', 'rerank', 'dense'):
        ranks.append(_rank(_dense_scores(query), mask))
    score = {}
    for r in ranks:
        for pos, i in enumerate(r):
            score[i] = score.get(i, 0) + 1 / (rrf_k + pos + 1)
    ordered = sorted(score, key=score.get, reverse=True)
    if mode == 'rerank':
        cand = ordered[:n_candidates]
        rr = reranker.predict([(query, cards['search_text'].iloc[i]) for i in cand], batch_size=8)
        score = {i: float(s) for i, s in zip(cand, rr)}
        ordered = sorted(cand, key=score.get, reverse=True)
    return [(cards['id'].iloc[i], round(score[i], 4)) for i in ordered[:k]]


# ---------- 검색 캐시 ----------
# search_ids 결과를 메모리와 cache/search_cache.json에 저장해 다음 실행에서도 재사용합니다.
# 키: cards.csv 내용 해시 + (질문, 상품유형, k, mode, rrf_k, n_candidates) + 모델 이름
# → 카드가 바뀌면 해시가 달라져 예전 결과는 쓰이지 않고, 불러올 때 지워집니다.
import hashlib
import os
import threading

CARDS_HASH = hashlib.sha256((DATA_DIR / 'cards.csv').read_bytes()).hexdigest()[:16]
SEARCH_CACHE_PATH = CACHE_DIR / 'search_cache.json'
_MODEL_TAG = 'bge-m3+bge-reranker-v2-m3'
_cache_lock = threading.Lock()
cache_stats = {'hit': 0, 'miss': 0}


def _load_search_cache():
    try:
        data = json.loads(SEARCH_CACHE_PATH.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in data.items() if json.loads(k)[0] == CARDS_HASH}   # 다른 카드 버전의 결과는 버림


_search_cache = _load_search_cache()


def _cache_key(query, product_type, k, mode, rrf_k, n_candidates):
    return json.dumps([CARDS_HASH, query, product_type, k, mode, rrf_k, n_candidates, _MODEL_TAG], ensure_ascii=False)


def _save_search_cache():
    tmp = SEARCH_CACHE_PATH.with_suffix(f'.tmp{os.getpid()}')
    tmp.write_text(json.dumps(_search_cache, ensure_ascii=False, indent=0), encoding='utf-8')
    os.replace(tmp, SEARCH_CACHE_PATH)      # 쓰는 도중 멈춰도 파일이 깨지지 않게


def search_ids_cached(query, product_type='전체', k=8, mode='rerank', rrf_k=60, n_candidates=20):
    """search_ids와 같은 결과를 캐시에서 돌려줍니다. 없으면 검색 후 저장."""
    key = _cache_key(query, product_type, k, mode, rrf_k, n_candidates)
    with _cache_lock:
        if key in _search_cache:
            cache_stats['hit'] += 1
            return [tuple(x) for x in _search_cache[key]]
    res = search_ids(query, product_type, k=k, mode=mode, rrf_k=rrf_k, n_candidates=n_candidates)
    with _cache_lock:
        cache_stats['miss'] += 1
        _search_cache[key] = [list(x) for x in res]
        _save_search_cache()
    return res


def search(query, product_type='전체', k=5, mode='rerank'):
    """보기 좋은 표로 돌려줍니다."""
    hits = search_ids(query, product_type, k=k, mode=mode)
    out = card_by_id.loc[[h[0] for h in hits], ['조 제목', '상품유형', '본문']].reset_index()
    out['score'] = [h[1] for h in hits]
    out['본문'] = out['본문'].str.slice(0, 80) + '…'
    return out
