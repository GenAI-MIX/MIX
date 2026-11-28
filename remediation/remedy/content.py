"""① 내용 수정: 05 LLM 판정 결과(rag/outputs/{ad_id}__{model}.json)를 읽어
- B(필수 내용)·D1(심사필) 누락 → phrases.yaml 표준 문구를 고지 문단에 '추가'
- C(금지 표현)·E → 카피는 자동으로 바꾸지 않고, 해당 광고 문구 위치 표시 + 고칠 방향 제안
"""
import json
import re
from pathlib import Path

ACT = ("위반", "확인필요")


def load_judge(judge_dir, ad_id):
    """같은 광고의 판정 파일이 여러 모델로 있으면 가장 최근 것."""
    files = sorted(Path(judge_dir).glob(f"{ad_id}__*.json"), key=lambda p: p.stat().st_mtime)
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _norm(s):
    return re.sub(r"[\s·.,:;()\[\]※*\-]", "", s or "")


def find_blocks(ad, quote):
    """ad_quote가 들어 있는 OCR 블록 (띄어쓰기·기호 무시)."""
    q = _norm(quote)
    if len(q) < 2:
        return []
    hit = [b for b in ad["ocr_blocks"] if q in _norm(b["text"]) or (len(_norm(b["text"])) >= 4 and _norm(b["text"]) in q)]
    return hit


def plan(ad, judge, phrases):
    """반환: {'add': [{'check_id','verdict','text','placeholder','reason'}], 'flag': [...], 'product': ...}"""
    out = {"add": [], "flag": [], "product": None, "model": None}
    if not judge:
        return out
    out["product"], out["model"] = judge.get("product"), judge.get("model")
    ptype = judge.get("product") or ""
    for it in judge.get("items", []):
        cid, v = it.get("check_id"), it.get("verdict")
        if v not in ACT:
            continue
        rule = phrases["add"].get(cid)
        if rule is not None:
            if "text" in rule:
                if rule.get("products") and ptype not in rule["products"]:
                    continue
                text = rule["text"]
            else:                                   # 상품 유형별 문구 (B6)
                text = rule.get(ptype, rule.get("default"))
            text = text.format(brand=ad.get("brand") or judge.get("brand") or "【판매회사명】")
            if _norm(text) and _norm(text)[:12] in _norm(" ".join(b["text"] for b in ad["ocr_blocks"])):
                continue                            # 이미 거의 같은 문장이 있음
            out["add"].append({"check_id": cid, "item": it.get("item"), "verdict": v, "text": text,
                               "placeholder": "【" in text, "reason": it.get("reason", ""),
                               "card_ids": it.get("card_ids", "")})
        elif cid in phrases["suggest"]:
            out["flag"].append({"check_id": cid, "item": it.get("item"), "verdict": v,
                                "ad_quote": it.get("ad_quote", ""), "suggest": phrases["suggest"][cid],
                                "boxes": [b["bbox"] for b in find_blocks(ad, it.get("ad_quote", ""))],
                                "reason": it.get("reason", ""), "card_ids": it.get("card_ids", "")})
    return out
