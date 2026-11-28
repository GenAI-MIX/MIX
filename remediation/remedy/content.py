"""① 내용 수정: 05 LLM 판정 결과(rag/outputs/{ad_id}__{model}.json)를 읽어
- B(필수 내용)·D1(심사필) 누락 → phrases.yaml 표준 문구를 고지 문단에 '추가'
- C(금지 표현)·E → 카피는 자동으로 바꾸지 않고, 해당 광고 문구 위치 표시 + 고칠 방향 제안
"""
import json
import re
from pathlib import Path

ACT = ("위반", "확인필요")


def load_judge(judge_dir, ad_id, model=None):
    """05 판정 결과. model을 정하면 {ad_id}__{model}.json만 읽음 (모델이 섞이지 않게).
    정하지 않으면 파일이 하나일 때만 읽고, 여러 모델이 있으면 경고하고 가장 최근 것."""
    if model:
        p = Path(judge_dir) / f"{ad_id}__{model}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    files = sorted(Path(judge_dir).glob(f"{ad_id}__*.json"), key=lambda p: p.stat().st_mtime)
    if len(files) > 1:
        print(f"  [경고] {ad_id}: 판정 파일이 여러 모델 — config paths.judge_model로 하나를 정하세요")
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def _norm(s):
    return re.sub(r"[\s·.,:;()\[\]※*\-<>{}\"'/|~]", "", s or "")


def _looks_measure(q):
    """LLM이 광고 문구 대신 측정값을 옮긴 경우 (예: '"risk_headline_ratio": 0.173')."""
    return bool(re.search(r"[a-z_]{6,}\s*\"?\s*:", q or "")) or (q or "").count('"') >= 2


def find_blocks(ad, quote):
    """ad_quote가 들어 있는 OCR 블록. ' / '로 나뉜 문구는 조각마다, 여러 블록에 걸친 문구는 이어진 블록을 합쳐서 찾음."""
    if not quote or _looks_measure(quote):
        return []
    blocks = sorted(ad["ocr_blocks"], key=lambda b: (b["bbox"][1], b["bbox"][0]))
    norms = [_norm(b["text"]) for b in blocks]
    hits = []
    for piece in re.split(r"\s/\s", quote):
        q = _norm(piece)
        if len(q) < 2:
            continue
        found = False
        for i in range(len(blocks)):              # 1~6개 연속 블록을 합쳐서 비교
            acc = ""
            for j in range(i, min(i + 6, len(blocks))):
                acc += norms[j]
                if q in acc:
                    hits += [b["bbox"] for b in blocks[i:j + 1] if b["bbox"] not in hits]
                    found = True
                    break
                if len(acc) > len(q) + 40:
                    break
            if found:
                break
        if not found:   # 블록 하나가 문구 일부인 경우 (긴 인용)
            hits += [b["bbox"] for b, n in zip(blocks, norms) if len(n) >= 6 and n in q and b["bbox"] not in hits]
    return hits


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
                               "evidence_issue": it.get("issues") or None,
                               "card_ids": it.get("card_ids", "")})
        elif cid in phrases["suggest"]:
            # C5(혜택·불이익 균형)는 05에서 코드로 정함: 원인이 표시 방법(A 항목)뿐이면 ② 표시 수정이 이미 고치므로 표현 검토에서 뺌
            if cid == "C5" and "C3" not in (it.get("measure") or ""):
                continue
            out["flag"].append({"check_id": cid, "item": it.get("item"), "verdict": v,
                                "ad_quote": it.get("ad_quote", ""), "suggest": phrases["suggest"][cid],
                                "boxes": find_blocks(ad, it.get("ad_quote", "")),
                                "evidence_issue": it.get("issues") or None,
                                "reason": it.get("reason", ""), "card_ids": it.get("card_ids", "")})
    return out
