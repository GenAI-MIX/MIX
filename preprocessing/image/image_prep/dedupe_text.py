import csv
import json
import re
from pathlib import Path

# OCR 텍스트 기반 중복·같은 캠페인 탐지

def _grams(text, n=3):
    t = re.sub(r"\s+", "", text)
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def _jaccard(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0


def find_text_duplicates(json_dir, out_csv, cfg):
    docs = {}
    for jp in sorted(Path(json_dir).glob("*.json")):
        d = json.loads(jp.read_text(encoding="utf-8"))
        body = " ".join(b["text"] for b in d["ocr_blocks"] if b["type"] != "risk_notice")
        risk = " ".join(b["text"] for b in d["ocr_blocks"] if b["type"] == "risk_notice")
        docs[d["ad_id"]] = (_grams(body), _grams(body + risk))
    ids = sorted(docs)
    rows = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if min(len(docs[a][1]), len(docs[b][1])) < cfg["min_grams"]:
                continue
            s_body, s_all = _jaccard(docs[a][0], docs[b][0]), _jaccard(docs[a][1], docs[b][1])
            # 본문이 비슷하거나, 스캔 품질 차이로 본문 OCR이 달라도 전체 텍스트가 아주 비슷하면 같은 문구
            if s_body >= cfg["same_ad_body_sim"] or s_all >= cfg["same_ad_all_sim"]:
                kind = "문구 거의 동일"
            elif s_all >= cfg["same_campaign_all_sim"]:
                kind = "고지문 공유"
            else:
                continue
            rows.append({"ad_a": a, "ad_b": b, "kind": kind, "body_sim": round(s_body, 3), "all_sim": round(s_all, 3)})
    rows.sort(key=lambda r: (r["kind"] != "문구 거의 동일", -r["body_sim"]))
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=["ad_a", "ad_b", "kind", "body_sim", "all_sim"])
        wr.writeheader()
        wr.writerows(rows)
    return rows
