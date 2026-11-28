"""06 수정안 사례집 PDF: 방법 정리(이미지·영상) + 이미지 20개 + 영상 20개.
상황·방법·결과가 겹치지 않게 고른다.

  python report/make_casebook.py
"""
import collections
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import (BLUE, GREEN, ORANGE, PURPLE, RED, SUB, INK, H, W, OUT, ROOT, VOUT, F,  # noqa: E402
                         bullets, case_page, fit, fmtv, header, image_lines, load_json, page, table, video_page)

HERE = Path(__file__).resolve().parent
PROFILE = "A4"

# 영상 20개: 결과·방법이 겹치지 않게 (방법 태그는 결과 JSON의 편집 지시에서 확인한 것)
VIDEO_PICK = [
    ("V111", "엔드 카드 당기기 · 원래 글자색 그대로"), ("V124", "엔드 카드 당기기 · 광고에 쓰인 색으로 글자색"),
    ("V143", "엔드 카드 당기기 · 글자색 명도 조정"), ("V133", "엔드 카드 당기기 · 글자 뒤 국소 톤다운"),
    ("V110", "빈 장면에 자막 이어 붙이기(자리 옮김) · 명도 조정"), ("V118", "빈 장면에 자막 이어 붙이기 · 광고 색 · 움직이는 배경"),
    ("V175", "빈 장면에 자막 이어 붙이기 · 원래 색 · 정지 화면"), ("V103", "움직이는 배경 · 원래 글자 블러 + 위쪽 자리"),
    ("V105", "저해상도(640×480) · 자리 옮김 · 국소 톤다운"), ("V127", "글자 크기·명도만 조정 (정지 화면)"),
    ("V135", "글자 크기·명도만 조정 (움직이는 배경)"),
    ("V108", "통과 못 함 · 화면 전체에 손글씨 그래픽"), ("V144", "통과 못 함 · 어느 자리에 두어도 겹침"),
    ("V138", "통과 못 함 · 고지 자리가 빈 장면 없음"), ("V129", "통과 못 함 · 4배 이상 키워야 함"),
    ("V140", "통과 못 함 · 빈 장면이 일부만 있음"),
    ("V101", "원본 유지"), ("V112", "원본 유지"), ("V109", "고지 없음"), ("V117", "고지 없음"),
]

COLOR = {"수정안": BLUE, "수정안 + 표현 검토": BLUE, "표현 검토": ORANGE, "원본 유지": GREEN, "디자이너 검토": PURPLE, "고지 없음": ORANGE}


def pick_images(n=20):
    """A4 결과에서 상황 × 결정 × 추천 방법 × 대비 처리가 겹치지 않게 n개."""
    recs = [load_json(p) for p in sorted((OUT / PROFILE / "json").glob("*.json"))]

    def key(r):
        best = next((c for c in r.get("candidates", []) if c.get("recommended")), {})
        tr = next((s.get("contrast_treatment") for s in best.get("steps", []) if s.get("contrast_treatment")), None)
        return (r.get("situation"), r["decision"], best.get("name"), (tr or "")[:6])
    quota = {"수정안": 11, "원본 유지": 2, "표현 검토": 2, "디자이너 검토": 4, "고지 없음": 1}
    got, seen_key, seen_method, seen_sit = [], set(), collections.Counter(), collections.Counter()
    # 1차: 방법·상황이 새로운 것부터
    for r in sorted(recs, key=lambda r: (seen_sit[r.get("situation")], r["ad_id"])):
        k = key(r)
        dec = "수정안" if r["decision"].startswith("수정안") else r["decision"]
        if quota.get(dec, 0) <= 0 or k in seen_key:
            continue
        if dec == "수정안" and seen_method[k[2]] >= 2 and len(got) < n - 3:
            continue
        if seen_sit[k[0]] >= 3:
            continue
        got.append(r); seen_key.add(k); seen_method[k[2]] += 1; seen_sit[k[0]] += 1; quota[dec] -= 1
        if len(got) >= n:
            break
    for r in recs:      # 모자라면 채움
        if len(got) >= n:
            break
        if r not in got:
            got.append(r)
    order = {"원본 유지": 0, "수정안": 1, "수정안 + 표현 검토": 1, "표현 검토": 2, "디자이너 검토": 3, "고지 없음": 4}
    return sorted(got, key=lambda r: (order.get(r["decision"], 9), r.get("recommended") or "", r["ad_id"]))


def methods_image(pages):
    im, d = page(); header(d, "방법 정리 ① 정적 광고물 (이미지)", len(pages) + 1)
    table(d, 110, 200, [
        ["단계", "하는 일"],
        ["1. 진단", "05와 같은 기준으로 A2 명암비·A3 글자 크기(판형별)·A4 헤드라인 대비 판정. 모두 통과면 원본 유지"],
        ["2. 상황 분류", "매체(세로·배너·상세페이지) × 고지 컨테이너(푸터 패널·박스·사진 위·문서형·표·흩어진 각주) × 배경(단색·그라데이션·사진)"],
        ["3. 글꼴", "글꼴 11종(컨셉별) 중 원본 고지 글자의 폭 비율·획 굵기·획 모양이 가장 비슷한 것 하나를 광고 전체에"],
        ["4. 대비 처리", "단색: 글자 명도 / 띠·박스: 글자·띠 중 덜 바뀌는 쪽 / 사진: 아래쪽을 원래 톤으로 페이드 (새 박스 없음)"],
        ["5. 후보안", "줄 간격(A) · 여백 회수(B) · 사진 다듬기(T) · 비주얼 축소(C) · 조합(D) · 상세페이지 높이(H) — 모두 원본과 같은 규격"],
        ["6. 재측정·추천", "같은 계산식으로 다시 재서 통과한 안 중 원본을 가장 덜 바꾸는 안 추천. 없으면 디자이너 브리프"],
    ], [330, 1780], size=27)
    table(d, 110, 830, [
        ["판형", "모바일", "B5", "A4", "B4", "A3", "신문 전면", "A2"],
        ["A3 기준", "5.9pt (390pt 폭)", "2.13mm", "2.13mm", "2.13mm", "2.13mm", "2.36mm", "2.13mm"],
    ], [200, 330, 220, 220, 220, 220, 260, 220], size=26)
    bullets(d, 110, 1040, [
        "광고 규격(px)은 바꾸지 않음. 상세페이지만 높이가 자유로운 매체라 예외.",
        "헤드라인·인물·로고는 줄이거나 가리지 않음. 여백 회수·사진 다듬기는 글자가 없는 줄만.",
        "같은 광고라도 판형마다 결과가 다름 → 판형 7개 모두 결과를 만듦. 사례집은 A4 기준.",
    ], size=28, width=2100)
    pages.append(im)


def methods_video(pages):
    im, d = page(); header(d, "방법 정리 ② 영상 광고", len(pages) + 1)
    table(d, 110, 200, [
        ["상황", "방법"],
        ["노출 시간 부족 (A5)", "① 고지 자리가 빈 장면에 같은 자막을 이어 붙임 (그 장면에서 빈 자리로 옮기기 포함)  ② 앞 컷 끝을 다듬어 엔드 카드를 일찍 시작 — 음성·전체 길이 그대로"],
        ["너무 빨리 지나감 (A6)", "같은 방법으로 읽을 시간 확보"],
        ["글자가 작음 (A7·A4)", "원래 자리·단 폭 → 아래 안전선 → 위 안전선 순으로 실제로 비어 있는 폭을 재서 그 폭에 맞춰 다시 씀"],
        ["명암비 부족 (A2)", "박스·띠 없이 글자색으로: 원래 색 → 광고에 이미 쓰인 색 → 원래 색의 명도. 마지막 수단으로만 글자 바로 뒤 국소 톤다운"],
        ["원래 작은 글자", "정지 화면(엔드 카드): 지운 배경 위에 다시 씀 / 움직이는 화면: 그 자리만 블러 (헤드라인과 겹치는 쪽은 제외)"],
        ["흐르는 자막", "같은 띠 자리에 고정 자막으로"],
        ["결과 형태", "고지 화면 수정 전/후 이미지 + 자막 시간대 전/후 타임라인 + 편집 지시 (수정 mp4는 선택)"],
    ], [420, 1690], size=26)
    bullets(d, 110, 1120, [
        "영상 길이·화면 크기는 그대로. 위험고지 자막만 고침.",
        "고친 화면을 같은 계산식으로 다시 재서 명암비·글자 크기·헤드라인 대비 확인, 노출·속도는 새 타임라인으로 계산.",
        "어느 자리에 두어도 다른 글자와 겹치거나 빈 장면이 없으면 디자이너 검토 (필요한 시간·자리 안내).",
    ], size=28, width=2100)
    pages.append(im)


def overview(pages, img_cases, vid_cases):
    im, d = page(); header(d, "사례 목록 (이미지 20 · 영상 20)", len(pages) + 1)
    rows = [["이미지", "상황", "결과"]]
    for r in img_cases:
        best = next((c for c in r.get("candidates", []) if c.get("recommended")), None)
        res = r["decision"] + (f" · {best['label'].split(' · ')[-1]}" if best else "")
        rows.append([r["ad_id"], (r.get("situation") or "-")[:22], (res[:30], COLOR.get(r["decision"], INK))])
    table(d, 110, 190, rows, [150, 520, 440], size=21)
    rows = [["영상", "무엇을 보여주나", "결과"]]
    for ad_id, what in vid_cases:
        p = VOUT / "json" / f"{ad_id}.json"
        dec = load_json(p)["decision"] if p.exists() else "-"
        rows.append([ad_id, what[:26], (dec, COLOR.get(dec, INK))])
    table(d, 1240, 190, rows, [130, 640, 220], size=21)
    pages.append(im)


def video_basic_page(im, d, ad_id, title):
    """원본 유지·고지 없음 영상: 대표 화면 + 판정."""
    meta = load_json(ROOT / "data/processed/video/json" / f"{ad_id}.json")
    rec = load_json(VOUT / "json" / f"{ad_id}.json")
    risk = [c for c in meta["cards"] if c["type"] == "risk_notice"]
    sid = risk[0]["screen_id"] if risk else (meta["screens"][-1]["id"] if meta["screens"] else None)
    fr = next((s["frame"] for s in meta["screens"] if s["id"] == sid), None)
    lines = [f"{meta.get('brand')} · {meta.get('media')} · {meta['duration']:.1f}초 · {meta['size'][0]}×{meta['size'][1]}"]
    for c in rec["checks_before"]:
        ok = "통과" if c["ok"] else ("미달" if c["ok"] is False else "측정 불가")
        lines.append(f"{c['id']} {c['name']}: {fmtv(c)} ({ok}, 기준 {c['threshold']})")
    lines.append(rec.get("note") or "")
    y = bullets(d, 110, 200, [l for l in lines if l], size=28, width=900)
    if fr and (ROOT / fr).exists():
        t = fit(Image.open(ROOT / fr).convert("RGB"), W - 1150, H - 300)
        im.paste(t, (1040, 200))


def video_case(im, d, rec, color, lines):
    """영상 사례: 왼쪽 설명 · 오른쪽 비교 패널(작게) · 아래 고지 부분 확대 전/후 (글자가 읽히는 배율로)."""
    pan = Image.open(VOUT / "viz" / f"{rec['ad_id']}.jpg").convert("RGB")
    pan = fit(pan, W - 110 - 900, 900)
    px = W - 110 - pan.width
    im.paste(pan, (px, 190)); d.rectangle((px - 2, 188, px + pan.width + 1, 191 + pan.height), outline=(220, 223, 230), width=2)
    d.rectangle((110, 200, 122, 200 + 46 * len(lines)), fill=color)
    y = bullets(d, 140, 200, lines, size=27, width=px - 200)
    d.text((140, y + 10), "편집 지시", font=F("SemiBold", 28), fill=INK)
    bullets(d, 140, y + 58, rec.get("edit_sheet", [])[:4] + ["불가: " + x for x in rec.get("fails", [])][:2],
            size=23, width=px - 200, color=SUB)
    fr = (rec.get("frames") or [None])[0]
    ms = [m for m in rec.get("measured", []) if fr and abs(m.get("t", -1) - fr["t"]) < 0.05]
    if not fr or not ms:
        return
    x0 = min(m["bbox"][0] for m in ms); y0 = min(m["bbox"][1] for m in ms)
    x1 = max(m["bbox"][2] for m in ms); y1 = max(m["bbox"][3] for m in ms)
    b = Image.open(ROOT / fr["before"]).convert("RGB"); a = Image.open(ROOT / fr["after"]).convert("RGB")
    pad = int((y1 - y0) * 0.35) + 12
    zh = H - 190 - pan.height - 120                       # 확대에 쓸 높이
    zw = (W - 220 - 60) // 2
    # 글자가 원래 크기 이상으로 보이도록: 확대 칸 폭에 맞춰 고지의 앞부분만 자름
    cw = min(x1 - x0 + 2 * pad, int(zw * (y1 - y0 + 2 * pad) / max(zh, 1)) if zh > 0 else zw)
    cw = max(cw, int(zw * 0.5))
    box = (max(0, x0 - pad), max(0, y0 - pad), min(a.width, x0 - pad + cw), min(a.height, y1 + pad))
    zb, za = fit(b.crop(box), zw, zh), fit(a.crop(box), zw, zh)
    yz = 190 + pan.height + 70
    d.text((110, yz - 44), f"고지 부분 확대 · {fr['where']} {fr['t']:.1f}초 (왼쪽 원본 → 오른쪽 수정)", font=F("SemiBold", 26), fill=INK)
    im.paste(zb, (110, yz)); d.rectangle((109, yz - 1, 110 + zb.width, yz + zb.height), outline=(220, 223, 230), width=2)
    im.paste(za, (110 + zw + 60, yz)); d.rectangle((109 + zw + 60, yz - 1, 110 + zw + 60 + za.width, yz + za.height), outline=BLUE, width=3)


def main():
    pages = []
    im, d = page()
    d.rectangle((0, 0, W, H), fill=(245, 248, 255)); d.rectangle((0, 0, 30, H), fill=BLUE)
    d.text((200, 560), "06 수정안 사례집", font=F("Bold", 96), fill=INK)
    d.text((200, 700), "정적 광고물 20개 · 영상 광고 20개 — 어떤 광고가 어떻게 고쳐지는지", font=F("Medium", 46), fill=BLUE)
    d.text((200, 800), "상황·방법·결과가 겹치지 않게 골랐음. 이미지는 A4 인쇄 판형 기준.", font=F("Regular", 34), fill=SUB)
    pages.append(im)
    methods_image(pages)
    methods_video(pages)
    imgs = pick_images()
    overview(pages, imgs, VIDEO_PICK)
    for i, r in enumerate(imgs, 1):
        im, d = page()
        header(d, f"이미지 {i:02d} · {r['ad_id']} · {r['decision']}", len(pages) + 1)
        case_page(im, d, OUT / PROFILE / r["ad_id"], COLOR.get(r["decision"], BLUE), image_lines(r, "A4 인쇄"))
        pages.append(im)
    for i, (ad_id, what) in enumerate(VIDEO_PICK, 1):
        rec = load_json(VOUT / "json" / f"{ad_id}.json")
        im, d = page()
        header(d, f"영상 {i:02d} · {ad_id} · {what}", len(pages) + 1)
        if rec["decision"] in ("원본 유지", "고지 없음"):
            video_basic_page(im, d, ad_id, what)
        else:
            after = {c["id"]: c for c in rec.get("checks_after") or []}
            lines = [f"{c['id']} {c['name']}: {fmtv(c)} → {fmtv(after[c['id']]) if c['id'] in after else '-'}"
                     for c in rec["checks_before"] if c["ok"] is False] or ["기준 미달 항목 없음"]
            video_case(im, d, rec, COLOR.get(rec["decision"], BLUE), lines)
        pages.append(im)
    dst = HERE / "06_수정안_사례집.pdf"
    pages[0].save(dst, save_all=True, append_images=pages[1:], resolution=200)
    print(dst, len(pages), "pages")


if __name__ == "__main__":
    main()
