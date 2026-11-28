"""상황 진단 (DESIGN_SYSTEM.md 2장): 매체·규격 / 고지 컨테이너 / 배경 / 고지 구조.
수정 방법을 고를 때(대비 처리, 공간 확보) 이 결과를 쓴다."""
import numpy as np

from .measure import ink_mask, to_gray

CONTAINER = {"footer": "브랜드 푸터 패널", "box": "박스·카드 안", "photo": "사진 위 직접", "doc": "문서형 본문",
             "table": "표·조건과 섞임", "scattered": "흩어진 각주", "plain": "단색·그라데이션 배경 위"}


def canvas_kind(W, H):
    a = H / W
    return "detail" if a > 2.0 else ("portrait" if a > 1.15 else ("square" if a > 0.85 else "banner"))


def classify(ad, img):
    """img: RGB numpy. 반환 dict(canvas, container, background, structure, band)."""
    H, W = img.shape[:2]
    canvas = canvas_kind(W, H)
    paras = [p for p in ad["paragraphs"] if p["type"] == "risk_notice"]
    if not paras:
        return {"canvas": canvas, "container": None, "background": None, "structure": None, "band": None}
    main = max(paras, key=lambda p: len(p["text"]))
    x0, y0, x1, y1 = [int(v) for v in main["bbox"]]
    blocks = {b["id"]: b for b in ad["ocr_blocks"]}
    hs = [blocks[i]["h_px"] for i in main["block_ids"] if i in blocks and blocks[i].get("h_px")]
    lh = int(np.median(hs)) if hs else 12
    # 글자 픽셀을 뺀 고지 주변 배경
    text_mask = np.zeros((H, W), bool)
    for b in ad["ocr_blocks"]:
        bx0, by0, bx1, by1 = [int(v) for v in b["bbox"]]
        text_mask[max(0, by0 - 2):by1 + 2, max(0, bx0 - 2):bx1 + 2] = True
    pad = max(2 * lh, 10)
    ry0, ry1, rx0, rx1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    # 고지 영역 안에서 글자(잉크)와 그 주변 2px를 뺀 픽셀 = 배경
    inner = img[y0:y1, x0:x1]
    ink = ink_mask(to_gray(inner))
    k = np.pad(ink, 2)
    near = np.zeros_like(ink)
    for dy in range(5):
        for dx in range(5):
            near |= k[dy:dy + ink.shape[0], dx:dx + ink.shape[1]]
    px = inner[~near].astype(float)
    if len(px) < 50:
        px = inner.reshape(-1, 3).astype(float)
    med = np.median(px, axis=0)
    flat = float((np.abs(px - med).sum(axis=1) < 36).mean())
    lum = px @ [0.299, 0.587, 0.114]
    background = "flat" if flat > 0.8 else ("gradient" if flat > 0.5 and lum.std() < 25 else "photo")
    # 띠/박스: 고지 위쪽으로 같은 색이 이어지는 구간 → 그 위에서 색이 바뀌면 경계
    band = None
    if background != "photo":
        top = y0
        for y in range(y0 - 1, max(0, y0 - 40 * lh), -1):
            row = img[y, rx0:rx1].astype(float)
            rm = ~text_mask[y, rx0:rx1]
            if rm.sum() < 10:
                continue
            if (np.abs(row[rm] - med).sum(axis=1) < 40).mean() < 0.6:
                break
            top = y
        bottom = y1
        for y in range(y1, min(H, y1 + 20 * lh)):
            row = img[y, rx0:rx1].astype(float)
            rm = ~text_mask[y, rx0:rx1]
            if rm.sum() < 10:
                continue
            if (np.abs(row[rm] - med).sum(axis=1) < 40).mean() < 0.6:
                break
            bottom = y
        # 좌우 범위: 띠 중간 줄에서 같은 색이 이어지는 폭
        ym = (top + y0) // 2 if top < y0 - 2 else max(0, y0 - 2)
        row = img[ym].astype(float)
        same = np.abs(row - med).sum(axis=1) < 40
        l = x0
        while l > 0 and same[l - 1]:
            l -= 1
        r = x1
        while r < W - 1 and same[r + 1]:
            r += 1
        band = [int(l), int(top), int(r), int(bottom)]
    global_med = np.median(img[::4, ::4].reshape(-1, 3).astype(float), axis=0)
    distinct = np.abs(med - global_med).sum() > 60
    risk_rows = {round(p["bbox"][1] / H, 1) for p in paras}
    n_other_in = sum(1 for b in ad["ocr_blocks"] if b["type"] != "risk_notice"
                     and b["bbox"][1] >= ry0 and b["bbox"][3] <= ry1 and b["bbox"][0] >= rx0 and b["bbox"][2] <= rx1)
    if canvas == "detail":
        container = "doc"
    elif len(paras) >= 3 and len(risk_rows) >= 3:
        container = "scattered"
    elif background == "photo":
        container = "photo"
    elif n_other_in >= 6:
        container = "table"
    elif band and (band[2] - band[0]) > 0.85 * W and band[3] > 0.8 * H and distinct:
        container = "footer"
    elif band and distinct:
        container = "box"
    else:
        container = "plain"
    lines = [blocks[i] for i in main["block_ids"] if i in blocks]
    xs = [b["bbox"][0] for b in lines]
    structure = "list" if any(b["text"].lstrip()[:1] in "※*•·-▶■" for b in lines[1:]) else "paragraph"
    if len(lines) > 3 and len({round(x / W, 1) for x in xs}) >= 2:
        structure = "columns"
    return {"canvas": canvas, "container": container, "background": background, "structure": structure,
            "band": band, "line_h": lh}


def label(s):
    if not s or not s.get("container"):
        return "고지 없음"
    cv = {"detail": "상세페이지", "portrait": "세로", "square": "정사각", "banner": "가로 배너"}[s["canvas"]]
    bg = {"flat": "단색", "gradient": "그라데이션", "photo": "사진"}[s["background"]]
    return f"{cv} · {CONTAINER[s['container']]} · {bg} 배경"
