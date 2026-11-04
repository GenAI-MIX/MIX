from .geometry import UnionFind


def _cy(b):
    return (b["bbox"][1] + b["bbox"][3]) / 2


def reading_order(blocks):
    bs = sorted(blocks, key=_cy)
    rows, cur = [], []
    for b in bs:
        if cur and abs(_cy(b) - _cy(cur[-1])) > 0.5 * min(b["h_px"], cur[-1]["h_px"]):
            rows.append(cur)
            cur = []
        cur.append(b)
    if cur:
        rows.append(cur)
    return [b for r in rows for b in sorted(r, key=lambda b: b["bbox"][0])]


def group_paragraphs(blocks, cfg):
    n = len(blocks)
    uf = UnionFind(n)
    for i in range(n):
        bi, hi = blocks[i]["bbox"], blocks[i]["h_px"]
        for j in range(i + 1, n):
            bj, hj = blocks[j]["bbox"], blocks[j]["h_px"]
            if max(hi, hj) / max(1e-6, min(hi, hj)) > cfg["height_ratio_max"]:
                continue
            mh = (hi + hj) / 2
            if abs(_cy(blocks[i]) - _cy(blocks[j])) < 0.5 * mh:          # 같은 줄
                gap_x = max(bi[0], bj[0]) - min(bi[2], bj[2])
                if gap_x <= cfg["same_line_gap_ratio"] * mh:
                    uf.union(i, j)
                continue
            if min(bi[2], bj[2]) - max(bi[0], bj[0]) <= 0:                # 가로로 겹치지 않음
                continue
            upper, lower = (bi, bj) if _cy(blocks[i]) < _cy(blocks[j]) else (bj, bi)
            gap = lower[1] - upper[3]
            if -0.6 * mh <= gap <= cfg["line_gap_ratio"] * mh:
                uf.union(i, j)
    groups = sorted(uf.groups(), key=lambda g: min(blocks[k]["id"] for k in g))
    pid = [0] * n
    for k, g in enumerate(groups):
        for i in g:
            pid[i] = k
    return pid


def build_text_paragraphs(blocks, cfg):
    n = len(blocks)
    uf = UnionFind(n)
    for i in range(n):
        a, A = blocks[i], blocks[i]["bbox"]
        for j in range(i + 1, n):
            b, B = blocks[j], blocks[j]["bbox"]
            ha, hb = a["text_h"], b["text_h"]
            h = max(ha, hb)
            bh = max(a["h_px"], b["h_px"])
            if abs(_cy(a) - _cy(b)) < 0.5 * min(a["h_px"], b["h_px"]):
                if max(B[0] - A[2], A[0] - B[2]) <= cfg["same_line_gap_ratio"] * bh:
                    uf.union(i, j)
                continue
            lower, upper = (b, a) if _cy(b) > _cy(a) else (a, b)
            short_tail = (len(lower["text"].replace(" ", "")) <= cfg["short_line_chars"]
                          and lower["text_h"] < upper["text_h"])
            limit = cfg["short_line_ratio_max"] if short_tail else cfg["height_ratio_max"]
            if max(ha, hb) / max(1e-6, min(ha, hb)) > limit:
                continue
            if abs(_cy(a) - _cy(b)) > cfg["pitch_ratio"] * h:                  # 줄 간격이 너무 넓음
                continue
            ox = min(A[2], B[2]) - max(A[0], B[0])
            if ox <= 0:
                continue
            if ox / max(A[2] - A[0], B[2] - B[0]) >= cfg["span_ratio"] or abs(A[0] - B[0]) <= cfg["indent_ratio"] * h:
                uf.union(i, j)
    groups = [sorted(g, key=lambda k: blocks[k]["id"]) for g in uf.groups()]
    return sorted(groups, key=lambda g: blocks[g[0]]["id"])
