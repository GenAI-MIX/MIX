import numpy as np
from image_prep.geometry import UnionFind, overlap_min, union_box


def frame_blocks(boxes, W, H, cfg):
    """한 프레임의 검출 박스 → 작은 글씨 줄을 묶은 블록 목록."""
    lines = [b for b in boxes
             if cfg["min_h_ratio"] * H <= b[4] <= cfg["max_h_ratio"] * H and b[2] - b[0] >= cfg["min_line_aspect"] * b[4]]
    edge = cfg["edge_touch_ratio"] * W
    touches = [b[0] <= edge or b[2] >= W - edge for b in lines]
    uf = UnionFind(len(lines))
    for i, a in enumerate(lines):
        for j in range(i + 1, len(lines)):
            b = lines[j]
            if touches[i] != touches[j] or max(a[4], b[4]) / min(a[4], b[4]) > cfg["height_ratio_max"]:
                continue
            h = (a[4] + b[4]) / 2
            y_ov = min(a[3], b[3]) - max(a[1], b[1])
            if y_ov > 0.5 * min(a[3] - a[1], b[3] - b[1]):        
                if max(a[0], b[0]) - min(a[2], b[2]) <= cfg["same_line_gap_ratio"] * h:
                    uf.union(i, j)
            elif -y_ov <= cfg["line_gap_ratio"] * h and (min(a[2], b[2]) > max(a[0], b[0]) or abs(a[0] - b[0]) <= h):
                uf.union(i, j)                                    
    blocks = []
    for g in uf.groups():
        ls = [lines[i] for i in g]
        if max(l[2] - l[0] for l in ls) < cfg["min_width_ratio"] * W and not touches[g[0]]:
            continue                                               
        rows = {round((l[1] + l[3]) / 2 / max(l[4], 1)) for l in ls}
        blocks.append({"bbox": union_box(ls), "lines": [l[:4] for l in ls], "n_lines": len(rows),
                       "line_h": float(np.median([l[4] for l in ls]))})
    return blocks


def _same_place(a, b, cfg):
    ha, hb = a["bbox"][3] - a["bbox"][1], b["bbox"][3] - b["bbox"][1]
    return (max(ha, hb) / max(1.0, min(ha, hb)) <= cfg["match_height_ratio"]
            and overlap_min(a["bbox"], b["bbox"]) >= cfg["match_overlap"])


def track_blocks(det_frames, W, H, cfg):
    tracks = []
    for fr in det_frames:
        t, used = fr["t"], set()
        blocks = frame_blocks(fr["boxes"], W, H, cfg)
        for tr in sorted(tracks, key=lambda tr: -tr["items"][-1][0]):
            last_t, last_b = tr["items"][-1]
            if last_t >= t or t - last_t > cfg["max_gap_s"] + 1e-6:
                continue
            cand = [k for k, b in enumerate(blocks) if k not in used and _same_place(last_b, b, cfg)]
            if cand:
                k = max(cand, key=lambda k: overlap_min(last_b["bbox"], blocks[k]["bbox"]))
                tr["items"].append((t, blocks[k]))
                used.add(k)
        tracks += [{"items": [(t, b)]} for k, b in enumerate(blocks) if k not in used]
    for tr in tracks:
        tr["bbox"] = union_box([b["bbox"] for _, b in tr["items"]])
        tr["line_h"] = float(np.median([b["line_h"] for _, b in tr["items"]]))
        tr["t0"], tr["t1"] = tr["items"][0][0], tr["items"][-1][0]
    tracks.sort(key=lambda tr: -len(tr["items"]))
    kept = []
    for tr in tracks:
        if not any(k["t0"] <= tr["t0"] and tr["t1"] <= k["t1"] and len(tr["items"]) <= cfg["absorb_max_frames"]
                   and overlap_min(tr["bbox"], k["bbox"]) >= cfg["absorb_overlap"] for k in kept):
            kept.append(tr)
    return sorted(kept, key=lambda tr: (tr["t0"], tr["bbox"][1]))


def track_regions(tracks, W, H, cfg):
    step = 1 / cfg["det_fps"]
    for tr in tracks:
        m = cfg["margin_ratio"] * tr["line_h"]
        x0, y0, x1, y1 = tr["bbox"]
        tr["region"] = [max(0, int(x0 - m)) // 2 * 2, max(0, int(y0 - m)) // 2 * 2,
                        min(W, int(np.ceil(x1 + m))) // 2 * 2, min(H, int(np.ceil(y1 + m))) // 2 * 2]
        tr["window"] = (tr["t0"] - step, tr["t1"] + step)


def collect_crops(frame_iter, tracks, scale):

    for tr in tracks:
        tr["scan"] = []
    row_max = col_max = None
    for t, g in frame_iter:
        row_max = g.max(axis=1) if row_max is None else np.maximum(row_max, g.max(axis=1))
        col_max = g.max(axis=0) if col_max is None else np.maximum(col_max, g.max(axis=0))
        for tr in tracks:
            if tr["window"][0] - 1e-6 <= t <= tr["window"][1] + 1e-6:
                x0, y0, x1, y1 = (round(v * scale) for v in tr["region"])
                tr["scan"].append((t, g[y0:y1, x0:x1]))
    return row_max, col_max


def edge_map(g):
    g = g.astype(np.float32)
    e = np.zeros_like(g)
    e[:, :-1] += np.abs(np.diff(g, axis=1))
    e[:-1, :] += np.abs(np.diff(g, axis=0))
    return e


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def profile_shift(pa, pb, max_shift):
    n = len(pa)
    best = (0, -1.0)
    for v in range(-max_shift, max_shift + 1):
        a, b = (pa[v:], pb[:n - v]) if v >= 0 else (pa[:n + v], pb[-v:])
        c = ncc(a, b)
        if c > best[1]:
            best = (v, c)
    return best


def crawl_speed(scan, scan_fps, scale, cfg):
    """연속 scan 프레임 사이 가로 이동량으로 흐르는 자막 판정. 원본 px/초 (왼쪽으로 흐르면 양수) 또는 None."""
    profs = [edge_map(g).sum(axis=0) for _, g in scan]
    if len(profs) < 3:
        return None
    max_shift = max(2, int(len(profs[0]) * cfg["crawl_max_shift_ratio"]))
    shifts = [v for v, c in (profile_shift(a, b, max_shift) for a, b in zip(profs, profs[1:]))
              if c >= cfg["crawl_min_corr"]]
    if len(shifts) < max(2, cfg["crawl_min_frac"] * (len(profs) - 1)):
        return None
    v = float(np.median(shifts))
    moving = np.mean([np.sign(s) == np.sign(v) and s != 0 for s in shifts])
    speed = v * scan_fps / scale
    if moving < cfg["crawl_min_frac"] or abs(speed) < cfg["crawl_min_speed"]:
        return None
    return round(speed, 1)


def crawl_like(tr, speed, W, cfg):
    blocks = [b for _, b in tr["items"]]
    edge = cfg["edge_touch_ratio"] * W
    touch = np.mean([b["bbox"][0] <= edge or b["bbox"][2] >= W - edge for b in blocks])
    return (speed >= cfg["crawl_min_speed"]
            and np.median([b["bbox"][2] - b["bbox"][0] for b in blocks]) >= cfg["crawl_min_width_ratio"] * W
            and touch >= cfg["crawl_min_edge_frac"] and np.median([b["n_lines"] for b in blocks]) == 1)


def _refine(scan, ok, t_in, t_out_prev, t_out_next, t_in_last):
    ts = [t for t, _ in scan]
    start = t_in
    for t in sorted((t for t in ts if t_out_prev < t < t_in), reverse=True):
        if not ok(t):
            break
        start = t
    end = t_in_last
    for t in sorted(t for t in ts if t_in_last < t < t_out_next):
        if not ok(t):
            break
        end = t
    return start, end


def split_cards(tr, cfg, scan_fps, scale, duration, W):
    step = 1 / cfg["det_fps"]
    frame_dt = 1 / scan_fps
    by_t = dict(tr["scan"])
    near = lambda t: by_t[min(by_t, key=lambda s: abs(s - t))]
    det_ts = [t for t, _ in tr["items"]]
    inside = [(t, g) for t, g in tr["scan"] if tr["t0"] - 1e-6 <= t <= tr["t1"] + 1e-6]
    speed = crawl_speed(inside, scan_fps, scale, cfg)
    if speed is not None and not crawl_like(tr, speed, W, cfg):
        speed = None                                                # 움직이는 장면 속 글자 → static으로 처리

    if speed is not None:
        energy = {t: float(edge_map(g).mean()) for t, g in tr["scan"]}
        ref_e = np.median([energy[t] for t, _ in inside])
        ok = lambda t: energy[t] >= cfg["presence_energy_ratio"] * ref_e
        start, end = _refine(tr["scan"], ok, tr["t0"], tr["t0"] - step, tr["t1"] + step, tr["t1"])
        groups = [(det_ts, start, end, None)]
    else:
        # 1) det 시점 단위로 문구가 바뀐 곳에서 나눔
        edges = {t: edge_map(near(t)) for t in det_ts}
        runs = [[det_ts[0]]]
        for a, b in zip(det_ts, det_ts[1:]):
            if ncc(edges[a], edges[b]) < cfg["split_ncc"]:
                runs.append([])
            runs[-1].append(b)
        # 2) run마다 기준 edge map(중앙값 → 움직이는 배경은 지워지고 글자만 남음)으로 경계 다듬기
        groups = []
        for i, run in enumerate(runs):
            ref = np.median(np.stack([edges[t] for t in run]), axis=0)
            prev_t = runs[i - 1][-1] if i > 0 else run[0] - step
            next_t = runs[i + 1][0] if i + 1 < len(runs) else run[-1] + step
            scores = {t: ncc(edge_map(g), ref) for t, g in tr["scan"]}
            ok = lambda t, s=scores: s[t] >= cfg["present_ncc"]
            start, end = _refine(tr["scan"], ok, run[0], prev_t, next_t, run[-1])
            groups.append((run, start, end, scores))
        for i in range(1, len(groups)):                             # 이웃 카드끼리 겹치면 가운데에서 자름
            (ra, sa, ea, ca), (rb, sb, eb, cb) = groups[i - 1], groups[i]
            if ea >= sb:
                mid = round((ra[-1] + rb[0]) / 2, 3)
                groups[i - 1], groups[i] = (ra, sa, min(ea, mid), ca), (rb, max(sb, mid + frame_dt), eb, cb)

    cards = []
    for run, start, end, scores in groups:
        t_end = min(duration, end + frame_dt)                       # 마지막 프레임이 화면에 머무는 시간까지
        rep_t = (max((t for t in scores if start <= t <= end), key=scores.get) if scores
                 else run[len(run) // 2])
        blocks = [b for t, b in tr["items"] if t in run]
        cards.append({"t_start": round(start, 2), "t_end": round(t_end, 2), "duration": round(t_end - start, 2),
                      "kind": "crawl" if speed is not None else "static", "crawl_px_s": speed,
                      "bbox": [round(v) for v in union_box([b["bbox"] for b in blocks])],
                      "n_lines": int(np.median([b["n_lines"] for b in blocks])),
                      "line_h_px": round(float(np.median([b["line_h"] for b in blocks])), 1),
                      "n_det": len(run), "rep_t": round(rep_t, 2)})
    return cards


def interval_iou(a, b):
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return max(0.0, inter) / (max(a[1], b[1]) - min(a[0], b[0]))


def group_screens(cards, min_iou):
    uf = UnionFind(len(cards))
    for i, a in enumerate(cards):
        for j in range(i + 1, len(cards)):
            b = cards[j]
            if interval_iou((a["t_start"], a["t_end"]), (b["t_start"], b["t_end"])) >= min_iou:
                uf.union(i, j)
    screens = []
    for k, g in enumerate(sorted(uf.groups(), key=lambda g: min(cards[i]["t_start"] for i in g))):
        main = max(g, key=lambda i: (cards[i]["kind"] == "static", cards[i]["n_lines"], cards[i]["duration"]))
        for i in g:
            cards[i]["screen_id"] = k
        screens.append({"id": k, "t_start": min(cards[i]["t_start"] for i in g),
                        "t_end": max(cards[i]["t_end"] for i in g), "rep_t": cards[main]["rep_t"],
                        "card_ids": sorted(cards[i]["id"] for i in g)})
    return screens


def union_length(intervals):
    total, cur = 0.0, None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0.0)


def content_box(row_max, col_max, scale, W, H, thr):
    rows, cols = np.where(row_max > thr)[0], np.where(col_max > thr)[0]
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, W, H]
    return [int(cols[0] / scale), int(rows[0] / scale), min(W, int(np.ceil((cols[-1] + 1) / scale))),
            min(H, int(np.ceil((rows[-1] + 1) / scale)))]
