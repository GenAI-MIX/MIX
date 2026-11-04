import re
import numpy as np
import yaml
from .geometry import UnionFind

RULE_TYPES = ("risk_notice", "return_claim", "other")
SPLIT_WINDOW = 15


def strip_text(text):
    return re.sub(r"\s+", "", text).lower()


def _fuzzy_regex(kw):
    alts = [re.escape(kw[:i]) + ".?" + re.escape(kw[i + 1:]) for i in range(len(kw))]
    alts += [re.escape(kw[:i]) + "." + re.escape(kw[i:]) for i in range(1, len(kw))]
    return re.compile("|".join(alts))


class Rules:
    def __init__(self, path, fuzzy_min_len):
        with open(path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        self.keywords = {t: [strip_text(k) for k in (d.get(t) or {}).get("keywords") or []] for t in RULE_TYPES}
        self.patterns = {t: [re.compile(p) for p in (d.get(t) or {}).get("patterns") or []] for t in RULE_TYPES}
        self.brands = [strip_text(b) for b in (d.get("other") or {}).get("brands") or []]
        self.fuzzy = [(k, _fuzzy_regex(k)) for k in self.keywords["risk_notice"] if len(k) >= fuzzy_min_len]

    def match(self, ns):
        for t in RULE_TYPES:
            for k in self.keywords[t]:
                if k in ns:
                    return t, f"kw:{k}"
            for p in self.patterns[t]:
                if p.search(ns):
                    return t, f"re:{p.pattern}"
            if t == "risk_notice":
                for k, rx in self.fuzzy:
                    if rx.search(ns):
                        return t, f"kw~:{k}"
            if t == "other":
                for b in self.brands:
                    if b in ns and len(ns) <= len(b) + 3:
                        return t, f"brand:{b}"
        return None, None

    def split_hit(self, upper_ns, lower_ns):
        a, b = upper_ns[-SPLIT_WINDOW:], lower_ns[:SPLIT_WINDOW]
        joined = a + b
        for k in self.keywords["risk_notice"]:
            if k in joined and k not in a and k not in b:
                return k
        return None


MEANINGFUL = re.compile(r"[가-힣%원]")
CORE_CHARS = re.compile(r"[0-9A-Za-z가-힣]")


def split_graphic_text(blocks, cfg):
    short = []
    for i, b in enumerate(blocks):
        core = len(CORE_CHARS.findall(b["text"]))
        if not MEANINGFUL.search(b["text"]) and core <= cfg["max_core_chars"]:
            short.append((i, core))
    drop = {i for i, core in short if core <= 1}
    cand = [i for i, core in short if core > 1]
    if cand:
        uf = UnionFind(len(short))
        idx = [i for i, _ in short]
        for a in range(len(idx)):
            for c in range(a + 1, len(idx)):
                ba, bc = blocks[idx[a]], blocks[idx[c]]
                dx = (ba["bbox"][0] + ba["bbox"][2] - bc["bbox"][0] - bc["bbox"][2]) / 2
                dy = (ba["bbox"][1] + ba["bbox"][3] - bc["bbox"][1] - bc["bbox"][3]) / 2
                if (dx * dx + dy * dy) ** 0.5 <= cfg["cluster_dist_ratio"] * max(ba["h_px"], bc["h_px"]):
                    uf.union(a, c)
        for g in uf.groups():
            if len(g) >= cfg["cluster_min"]:
                drop |= {idx[k] for k in g}
    return [b for i, b in enumerate(blocks) if i not in drop], [b for i, b in enumerate(blocks) if i in drop]


def _aligned(a, c, h, span_ratio):
    ox = min(a[2], c[2]) - max(a[0], c[0])
    if ox <= 0:
        return False
    return ox / max(a[2] - a[0], c[2] - c[0]) >= span_ratio or abs(a[0] - c[0]) <= 1.5 * h


def find_disclosure_regions(blocks, max_h, ns, cfg):
    small = [i for i, b in enumerate(blocks) if b["text_h"] <= cfg["max_height_ratio"] * max_h]
    uf = UnionFind(len(small))
    for a in range(len(small)):
        ba = blocks[small[a]]
        for c in range(a + 1, len(small)):
            bc = blocks[small[c]]
            ha, hc = ba["text_h"], bc["text_h"]
            if max(ha, hc) / max(1e-6, min(ha, hc)) > cfg["height_ratio_max"]:
                continue
            h = (ha + hc) / 2
            A, C = ba["bbox"], bc["bbox"]
            if abs((A[1] + A[3]) - (C[1] + C[3])) / 2 < 0.5 * h:            # 같은 줄
                if max(C[0] - A[2], A[0] - C[2]) <= cfg["same_line_gap_ratio"] * h:
                    uf.union(a, c)
            elif max(C[1] - A[3], A[1] - C[3]) <= cfg["line_gap_ratio"] * h and _aligned(A, C, h, cfg["span_ratio"]):
                uf.union(a, c)
    regions = []
    for g in uf.groups():
        idx = [small[k] for k in g]
        med_chars = float(np.median([len(ns[i]) for i in idx]))
        if med_chars >= cfg["long_line_chars"]:
            regions.append(sorted(idx))
        elif len(idx) >= cfg["min_lines"] and (med_chars >= cfg["min_median_chars"]
                                               or len(idx) >= cfg["dense_min_lines"]):
            regions.append(sorted(idx))
    return sorted(regions, key=lambda r: r[0])


def _heading_kind(s, cfg):
    if len(s) > cfg["max_heading_chars"]:
        return None
    if any(k in s for k in cfg["event_headings"]):
        return "event"
    if any(k in s for k in cfg["risk_headings"]):
        return "risk"
    return None


def find_sections(blocks, ns, cfg):
    order = sorted(range(len(blocks)), key=lambda k: (blocks[k]["bbox"][1], blocks[k]["bbox"][0]))
    out = []
    for i in order:
        kind = _heading_kind(ns[i], cfg)
        if not kind:
            continue
        H = blocks[i]
        hx, hh = H["bbox"][0], H["text_h"]
        members, bottom = [], H["bbox"][3]
        for j in order:
            b = blocks[j]
            if j == i or b["bbox"][1] < H["bbox"][1] + 0.5 * hh:
                continue
            if b["bbox"][2] < hx or b["bbox"][0] < hx - hh:          # 섹션 단(column) 밖
                continue
            if b["bbox"][1] - bottom > cfg["gap_ratio"] * hh:        # 큰 빈 공간 → 섹션 끝
                break
            heading_like = (len(ns[j]) <= cfg["max_heading_chars"] and abs(b["bbox"][0] - hx) <= 0.5 * hh
                            and (_heading_kind(ns[j], cfg) or b["text_h"] >= 0.8 * hh))
            if heading_like:                                        # 다음 제목 → 섹션 끝
                break
            members.append(j)
            bottom = max(bottom, b["bbox"][3])
        out.append((kind, i, members))
    return out


def classify_blocks(blocks, rules, cfg):
    ns = [strip_text(b["text"]) for b in blocks]
    matched = [rules.match(s) for s in ns]

    by_para = {}
    for i, b in enumerate(blocks):
        by_para.setdefault(b["text_para_id"], []).append(i)
    for idxs in by_para.values():
        t, rule = rules.match("".join(ns[i] for i in idxs))
        if t == "risk_notice":
            for i in idxs:
                if matched[i][0] != "risk_notice":
                    matched[i] = ("risk_notice", f"paragraph_{rule}")

    low = cfg["low_conf"]
    ref = [b["text_h"] for b, (t, _), s in zip(blocks, matched, ns)
           if t not in ("risk_notice", "other") and b["confidence"] >= low and len(s) >= 2]
    max_h = max(ref) if ref else None

    for b, (t, rule) in zip(blocks, matched):
        big = max_h is not None and t not in ("risk_notice", "other") and b["text_h"] >= cfg["headline_size_ratio"] * max_h
        b["size_tier"] = "headline" if big else "body"
        b["type_hint"] = None
        if b["confidence"] < low:
            b["type"], b["type_rule"], b["type_hint"] = "unknown", "low_conf", t or ("headline" if big else None)
        elif t:
            b["type"], b["type_rule"] = t, rule
        elif big:
            b["type"], b["type_rule"] = "headline", "size"
        else:
            b["type"], b["type_rule"] = "unknown", None

    # 작은 글씨가 여러 줄 모인 고지 영역은 키워드와 상관없이 전부 risk_notice
    for b in blocks:
        b["disclosure_id"] = None
    if max_h:
        for k, idx in enumerate(find_disclosure_regions(blocks, max_h, ns, cfg["disclosure"])):
            for i in idx:
                b = blocks[i]
                b["disclosure_id"] = k
                if b["type"] != "risk_notice":
                    b["type_hint"] = b["type"] if b["type"] != "unknown" else b["type_hint"]
                    b["type"], b["type_rule"] = "risk_notice", "disclosure_region"

    # "투자 유의사항" 같은 제목 줄 아래 섹션은 통째로 risk_notice, "이벤트 유의사항" 섹션은 통째로 other
    for b in blocks:
        b["section"] = None
    for kind, head, members in find_sections(blocks, ns, cfg["sections"]):
        new_type = "risk_notice" if kind == "risk" else "other"
        for i in [head] + members:
            b = blocks[i]
            b["section"] = kind
            if b["type"] != new_type:
                b["type_hint"] = b["type"] if b["type"] != "unknown" else b["type_hint"]
                b["type"], b["type_rule"] = new_type, f"{kind}_section"
    return max_h
