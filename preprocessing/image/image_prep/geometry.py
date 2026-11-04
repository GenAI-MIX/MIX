import numpy as np


def quad_bbox(quad):
    q = np.asarray(quad, dtype=float)
    return [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]


def quad_height(quad):
    q = np.asarray(quad, dtype=float)
    return float((np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2)


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def overlap_min(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    small = min(box_area(a), box_area(b))
    return ix * iy / small if small > 0 else 0.0


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def clip_box_int(b, w, h):
    return [max(0, int(np.floor(b[0]))), max(0, int(np.floor(b[1]))),
            min(w, int(np.ceil(b[2]))), min(h, int(np.ceil(b[3])))]


def to_gray(rgb):
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.uint8)


def otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = hist.sum()
    if total == 0:
        return 127
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    sigma_b[~np.isfinite(sigma_b)] = -1
    return int(np.argmax(sigma_b))


def ink_mask(gray_region):
    t = otsu_threshold(gray_region)
    hi = gray_region > t
    ring = np.concatenate([hi[:2].ravel(), hi[-2:].ravel(), hi[:, :2].ravel(), hi[:, -2:].ravel()])
    bg_is_hi = ring.mean() > 0.5
    return ~hi if bg_is_hi else hi


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)

    def groups(self):
        out = {}
        for i in range(len(self.p)):
            out.setdefault(self.find(i), []).append(i)
        return list(out.values())
