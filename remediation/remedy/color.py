"""색 계산. 수정은 '원본 글자색의 색상(a*, b*)은 그대로 두고 밝기(L*)만' 옮기는 것이 원칙."""
import numpy as np

_M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
_WHITE = np.array([0.95047, 1.0, 1.08883])


def _lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb):
    return float(_lin(rgb) @ [0.2126, 0.7152, 0.0722])


def contrast(a, b):
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def to_lab(rgb):
    xyz = (_M @ _lin(rgb)) / _WHITE
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.array([116 * f[1] - 16, 500 * (f[0] - f[1]), 200 * (f[1] - f[2])])


def from_lab(lab):
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f ** 3 > 0.008856, f ** 3, (f - 16 / 116) / 7.787) * _WHITE
    lin = np.clip(np.linalg.solve(_M, xyz), 0, 1)
    c = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(int)


def delta_e(a, b):
    return float(np.linalg.norm(to_lab(a) - to_lab(b)))


def fit_ink(fg, bg, target):
    """fg와 같은 색상을 유지하며 bg 대비 target 명암비를 넘는, fg에 가장 가까운 색."""
    fg, bg = [int(v) for v in fg], [int(v) for v in bg]
    if contrast(fg, bg) >= target:
        return fg
    lab = to_lab(fg)
    best = None
    for direction in (-1, 1) if luminance(bg) > luminance(fg) else (1, -1):
        for step in np.arange(1, 101, 0.5):
            L = lab[0] + direction * step
            if not 0 <= L <= 100:
                break
            for chroma in (1.0, 0.8, 0.6, 0.4, 0.2, 0.0):   # 색역 밖이면 채도만 조금씩 줄임
                c = from_lab([L, lab[1] * chroma, lab[2] * chroma])
                if contrast(c, bg) >= target:
                    best = [int(v) for v in c]
                    break
            if best:
                return best
    return [0, 0, 0] if luminance(bg) > 0.18 else [255, 255, 255]
