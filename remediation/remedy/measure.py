"""재측정: 전처리(preprocessing/image/image_prep/metrics.py)와 같은 계산식을 그대로 가져다 쓴다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "preprocessing" / "image"))
from image_prep.metrics import analyze_block, _glyph_height  # noqa: E402
from image_prep.geometry import ink_mask, to_gray  # noqa: E402

PEAK = 0.25


def glyph_height(rows):
    return _glyph_height(rows, PEAK)


__all__ = ["analyze_block", "glyph_height", "ink_mask", "to_gray"]
