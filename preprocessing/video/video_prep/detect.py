import os
import numpy as np
from image_prep.geometry import quad_bbox, quad_height


class TextDetector:
    def __init__(self, **kwargs):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr
        from paddleocr import TextDetection

        self._params = dict(kwargs)
        self._version = paddleocr.__version__
        self._det = TextDetection(**kwargs)

    def detect(self, rgb):
        r = self._det.predict(np.ascontiguousarray(rgb[:, :, ::-1]))[0]
        out = []
        for poly, score in zip(r["dt_polys"], r["dt_scores"]):
            q = np.asarray(poly, dtype=float).tolist()
            out.append([round(v, 1) for v in quad_bbox(q)] + [round(quad_height(q), 1), round(float(score), 3)])
        return out

    def describe(self):
        return {"engine": "paddleocr.TextDetection", "version": self._version, **self._params}
