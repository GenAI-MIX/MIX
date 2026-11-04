import os
import numpy as np
from .base import OCRBlock, OCREngine


class PaddleOCREngine(OCREngine):
    name = "paddleocr"

    def __init__(self, **kwargs):
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        import paddleocr
        from paddleocr import PaddleOCR

        self._params = dict(kwargs)
        self._version = paddleocr.__version__
        self._ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                              use_textline_orientation=False, **kwargs)

    def recognize(self, image_rgb):
        bgr = np.ascontiguousarray(image_rgb[:, :, ::-1])   
        r = self._ocr.predict(bgr)[0]
        out = []
        for text, score, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"]):
            if text.strip():
                out.append(OCRBlock(text=text, quad=np.asarray(poly, dtype=float).tolist(), confidence=float(score)))
        return out

    def describe(self):
        return {"engine": self.name, "version": self._version, **self._params}
