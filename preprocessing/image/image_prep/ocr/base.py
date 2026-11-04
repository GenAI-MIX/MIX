from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np


@dataclass
class OCRBlock:
    text: str
    quad: list        
    confidence: float


class OCREngine(ABC):
    name = "base"

    @abstractmethod
    def recognize(self, image_rgb: np.ndarray) -> list[OCRBlock]:
        """RGB uint8 (H, W, 3) 이미지에서 줄 단위 텍스트 블록을 반환한다."""

    def describe(self) -> dict:
        return {"engine": self.name}


def build_engine(ocr_cfg) -> OCREngine:
    engine = ocr_cfg["engine"]
    if engine == "paddle":
        from .paddle import PaddleOCREngine
        return PaddleOCREngine(**ocr_cfg.get("paddle", {}))
    raise ValueError(f"알 수 없는 OCR 엔진: {engine}")
