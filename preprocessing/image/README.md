# 이미지 광고 전처리 (포스터·배너·상세페이지)

## 폴더 구성

```
preprocessing/image/
├─ README.md           
├─ config.yaml          ← 모든 설정값 (경로·업스케일·분류 기준·매체별 표시 폭 등)
├─ preprocess.py        ← 실행 진입점
├─ rules/keywords.yaml  ← 분류 키워드 (코드 수정 없이 편집 가능)
└─ image_prep/        
```


## 처리 단계

| 단계 | 내용 | 코드 |
|---|---|---|
| 0. 정리 | ad_id 부여, SHA-256으로 완전 중복 제외, EXIF 회전·sRGB 변환·투명 배경 합성 | `image_prep/normalize.py` |
| 1. OCR 사본 | Lanczos 업스케일(폭 4000px, 최대 3배, AI 초해상도 미사용) 후 가로 전체 폭의 띠로 세로 분할(겹침 400px 이상). 긴 고지문 줄이 가로로 잘리지 않음 | `image_prep/ocr_input.py` |
| 2. OCR | 축소본(아주 큰 헤드라인)·원본(중간 글자)·업스케일 띠(작은 글자) 3패스를 원본 좌표로 병합 | `image_prep/ocr/` |
| 3. 분류 | 아래 "분류 규칙" 참고 | `image_prep/classify.py`, `image_prep/layout.py`, `rules/keywords.yaml` |
| 4. 측정 | 글자 높이(실제 글자 픽셀), 표시 크기 = 글자 높이 × 표시 폭 / 원본 폭, 위험고지/헤드라인 높이비, WCAG 명암비 | `image_prep/metrics.py` |
| 5. 패키지 | 위험고지 영역 crop (경계에 걸친 텍스트 블록과 같은 줄의 이어지는 칸까지 포함해 잘리지 않게), 광고별 JSON, 요약 CSV, 텍스트 기준 중복 목록 | `image_prep/package.py`, `image_prep/dedupe_text.py` |
| 검수 | type별 색으로 OCR 박스를 그린 이미지 | `image_prep/viz.py` |



### 분류 규칙 (순서대로 적용)

1. **그래픽 글자 제외**: 그림 속 기호·숫자 조각(계산기 버튼, 로고 기호 등)은 JSON에서 뺍니다.
2. **줄 단위 키워드**: `risk_notice > return_claim > other` 순으로 `rules/keywords.yaml`의 키워드·정규식을 봅니다. 키워드에 안 걸린 큰 글씨는 `headline`, 나머지는 `unknown`(나중에 LLM이 분류).
3. **문단 단위 키워드**: OCR 줄을 문장·문단으로 다시 이은 뒤, 문단 안 어디든 위험고지 키워드가 있으면 문단의 모든 줄을 `risk_notice`로 합니다. 한 문장 안에서 줄마다 판정이 갈리지 않습니다.
4. **작은 글씨 고지 영역**: 가장 큰 글씨의 0.3배 이하 글씨가 촘촘히 모인 덩어리는 영역 안 모든 줄을 `risk_notice`로 합니다. 조건: 4줄 이상이면서 줄당 40자 이상(문단형) 또는 15줄 이상(여러 단), 혹은 줄 수와 상관없이 줄당 50자 이상. 상품 조건 표·이벤트 박스처럼 칸이 짧은 것은 제외됩니다.
5. **제목 기준 섹션**: "투자 유의사항" 같은 제목 줄 아래 섹션은 다음 제목 전까지 통째로 `risk_notice`, "이벤트 유의사항" 섹션은 통째로 `other`입니다.

기준값은 모두 `config.yaml`(`text_paragraph`, `classify.disclosure`, `classify.sections` 등)에서 조정할 수 있습니다.

## 출력 (`data/processed/image/`)

| 경로 | 내용 |
|---|---|
| `json/{ad_id}.json` | 광고 1건의 OCR 블록·문단·측정값·crop 경로 (아래 예시) |
| `summary.csv` | 전체 광고 요약 1행씩 |
| `duplicates.csv` | OCR 텍스트 유사도 기준 "문구 거의 동일"·"고지문 공유" 광고 쌍. 크롭·해상도가 달라 이미지 해시로 못 잡는 중복(예: P087 ≈ AD46)과 같은 캠페인 시안 변형을 찾음. train/test 분리 시 같은 쪽에 둘 것 |
| `crops/{ad_id}_risk_XX.png` | 위험고지 영역 crop (VLM 입력) |
| `viz/{ad_id}.png` | 검수용 시각화. 빨강 risk_notice · 파랑 headline · 주황 return_claim · 회색 other · 보라 unknown · 연회색 점선 그래픽 제외 · 초록 점선 crop 범위 |
| `ocr_raw/{ad_id}.json` | OCR 원본 결과 캐시 (좌표 4점, 패스별 통계 등 디버깅 정보 포함) |
| `normalized/{ad_id}.png` | 회전·색공간 변환·투명 배경 합성 등으로 원본과 달라진 경우에만 생성 |

```json
{
  "ad_id": "P003", "media": null, "product": null, "brand": null, "size": [2115, 2967], "format": "JPEG",
  "image": "data/raw/images/P003.jpg",
  "ocr_blocks": [
    {"id": 11, "text": "*이 금융투자상품은 예금자보호법에 따라 …", "bbox": [112, 2643, 1997, 2666],
     "conf": 0.91, "type": "risk_notice", "h_px": 15.0, "contrast": 10.8}
  ],
  "paragraphs": [
    {"id": 7, "type": "risk_notice", "text": "*이 금융투자상품은 예금자보호법에 따라 … 광고물 유효기간 2023.6.30까지",
     "bbox": [112, 2643, 1998, 2726], "block_ids": [11, 12, 13, 14]}
  ],
  "metrics": {"risk_headline_ratio": 0.142, "risk_contrast_min": 10.85,
              "mobile_feed": {"width_pt": 390, "risk_min_pt": 2.77, "risk_median_pt": 2.95, "headline_max_pt": 20.84}},
  "crops": [{"path": "data/processed/image/crops/P003_risk_00.png", "bbox": [61, 2527, 1760, 2647]},
            {"path": "data/processed/image/crops/P003_risk_01.png", "bbox": [94, 2624, 2016, 2744]}],
  "vlm_judgment": null, "label": null
}
```

- `ocr_blocks`는 OCR이 읽은 **한 줄 단위** 블록입니다 (OCR은 한 줄씩 검출하므로 문장이 줄마다 나뉨). 크기·명암비는 줄 단위로 잽니다.
- `paragraphs`는 이어지는 줄과 같은 줄에서 쪼개진 조각을 **문장·문단으로 다시 이은 것**입니다 (`text`: 이어진 전체 문장, `block_ids`: 포함된 줄). 다음 단계 LLM·RAG에는 이 문단 텍스트를 쓰면 됩니다.
- `image`는 VLM에 넣을 전체 이미지입니다. 정규화로 바뀐 게 없으면 원본 경로를 가리킵니다.
- `bbox`는 원본 이미지 픽셀 좌표 `[x0, y0, x1, y1]`입니다.
- `h_px`는 OCR 박스가 아니라 박스 안 **실제 글자 픽셀 높이(잉크 높이)**입니다. 사진이 겹쳐 측정이 튀는 줄은 같은 문단 줄들의 중앙값으로 대신합니다.
- `contrast`는 WCAG 명암비입니다. 사진 배경처럼 측정을 신뢰하기 어려우면 `null`입니다.
- `metrics`의 `*_pt`는 폭 390으로 줄였을 때의 글자 높이입니다 (폰 화면 390px 폭 환산과 같은 숫자). `risk_headline_ratio`는 위험고지 글자 높이 중앙값 ÷ 가장 큰 헤드라인 글자 높이입니다.
- 그래픽 글자로 제외된 블록은 JSON에 없고 `ocr_raw/`와 viz에서 확인할 수 있습니다.
