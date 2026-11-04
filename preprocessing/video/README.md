# 영상 광고 전처리 (TV CF·온라인 영상)

## 카드

같은 고지 문구가 같은 자리에 떠 있는 구간입니다. 영상의 고지는 크게 세 가지 형태라 이렇게 나눕니다.

| kind | 예 | 처리 |
|---|---|---|
| `static` | 화면 하단 2~4줄 고지 (V107 9.8초~), 엔드카드 고지 (V174) | 문구가 바뀌면 카드를 나눔. 대표 프레임 1장으로 OCR |
| `crawl` | 한 줄로 오른쪽→왼쪽으로 흐르는 고지 (V107 0~9.6초, V177) | 한 프레임엔 문장 일부만 있음. `crawl_px_s`(흐르는 속도) 기록. V3에서 프레임을 이어 붙여 OCR |

같은 시간에 함께 떴다가 사라지는 카드(엔드카드처럼 여러 블록으로 나뉜 고지)는 하나의 **화면(screen)**으로 묶고, 화면마다 원본 해상도 대표 프레임을 저장합니다.

## 처리 단계

| 단계 | 내용 | 코드 |
|---|---|---|
| V0. 정리 | 원본 CSV 기준 ad_id 부여, SHA-256 중복 검사, ffprobe(해상도·fps·길이·음성), 메타 CSV 동기화 | `video_prep/probe.py` |
| V1. 프레임 | ffmpeg로 디코딩 (OpenCV 미사용). 검출용 0.5초 간격 원본 해상도 RGB, 비교용 0.1초 간격 960px 회색조 (카드 후보 영역만 메모리에 보관) | `video_prep/frames.py` |
| V2-a. 검출 | PaddleOCR 글자 영역 검출만 (인식 없음 — 프레임마다 인식까지 하면 80개에 몇 시간) | `video_prep/detect.py` |
| V2-b. 카드 | 아래 "카드 분할 규칙" | `video_prep/cards.py` |
| V3. 글자·분류·측정 | 아래 "카드 읽기" | `video_prep/text.py` (+ `image_prep`) |
| 패키지 | 광고별 JSON, 요약 CSV, 화면별 대표 프레임, 흐르는 자막 띠, 검수용 타임라인 | `video_prep/package.py`, `video_prep/viz.py` |

### 카드 분할 규칙 (순서대로)

1. **줄 선택**: 검출 박스 중 작은 글씨(1080p 기준 박스 높이 약 9~75px)이면서 가로로 긴 줄(화면 폭 15% 이상)만 씁니다. 큰 헤드라인과 장면 속 짧은 글자 조각은 여기서 빠집니다. 후보 단계라 놓치지 않는 쪽으로 넓게 잡았습니다 (한 줄짜리 큰 고지가 70px인 경우가 있음, V169). 그래서 중간 크기 카피·대사 자막도 후보에 들어오고, V3 분류에서 걸러야 합니다.
2. **블록**: 위아래로 붙은 줄을 묶습니다. 단, 화면 좌우 끝에 닿는 줄(화면보다 긴 흐르는 자막)은 고정 고지와 묶지 않습니다 (V101처럼 고정 고지 바로 아래로 자막이 흐르는 경우).
3. **시간 추적**: 0.5초 간격 프레임에서 같은 자리의 블록을 잇습니다. 검출이 1초 이하로 빠져도 이어 붙이고, 한두 프레임만 쪼개져 생긴 조각은 버립니다.
4. **흐르는 자막 판정**: 0.1초 간격 프레임에서 블록 영역의 열 방향 edge 프로파일이 일정하게 움직이고, 그 움직임이 **오른쪽→왼쪽 초당 60px 이상 · 블록 폭이 화면 절반 이상 · 화면 좌우 끝에 닿은 한 줄**이면 `crawl`입니다. 카메라 이동·미끄러져 들어오는 카피·확대되는 로고도 가로로 움직이므로, 이 조건이 없으면 흐르는 자막으로 잘못 잡힙니다 (실제 자막 초당 120~380px, 장면 속 글자 20~30px).
5. **문구 변경**: `static`은 이웃한 검출 시점의 edge map 상관(NCC)이 0.5 미만이면 문구가 바뀐 것으로 보고 카드를 나눕니다. 같은 문구는 배경이 움직여도 0.8~1.0, 바뀌면 0.4 이하로 나옵니다.
6. **경계 다듬기**: 카드의 검출 시점 edge map 중앙값(움직이는 배경은 지워지고 글자만 남음)을 기준으로, 0.1초 간격 프레임이 기준과 닮은 동안을 카드가 떠 있는 시간으로 봅니다. 그래서 경계 오차는 0.1초 안팎입니다.

기준값은 모두 `config.yaml`의 `cards`, `scan`에서 조정합니다.

### 카드 읽기 (V3)

- **static 카드**: 화면 대표 프레임을 이미지 광고와 같은 방식으로 읽습니다. 다만 프레임 전체를 확대해 읽는 대신, **카드가 있는 가로 띠만 이미지와 같은 배율로 확대**해 읽고 프레임 전체는 축소본(큰 글씨)으로 한 번 읽습니다 (화면당 약 15초 → 4초). 그 뒤는 이미지와 똑같이 그래픽 글자 제외 → 문단 → 잉크 높이·대비 → `keywords.yaml` 분류를 하고, OCR 줄을 카드에 배정합니다. 카드 type은 카드 안 줄들의 type 중 글자 수가 가장 많은 것입니다. 단 **카드 문구에 위험고지 키워드(`keywords.yaml` risk_notice)가 하나도 없으면 위험고지 카드로 보지 않습니다** (`type_rule: no_risk_keyword`). 이미지의 '작은 글씨 고지 영역' 규칙은 키워드 없이도 risk_notice로 보는데, 영상에는 화면 속 기사·앱 화면처럼 작은 글씨가 빽빽한 장면이 많기 때문입니다 (V152 신문 기사).
- **crawl 카드**: 0.5초마다 띠를 잘라 가로 이동량만큼 새로 들어온 부분을 이어 붙인 긴 띠(`crawl/`)를 만들고, 1920px씩 480px 겹쳐 잘라 OCR한 뒤 겹쳐 읽힌 글자를 한 번만 남깁니다. 한 프레임엔 문장 일부만 있지만 이렇게 하면 문장 전체를 얻습니다 (V177: 60초 동안 흐른 고지 전체). type은 이어진 문장 전체에 `keywords.yaml` 규칙을 적용해 정합니다.
- **측정** (위험고지 줄만): 글자 수, **초당 글자 수 = 위험고지 글자 수 ÷ 카드가 떠 있는 시간**, 잉크 높이(px), 표시 크기(pt, `display_profiles`: 폰 세로 390 · 가로 844), WCAG 대비, 같은 화면 헤드라인 대비 비율. 원본 CSV의 초당 글자 수와 달리 카피·로고 글자를 세지 않습니다.
- **심의필·심사필 번호**: 문단 텍스트와 흐르는 자막 문장에서 정규식으로 뽑습니다.

## 출력 (`data/processed/video/`)

| 경로 | 내용 |
|---|---|
| `json/{ad_id}.json` | 영상 1건의 메타·화면·카드·위험고지 노출·OCR 줄·문단·지표 (아래 예시) |
| `summary.csv` | 전체 영상 요약 1행씩 (위험고지 노출 시간·최대 초당 글자 수·최소 글자 크기·대비·심의필 번호 등) |
| `frames/{ad_id}_sXX.png` | 화면별 원본 해상도 대표 프레임 (OCR 입력) |
| `crawl/{ad_id}_cXX.png` | 흐르는 자막을 이어 붙인 긴 띠 (OCR 입력) |
| `viz/{ad_id}.png` | 검수용 타임라인. 회색 원본 CSV 구간 · **빨강 위험고지 카드** · 연파랑 static · 연주황 crawl 후보, 아래에 화면별 대표 프레임과 카드 박스 |
| `scan/{ad_id}.json` | 검출 캐시 (0.5초마다 박스 목록) |
| `ocr_raw/{ad_id}.json` | OCR 캐시 (화면·띠 조각별 원본 OCR 결과) |

```json
{
  "ad_id": "V107", "media": "tv_cf", "product": null, "brand": "KB국민은행", "title": "...",
  "video": "data/raw/videos/영상/107_KB국민은행_0cxgu-Gx5gc.mp4", "size": [1920, 1080], "fps": 23.976,
  "duration": 15.015, "has_audio": true, "content_bbox": [0, 0, 1920, 1080],
  "review_no": ["심사필제26-05337호", "심의필제2026-5134-2호"],
  "csv": {"risk": "있음", "intervals": [[0.0, 15.0]], "review_no": "한국금융투자협회 심사필 제26-05337호"},
  "screens": [
    {"id": 4, "t_start": 9.8, "t_end": 15.02, "rep_t": 14.7, "card_ids": [5], "frame": "data/processed/video/frames/V107_s04.png", "headline_h_px": 69.0}
  ],
  "cards": [
    {"id": 0, "screen_id": 0, "t_start": 0.0, "t_end": 9.6, "duration": 9.6, "kind": "crawl", "crawl_px_s": 120.0,
     "bbox": [0, 1019, 1920, 1051], "n_lines": 1, "line_h_px": 23.5, "n_det": 20, "rep_t": 5.0,
     "type": "risk_notice", "text": "※가입 전 설명 청취 및 상품설명서·약관 필독 ※(DC/기업형IRP/개인형IRP)이 퇴직연금은 … 보호되지 않습니다.",
     "strip_w_px": 3228, "risk_chars": 183, "risk_cps": 19.1, "risk_text_h_px": {"min": 16.0, "median": 16.0},
     "risk_display_pt": {"mobile_feed": {"min": 3.25, "median": 3.25}, "mobile_landscape": {"min": 7.03, "median": 7.03}},
     "risk_contrast_min": 15.15, "risk_headline_ratio": 0.32, "strip": "data/processed/video/crawl/V107_c00.png"},
    {"id": 5, "screen_id": 4, "t_start": 9.8, "t_end": 15.02, "duration": 5.21, "kind": "static", "crawl_px_s": null,
     "bbox": [71, 986, 1857, 1055], "n_lines": 3, "line_h_px": 23.0, "n_det": 11, "rep_t": 14.7,
     "type": "risk_notice", "text": "※과세기준 및 방법 향후 변동 가능 ※연금 외 수령 시 … 광고물 유효기간 2026.10.26.~2027.09.30.까지",
     "block_ids": [5, 6, 7], "risk_chars": 336, "risk_cps": 64.5, "risk_text_h_px": {"min": 23.5, "median": 24.0},
     "risk_display_pt": {"mobile_feed": {"min": 4.77, "median": 4.88}, "mobile_landscape": {"min": 10.33, "median": 10.55}},
     "risk_contrast_min": null, "risk_headline_ratio": 0.348}
  ],
  "risk_exposures": [
    {"id": 0, "t_start": 0.0, "t_end": 9.6, "duration": 9.6, "kind": "crawl", "card_ids": [0], "risk_chars": 183, "risk_cps": 19.1, "bbox": [0, 1019, 1920, 1051], "text_card_id": 0, "text": "※가입 전 설명 청취 및 …"},
    {"id": 1, "t_start": 9.8, "t_end": 15.02, "duration": 5.22, "kind": "static", "card_ids": [5], "risk_chars": 336, "risk_cps": 64.4, "bbox": [71, 986, 1857, 1055], "text_card_id": 5, "text": "※과세기준 및 방법 …"}
  ],
  "ocr_blocks": [
    {"screen_id": 0, "id": 0, "text": "*b KB국민은행", "bbox": [68, 49, 371, 99], "conf": 0.99, "type": "other", "h_px": 34.0, "contrast": 16.8, "card_id": null}
  ],
  "paragraphs": [
    {"screen_id": 0, "id": 0, "type": "other", "text": "*b KB국민은행", "bbox": [68, 49, 371, 99], "block_ids": [0]}
  ],
  "metrics": {"n_cards": 11, "n_static": 10, "n_crawl": 1, "candidate_exposure_s": 14.82, "csv_exposure_s": 15.0,
              "csv_coverage": 0.987, "min_line_h_px": 22.5,
              "n_risk_cards": 2, "n_risk_crawl": 1, "n_risk_exposures": 2, "risk_short_exposures": 0,
              "risk_exposure_s": 14.82, "risk_csv_coverage": 0.987, "risk_chars": 519, "risk_cps_max": 64.4,
              "risk_unreadable_exposures": 2, "risk_text_h_min_px": 16.0,
              "risk_display_min_pt": {"mobile_feed": 3.25, "mobile_landscape": 7.03},
              "risk_contrast_min": 15.15, "risk_headline_ratio_min": 0.32},
  "label": null
}
```
(실제 V107에는 로고·헤드라인 등 후보 카드가 11개, 화면이 9개 있고 위는 일부만 보인 것입니다.)

- 시간은 모두 초입니다. `t_end`는 마지막으로 보인 프레임이 화면에서 사라지는 시점입니다 (마지막 프레임 + 0.1초).
- `bbox`는 원본 해상도 픽셀 좌표 `[x0, y0, x1, y1]`입니다.
- `line_h_px`는 카드 분할용 **검출 박스 높이**입니다. 글자 크기 지표는 `risk_text_h_px`(이미지 JSON `h_px`와 같은 잉크 높이)를 씁니다.
- `ocr_blocks`·`paragraphs`는 이미지 JSON과 같은 형식에 `screen_id`(어느 화면의 글자인지)와 `card_id`(어느 카드에 배정됐는지, 카드 밖이면 null)를 더한 것입니다. id는 화면 안에서의 번호입니다.
- `risk_exposures`는 같은 위험고지 문구가 여러 카드로 쪼개져 잡힌 것(나타나는 전환 프레임, 움직이는 고지)을 한 노출로 묶은 것입니다. **초당 글자 수(`risk_cps`)는 이 노출 단위로 잽니다.** 1초 미만 노출(`risk_short_exposures`)은 흐린 고지가 끊겨 잡혔거나 장면 속 글자일 수 있어 확인이 필요합니다.
- `risk_display_pt`는 잉크 높이 × 표시 폭 / 영상 폭입니다. `mobile_feed`(폰 세로, 390)는 이미지 광고와 같은 기준입니다.
- `risk_contrast_min`이 null이면 고지가 영상 위에 겹쳐 있어 대비 측정을 신뢰할 수 없는 경우입니다 (이미지와 같은 기준).
- `review_no`는 OCR 글자에서 뽑은 심의필·심사필 번호(띄어쓰기 제거)입니다. `csv.review_no`는 원본 CSV 값입니다.
- `csv`는 원본 CSV의 자동 분석값(macOS Vision OCR, 0.5초 단위, 고지 키워드 기준)입니다. 정답이 아니라 비교용입니다. CSV는 "금리·수수료"가 들어간 광고 카피까지 고지로 잡은 경우가 있습니다 (V125·V134·V147·V161).
- `content_bbox`는 모든 프레임에서 어두운 위아래·좌우 띠(letterbox)를 뺀 영역입니다.
