# 06 수정안 생성 · 시각화 (이미지·영상 광고)

위험고지 표시가 기준에 못 미치는 광고만, **원본 디자인의 결을 지키면서** 고친 수정안을 만든다.
설계 원칙과 상황별 처방은 [DESIGN_SYSTEM.md](DESIGN_SYSTEM.md) 참고.

- 기준을 이미 통과한 광고는 고치지 않음 → **원본 유지**
- 광고 **규격(px)·영상 길이는 그대로**. 필요한 공간은 광고 안에서 찾음
- 광고마다 **상황(매체·고지 컨테이너·배경)을 먼저 분류**하고, 상황에 맞는 디자인 방법을 조합해 여러 후보안을 만듦
- 고친 뒤 **전처리와 같은 계산식**(`preprocessing/image/image_prep/metrics.py`)으로 다시 재서 통과한 안 중 원본을 가장 덜 바꾸는 안을 추천
- 어떤 안으로도 안 되면 억지로 맞추지 않고 **디자이너 브리프**(필요한 공간·안 된 이유)를 남김
- 05 LLM 판정 결과가 있으면 필수 문구 추가(B·D1)와 표현 위치 표시(C·E)도 함께 함

## 폴더 구성

```
remediation/
├─ README.md
├─ DESIGN_SYSTEM.md        ← 상황 진단 축, 디자인 방법, 상황별 처방, 판형 기준, 영상 체계
├─ config.yaml             ← 기준값(05와 동일)·판형 프로필·단계 한도·글꼴 라이브러리
├─ remediate.py            ← 실행 (이미지, 판형별)
├─ remediate_video.py      ← 실행 (영상)
├─ export_summary.py       ← 리포트 요약용 통합 결과 파일 (report_input.json)
├─ rules/phrases.yaml      ← ① 내용: 항목별 추가 문구(B·D1)·표현 수정 방향(C·E)
├─ fonts/                  ← 글꼴 11종 (모두 OFL, OFL.txt) — 개발 환경과 상관없이 같은 결과
├─ examples/judge_sample/  ← 05 결과 형식의 테스트용 샘플 (실제 판정 아님)
├─ report/                 ← 팀 내부 확인용 PDF와 생성 스크립트
└─ remedy/
   ├─ situation.py  ← 상황 분류 (매체·컨테이너·배경·고지 구조)
   ├─ diagnose.py   ← A2·A3·A4 판정 (판형 단위로 변환), 고칠 문단·목표 크기
   ├─ color.py      ← WCAG 명암비, LAB, 색상 유지 명도 조정, 광고 팔레트·글자색 고르기
   ├─ render.py     ← 글꼴 라이브러리, 재조판, 상황별 대비 처리, 공간 확보(여백 회수·사진 다듬기·비주얼 축소)
   ├─ content.py    ← 05 판정 결과 읽기 (문구 추가·표현 표시)
   ├─ measure.py    ← 전처리 측정 함수 재사용
   ├─ verify.py     ← 재측정·재판정
   ├─ video.py      ← 영상: 타임라인 계획, 고지 자막 자리·글자색, 엔드 카드 당기기
   └─ viz.py        ← 후보안 비교 패널
```

## 판정 기준

| 항목 | 지표 | 기준 |
|---|---|---|
| A2 명암비 | 위험고지 최소 WCAG 명암비 | ≥ 3.0 |
| A3 글자 크기 | 위험고지 글자 높이 중앙값 (판형별) | 모바일 5.9pt(390pt 폭) / 인쇄 9pt·신문 10pt → 실제 글자 높이 2.13mm·2.36mm |
| A4 헤드라인 대비 | 위험고지/헤드라인 높이비 | ≥ 0.075 (영상 0.10) |
| 영상 A5·A6·A7 | 노출 비율·초당 글자 수·가로 화면 글자 | ≥ 1/3 · ≤ 197 · ≥ 5.06pt |

### 판형 프로필

같은 이미지라도 출력 판형에 따라 실제 글자 크기가 달라서, **모바일·B5·A4·B4·A3·신문 전면·A2** 7개 판형마다 따로 판정하고 수정안을 만든다 (`config.yaml profiles`).
인쇄 글자 높이 = 기준 pt × 25.4/72 × 0.67 (한글 실제 글자 높이 / 글꼴 크기, 실측).

## 이미지: 후보안

| 후보 | 방법 |
|---|---|
| 대비 처리 | 상황별 — 단색: 글자 명도 / 띠·박스: 글자·띠 중 덜 바뀌는 쪽 / 사진: 아래쪽 톤 페이드. 그린 뒤 같은 식으로 재서 모자라면 한 단계씩 더 |
| A 줄 간격 | 고지 영역 안에서 줄 간격만 정리 (글자 높이의 1.25배까지) |
| B 여백 회수 | A + 위아래와 거의 같은 '평평한 배경 줄'을 여러 곳에서 조금씩 회수 |
| T 사진 다듬기 | A + 고지 바로 위 사진의 아래쪽을 조금 잘라냄 (높이의 6%까지, 글자 없는 줄만) |
| C 비주얼 축소 | A + 고지 위쪽 내용을 같은 비율로 축소 (90%까지), 테두리는 세로로만 |
| D 조합 | B → T → C를 나눠 씀 (축소 86%까지) |
| H 높이 확장 | 상세페이지(폭만 고정)는 높이를 늘림 |

추천 순서: H → A → B → T → C → D (통과한 것 중 원본을 덜 건드리는 순).

### 글꼴

`fonts/`의 11종(Pretendard, Noto Sans KR, IBM Plex Sans KR, Gothic A1, 나눔고딕, 고운돋움, Noto Serif KR, 나눔명조, 고운바탕, 나눔손글씨 펜, 도현)을 **디자인 컨셉별로** 두고,
광고마다 가장 긴 위험고지 줄의 **글자 폭 비율(자간 ±6% 허용)·획 굵기·세로 획 분포**가 가장 비슷한 글꼴·굵기 하나를 골라 그 광고 전체에 씀.
고지문은 가독성이 우선이라 명조·손글씨·제목용은 확실히 더 비슷할 때만 고르고, 원본 글자가 아주 작으면(20px 안팎) 고딕 안에서만 고름.

## 영상

영상 자체를 다시 만들지 않고, **위험고지가 나온 화면의 수정 전/후 이미지 + 타임라인(자막 시간대 전/후)**을 만든다 (`--render`를 주면 수정 mp4도 생성).

| 항목 | 처방 |
|---|---|
| A5 노출·A6 속도 | ① 고지 자리가 빈 장면에 같은 자막을 이어서 표시 (그 장면에서 빈 자리로 옮기기 포함) ② 앞 컷 끝을 다듬고 엔드 카드를 일찍 시작 (음성·길이 그대로) |
| A7 크기·A4 대비 | 원래 단·기준선 → 아래 안전선 → 위 안전선 순으로 실제로 비어 있는 폭을 재서 그 폭에 맞춰 다시 씀 |
| A2 명암비 | 박스 없이 글자색으로 — 원래 색 → 광고에 쓰인 색 → 원래 색의 명도 조정. 마지막 수단으로만 글자 바로 뒤 국소 톤다운 |
| 원래 작은 글자 | 정지 화면은 지운 배경 위에 다시 쓰고, 움직이는 화면은 그 자리만 블러 (헤드라인 등 다른 글자와 겹치는 쪽은 제외) |
| 흐르는 자막 | 같은 띠 자리에 고정 자막으로 다시 씀 |

## 실행

```bash
cd remediation
python remediate.py --all                         # 이미지 전체 × 판형 7개
python remediate.py --ids P021 --profiles A4,A3    # 일부
python remediate_video.py --all                   # 영상 전체 (ffmpeg 필요)
python export_summary.py                          # 리포트 요약용 report_input.json
python report/make_report.py                      # 팀 내부 확인용 PDF
python remediate.py --ids P021,AD07 --judge-dir examples/judge_sample --profiles A4 --out ../data/processed/remediation/content_sample   # ① 내용 수정 시험
```

필요 패키지: `numpy`, `pillow`, `pyyaml` (requirements.txt에 있음), 영상은 `ffmpeg`.

## 출력 (`data/processed/remediation/`)

| 경로 | 내용 |
|---|---|
| `<판형>/json/{ad_id}.json` | 상황·글꼴·결정·판정 전/후·후보안별 통과 여부/불가 이유/지표·추천안·디자이너 브리프·① 내용 |
| `<판형>/viz/{ad_id}.jpg` | 비교 패널 (git에는 A4·모바일만) |
| `<판형>/fixed/` | 추천안·후보안 원본 크기 이미지 (용량이 커서 git 제외, 실행하면 생성) |
| `<판형>/summary.csv`, `profiles_summary.csv` | 판형별 요약, 광고 × 판형 결정표 |
| `video/json`, `video/viz`, `video/frames` | 영상 결과, 비교 패널, 고지 화면 수정 전/후 이미지 |
| `report_input.json` | **리포트 요약용 통합 파일** (아래) |

## 리포트 연동 (`report_input.json`)

서비스 결과 화면의 리포트 요약은 이 파일 하나만 읽으면 되도록 모아 둠 (`python export_summary.py`).

```jsonc
{
  "generated_at": "...",
  "thresholds": {"contrast": 3.0, "ratio_image": 0.075, ...},
  "profiles": {"mobile": "모바일 피드", "A4": "A4 인쇄", ...},
  "images": [{
    "ad_id": "P021", "kind": "image", "situation": "세로 · 브랜드 푸터 패널 · 단색 배경",
    "content_add": [...], "content_flag": [...],             // ① 내용 (05 판정이 있을 때)
    "profiles": {
      "A4": {
        "label": "A4 인쇄", "decision": "수정안",              // 원본 유지 / 수정안 / 디자이너 검토 / 고지 없음 ...
        "checks_before": [{"id": "A2", "name": "명암비", "value": 2.54, "threshold": 3.0, "ok": false}, ...],
        "checks_after":  [...],
        "recommended": {"name": "A", "label": "A안 · 고지 영역 안에서 줄 간격 정리", "changed_pct": 3.1,
                        "size_factor": 1.25, "rows_removed": null, "rows_trimmed": null, "visual_scale": null,
                        "contrast_treatment": "띠·박스 색 명도 조정 ..."},
        "candidates": [{"name": "B", "label": "...", "passed": false, "why_not": ["평평한 배경 줄이 ..."]}, ...],
        "designer_brief": null,
        "panel_image": "data/processed/remediation/A4/viz/P021.jpg",
        "fixed_image": "data/processed/remediation/A4/fixed/P021.png"
      }, ...
    }
  }],
  "videos": [{
    "ad_id": "V110", "kind": "video", "brand": "...", "duration": 15.0, "decision": "수정안",
    "checks_before": [...], "checks_after": [...],
    "timeline": {"before": [[10.4, 13.0]], "after": [[10.4, 13.0], [5.1, 7.0]], "end_card_hold": null},
    "frames": [{"card": 9, "where": "원래 구간", "t": 11.9, "before": "...jpg", "after": "...jpg"}],
    "edit_sheet": ["자막 9: ..."], "why_not": [], "panel_image": "..."
  }]
}
```

## 한계

- 다시 쓰는 문구는 OCR 결과라 저해상도(예: 640×480 영상)에서는 오타가 그대로 나옴 → 원문 문구를 입력으로 받는 단계가 필요
- 얼굴 위치는 따로 찾지 않음 (글자 블록만 보호). 인물 위 배치는 패널로 확인 필요
- 로고·QR 재배치, 다단 전환, 흩어진 각주 통합은 아직 미구현 (DESIGN_SYSTEM.md 8장)
- 표준 문구(`rules/phrases.yaml`)는 준법 검토 전 초안
