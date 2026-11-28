# 04 규정 기반 RAG · 05 LLM 판단

## 파일
| 파일 | 내용 |
|---|---|
| `04_rag_search.ipynb` | 검색 노트북: BM25 + BGE-M3(FAISS) + 리랭커, 평가 |
| `rag_search.py` | 04의 검색 장치를 다른 노트북에서 불러 쓰는 모듈 |
| `05_llm_judge.ipynb` | 05 LLM 판단 노트북: 단계별 설명과 광고 1건 판정 |
| `judge.py` | 05의 판정 함수 모듈 (노트북·`run_judge.py`가 import) |
| `run_judge.py` | 05 판정을 여러 광고에 실행하는 명령줄 실행기 |
| `outputs/` | 05 판정 결과 `{ad_id}__{model}.json` · `.csv` |
| `data/cards.csv` | 규정 카드 190장 (Google 시트 '규정카드' 탭) |
| `data/eval_checks.csv` | 검색 평가 질문 30개와 정답 카드 |
| `cache/` | FAISS 인덱스 캐시 (자동 생성, 커밋 금지) |

## 맥(Apple Silicon)에서 실행
레포 루트(`MIX/`)에서:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install rank_bm25 kiwipiepy sentence-transformers faiss-cpu pandas ipykernel
```
- VS Code에서 `rag/04_rag_search.ipynb`를 열고 커널로 `.venv`를 고릅니다.
- 작업 디렉터리가 `MIX/rag/`인지 확인하고 셀을 위에서부터 실행합니다.
- 첫 실행 때 BGE-M3(약 2.3GB)와 리랭커(약 2.2GB) 모델을 내려받습니다.

## 카드가 바뀌면

Google 시트의 ‘규정카드’ 탭을 CSV로 내려받아 `rag/data/cards.csv`를 덮어씁니다. 카드 ID를 유지한 채 본문을 수정했다면 기존 `rag/cache/` 인덱스를 지우고 다시 실행합니다.

## 05 LLM 판단

`rag/05_llm_judge.ipynb`는 `rag_search.py`로 근거 카드를 찾고, `rag/data/checks.csv`의 30개 항목에 따라 광고를 판정합니다. 04와 같은 가상환경에 아래 패키지를 추가합니다.

```bash
python -m pip install openai anthropic python-dotenv
```

`rag/.env`에 사용하려는 서비스의 키를 설정합니다. `openai`로 실행할 때는 `OPENAI_API_KEY`, `claude`로 실행할 때는 `ANTHROPIC_API_KEY`를 사용합니다.

VS Code에서 05 노트북을 열고 04와 같은 커널을 선택합니다. 작업 디렉터리가 `MIX/rag/`인지 확인한 뒤 셀을 위에서부터 실행합니다. 광고 JSON은 레포의 `data/processed/{image,video}/json/`에서 읽습니다. 데이터가 별도 체크아웃에 있으면 `rag/.env`에 `MIX_REPO_DIR`로 해당 레포 경로를 지정합니다.

`judge_ad('AD01', 'openai')`로 광고 1건을 판정할 수 있습니다. 결과 JSON과 CSV는 `rag/outputs/`에 저장됩니다.

판정 함수는 `rag/judge.py`에 있습니다. 05 노트북과 `run_judge.py`가 이 모듈을 함께 import하므로, 판정 로직은 `judge.py`에서 고칩니다.

### 여러 광고 한 번에 판정 (`run_judge.py`)

`rag/` 폴더에서 실행합니다(같은 가상환경 사용).

```bash
cd rag
python run_judge.py --ids AD01,V101                       # 지정한 광고
python run_judge.py --all-images --skip-existing          # 이미지 광고 전체, 결과가 이미 있으면 건너뜀
python run_judge.py --all-videos --product 투자성          # 영상 중 투자성 상품만
python run_judge.py --all-images --provider claude --model claude-sonnet-5-5
python run_judge.py --all-images --skip-existing --workers 6   # 광고 6건을 동시에 판정
```

| 옵션 | 내용 |
|---|---|
| `--ids` | 쉼표로 구분한 광고 ID 목록 |
| `--all-images` / `--all-videos` | `data/processed/{image,video}/json/` 전체 (`--ids`와 함께 쓰면 합집합) |
| `--product` | 상품유형 필터 (`투자성`·`보장성`·`대출성`·`예금성`·`범위밖`, 아래 상품유형 분류 기준) |
| `--provider` | `openai`(기본) 또는 `claude` |
| `--model` | 모델 이름. 기본값은 openai `gpt-6-luna`, claude `claude-haiku-5-5` |
| `--skip-existing` | `{out-dir}/{ad_id}__{model}.json`이 있으면 건너뜀. 중간에 멈췄거나 실패한 광고만 다시 돌릴 때 사용 |
| `--out-dir` | 결과 폴더 (기본 `rag/outputs`) |
| `--workers` | 동시에 판정할 광고 수 (기본 1). 근거 검색은 한 번에 한 건씩, LLM 호출만 동시에 실행 |
| `--price-in` / `--price-out` | 1M 토큰당 달러. `run_judge.py`의 `PRICES` 표에 없는 모델의 비용을 계산할 때 사용 |

광고마다 상품유형, 매체, 판정 개수(위반·확인필요·적합·해당없음), 토큰, 누적 토큰과 예상 비용, 남은 시간을 출력합니다. 끝나면 전체 판정 분포를 출력합니다. 예상 비용은 정가 기준이며 캐시 할인은 반영하지 않습니다. 한 건이 실패해도 나머지는 계속 진행합니다. 실패한 광고 목록은 마지막에 출력되며, 같은 명령에 `--skip-existing`을 붙여 다시 실행하면 됩니다.

참고로 gpt-6-luna는 광고 1건에 입력 약 2.5만, 출력 약 8천 토큰(약 $0.007)을 씁니다.

**상품유형 분류.** 상품유형은 2단계로 정합니다(`judge.infer_product_detail`).
1. JSON의 `product`가 `ETF`·`증권`이면 투자성으로 정합니다.
2. 키워드(`KEYWORDS`): 한 유형만 나오고 '연금'이 없으면 그 유형으로 확정합니다.
3. 그 밖(여러 유형이 섞임, '연금' 있음, 아무 유형도 없음)에는 gpt-6-luna가 광고 문구 앞부분 1500자와 브랜드를 보고 정합니다. 결과는 `rag/cache/product_cache.json`에 저장해 재사용합니다.

키워드를 찾기 전에 예금자보호 고지문(예금자보호법, 예금보험공사)과 세금 안내(건강·연금보험료) 문구는 지웁니다. 판정 JSON의 `product_by`(json/keyword/llm/override), `product_reason`, `product_keywords`에 근거가 남습니다. 분류가 틀리면 `judge.py`의 `PRODUCT_OVERRIDE`에 적습니다. 분류에는 OpenAI 키가 필요합니다.

**검색 캐시.** 근거 카드 검색 질문은 `checks.csv`의 `query` 열이라 광고와 관계없이 같습니다. 그래서 `search_ids` 결과를 `rag/cache/search_cache.json`에 저장해 두고 다음 실행(노트북, 모델 비교 등)에서도 재사용합니다(`rag_search.search_ids_cached`). 캐시 키에는 질문, 상품유형, `k`, `mode`, `rrf_k`, `n_candidates`, 모델 이름과 함께 `cards.csv` 내용의 해시가 들어갑니다. 카드를 고치면 예전 결과는 자동으로 쓰이지 않습니다. 검색 결과는 캐시가 없을 때와 같습니다. 검색 로직이나 모델을 바꿨다면 `search_cache.json`을 지웁니다. 판정 JSON의 `retrieved_cards`에는 묶음(AD/B/CE)별로 LLM에 넘긴 카드 목록이 남습니다.

`rag/outputs/`의 `.json`(06 수정안 `remediation/`의 입력), `.csv`, 보고서 `.md`는 커밋하고 그 밖의 파일은 커밋하지 않습니다(`rag/.gitignore`).

Google 시트의 ‘검사기준표’를 수정했다면 `rag/data/checks.csv`에서 해당 항목을 업데이트합니다. 노트북이 사용하는 추가 열도 유지합니다.

## 위반 예시 평가 (`eval/`)

심의를 통과한 광고의 전처리 JSON을 일부러 고친 '위반 예시'로 05 판단의 탐지율을 잽니다. 이미지와 영상 파일은 건드리지 않습니다.

```bash
cd rag
python eval/make_variants.py                                   # 변형본 + 정답표 만들기 (LLM 호출 없음)
python eval/evaluate.py --provider openai --model gpt-6-luna   # 판정 + 보고서
python eval/evaluate.py --model gpt-6-luna --report-only       # 있는 결과로 보고서만 다시
python eval/evaluate.py --model gpt-6-luna --rejudge --ids AD01__m_B1,V101__m_B3 --baseline eval/runs/gpt-6-luna/results_v1.csv   # 바뀐 변형만 다시 판정 + 이전 대비 표
python eval/evaluate.py --model gpt-6-luna --runs 3 --single-report eval/runs/gpt-6-luna/report_1run.md   # 3회 반복 판정, 평균·최소~최대 요약
python run_judge.py --ad-dir eval/variants --ids AD01__m_B3 --out-dir eval/runs/test   # 변형 1건만 판정
```

| 파일 | 내용 |
|---|---|
| `eval/make_variants.py` | 원본 10건(AD01~03, V101~V126 중 7건)에 변형 9종을 적용 |
| `eval/variants/{image,video}/json/` | 변형 JSON `{원본}__{변형}.json`, 원본 `{원본}__original.json`. 상품유형은 `product_override`로 원본 판정값에 고정 |
| `eval/variants/skipped.csv` | 적용하지 않은 조합과 이유 (원본 판정에서 target 항목이 적합이 아니면 건너뜀) |
| `eval/labels.csv` | 정답표: `variant_id`, `base_id`, `mutation`, `target_check`, `expected_verdicts`, `collateral_removed`(다른 필수 문구까지 지워졌으면 기록), `delete_mode`(구간 통째 삭제 · 부분 삭제 수), `note` |
| `eval/evaluate.py` | 변형 판정 후 지표 계산. `--workers`, `--rejudge`, `--ids`(재판정할 변형), `--no-repeat`, `--report-only`, `--baseline`(이전 results.csv), `--note`(이전 대비 바뀐 점), `--runs N`(N회 반복 판정, `runs/{model}/xN/r1..rN`), `--single-report`(반복 평가 보고서 아래에 붙일 1회 실행 보고서) |
| `eval/runs/{model}/` | 판정 결과, `results.csv`(변형별 정확 판정·검토 전달·재확인 낮춤·판정·이유·부작용), `results_v1.csv`(첫 실행), `repeat/`(원본 반복 판정) |
| `eval/runs/{model}/x3/r1..r3/` | 3회 반복 판정 결과와 회차별 `results.csv`. `report_1run.md`는 B1 기준 보완 전 1회 실행 보고서 |
| `eval/report_{model}.md` | 보고서 |

**변형 9종**
- **표시 방법(측정값 변경):** `m_A2` 명암비 1.5 · `m_A3` 위험고지 글자와 헤드라인 비율 ×0.5(이미지) · `m_A5` 노출 시간 = 영상 길이×0.2(영상)
- **필수 문구 삭제:** `m_B1` 설명서·약관 권유 · `m_B3` 원금손실·투자자 귀속 · `m_B6` 예금자보호 · `m_D1` 심사필 문구와 `review_no`
- **금지 표현 추가(맨 앞 헤드라인):** `m_C2` "목표수익률 연 15%" · `m_C3` "국내 유일 최고 수익 ETF"

영상(`tv_cf`)의 `m_D1` 기대 판정은 `해당없음`입니다. 방송 광고는 공간 제약 시 심사필 표시를 생략할 수 있기 때문입니다(K-2-47-3). 05 규칙도 같습니다.

**문구 삭제 방법.** `paragraphs`, `ocr_blocks`, 영상 `cards`(crawl 포함), `risk_exposures`, `review_no`, `csv.review_no`에서 지웁니다.
- 글을 구간으로 나눕니다. 구분 기준은 `※`·`•`·`*`·`|`, OCR이 글머리를 숫자로 읽은 `1`, 문장 끝('다.', '니다' 뒤)입니다. `·`는 '투자설명서·집합투자규약'처럼 조항 안에서도 쓰여 구분자로 보지 않습니다.
- 해당 문구가 든 구간은 통째로 지웁니다.
- 같은 구간에 다른 필수 문구(B1·B3·B5·B6·B8·D1)가 함께 있으면 해당 조항만 지우고, 남는 이음말('및', 쉼표 등)과 지운 자리 앞뒤 3글자 이하 조각을 정리합니다. 이 정리 때문에 다른 필수 문구가 사라지면 조각 정리는 하지 않습니다.
- 대상 단어가 하나라도 남으면 멈춥니다. 다른 필수 문구가 같이 지워졌는지는 필드별로 검사해 `collateral_removed`에 남깁니다.

**지표**
- (가) 변형 종류별·전체 탐지율
  - 정확 판정률: target 판정이 기대 판정과 정확히 일치한 비율입니다.
  - 검토 전달률: target이 `위반`이나 `확인필요`로 나와 사람의 검토 목록에 오른 비율입니다. 기대가 `해당없음`이면 `해당없음`일 때 성공입니다.
  - 재확인으로 낮춤: 위반 재확인에서 `위반`이 `확인필요`로 낮아진 target 수입니다.
- (나) 원본 오탐: 원본에서 나온 `위반` 수입니다.
- (다) 부작용: target 외 항목이 원본과 달라진 수입니다. 연관 항목(C5, D2 등 코드로 따라 바뀌는 항목)은 따로 셉니다. 원본 반복 판정에서 달라지는 수를 기준선으로 함께 보여 줍니다.
- (라) 근거 검증 통과율, 위반 재확인 횟수, 비용.
