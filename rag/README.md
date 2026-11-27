# 04 규정 기반 RAG · 05 LLM 판단

## 파일
| 파일 | 내용 |
|---|---|
| `04_rag_search.ipynb` | 검색 노트북: BM25 + BGE-M3(FAISS) + 리랭커, 평가 |
| `rag_search.py` | 04의 검색 장치를 다른 노트북에서 불러 쓰는 모듈 |
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

Google 시트의 ‘검사기준표’를 수정했다면 `rag/data/checks.csv`에서 해당 항목을 업데이트합니다. 노트북이 사용하는 추가 열도 유지합니다.
