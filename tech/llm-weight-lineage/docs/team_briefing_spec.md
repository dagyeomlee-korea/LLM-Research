# Qwen3.5 계보 분석 팀 브리핑 명세

## 1. 목적

Qwen3.5 10개 모델의 가중치 분석 결과를 LLM과 관련 논문을 모르는 팀원이
10분 안에 이해할 수 있도록 설명한다. 발표 후 확인할 수 있는 상세 보고서와
회의용 슬라이드 원고를 함께 제공한다.

## 2. 독자와 설명 원칙

- 독자: LLM 구조, safetensors, 계보 복원 논문을 읽지 않은 팀원
- 모든 기술용어는 `쉬운 정의 -> 비유 -> 이 분석에서의 역할` 순서로 설명한다.
- 수식은 weight distance 등 재현에 필요한 최소 범위만 사용한다.
- 모델 이름의 `JP-Tuned`, `abliterated`는 제작자가 붙인 명칭으로 취급하며,
  실제 데이터나 기능 효과를 가중치만으로 확정하지 않는다.

## 3. 산출물

1. `qwen35_lineage_team_report.md`
   - 배경, 분석 파이프라인, 용어 사전, 알고리즘 원리, 결과표, MotherTree,
     계보 해석, 한계, 예상 질문을 포함한다.
2. `qwen35_lineage_10min_slides.md`
   - 10분 발표용 8장 구성, 슬라이드 문구, 발표자 노트, Mermaid 도식을 포함한다.
3. `Qwen3.5_10-model_lineage_team_report.docx`
   - 팀원이 별도 Markdown 도구 없이 열 수 있는 최종 공유본이다.
   - 10개 모델 전체 판정표, 10→7 dedup 설명, 알고리즘 흐름도, MotherTree PNG,
     기술용어 해설, 수치 결과, 증거 수준, 한계와 예상 질문을 포함한다.
4. `assets/qwen35_mothertree.png`, `assets/qwen35_analysis_pipeline.png`
   - DOCX에 삽입되는 고해상도 도식이다.

DOCX 생성은 단일 목적 Python 생성기를 사용한다. 기본 표준은 입력 데이터와 문서
구성을 상수·함수로 분리하고, 재실행 시 같은 경로를 안전하게 덮어쓰며, 함수마다
목적·입력·처리·반환/부작용을 설명하는 한국어 구조화 docstring을 유지하는 것이다.

## 4. 증거 표기 규칙

| 등급 | 의미 | 도식 표기 |
|---|---|---|
| 확정 | SHA-256이 동일해 다운로드한 weight 파일 전체 byte가 동일 | 굵은 실선/동일 그룹 |
| 강한 후보 | 같은 family에서 최근접이고 변화 위치·형태가 구조적으로 설명됨 | 점선 화살표 |
| 참고 후보 | Base/Instruct 명칭과 거리로 연결되지만 시간 방향을 weight만으로 확정 못함 | 회색 점선 화살표 |

MotherTree는 법적 출처나 실제 학습 이력을 증명하는 도구가 아니라, 현재 가중치
증거로 가장 타당한 관계를 정리한 `계보 후보 그래프`로 정의한다.

## 5. 포함할 검증 수치

- 입력: 0.8B 5개, 2B 5개, 약 29.35GB
- 통계: 5,600 tensor row, 오류 0
- SHA-256: 0.8B exact mirror 1개, 2B exact mirror 2개
- 고유 모델: 7개, family 내부 비교 pair 9개
- weight distance: 4,824 tensor-pair row, shape/dtype skip 0
- CloudGoat: Instruct 최근접, language-core 거리 비율 89.12%, CKA 0.991 이상
- Huihui: O/down 중심 변경, 100개 delta 모두 top-1 energy 99.46% 이상
- 검증: NAS 테스트 13개 통과, 고급 분석 재실행 결과 일치

10개 모델은 도식과 판정표에서 모두 개별 노드로 유지한다. 다만 pairwise 계산에서는
다음 SHA-256 동치 관계를 한 대표 weight로 축약한다.

- 0.8B: 5개 배포본 중 공식 Instruct와 unsloth가 동일해 고유 weight 4개
- 2B: 5개 배포본 중 공식 Instruct, unsloth, hamishivi가 동일해 고유 weight 3개
- 합계: 배포본 10개, 고유 weight `4 + 3 = 7개`

최종 판정표에는 10개 각각의 family, 기준 anchor, 관계 유형, 핵심 증거, 증거 수준을
누락 없이 표시한다.

## 6. 발표 성공 기준

발표를 들은 팀원이 다음 네 질문에 답할 수 있어야 한다.

1. 왜 저장소 이름이 다르다고 서로 다른 모델이라고 볼 수 없는가?
2. weight distance, kurtosis, CKA, SVD가 각각 무엇을 측정하는가?
3. CloudGoat와 Huihui의 변경 방식은 어떻게 다른가?
4. MotherTree의 실선과 점선은 왜 증거 수준이 다른가?

최종 DOCX는 다음 구조 검사를 통과해야 한다.

- ZIP/OpenXML 문서로 정상 열림
- 제목·본문·표·목록 스타일 존재
- 두 PNG 도식이 문서 내부 media로 포함됨
- 10개 모델 판정표가 누락 없이 포함됨
- 결론에서 exact mirror, CloudGoat, Huihui의 차이가 먼저 제시됨

## 7. 범위 밖

- 모델 응답 품질, 안전성, 일본어 성능의 행동 평가
- 공개 시점과 model card를 포함한 외부 provenance 검증
- 저작권·라이선스·무단 복제 여부의 법적 판단
- 0.8B와 2B weight의 직접 원소 대 원소 거리 비교
