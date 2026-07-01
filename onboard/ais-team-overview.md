# AIS팀 소개 및 제품화 현황

AIS팀은 쿤텍의 AI를 이용한 사업들을 수행하기 위해 만들어진 AI 사업 조직입니다.

현재 Local LLM을 이용하여 솔루션을 어떻게 만들고 그것을 제품화할지 고민하며, 각 구성원들이 업무에 임하고 있습니다.

저희는 기본적으로 Cloud 기반 LLM을 적극 이용하여 기획 및 개발을 하고 있으며, LLM의 부작용 또한 경험하고 있습니다.

## 1. 주요 역할 분담

- 국내 각 기술 과제들에 대한 제안 및 입찰 자료 작성
- 각종 전시회 참여 및 부스 운영
- LLM에 대한 부분 학습 모델 생성
- 로컬 LLM을 이용한 제품화 기획
- 온프렘 및 클라우드 연동 제품 개발
- 사내 IT 인프라 관리
- Cloud IT 인프라 관리

## 2. Prototype 및 제품화 현황

현재까지 prototype 또는 제품화 중인 목록은 다음과 같습니다.

| 번호 | 제품 이름 | 진행도 | 설명 |
| --- | --- | --- | --- |
| 1 | [AIBOM](https://coontec.atlassian.net/wiki/spaces/AIS/folder/1160478724?atlOrigin=eyJpIjoiMWY3MzYzMzYxMDI0NDE3ZWEwZmZjNGZkNTY0YmQ0NmQiLCJwIjoiYyJ9) | beta | AIBOM 개념을 제품화한 솔루션/서비스 |
| 2 | [N2SF DOCS](https://coontec.atlassian.net/wiki/spaces/AIS/pages/1112506369) | prototype | LLM 기반 text 구문 분석을 이용한 문서 CSO 분류기 |
| 3 | [LLM Gate](https://coontec.atlassian.net/wiki/spaces/AIS/pages/1270120450) | beta | RMS(반입시스템)에서 사용되는 LLM 분석(동적/정적) gate 솔루션 |
| 4 | [AI Redteam](https://coontec.atlassian.net/wiki/x/AQBDRQ) | prototype | LLM 모델을 레드티밍 공격 후 분석하는 솔루션 |
| 5 | AI Guard | prototype | AI 질의 또는 응답에 대한 필터/차단 솔루션 |
| 6 | [SPLX / AI Guardrail](https://coontec.atlassian.net/wiki/spaces/AIS/folder/1097531411?atlOrigin=eyJpIjoiYjJhNGE4MzJlNTAyNDFiYmI1NWQyYzQxYWY0MGU4MjYiLCJwIjoiYyJ9) | production | 지스케일러 제품 협업 판매용 |

## 3. 내부 운영 시스템 목록

제품화를 위해 내부 운영 중인 [시스템 목록](https://coontec.atlassian.net/wiki/spaces/AIS/folder/1093894152?atlOrigin=eyJpIjoiNGRiY2M5ZjAwYjFkNDA1ZTkyOTIwOWU5OGYzZTIxYzUiLCJwIjoiYyJ9)은 다음과 같습니다.

| 번호 | 이름 | 설명 |
| --- | --- | --- |
| 1 | probe-loader | probe 스크랩 데이터 DB 적재 |
| 2 | aibom-scraper / aibom-etl / aibom-loader / aibom-admin | 클라우드 인프라 기반 AI BOM 데이터 스크랩 및 raw 데이터 사내 DB 적재<br>AI BOM DB 생성 및 데이터 관리 툴 |
| 3 | LLM QA Bench | LLM에 대한 사내 기준 성능 벤치 및 QA 툴 |
