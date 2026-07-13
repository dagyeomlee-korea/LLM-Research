# Weight Distance 분석 명세

## 목적

- SHA-256 중복 제거 후 Qwen3.5 고유 모델 7개의 실제 weight 원소 거리를 계산한다.
- 0.8B와 2B family를 분리하고 family 내부에서만 동일 이름·형상 tensor를 비교한다.
- 전체 tensor 관점과 language core 관점을 분리해 vision/embedding/기타 대형 tensor의 지배 효과를 확인한다.
- 기존 tensor별 kurtosis 차이와 실제 weight distance가 같은 결론을 지지하는지 비교한다.

## 비교 대상

- 0.8B 고유 모델 4개: 공식 Instruct, 공식 Base, CloudGoat JP-Tuned, Huihui abliterated
- 2B 고유 모델 3개: 공식 Instruct, 공식 Base, Huihui abliterated
- SHA-256 동일 복제본은 계산하지 않고 대표 모델과 0거리로 매핑한다.
- 0.8B와 2B 사이의 직접 원소 비교는 수행하지 않는다.

## View 정의

### all

- 이름·shape·dtype이 같은 모든 tensor를 포함한다.
- vision, embedding, MTP, language model을 모두 포함한다.

### language_core

- tensor 이름이 `model.language_model.layers.`로 시작한다.
- module type이 attention Q/K/V/O, MLP gate/up/down, norm 중 하나다.
- embedding, vision, MTP, 기타 tensor는 제외한다.

## Tensor별 거리

동일 원소 위치의 weight를 `A`, `B`라고 할 때:

```text
l2 = ||A - B||_F
relative_l2_a = ||A - B||_F / (||A||_F + epsilon)
symmetric_l2 = ||A - B||_F / sqrt(||A||_F^2 + ||B||_F^2 + epsilon)
cosine_distance = 1 - <A, B> / (||A||_F ||B||_F + epsilon)
kurtosis_abs_delta = |kurtosis(A) - kurtosis(B)|
```

## 집계

- parameter-weighted global distance: tensor별 제곱합·dot을 합산한 뒤 거리 계산
- tensor-balanced distance: tensor별 `symmetric_l2`의 median과 p95
- module distance: module type별 global/median/p95
- kurtosis distance: 동일 tensor의 absolute delta median/p95
- distance matrix: family와 view별 global symmetric L2, cosine distance, median kurtosis delta

global 값은 큰 tensor 영향이 크고 tensor median은 각 tensor를 동일 가중한다. 두 값을 함께 보고한다.

## 처리 방식

- 두 safetensors를 mmap으로 열고 같은 tensor를 제한된 NumPy chunk로 동시에 순회한다.
- 모델 전체 weight를 RAM에 적재하지 않는다.
- 모델 쌍은 NAS 메모리 보호를 위해 순차 처리한다.
- tensor별 결과는 JSONL, pair/module/matrix 결과는 JSON으로 기록한다.

## 검증 기준

- 동일 파일 self-distance는 L2 0, cosine distance 0이다.
- A-B와 B-A의 symmetric L2/cosine distance는 동일하다.
- 모든 pair에서 공통 tensor 수, shape/dtype skip 수, 비교 원소 수를 기록한다.
- NaN/Inf distance가 없어야 한다.
- synthetic F32/BF16 fixture에서 수작업 거리와 일치해야 한다.

## 해석 제한

- 낮은 weight distance는 계보 근거지만 parent-child 방향을 단독 확정하지 않는다.
- cosine distance는 scale 변화에 둔감하므로 L2와 함께 사용한다.
- kurtosis distance는 분포 형태 변화이며 실제 원소별 이동량을 대신하지 않는다.
- 모델 공개 시점, 학습 claim, CKA/subspace 분석을 결합해야 최종 계보 판단이 가능하다.

## 실행 결과 (2026-07-13)

- 고유 모델 7개, family 내부 9 pair를 계산했다.
- tensor-pair 4,824행, shape/dtype skip 0, 비유한 matrix 값 0이다.
- full 실행은 7분 31초, exit code 0, 최대 RSS 약 3.44GiB, swap 0이었다.
- 0.8B Instruct 기준 all/core symmetric L2는 Base 0.02004/0.04729, CloudGoat 0.01075/0.04215, Huihui 0.00301/0.01153이다.
- 2B Instruct 기준 all/core symmetric L2는 Base 0.02178/0.05658, Huihui 0.00255/0.00936이다.
- 결과는 NAS `~/qwen35-weight-distance-results`의 JSON/JSONL에 저장했다.
