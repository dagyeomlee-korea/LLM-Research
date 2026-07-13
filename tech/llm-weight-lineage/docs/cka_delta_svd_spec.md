# CKA + Delta SVD 고급 Fingerprint 명세

## 목적

- 실제 weight distance 다음 단계로 행렬 구조 유사도와 update subspace를 분석한다.
- CloudGoat 0.8B의 광범위 core 변경은 sampled linear CKA로 평가한다.
- Huihui 0.8B/2B의 O/down projection 변경은 randomized SVD로 low-rank·공통 변환 가능성을 평가한다.
- Base→Instruct delta와 Instruct→후보 delta의 cosine alignment를 계산해 update 방향 관계를 확인한다.

## 대상

### CloudGoat

- anchor: `Qwen/Qwen3.5-0.8B`
- candidate: `CloudGoat/Qwen3.5-0.8B-JP-Tuned-v1.0`
- module: Q/K/V/O, MLP gate/up/down 중 실제 weight distance가 0보다 큰 rank-2 tensor

### Huihui

- anchor: 공식 Qwen3.5 Instruct 0.8B/2B
- candidate: 동일 크기 Huihui abliterated
- module: 실제로 변경된 `attn_o`, `mlp_down`
- Base anchor도 함께 읽어 delta alignment를 계산한다.

## Linear CKA

같은 tensor row index에서 최대 2,048행을 균등 간격으로 선택한다.

```text
Xc = X - column_mean(X)
Yc = Y - column_mean(Y)
linear_CKA = ||Xc^T Yc||_F^2 /
             (||Xc^T Xc||_F * ||Yc^T Yc||_F)
cka_distance = 1 - linear_CKA
```

- 동일 weight는 CKA 1이다.
- CKA는 scale과 일부 isotropic 변화에 둔감하므로 L2와 함께 사용한다.
- row sampling은 tensor 이름과 무관한 균등 index로 결정해 재현성을 유지한다.
- shape가 다르거나 rank-2가 아닌 tensor는 제외한다.

## Randomized Delta SVD

```text
D = W_candidate - W_instruct
D ≈ U_k S_k V_k^T
```

- 기본 rank `k=16`, oversampling `p=8`, power iteration 1회
- tensor 이름 SHA-256에서 deterministic seed 생성
- 결과: top singular values, top-k energy fraction, 50/80/90% energy 도달 rank 또는 `>k`
- 원본 tensor는 float32로 한 개씩 로드하고 처리 후 해제한다.

top-k energy fraction이 높고 작은 rank에서 80~90%에 도달하면 low-rank update 후보로 본다. 도달하지 못하면 광범위/high-rank 변화로 분류한다.

## Delta Alignment

```text
D_base_to_instruct = W_instruct - W_base
D_instruct_to_candidate = W_candidate - W_instruct
alignment = cosine(D_base_to_instruct, D_instruct_to_candidate)
```

- `+1`: 같은 방향
- `0`: 거의 직교
- `-1`: 반대 방향

alignment는 방향성 단서이며 학습 순서나 parent-child를 단독 확정하지 않는다.

## 출력

- tensor별 CKA/L2/kurtosis 결합 JSONL
- tensor별 delta SVD singular value/energy/alignment JSONL
- module별 median/p95와 energy summary JSON
- 상세 Markdown 보고서

## 검증

- 동일 matrix CKA는 1이다.
- 상수 scale matrix CKA는 1에 가깝다.
- synthetic rank-1 delta의 top-1 energy는 1에 가깝다.
- 같은 방향/반대 방향 delta alignment는 각각 +1/-1이다.
- 실제 BF16 tensor에서 NaN/Inf가 없고 동일 seed 재실행 결과가 같아야 한다.

## 자원 정책

- NAS는 2코어·3.7GiB RAM이므로 tensor 하나씩 순차 처리한다.
- 전체 모델 병렬 처리와 full exact SVD는 사용하지 않는다.
- 큰 tensor 처리 후 배열을 즉시 해제한다.
