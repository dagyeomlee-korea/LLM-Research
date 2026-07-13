from __future__ import annotations

from typing import Any

from filecollector.schemas.weight_statistics import TensorStatistics


def compare_kurtosis(
    stats_a: list[TensorStatistics],
    stats_b: list[TensorStatistics],
    epsilon: float = 1e-12,
) -> list[dict[str, Any]]:
    """
    purpose: 모델 A/B의 공통 tensor에 대해 kurtosis delta와 relative delta를 계산한다.
    input: 두 모델의 `TensorStatistics` 리스트와 0 division 방지 epsilon.
    processing: tensor_name 교집합을 만들고 `kurtosis(B)-kurtosis(A)`와 상대 변화율을 계산한다.
    return/side effects: pairwise kurtosis delta row dict 리스트를 반환하며 외부 상태는 변경하지 않는다.
    """

    index_a = {row.tensor_name: row for row in stats_a if row.kurtosis is not None}
    index_b = {row.tensor_name: row for row in stats_b if row.kurtosis is not None}
    rows: list[dict[str, Any]] = []
    for tensor_name in sorted(set(index_a) & set(index_b)):
        a = index_a[tensor_name]
        b = index_b[tensor_name]
        kurtosis_a = float(a.kurtosis)
        kurtosis_b = float(b.kurtosis)
        delta = kurtosis_b - kurtosis_a
        rows.append(
            {
                "model_a_repo_id": a.repo_id,
                "model_b_repo_id": b.repo_id,
                "revision_a": a.revision,
                "revision_b": b.revision,
                "layer_idx": a.layer_idx,
                "module_type": a.module_type,
                "tensor_name_a": a.tensor_name,
                "tensor_name_b": b.tensor_name,
                "kurtosis_a": kurtosis_a,
                "kurtosis_b": kurtosis_b,
                "delta_kurtosis": delta,
                "relative_delta_kurtosis": delta / (abs(kurtosis_a) + epsilon),
            }
        )
    return rows
