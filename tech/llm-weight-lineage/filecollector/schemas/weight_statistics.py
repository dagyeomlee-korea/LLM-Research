from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class TensorStatistics:
    """
    purpose: 텐서 하나의 fingerprint 통계와 저장 메타데이터를 표현한다.
    input: 분석 대상 repo/file/tensor 식별자, dtype/shape, 통계 값.
    processing: service 계층에서 계산한 값을 repository/CLI 계층에 전달 가능한 불변 객체로 묶는다.
    return/side effects: `to_dict` 호출 시 직렬화 가능한 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    repo_id: str
    revision: str | None
    file_path: str
    tensor_name: str
    layer_idx: int | None
    module_type: str
    dtype: str
    shape: list[int]
    num_elements: int
    mean: float | None
    std: float | None
    skewness: float | None
    kurtosis: float | None
    excess_kurtosis: float | None
    l2_norm: float | None
    max_abs: float | None
    q99_abs: float | None
    q999_abs: float | None
    sparsity: float | None
    metadata: dict[str, Any]
    analyzed_at: datetime

    @classmethod
    def empty(
        cls,
        *,
        repo_id: str,
        revision: str | None,
        file_path: str,
        tensor_name: str,
        layer_idx: int | None,
        module_type: str,
        dtype: str,
        shape: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> "TensorStatistics":
        """
        purpose: 값이 없는 텐서 또는 지원하지 않는 dtype의 저장 행을 만든다.
        input: repo/file/tensor 식별자와 텐서 기본 메타데이터.
        processing: 원소 수를 shape에서 계산하고 통계 필드는 None으로 채운다.
        return/side effects: `TensorStatistics` 인스턴스를 반환하며 외부 상태는 변경하지 않는다.
        """

        num_elements = 1
        for dim in shape:
            num_elements *= dim
        return cls(
            repo_id=repo_id,
            revision=revision,
            file_path=file_path,
            tensor_name=tensor_name,
            layer_idx=layer_idx,
            module_type=module_type,
            dtype=dtype,
            shape=shape,
            num_elements=num_elements,
            mean=None,
            std=None,
            skewness=None,
            kurtosis=None,
            excess_kurtosis=None,
            l2_norm=None,
            max_abs=None,
            q99_abs=None,
            q999_abs=None,
            sparsity=None,
            metadata=metadata or {},
            analyzed_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """
        purpose: CLI 출력과 테스트 검증에 사용할 직렬화 dict를 만든다.
        input: 현재 `TensorStatistics` 인스턴스.
        processing: dataclass를 dict로 변환하고 datetime은 ISO 문자열로 변환한다.
        return/side effects: 새 dict를 반환하며 외부 상태는 변경하지 않는다.
        """

        data = asdict(self)
        data["analyzed_at"] = self.analyzed_at.isoformat()
        return data
