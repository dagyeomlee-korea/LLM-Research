from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from filecollector.analysis.safetensors_stream import (
    SafeTensorFile,
    TensorInfo,
    finite_or_none,
    validate_tensor_size,
)
from filecollector.analysis.tensor_classifier import classify_tensor
from filecollector.schemas.weight_statistics import TensorStatistics


class WeightStatisticsService:
    """
    purpose: safetensors 파일의 tensor별 weight fingerprint 통계를 계산한다.
    input: repo_id, safetensors path, histogram bin 수, chunk byte 크기.
    processing: 텐서를 하나씩 열고 두 번 순회해 moment 통계와 절대값 분위수를 계산한다.
    return/side effects: `TensorStatistics` iterator를 반환하며 저장은 repository 계층에 위임한다.
    """

    def __init__(
        self,
        histogram_bins: int = 8192,
        chunk_bytes: int = 8 * 1024 * 1024,
        engine: str = "auto",
    ) -> None:
        if histogram_bins < 2:
            raise ValueError("histogram_bins must be >= 2")
        if engine not in {"auto", "python", "numpy"}:
            raise ValueError("engine must be one of: auto, python, numpy")
        self.histogram_bins = histogram_bins
        self.chunk_bytes = chunk_bytes
        self.engine = engine

    def analyze_file(
        self,
        repo_id: str,
        path: str | Path,
        revision: str | None = None,
    ) -> Iterator[TensorStatistics]:
        """
        purpose: safetensors 파일 하나의 모든 텐서 통계를 순차 생성한다.
        input: Hugging Face repo_id와 로컬 safetensors 파일 경로.
        processing: 파일을 mmap으로 열고 텐서 단위로 validate/classify/statistics 계산을 수행한다.
        return/side effects: tensor별 `TensorStatistics`를 yield하며 파일 리소스는 context manager로 닫는다.
        """

        file_path = str(path)
        with SafeTensorFile(path, chunk_bytes=self.chunk_bytes) as st_file:
            for tensor in st_file.tensors():
                layer_idx, module_type = classify_tensor(tensor.name)
                try:
                    validate_tensor_size(tensor)
                    yield self._analyze_tensor(
                        repo_id,
                        revision,
                        file_path,
                        st_file,
                        tensor,
                        layer_idx,
                        module_type,
                    )
                except ValueError as exc:
                    yield TensorStatistics.empty(
                        repo_id=repo_id,
                        revision=revision,
                        file_path=file_path,
                        tensor_name=tensor.name,
                        layer_idx=layer_idx,
                        module_type=module_type,
                        dtype=tensor.dtype,
                        shape=tensor.shape,
                        metadata={"error": str(exc)},
                    )

    def _analyze_tensor(
        self,
        repo_id: str,
        revision: str | None,
        file_path: str,
        st_file: SafeTensorFile,
        tensor: TensorInfo,
        layer_idx: int | None,
        module_type: str,
    ) -> TensorStatistics:
        """
        purpose: 텐서 하나의 통계 값을 계산해 schema 객체로 변환한다.
        input: repo/file 식별자, 열린 safetensors reader, tensor metadata, tensor category.
        processing: 1차 순회로 raw moment와 max/sparsity를 계산하고 2차 순회로 abs quantile histogram을 만든다.
        return/side effects: `TensorStatistics` 인스턴스를 반환하며 저장은 수행하지 않는다.
        """

        use_numpy = self._use_numpy_engine()
        first_pass = (
            _collect_moments_numpy(st_file.iter_numpy_chunks(tensor))
            if use_numpy
            else _collect_moments(st_file.iter_values(tensor))
        )
        if first_pass.count == 0:
            return TensorStatistics.empty(
                repo_id=repo_id,
                revision=revision,
                file_path=file_path,
                tensor_name=tensor.name,
                layer_idx=layer_idx,
                module_type=module_type,
                dtype=tensor.dtype,
                shape=tensor.shape,
            )

        if use_numpy:
            q99_abs, q999_abs = _estimate_abs_quantiles_numpy(
                st_file.iter_numpy_chunks(tensor),
                max_abs=first_pass.max_abs,
                bins=self.histogram_bins,
                quantiles=(0.99, 0.999),
            )
        else:
            q99_abs, q999_abs = _estimate_abs_quantiles(
                st_file.iter_values(tensor),
                max_abs=first_pass.max_abs,
                bins=self.histogram_bins,
                quantiles=(0.99, 0.999),
            )
        mean, std, skewness, kurtosis, excess_kurtosis = first_pass.finalize()
        return TensorStatistics(
            repo_id=repo_id,
            revision=revision,
            file_path=file_path,
            tensor_name=tensor.name,
            layer_idx=layer_idx,
            module_type=module_type,
            dtype=tensor.dtype,
            shape=tensor.shape,
            num_elements=first_pass.count,
            mean=finite_or_none(mean),
            std=finite_or_none(std),
            skewness=finite_or_none(skewness),
            kurtosis=finite_or_none(kurtosis),
            excess_kurtosis=finite_or_none(excess_kurtosis),
            l2_norm=finite_or_none(math.sqrt(first_pass.sum_x2)),
            max_abs=finite_or_none(first_pass.max_abs),
            q99_abs=finite_or_none(q99_abs),
            q999_abs=finite_or_none(q999_abs),
            sparsity=finite_or_none(first_pass.zero_count / first_pass.count),
            metadata={"quantile_method": f"histogram:{self.histogram_bins}"},
            analyzed_at=datetime.now(timezone.utc),
        )

    def _use_numpy_engine(self) -> bool:
        """
        purpose: 설정과 설치 상태를 기준으로 NumPy 엔진 사용 여부를 결정한다.
        input: 생성자에서 받은 `engine` 값과 현재 Python 환경의 NumPy 설치 상태.
        processing: `python`은 비활성화하고 `numpy`는 필수 import, `auto`는 선택적 import를 수행한다.
        return/side effects: NumPy 사용 여부를 반환하며 `numpy` 강제 모드의 미설치 시 RuntimeError를 발생시킨다.
        """

        if self.engine == "python":
            return False
        try:
            import numpy  # noqa: F401
        except ImportError as exc:
            if self.engine == "numpy":
                raise RuntimeError("NumPy engine requires the numpy package") from exc
            return False
        return True


class _MomentAccumulator:
    """
    purpose: streaming 값에서 raw moment 기반 통계 계산에 필요한 누적값을 보관한다.
    input: 텐서 값 stream.
    processing: count, sum(x), sum(x^2), sum(x^3), sum(x^4), max_abs, zero_count를 누적한다.
    return/side effects: `finalize`에서 평균/표준편차/왜도/첨도를 반환하며 외부 상태는 변경하지 않는다.
    """

    def __init__(self) -> None:
        self.count = 0
        self.zero_count = 0
        self.sum_x = 0.0
        self.sum_x2 = 0.0
        self.sum_x3 = 0.0
        self.sum_x4 = 0.0
        self.max_abs = 0.0

    def add(self, value: float) -> None:
        """
        purpose: 단일 값을 moment 누적값에 반영한다.
        input: float로 변환된 텐서 원소.
        processing: powers와 zero/max_abs 카운터를 갱신한다.
        return/side effects: 내부 누적 상태를 변경하고 반환값은 없다.
        """

        self.count += 1
        if value == 0.0:
            self.zero_count += 1
        abs_value = abs(value)
        if abs_value > self.max_abs:
            self.max_abs = abs_value
        x2 = value * value
        self.sum_x += value
        self.sum_x2 += x2
        self.sum_x3 += x2 * value
        self.sum_x4 += x2 * x2

    def finalize(self) -> tuple[float, float, float, float, float]:
        """
        purpose: raw moment 누적값을 최종 분포 통계로 변환한다.
        input: 누적된 count/sum/power sums.
        processing: population variance, central moment 3/4, skewness, kurtosis를 계산한다.
        return/side effects: mean/std/skewness/kurtosis/excess_kurtosis tuple을 반환한다.
        """

        if self.count == 0:
            return math.nan, math.nan, math.nan, math.nan, math.nan
        n = self.count
        mean = self.sum_x / n
        ex2 = self.sum_x2 / n
        ex3 = self.sum_x3 / n
        ex4 = self.sum_x4 / n
        variance = max(ex2 - mean * mean, 0.0)
        std = math.sqrt(variance)
        if variance == 0.0:
            return mean, std, 0.0, 0.0, -3.0
        central3 = ex3 - 3.0 * mean * ex2 + 2.0 * mean**3
        central4 = ex4 - 4.0 * mean * ex3 + 6.0 * mean * mean * ex2 - 3.0 * mean**4
        skewness = central3 / (std**3)
        kurtosis = central4 / (variance * variance)
        return mean, std, skewness, kurtosis, kurtosis - 3.0


def _collect_moments(values: Iterable[float]) -> _MomentAccumulator:
    """
    purpose: 값 stream을 한 번 순회해 moment accumulator를 만든다.
    input: float 값 iterable.
    processing: 각 값을 `_MomentAccumulator.add`에 전달한다.
    return/side effects: 누적이 완료된 accumulator를 반환한다.
    """

    acc = _MomentAccumulator()
    for value in values:
        acc.add(value)
    return acc


def _collect_moments_numpy(chunks: Iterable[object]) -> _MomentAccumulator:
    """
    purpose: NumPy 값 chunk에서 raw moment 누적값을 벡터 연산으로 계산한다.
    input: float64 NumPy 배열 iterable.
    processing: chunk별 count, 합, 제곱합, 3·4제곱합, max_abs, zero_count를 누적한다.
    return/side effects: `_MomentAccumulator`를 반환하며 입력 배열과 외부 상태는 변경하지 않는다.
    """

    import numpy as np

    acc = _MomentAccumulator()
    for values in chunks:
        array = np.asarray(values, dtype=np.float64)
        if array.size == 0:
            continue
        squared = array * array
        acc.count += int(array.size)
        acc.zero_count += int(np.count_nonzero(array == 0.0))
        acc.sum_x += float(np.sum(array, dtype=np.float64))
        acc.sum_x2 += float(np.sum(squared, dtype=np.float64))
        acc.sum_x3 += float(np.sum(squared * array, dtype=np.float64))
        acc.sum_x4 += float(np.sum(squared * squared, dtype=np.float64))
        acc.max_abs = max(acc.max_abs, float(np.max(np.abs(array))))
    return acc


def _estimate_abs_quantiles(
    values: Iterable[float],
    *,
    max_abs: float,
    bins: int,
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    """
    purpose: 전체 값을 저장하지 않고 절대값 분위수를 근사 계산한다.
    input: 값 stream, 1차 순회에서 얻은 max_abs, histogram bin 수, quantile 목록.
    processing: `[0, max_abs]` 구간 histogram을 만들고 누적 count가 목표 rank를 넘는 bin 상한을 반환한다.
    return/side effects: quantile별 근사값 tuple을 반환하며 외부 상태는 변경하지 않는다.
    """

    if max_abs <= 0.0:
        return tuple(0.0 for _ in quantiles)
    counts = [0] * bins
    total = 0
    scale = (bins - 1) / max_abs
    for value in values:
        idx = min(int(abs(value) * scale), bins - 1)
        counts[idx] += 1
        total += 1
    if total == 0:
        return tuple(math.nan for _ in quantiles)
    sorted_quantiles = sorted(enumerate(quantiles), key=lambda item: item[1])
    output = [0.0] * len(quantiles)
    cumulative = 0
    cursor = 0
    for idx, count in enumerate(counts):
        cumulative += count
        while cursor < len(sorted_quantiles):
            original_index, quantile = sorted_quantiles[cursor]
            target = max(1, math.ceil(total * quantile))
            if cumulative < target:
                break
            output[original_index] = idx / scale
            cursor += 1
    return tuple(output)


def _estimate_abs_quantiles_numpy(
    chunks: Iterable[object],
    *,
    max_abs: float,
    bins: int,
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    """
    purpose: NumPy chunk로 기존 histogram 정의와 동일한 절대값 분위수를 계산한다.
    input: float64 배열 iterable, max_abs, bin 수, quantile 목록.
    processing: 기존 scalar 식과 같은 bin index를 벡터 계산하고 누적 histogram을 분위수로 변환한다.
    return/side effects: quantile별 근사값 tuple을 반환하며 외부 상태는 변경하지 않는다.
    """

    import numpy as np

    if max_abs <= 0.0:
        return tuple(0.0 for _ in quantiles)
    counts = np.zeros(bins, dtype=np.int64)
    total = 0
    scale = (bins - 1) / max_abs
    for values in chunks:
        array = np.asarray(values, dtype=np.float64)
        indices = np.minimum((np.abs(array) * scale).astype(np.int64), bins - 1)
        counts += np.bincount(indices, minlength=bins)
        total += int(array.size)
    return _quantiles_from_counts(counts.tolist(), total, max_abs, bins, quantiles)


def _quantiles_from_counts(
    counts: list[int],
    total: int,
    max_abs: float,
    bins: int,
    quantiles: tuple[float, ...],
) -> tuple[float, ...]:
    """
    purpose: 누적 histogram count를 요청된 분위수 값으로 변환한다.
    input: bin count, 전체 원소 수, max_abs, bin 수, quantile 목록.
    processing: quantile 목표 rank를 처음 만족하는 기존 정의의 bin 상한을 찾는다.
    return/side effects: quantile별 값 tuple을 반환하며 외부 상태는 변경하지 않는다.
    """

    if total == 0:
        return tuple(math.nan for _ in quantiles)
    scale = (bins - 1) / max_abs
    sorted_quantiles = sorted(enumerate(quantiles), key=lambda item: item[1])
    output = [0.0] * len(quantiles)
    cumulative = 0
    cursor = 0
    for idx, count in enumerate(counts):
        cumulative += count
        while cursor < len(sorted_quantiles):
            original_index, quantile = sorted_quantiles[cursor]
            target = max(1, math.ceil(total * quantile))
            if cumulative < target:
                break
            output[original_index] = idx / scale
            cursor += 1
    return tuple(output)
