from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from filecollector.analysis.safetensors_stream import SafeTensorFile, TensorInfo
from filecollector.analysis.tensor_classifier import classify_tensor


_LANGUAGE_CORE_MODULES = {
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_o",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
    "norm",
}


@dataclass
class _DistanceAccumulator:
    """
    purpose: 두 weight 배열의 L2·norm·dot 계산에 필요한 합계를 누적한다.
    input: 같은 위치의 NumPy chunk 쌍.
    processing: element 수, A/B 제곱합, 차이 제곱합, dot product를 float64로 누적한다.
    return/side effects: `metrics`에서 거리 dict를 반환하며 인스턴스 내부 상태만 변경한다.
    """

    elements: int = 0
    norm_a2: float = 0.0
    norm_b2: float = 0.0
    diff2: float = 0.0
    dot: float = 0.0

    def add(self, values_a: Any, values_b: Any) -> None:
        """
        purpose: 같은 길이의 NumPy weight chunk를 거리 누적값에 반영한다.
        input: float64로 변환 가능한 NumPy 배열 두 개.
        processing: vectorized multiply/subtract/reduction으로 제곱합과 dot을 계산한다.
        return/side effects: 내부 누적 상태를 변경하고 반환값은 없다.
        """

        import numpy as np

        array_a = np.asarray(values_a, dtype=np.float64)
        array_b = np.asarray(values_b, dtype=np.float64)
        if array_a.shape != array_b.shape:
            raise ValueError(f"chunk shape mismatch: {array_a.shape} != {array_b.shape}")
        difference = array_a - array_b
        self.elements += int(array_a.size)
        self.norm_a2 += float(np.sum(array_a * array_a, dtype=np.float64))
        self.norm_b2 += float(np.sum(array_b * array_b, dtype=np.float64))
        self.diff2 += float(np.sum(difference * difference, dtype=np.float64))
        self.dot += float(np.sum(array_a * array_b, dtype=np.float64))

    def merge(self, other: "_DistanceAccumulator") -> None:
        """
        purpose: tensor별 누적값을 module/global 누적값으로 합친다.
        input: 다른 `_DistanceAccumulator`.
        processing: 모든 scalar 합계를 현재 인스턴스에 더한다.
        return/side effects: 현재 인스턴스 내부 상태를 변경하고 반환값은 없다.
        """

        self.elements += other.elements
        self.norm_a2 += other.norm_a2
        self.norm_b2 += other.norm_b2
        self.diff2 += other.diff2
        self.dot += other.dot

    def metrics(self) -> dict[str, float | int | None]:
        """
        purpose: 누적 제곱합과 dot을 normalized L2·cosine 거리로 변환한다.
        input: 인스턴스에 누적된 element/norm/diff/dot 값.
        processing: Frobenius norm, A 기준 relative L2, symmetric L2, cosine distance를 계산한다.
        return/side effects: 거리 dict를 반환하며 외부 상태는 변경하지 않는다.
        """

        norm_a = math.sqrt(max(self.norm_a2, 0.0))
        norm_b = math.sqrt(max(self.norm_b2, 0.0))
        l2 = math.sqrt(max(self.diff2, 0.0))
        relative_l2_a = l2 / norm_a if norm_a > 0.0 else (0.0 if l2 == 0.0 else None)
        symmetric_denominator = math.sqrt(max(self.norm_a2 + self.norm_b2, 0.0))
        symmetric_l2 = l2 / symmetric_denominator if symmetric_denominator > 0.0 else 0.0
        cosine_denominator = norm_a * norm_b
        cosine_distance = 1.0 - self.dot / cosine_denominator if cosine_denominator > 0.0 else 0.0
        if abs(cosine_distance) < 1e-15:
            cosine_distance = 0.0
        return {
            "elements": self.elements,
            "norm_a": norm_a,
            "norm_b": norm_b,
            "l2": l2,
            "relative_l2_a": relative_l2_a,
            "symmetric_l2": symmetric_l2,
            "cosine_distance": cosine_distance,
        }


def build_parser() -> argparse.ArgumentParser:
    """
    purpose: family별 weight distance batch CLI 인자를 구성한다.
    input: 없음.
    processing: manifest, 통계 결과, SHA summary, 출력 디렉터리 인자를 등록한다.
    return/side effects: `ArgumentParser`를 반환하며 외부 상태는 변경하지 않는다.
    """

    parser = argparse.ArgumentParser(description="Compute pairwise weight and kurtosis distances.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--analysis-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    """
    purpose: tensor-balanced 거리의 선형 보간 분위수를 계산한다.
    input: 유한 float iterable과 0~1 quantile.
    processing: 정렬된 rank 사이를 선형 보간한다.
    return/side effects: 값이 없으면 None, 있으면 float를 반환하며 외부 상태는 변경하지 않는다.
    """

    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _is_language_core(tensor_name: str, module_type: str) -> bool:
    """
    purpose: tensor가 language core view에 포함되는지 판정한다.
    input: tensor 이름과 classifier module type.
    processing: language layer prefix와 허용 attention/MLP/norm module 집합을 함께 검사한다.
    return/side effects: 포함 여부 bool을 반환하며 외부 상태는 변경하지 않는다.
    """

    return tensor_name.startswith("model.language_model.layers.") and module_type in _LANGUAGE_CORE_MODULES


def _tensor_map(st_file: SafeTensorFile) -> dict[str, TensorInfo]:
    """
    purpose: 열린 safetensors의 tensor 이름 조회 map을 만든다.
    input: context manager로 열린 `SafeTensorFile`.
    processing: tensor iterator를 이름 key dict로 변환한다.
    return/side effects: tensor map을 반환하며 파일은 읽기만 한다.
    """

    return {tensor.name: tensor for tensor in st_file.tensors()}


def _load_kurtosis(path: Path) -> dict[str, float | None]:
    """
    purpose: 기존 weight statistics JSONL에서 tensor별 kurtosis를 읽는다.
    input: 모델별 JSONL 경로.
    processing: 각 JSON row의 tensor_name과 kurtosis를 map으로 변환한다.
    return/side effects: kurtosis map을 반환하며 파일은 읽기만 한다.
    """

    output: dict[str, float | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        output[str(row["tensor_name"])] = row.get("kurtosis")
    return output


def compute_pair(
    entry_a: dict[str, Any],
    entry_b: dict[str, Any],
    kurtosis_a: dict[str, float | None],
    kurtosis_b: dict[str, float | None],
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    purpose: 같은 family 모델 쌍의 실제 weight와 kurtosis 거리를 tensor·view·module별 계산한다.
    input: 두 manifest entry, tensor별 kurtosis map, chunk 크기.
    processing: 같은 이름·shape·dtype tensor를 NumPy chunk로 동시 순회하고 거리 record와 집계를 만든다.
    return/side effects: tensor record list와 pair summary를 반환하며 두 원본 파일은 읽기만 한다.
    """

    records: list[dict[str, Any]] = []
    skipped_shape = 0
    skipped_dtype = 0
    with SafeTensorFile(entry_a["path"], chunk_bytes=chunk_bytes) as file_a, SafeTensorFile(
        entry_b["path"], chunk_bytes=chunk_bytes
    ) as file_b:
        tensors_a = _tensor_map(file_a)
        tensors_b = _tensor_map(file_b)
        common_names = sorted(tensors_a.keys() & tensors_b.keys())
        for tensor_name in common_names:
            tensor_a = tensors_a[tensor_name]
            tensor_b = tensors_b[tensor_name]
            if tensor_a.shape != tensor_b.shape:
                skipped_shape += 1
                continue
            if tensor_a.dtype.upper() != tensor_b.dtype.upper():
                skipped_dtype += 1
                continue
            accumulator = _DistanceAccumulator()
            chunks_a = file_a.iter_numpy_chunks(tensor_a)
            chunks_b = file_b.iter_numpy_chunks(tensor_b)
            for values_a, values_b in itertools.zip_longest(chunks_a, chunks_b):
                if values_a is None or values_b is None:
                    raise ValueError(f"chunk count mismatch for {tensor_name}")
                accumulator.add(values_a, values_b)
            _, module_type = classify_tensor(tensor_name)
            value_a = kurtosis_a.get(tensor_name)
            value_b = kurtosis_b.get(tensor_name)
            kurtosis_delta = (
                abs(float(value_a) - float(value_b)) if value_a is not None and value_b is not None else None
            )
            records.append(
                {
                    "model_a": entry_a["repo_id"],
                    "model_b": entry_b["repo_id"],
                    "family": entry_a["family"],
                    "tensor_name": tensor_name,
                    "module_type": module_type,
                    "language_core": _is_language_core(tensor_name, module_type),
                    "kurtosis_abs_delta": kurtosis_delta,
                    "raw": {
                        "elements": accumulator.elements,
                        "norm_a2": accumulator.norm_a2,
                        "norm_b2": accumulator.norm_b2,
                        "diff2": accumulator.diff2,
                        "dot": accumulator.dot,
                    },
                    **accumulator.metrics(),
                }
            )

    views = {
        "all": records,
        "language_core": [record for record in records if record["language_core"]],
    }
    return records, {
        "model_a": entry_a["repo_id"],
        "model_b": entry_b["repo_id"],
        "family": entry_a["family"],
        "common_tensor_names": len(set(kurtosis_a) & set(kurtosis_b)),
        "compared_tensors": len(records),
        "skipped_shape": skipped_shape,
        "skipped_dtype": skipped_dtype,
        "views": {name: _aggregate_records(view_records) for name, view_records in views.items()},
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    purpose: tensor distance record를 parameter-weighted global·tensor-balanced·module별 지표로 집계한다.
    input: 하나의 view에 포함된 tensor record list.
    processing: raw 합계를 merge하고 symmetric L2/cosine/kurtosis의 median·p95와 module 집계를 계산한다.
    return/side effects: view summary dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    global_accumulator = _DistanceAccumulator()
    module_accumulators: dict[str, _DistanceAccumulator] = defaultdict(_DistanceAccumulator)
    module_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        raw = record["raw"]
        accumulator = _DistanceAccumulator(
            elements=int(raw["elements"]),
            norm_a2=float(raw["norm_a2"]),
            norm_b2=float(raw["norm_b2"]),
            diff2=float(raw["diff2"]),
            dot=float(raw["dot"]),
        )
        global_accumulator.merge(accumulator)
        module_accumulators[record["module_type"]].merge(accumulator)
        module_records[record["module_type"]].append(record)
    return {
        "tensor_count": len(records),
        "changed_tensor_count": sum(1 for record in records if float(record["l2"]) > 0.0),
        "global": global_accumulator.metrics(),
        "tensor_balanced": _balanced_metrics(records),
        "modules": {
            module_type: {
                "tensor_count": len(module_records[module_type]),
                "changed_tensor_count": sum(
                    1 for record in module_records[module_type] if float(record["l2"]) > 0.0
                ),
                "global": accumulator.metrics(),
                "tensor_balanced": _balanced_metrics(module_records[module_type]),
            }
            for module_type, accumulator in sorted(module_accumulators.items())
        },
    }


def _balanced_metrics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """
    purpose: tensor 크기와 무관하게 각 tensor를 동일 가중한 거리 통계를 계산한다.
    input: tensor distance record list.
    processing: symmetric L2, cosine distance, kurtosis delta의 median과 p95를 계산한다.
    return/side effects: tensor-balanced 지표 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    symmetric = [float(record["symmetric_l2"]) for record in records]
    cosine = [float(record["cosine_distance"]) for record in records]
    kurtosis = [
        float(record["kurtosis_abs_delta"])
        for record in records
        if record["kurtosis_abs_delta"] is not None
    ]
    return {
        "symmetric_l2_median": statistics.median(symmetric) if symmetric else None,
        "symmetric_l2_p95": _percentile(symmetric, 0.95),
        "cosine_distance_median": statistics.median(cosine) if cosine else None,
        "cosine_distance_p95": _percentile(cosine, 0.95),
        "kurtosis_abs_delta_median": statistics.median(kurtosis) if kurtosis else None,
        "kurtosis_abs_delta_p95": _percentile(kurtosis, 0.95),
    }


def _build_matrices(entries: list[dict[str, Any]], pair_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    purpose: family·view별 pair summary를 대칭 거리 행렬로 변환한다.
    input: 고유 manifest entry list와 pair summary list.
    processing: diagonal 0과 pair 값을 global L2/cosine/tensor median/kurtosis matrix에 채운다.
    return/side effects: JSON 직렬화 가능한 matrix dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    result: dict[str, Any] = {}
    families = sorted({str(entry["family"]) for entry in entries})
    for family in families:
        family_entries = [entry for entry in entries if str(entry["family"]) == family]
        names = [entry["repo_id"] for entry in family_entries]
        index = {name: position for position, name in enumerate(names)}
        family_result: dict[str, Any] = {}
        for view in ("all", "language_core"):
            metrics = {
                "global_symmetric_l2": [[0.0 for _ in names] for _ in names],
                "global_cosine_distance": [[0.0 for _ in names] for _ in names],
                "tensor_median_symmetric_l2": [[0.0 for _ in names] for _ in names],
                "tensor_median_kurtosis_delta": [[0.0 for _ in names] for _ in names],
            }
            for pair in pair_summaries:
                if str(pair["family"]) != family:
                    continue
                left = index[pair["model_a"]]
                right = index[pair["model_b"]]
                summary = pair["views"][view]
                values = {
                    "global_symmetric_l2": summary["global"]["symmetric_l2"],
                    "global_cosine_distance": summary["global"]["cosine_distance"],
                    "tensor_median_symmetric_l2": summary["tensor_balanced"]["symmetric_l2_median"],
                    "tensor_median_kurtosis_delta": summary["tensor_balanced"]["kurtosis_abs_delta_median"],
                }
                for metric, value in values.items():
                    metrics[metric][left][right] = value
                    metrics[metric][right][left] = value
            family_result[view] = {"models": names, "metrics": metrics}
        result[family] = family_result
    return result


def run_batch(
    manifest: list[dict[str, Any]],
    statistics_dir: Path,
    analysis_summary: dict[str, Any],
    output_dir: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """
    purpose: SHA-256 고유 대표 모델의 family 내 모든 pair를 계산하고 거리 행렬을 생성한다.
    input: manifest, 통계 디렉터리, SHA summary, 출력 디렉터리, chunk 크기.
    processing: hash 중복 제거, family 조합 생성, pair 계산, tensor JSONL과 summary JSON 작성.
    return/side effects: 최종 summary dict를 반환하고 출력 디렉터리에 결과 파일을 생성한다.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_by_repo = {model["repo_id"]: model["sha256"] for model in analysis_summary["models"]}
    representatives: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for entry in manifest:
        file_hash = hash_by_repo[entry["repo_id"]]
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        representatives.append(entry)

    kurtosis_by_repo = {
        entry["repo_id"]: _load_kurtosis(statistics_dir / f"{entry['output_name']}.jsonl")
        for entry in representatives
    }
    pair_summaries: list[dict[str, Any]] = []
    tensor_output = output_dir / "tensor-weight-distances.jsonl.part"
    with tensor_output.open("w", encoding="utf-8") as handle:
        families = sorted({str(entry["family"]) for entry in representatives})
        for family in families:
            family_entries = [entry for entry in representatives if str(entry["family"]) == family]
            for entry_a, entry_b in itertools.combinations(family_entries, 2):
                started_at = time.monotonic()
                print(
                    f"pair_start family={family} model_a={entry_a['repo_id']} model_b={entry_b['repo_id']}",
                    file=sys.stderr,
                    flush=True,
                )
                records, pair_summary = compute_pair(
                    entry_a,
                    entry_b,
                    kurtosis_by_repo[entry_a["repo_id"]],
                    kurtosis_by_repo[entry_b["repo_id"]],
                    chunk_bytes=chunk_bytes,
                )
                for record in records:
                    record.pop("raw", None)
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                pair_summaries.append(pair_summary)
                elapsed = time.monotonic() - started_at
                all_global = pair_summary["views"]["all"]["global"]
                print(
                    "pair_done "
                    f"family={family} model_a={entry_a['repo_id']} model_b={entry_b['repo_id']} "
                    f"tensors={pair_summary['compared_tensors']} "
                    f"symmetric_l2={all_global['symmetric_l2']:.9g} "
                    f"cosine_distance={all_global['cosine_distance']:.9g} elapsed_seconds={elapsed:.3f}",
                    file=sys.stderr,
                    flush=True,
                )
    tensor_output.replace(output_dir / "tensor-weight-distances.jsonl")
    result = {
        "unique_model_count": len(representatives),
        "representatives": [entry["repo_id"] for entry in representatives],
        "byte_identical_groups": analysis_summary.get("byte_identical_groups", []),
        "pair_count": len(pair_summaries),
        "pairs": pair_summaries,
        "matrices": _build_matrices(representatives, pair_summaries),
    }
    (output_dir / "weight-distance-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """
    purpose: CLI 입력에서 7개 고유 모델의 pairwise weight/kurtosis distance 결과를 생성한다.
    input: CLI argv 또는 테스트용 argv list.
    processing: manifest·SHA summary를 읽고 `run_batch`를 실행한다.
    return/side effects: 성공 시 0을 반환하고 tensor JSONL·summary JSON을 생성한다.
    """

    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    analysis_summary = json.loads(Path(args.analysis_summary).read_text(encoding="utf-8"))
    run_batch(
        manifest,
        Path(args.statistics_dir),
        analysis_summary,
        Path(args.output_dir),
        chunk_bytes=args.chunk_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
