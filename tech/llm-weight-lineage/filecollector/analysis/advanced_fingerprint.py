from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from filecollector.analysis.safetensors_stream import SafeTensorFile, TensorInfo


_CLOUDGOAT_MODULES = {"attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down"}
_HUIHUI_MODULES = {"attn_o", "mlp_down"}


def build_parser() -> argparse.ArgumentParser:
    """
    purpose: CKA·delta SVD 고급 fingerprint CLI 인자를 구성한다.
    input: 없음.
    processing: manifest, statistics, weight-distance 결과, 출력과 sampling/SVD 옵션을 등록한다.
    return/side effects: `ArgumentParser`를 반환하며 외부 상태는 변경하지 않는다.
    """

    parser = argparse.ArgumentParser(description="Compute sampled CKA and randomized delta SVD fingerprints.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--statistics-dir", required=True)
    parser.add_argument("--weight-distance-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--svd-rank", type=int, default=16)
    parser.add_argument("--oversample", type=int, default=8)
    parser.add_argument("--power-iterations", type=int, default=1)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    return parser


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    """
    purpose: CKA/SVD module 집계에 사용할 선형 보간 분위수를 계산한다.
    input: 유한 float iterable과 0~1 quantile.
    processing: 정렬 rank 사이를 선형 보간한다.
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


def _tensor_lookup(st_file: SafeTensorFile, tensor_name: str) -> TensorInfo:
    """
    purpose: 열린 safetensors에서 지정 이름의 tensor metadata를 찾는다.
    input: 열린 `SafeTensorFile`과 tensor 이름.
    processing: tensor iterator를 순회해 exact name을 찾는다.
    return/side effects: `TensorInfo`를 반환하며 없으면 KeyError를 발생시킨다.
    """

    for tensor in st_file.tensors():
        if tensor.name == tensor_name:
            return tensor
    raise KeyError(tensor_name)


def _read_matrix(path: str, tensor_name: str, chunk_bytes: int) -> Any:
    """
    purpose: safetensors의 rank-2 tensor 하나를 float32 NumPy matrix로 읽는다.
    input: 파일 경로, tensor 이름, read chunk 크기.
    processing: mmap chunk를 preallocated float32 flat array에 복사하고 원래 shape로 변환한다.
    return/side effects: 독립 NumPy matrix를 반환하고 파일/mmap은 함수 종료 전에 닫는다.
    """

    import numpy as np

    with SafeTensorFile(path, chunk_bytes=chunk_bytes) as st_file:
        tensor = _tensor_lookup(st_file, tensor_name)
        if len(tensor.shape) != 2:
            raise ValueError(f"rank-2 tensor required: {tensor_name} shape={tensor.shape}")
        flat = np.empty(tensor.num_elements, dtype=np.float32)
        offset = 0
        for chunk in st_file.iter_numpy_chunks(tensor):
            size = int(chunk.size)
            flat[offset : offset + size] = chunk
            offset += size
        if offset != tensor.num_elements:
            raise ValueError(f"element count mismatch for {tensor_name}: {offset} != {tensor.num_elements}")
        return flat.reshape(tuple(tensor.shape))


def linear_cka(matrix_a: Any, matrix_b: Any, *, max_rows: int = 2048) -> dict[str, float | int]:
    """
    purpose: 같은 shape weight matrix의 deterministic sampled centered linear CKA를 계산한다.
    input: NumPy matrix 두 개와 최대 sample row 수.
    processing: 균등 row index를 선택하고 column centering 후 작은 Gram/covariance 경로로 CKA를 계산한다.
    return/side effects: CKA, CKA distance, sample 수 dict를 반환하며 입력 matrix는 변경하지 않는다.
    """

    import numpy as np

    values_a = np.asarray(matrix_a, dtype=np.float32)
    values_b = np.asarray(matrix_b, dtype=np.float32)
    if values_a.shape != values_b.shape or values_a.ndim != 2:
        raise ValueError(f"CKA requires equal rank-2 shapes: {values_a.shape} != {values_b.shape}")
    row_count = values_a.shape[0]
    sample_count = min(row_count, max_rows)
    indices = np.linspace(0, row_count - 1, num=sample_count, dtype=np.int64)
    sampled_a = values_a[indices].astype(np.float64)
    sampled_b = values_b[indices].astype(np.float64)
    sampled_a -= sampled_a.mean(axis=0, keepdims=True)
    sampled_b -= sampled_b.mean(axis=0, keepdims=True)
    if sample_count <= sampled_a.shape[1]:
        gram_a = sampled_a @ sampled_a.T
        gram_b = sampled_b @ sampled_b.T
        numerator = float(np.sum(gram_a * gram_b, dtype=np.float64))
        denominator_a = float(np.sum(gram_a * gram_a, dtype=np.float64))
        denominator_b = float(np.sum(gram_b * gram_b, dtype=np.float64))
    else:
        cross = sampled_a.T @ sampled_b
        covariance_a = sampled_a.T @ sampled_a
        covariance_b = sampled_b.T @ sampled_b
        numerator = float(np.sum(cross * cross, dtype=np.float64))
        denominator_a = float(np.sum(covariance_a * covariance_a, dtype=np.float64))
        denominator_b = float(np.sum(covariance_b * covariance_b, dtype=np.float64))
    denominator = math.sqrt(max(denominator_a * denominator_b, 0.0))
    cka = numerator / denominator if denominator > 0.0 else 1.0
    cka = min(max(cka, 0.0), 1.0)
    return {"cka": cka, "cka_distance": 1.0 - cka, "sample_rows": sample_count}


def randomized_delta_svd(
    delta: Any,
    *,
    rank: int = 16,
    oversample: int = 8,
    power_iterations: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """
    purpose: weight delta matrix의 dominant singular spectrum과 low-rank energy를 근사한다.
    input: rank-2 delta, target rank, oversampling, power iteration 수, deterministic seed.
    processing: randomized range finder, QR, 작은 projected SVD를 수행하고 total Frobenius energy와 비교한다.
    return/side effects: singular values, energy fraction, threshold rank dict를 반환하며 입력은 변경하지 않는다.
    """

    import numpy as np

    matrix = np.asarray(delta, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("randomized SVD requires a rank-2 matrix")
    target = min(rank, min(matrix.shape))
    projection_rank = min(target + oversample, min(matrix.shape))
    total_energy = float(np.sum(matrix * matrix, dtype=np.float64))
    if total_energy == 0.0:
        return {
            "singular_values": [0.0] * target,
            "top_1_energy_fraction": 0.0,
            "top_k_energy_fraction": 0.0,
            "energy_rank_50": 0,
            "energy_rank_80": 0,
            "energy_rank_90": 0,
            "total_delta_frobenius": 0.0,
        }
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((matrix.shape[1], projection_rank), dtype=np.float32)
    sample = matrix @ omega
    for _ in range(power_iterations):
        sample = matrix @ (matrix.T @ sample)
    basis, _ = np.linalg.qr(sample, mode="reduced")
    projected = basis.T @ matrix
    singular_values = np.linalg.svd(projected, compute_uv=False, full_matrices=False)[:target]
    cumulative = np.cumsum(singular_values.astype(np.float64) ** 2) / total_energy

    def threshold_rank(threshold: float) -> int | str:
        """
        purpose: 누적 singular energy가 threshold에 처음 도달하는 근사 rank를 찾는다.
        input: 0~1 energy threshold.
        processing: 계산된 cumulative energy 배열에서 첫 index를 찾는다.
        return/side effects: rank 정수 또는 target 밖이면 `>target` 문자열을 반환한다.
        """

        matches = np.flatnonzero(cumulative >= threshold)
        return int(matches[0] + 1) if matches.size else f">{target}"

    return {
        "singular_values": [float(value) for value in singular_values],
        "top_1_energy_fraction": float(cumulative[0]),
        "top_k_energy_fraction": float(cumulative[-1]),
        "energy_rank_50": threshold_rank(0.50),
        "energy_rank_80": threshold_rank(0.80),
        "energy_rank_90": threshold_rank(0.90),
        "total_delta_frobenius": math.sqrt(total_energy),
    }


def delta_alignment(base_to_instruct: Any, instruct_to_candidate: Any) -> float | None:
    """
    purpose: Base→Instruct와 Instruct→candidate weight delta 방향의 cosine alignment를 계산한다.
    input: 같은 shape delta matrix 두 개.
    processing: float64 dot과 L2 norm으로 cosine similarity를 계산한다.
    return/side effects: [-1,1] alignment 또는 zero norm이면 None을 반환하며 입력은 변경하지 않는다.
    """

    import numpy as np

    first = np.asarray(base_to_instruct, dtype=np.float32)
    second = np.asarray(instruct_to_candidate, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError(f"delta shape mismatch: {first.shape} != {second.shape}")
    dot = float(np.sum(first * second, dtype=np.float64))
    norm_first = math.sqrt(float(np.sum(first * first, dtype=np.float64)))
    norm_second = math.sqrt(float(np.sum(second * second, dtype=np.float64)))
    if norm_first == 0.0 or norm_second == 0.0:
        return None
    return min(max(dot / (norm_first * norm_second), -1.0), 1.0)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    purpose: 고급 분석 입력 JSONL을 row list로 파싱한다.
    input: JSONL 경로.
    processing: 빈 줄을 제외하고 각 줄을 JSON object로 변환한다.
    return/side effects: row list를 반환하며 파일은 읽기만 한다.
    """

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_kurtosis(path: Path) -> dict[str, float | None]:
    """
    purpose: 모델별 statistics JSONL에서 tensor kurtosis map을 만든다.
    input: statistics JSONL 경로.
    processing: tensor_name과 kurtosis를 추출한다.
    return/side effects: map을 반환하며 파일은 읽기만 한다.
    """

    return {row["tensor_name"]: row.get("kurtosis") for row in _load_jsonl(path)}


def _seed_for_name(tensor_name: str) -> int:
    """
    purpose: tensor 이름에서 randomized SVD 재현용 uint32 seed를 만든다.
    input: tensor 이름.
    processing: SHA-256 앞 4byte를 little-endian 정수로 변환한다.
    return/side effects: seed 정수를 반환하며 외부 상태는 변경하지 않는다.
    """

    return int.from_bytes(hashlib.sha256(tensor_name.encode("utf-8")).digest()[:4], "little")


def _summarize_records(records: list[dict[str, Any]], metric_fields: tuple[str, ...]) -> dict[str, Any]:
    """
    purpose: CKA/SVD tensor record를 전체·module별 median/p95/mean으로 집계한다.
    input: record list와 집계할 numeric field 목록.
    processing: None/비유한 값을 제외해 field별 count, mean, median, p95를 계산한다.
    return/side effects: 전체와 module summary dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    def aggregate(group: list[dict[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {"tensor_count": len(group)}
        for field in metric_fields:
            values = [
                float(record[field])
                for record in group
                if record.get(field) is not None and math.isfinite(float(record[field]))
            ]
            output[field] = {
                "count": len(values),
                "mean": statistics.fmean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "p95": _percentile(values, 0.95),
            }
        return output

    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        modules[record["module_type"]].append(record)
    return {
        "overall": aggregate(records),
        "modules": {module: aggregate(rows) for module, rows in sorted(modules.items())},
    }


def run_analysis(
    manifest: list[dict[str, Any]],
    statistics_dir: Path,
    weight_distance_rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_rows: int = 2048,
    svd_rank: int = 16,
    oversample: int = 8,
    power_iterations: int = 1,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict[str, Any]:
    """
    purpose: CloudGoat CKA와 Huihui delta SVD/alignment를 순차 실행하고 결과를 저장한다.
    input: manifest, statistics dir, weight-distance row, 출력과 CKA/SVD 설정.
    processing: 대상 tensor를 거리 결과에서 선택하고 matrix 단위로 읽어 CKA·SVD·alignment를 계산한다.
    return/side effects: summary dict를 반환하고 CKA/SVD JSONL 및 summary JSON을 생성한다.
    """

    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    entries = {entry["repo_id"]: entry for entry in manifest}
    kurtosis = {
        repo_id: _load_kurtosis(statistics_dir / f"{entry['output_name']}.jsonl")
        for repo_id, entry in entries.items()
    }

    instruct_08 = "Qwen/Qwen3.5-0.8B"
    cloudgoat = "CloudGoat/Qwen3.5-0.8B-JP-Tuned-v1.0"
    cloud_distance = [
        row
        for row in weight_distance_rows
        if row["model_a"] == instruct_08
        and row["model_b"] == cloudgoat
        and row["module_type"] in _CLOUDGOAT_MODULES
        and float(row["l2"]) > 0.0
    ]
    cka_records: list[dict[str, Any]] = []
    for index, distance_row in enumerate(cloud_distance, start=1):
        tensor_name = distance_row["tensor_name"]
        started_at = time.monotonic()
        matrix_a = _read_matrix(entries[instruct_08]["path"], tensor_name, chunk_bytes)
        matrix_b = _read_matrix(entries[cloudgoat]["path"], tensor_name, chunk_bytes)
        cka = linear_cka(matrix_a, matrix_b, max_rows=max_rows)
        value_a = kurtosis[instruct_08].get(tensor_name)
        value_b = kurtosis[cloudgoat].get(tensor_name)
        record = {
            "model_a": instruct_08,
            "model_b": cloudgoat,
            "tensor_name": tensor_name,
            "module_type": distance_row["module_type"],
            "shape": list(matrix_a.shape),
            "symmetric_l2": distance_row["symmetric_l2"],
            "cosine_distance": distance_row["cosine_distance"],
            "kurtosis_abs_delta": (
                abs(float(value_a) - float(value_b)) if value_a is not None and value_b is not None else None
            ),
            **cka,
        }
        cka_records.append(record)
        print(
            f"cka_done index={index}/{len(cloud_distance)} module={record['module_type']} "
            f"cka={record['cka']:.9g} tensor={tensor_name} elapsed_seconds={time.monotonic()-started_at:.3f}",
            file=sys.stderr,
            flush=True,
        )
        del matrix_a, matrix_b
        gc.collect()

    huihui_targets = [
        ("0.8B", "Qwen/Qwen3.5-0.8B-Base", instruct_08, "huihui-ai/Huihui-Qwen3.5-0.8B-abliterated"),
        ("2B", "Qwen/Qwen3.5-2B-Base", "Qwen/Qwen3.5-2B", "huihui-ai/Huihui-Qwen3.5-2B-abliterated"),
    ]
    svd_records: list[dict[str, Any]] = []
    for family, base_repo, instruct_repo, candidate_repo in huihui_targets:
        selected = [
            row
            for row in weight_distance_rows
            if row["model_a"] == instruct_repo
            and row["model_b"] == candidate_repo
            and row["module_type"] in _HUIHUI_MODULES
            and float(row["l2"]) > 0.0
        ]
        for index, distance_row in enumerate(selected, start=1):
            tensor_name = distance_row["tensor_name"]
            started_at = time.monotonic()
            matrix_base = _read_matrix(entries[base_repo]["path"], tensor_name, chunk_bytes)
            matrix_instruct = _read_matrix(entries[instruct_repo]["path"], tensor_name, chunk_bytes)
            matrix_candidate = _read_matrix(entries[candidate_repo]["path"], tensor_name, chunk_bytes)
            base_delta = matrix_instruct - matrix_base
            candidate_delta = matrix_candidate - matrix_instruct
            svd = randomized_delta_svd(
                candidate_delta,
                rank=svd_rank,
                oversample=oversample,
                power_iterations=power_iterations,
                seed=_seed_for_name(tensor_name),
            )
            alignment = delta_alignment(base_delta, candidate_delta)
            record = {
                "family": family,
                "base_repo": base_repo,
                "instruct_repo": instruct_repo,
                "candidate_repo": candidate_repo,
                "tensor_name": tensor_name,
                "module_type": distance_row["module_type"],
                "language_core": distance_row["language_core"],
                "shape": list(matrix_candidate.shape),
                "symmetric_l2": distance_row["symmetric_l2"],
                "cosine_distance": distance_row["cosine_distance"],
                "delta_alignment": alignment,
                **svd,
            }
            svd_records.append(record)
            print(
                f"svd_done family={family} index={index}/{len(selected)} module={record['module_type']} "
                f"top_k_energy={record['top_k_energy_fraction']:.9g} alignment={alignment} "
                f"tensor={tensor_name} elapsed_seconds={time.monotonic()-started_at:.3f}",
                file=sys.stderr,
                flush=True,
            )
            del matrix_base, matrix_instruct, matrix_candidate, base_delta, candidate_delta
            gc.collect()

    cka_path = output_dir / "cloudgoat-cka.jsonl"
    cka_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in cka_records), encoding="utf-8")
    svd_path = output_dir / "huihui-delta-svd.jsonl"
    svd_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in svd_records), encoding="utf-8")
    summary = {
        "cka": {
            "tensor_count": len(cka_records),
            "settings": {"max_rows": max_rows},
            "summary": _summarize_records(
                cka_records, ("cka", "cka_distance", "symmetric_l2", "cosine_distance", "kurtosis_abs_delta")
            ),
        },
        "delta_svd": {
            "tensor_count": len(svd_records),
            "settings": {
                "rank": svd_rank,
                "oversample": oversample,
                "power_iterations": power_iterations,
            },
            "summary": _summarize_records(
                svd_records,
                (
                    "top_1_energy_fraction",
                    "top_k_energy_fraction",
                    "delta_alignment",
                    "symmetric_l2",
                    "cosine_distance",
                ),
            ),
            "by_family": {
                family: _summarize_records(
                    [record for record in svd_records if record["family"] == family],
                    ("top_1_energy_fraction", "top_k_energy_fraction", "delta_alignment"),
                )
                for family in sorted({record["family"] for record in svd_records})
            },
        },
    }
    (output_dir / "advanced-fingerprint-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    """
    purpose: CLI 입력으로 CloudGoat CKA와 Huihui delta SVD 고급 fingerprint 결과를 생성한다.
    input: CLI argv 또는 테스트용 argv list.
    processing: manifest와 distance JSONL을 읽고 `run_analysis`를 실행한다.
    return/side effects: 성공 시 0을 반환하고 출력 JSONL/JSON 파일을 생성한다.
    """

    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    weight_distance_rows = _load_jsonl(Path(args.weight_distance_jsonl))
    run_analysis(
        manifest,
        Path(args.statistics_dir),
        weight_distance_rows,
        Path(args.output_dir),
        max_rows=args.max_rows,
        svd_rank=args.svd_rank,
        oversample=args.oversample,
        power_iterations=args.power_iterations,
        chunk_bytes=args.chunk_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
