from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


_FINGERPRINT_FIELDS = (
    "dtype",
    "shape",
    "num_elements",
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "excess_kurtosis",
    "l2_norm",
    "max_abs",
    "q99_abs",
    "q999_abs",
    "sparsity",
)


def build_parser() -> argparse.ArgumentParser:
    """
    purpose: 배치 JSONL 결과 요약 CLI 인자를 구성한다.
    input: 없음.
    processing: manifest, 결과 디렉터리, 출력 JSON, hash 계산 옵션을 등록한다.
    return/side effects: `ArgumentParser`를 반환하며 외부 상태는 변경하지 않는다.
    """

    parser = argparse.ArgumentParser(description="Summarize and compare weight-statistics batch results.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--hash-files", action="store_true")
    return parser


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    """
    purpose: 숫자 iterable의 선형 보간 분위수를 계산한다.
    input: 유한 숫자 iterable과 0~1 quantile.
    processing: 정렬 후 인접 rank 사이를 선형 보간한다.
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
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """
    purpose: 유한 숫자 목록의 count, mean, median, p95를 계산한다.
    input: 숫자 iterable.
    processing: NaN/Inf를 제외하고 statistics와 percentile을 적용한다.
    return/side effects: 요약 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite) if finite else None,
        "median": statistics.median(finite) if finite else None,
        "p95": _percentile(finite, 0.95),
    }


def _sha256(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    """
    purpose: 원본 safetensors의 byte 동일성 검증용 SHA-256을 계산한다.
    input: 파일 경로와 read chunk 크기.
    processing: 파일을 제한된 chunk로 읽어 digest를 누적한다.
    return/side effects: hex digest를 반환하며 원본 파일은 읽기만 한다.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """
    purpose: 모델별 JSONL 결과를 파싱하고 필수 식별 필드를 검증한다.
    input: JSONL 파일 경로.
    processing: 빈 줄을 제외하고 JSON object와 tensor_name 중복을 검사한다.
    return/side effects: row dict list를 반환하며 잘못된 결과에는 ValueError를 발생시킨다.
    """

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    names = [row.get("tensor_name") for row in rows]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError(f"invalid or duplicate tensor_name in {path}")
    return rows


def _summarize_model(entry: dict[str, Any], rows: list[dict[str, Any]], file_hash: str | None) -> dict[str, Any]:
    """
    purpose: 모델 하나의 전체·module별 텐서 통계와 첨도 outlier를 요약한다.
    input: manifest entry, 분석 row list, 선택적 SHA-256.
    processing: parameter 수, 오류 수, kurtosis/sparsity/q99 분포와 module 집계를 계산한다.
    return/side effects: 모델 요약 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    modules: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        modules[str(row["module_type"])].append(row)

    module_summary: dict[str, Any] = {}
    for module_type, module_rows in sorted(modules.items()):
        module_summary[module_type] = {
            "tensor_count": len(module_rows),
            "parameter_count": sum(int(row.get("num_elements") or 0) for row in module_rows),
            "kurtosis": _numeric_summary(row.get("kurtosis") for row in module_rows),
            "sparsity": _numeric_summary(row.get("sparsity") for row in module_rows),
            "q99_abs": _numeric_summary(row.get("q99_abs") for row in module_rows),
        }

    valid_kurtosis = [row for row in rows if row.get("kurtosis") is not None]
    top_outliers = sorted(valid_kurtosis, key=lambda row: float(row["kurtosis"]), reverse=True)[:5]
    return {
        "repo_id": entry["repo_id"],
        "family": entry["family"],
        "classification": entry["classification"],
        "sha256": file_hash,
        "file_bytes": Path(entry["path"]).stat().st_size,
        "tensor_count": len(rows),
        "expected_tensors": entry["expected_tensors"],
        "parameter_count": sum(int(row.get("num_elements") or 0) for row in rows),
        "error_rows": sum(1 for row in rows if row.get("metadata", {}).get("error")),
        "overall_kurtosis": _numeric_summary(row.get("kurtosis") for row in rows),
        "modules": module_summary,
        "top_kurtosis_outliers": [
            {
                "tensor_name": row["tensor_name"],
                "module_type": row["module_type"],
                "kurtosis": row["kurtosis"],
                "shape": row["shape"],
            }
            for row in top_outliers
        ],
    }


def _compare_models(
    model_entry: dict[str, Any],
    model_rows: list[dict[str, Any]],
    anchor_entry: dict[str, Any],
    anchor_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    purpose: 동일 family 모델과 공식 anchor의 tensor별 첨도 fingerprint 차이를 계산한다.
    input: 비교 모델·anchor의 manifest entry와 row list.
    processing: tensor_name 교집합에서 fingerprint 동일 여부와 signed/absolute kurtosis delta를 집계한다.
    return/side effects: 전체·module별 비교 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    model_by_name = {row["tensor_name"]: row for row in model_rows}
    anchor_by_name = {row["tensor_name"]: row for row in anchor_rows}
    common_names = sorted(model_by_name.keys() & anchor_by_name.keys())
    abs_deltas: list[float] = []
    signed_deltas: list[float] = []
    module_deltas: dict[str, list[float]] = defaultdict(list)
    identical = 0
    for name in common_names:
        model_row = model_by_name[name]
        anchor_row = anchor_by_name[name]
        if all(model_row.get(field) == anchor_row.get(field) for field in _FINGERPRINT_FIELDS):
            identical += 1
        model_kurtosis = model_row.get("kurtosis")
        anchor_kurtosis = anchor_row.get("kurtosis")
        if model_kurtosis is None or anchor_kurtosis is None:
            continue
        delta = float(model_kurtosis) - float(anchor_kurtosis)
        signed_deltas.append(delta)
        abs_deltas.append(abs(delta))
        module_deltas[str(model_row["module_type"])].append(abs(delta))
    return {
        "model_repo_id": model_entry["repo_id"],
        "anchor_repo_id": anchor_entry["repo_id"],
        "family": model_entry["family"],
        "common_tensors": len(common_names),
        "identical_fingerprint_tensors": identical,
        "changed_fingerprint_tensors": len(common_names) - identical,
        "absolute_kurtosis_delta": _numeric_summary(abs_deltas),
        "signed_kurtosis_delta": _numeric_summary(signed_deltas),
        "module_median_abs_kurtosis_delta": {
            module_type: statistics.median(values) for module_type, values in sorted(module_deltas.items())
        },
    }


def summarize(manifest: list[dict[str, Any]], result_dir: Path, hash_files: bool) -> dict[str, Any]:
    """
    purpose: 10개 모델 결과의 무결성·모델 요약·공식 anchor 비교·hash group을 통합 생성한다.
    input: manifest, 결과 디렉터리, 원본 hash 계산 여부.
    processing: JSONL을 검증하고 모델별 요약과 family별 Instruct/Base 비교를 계산한다.
    return/side effects: 최종 summary dict를 반환하며 hash 옵션일 때 원본 파일을 순차 읽는다.
    """

    rows_by_repo: dict[str, list[dict[str, Any]]] = {}
    model_summaries: list[dict[str, Any]] = []
    for entry in manifest:
        result_path = result_dir / f"{entry['output_name']}.jsonl"
        rows = _load_rows(result_path)
        if len(rows) != int(entry["expected_tensors"]):
            raise ValueError(f"row count mismatch for {entry['repo_id']}: {len(rows)}")
        rows_by_repo[entry["repo_id"]] = rows
        file_hash = _sha256(Path(entry["path"])) if hash_files else None
        model_summaries.append(_summarize_model(entry, rows, file_hash))

    entries_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in manifest:
        entries_by_family[str(entry["family"])].append(entry)
    comparisons: list[dict[str, Any]] = []
    for family_entries in entries_by_family.values():
        anchors = [
            entry
            for entry in family_entries
            if entry["classification"] in {"official_instruct", "official_base"}
        ]
        for entry in family_entries:
            for anchor in anchors:
                comparisons.append(
                    _compare_models(
                        entry,
                        rows_by_repo[entry["repo_id"]],
                        anchor,
                        rows_by_repo[anchor["repo_id"]],
                    )
                )

    hash_groups: dict[str, list[str]] = defaultdict(list)
    for model in model_summaries:
        if model["sha256"]:
            hash_groups[model["sha256"]].append(model["repo_id"])
    return {
        "model_count": len(manifest),
        "total_tensor_rows": sum(len(rows) for rows in rows_by_repo.values()),
        "models": model_summaries,
        "anchor_comparisons": comparisons,
        "byte_identical_groups": [repos for repos in hash_groups.values() if len(repos) > 1],
    }


def main(argv: list[str] | None = None) -> int:
    """
    purpose: manifest와 배치 결과를 읽어 상세 비교 summary JSON을 생성한다.
    input: CLI argv 또는 테스트용 argv list.
    processing: manifest parse, `summarize` 실행, UTF-8 JSON 저장을 수행한다.
    return/side effects: 성공 시 0을 반환하고 지정 출력 파일을 생성한다.
    """

    args = build_parser().parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    summary = summarize(manifest, Path(args.result_dir), args.hash_files)
    Path(args.output).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
