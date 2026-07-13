from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    """
    purpose: safetensors 10개 순차 분석용 CLI 인자를 구성한다.
    input: 없음.
    processing: manifest, output directory, engine, force 옵션을 등록한다.
    return/side effects: `ArgumentParser`를 반환하며 외부 상태는 변경하지 않는다.
    """

    parser = argparse.ArgumentParser(description="Run resumable weight-statistics batch analysis.")
    parser.add_argument("--manifest", required=True, help="JSON manifest path")
    parser.add_argument("--output-dir", required=True, help="Result directory")
    parser.add_argument("--engine", choices=("python", "numpy"), default="numpy")
    parser.add_argument("--force", action="store_true", help="Re-run completed entries")
    return parser


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    """
    purpose: JSON manifest를 읽고 필수 필드·중복·파일 경로를 검증한다.
    input: manifest 파일 경로.
    processing: list entry를 파싱하고 repo/output 중복과 safetensors 존재 여부를 검사한다.
    return/side effects: 정규화된 entry list를 반환하며 검증 실패 시 ValueError/FileNotFoundError를 발생시킨다.
    """

    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must be a non-empty JSON list")
    required = {"repo_id", "family", "classification", "path", "expected_tensors", "output_name"}
    repo_ids: set[str] = set()
    output_names: set[str] = set()
    for entry in entries:
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry missing fields: {sorted(missing)}")
        if entry["repo_id"] in repo_ids or entry["output_name"] in output_names:
            raise ValueError(f"duplicate manifest entry: {entry['repo_id']}")
        if not Path(entry["path"]).is_file():
            raise FileNotFoundError(entry["path"])
        repo_ids.add(entry["repo_id"])
        output_names.add(entry["output_name"])
    return entries


def _is_complete(entry: dict[str, Any], output_dir: Path) -> bool:
    """
    purpose: 모델 분석 결과가 완료 marker와 예상 행 수를 모두 만족하는지 확인한다.
    input: manifest entry와 결과 디렉터리.
    processing: JSONL 및 done marker 존재, success 상태, 결과 행 수를 비교한다.
    return/side effects: 완료 여부를 반환하며 외부 상태는 변경하지 않는다.
    """

    stem = entry["output_name"]
    result_path = output_dir / f"{stem}.jsonl"
    marker_path = output_dir / f"{stem}.done.json"
    if not result_path.is_file() or not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    return marker.get("status") == "success" and marker.get("rows") == entry["expected_tensors"]


def _run_entry(entry: dict[str, Any], output_dir: Path, engine: str) -> dict[str, Any]:
    """
    purpose: manifest의 모델 하나를 subprocess로 분석하고 원자적으로 결과를 확정한다.
    input: manifest entry, 결과 디렉터리, 분석 engine.
    processing: 임시 JSONL과 log를 기록하고 성공·행 수 일치 시 최종 파일과 done marker를 만든다.
    return/side effects: 실행 상태 dict를 반환하고 결과·로그·marker 파일을 생성한다.
    """

    stem = entry["output_name"]
    part_path = output_dir / f"{stem}.jsonl.part"
    result_path = output_dir / f"{stem}.jsonl"
    log_path = output_dir / f"{stem}.log"
    marker_path = output_dir / f"{stem}.done.json"
    started_at = datetime.now(timezone.utc).isoformat()
    command = [
        sys.executable,
        "-m",
        "filecollector.analysis.weight_stats",
        "--repo-id",
        entry["repo_id"],
        "--path",
        entry["path"],
        "--engine",
        engine,
        "--dry-run",
    ]
    with part_path.open("w", encoding="utf-8") as stdout, log_path.open("w", encoding="utf-8") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    rows = sum(1 for line in part_path.open(encoding="utf-8") if line.strip())
    status = "success" if result.returncode == 0 and rows == entry["expected_tensors"] else "failed"
    marker = {
        "repo_id": entry["repo_id"],
        "family": entry["family"],
        "classification": entry["classification"],
        "status": status,
        "returncode": result.returncode,
        "rows": rows,
        "expected_tensors": entry["expected_tensors"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if status == "success":
        part_path.replace(result_path)
    return marker


def _write_summary(entries: list[dict[str, Any]], output_dir: Path) -> None:
    """
    purpose: 현재 배치의 모델별 완료 상태를 하나의 summary JSON으로 기록한다.
    input: manifest entry list와 결과 디렉터리.
    processing: 각 done marker를 읽고 미실행 항목은 pending으로 표시한다.
    return/side effects: `batch-summary.json`을 원자적으로 갱신하고 반환값은 없다.
    """

    states: list[dict[str, Any]] = []
    for entry in entries:
        marker_path = output_dir / f"{entry['output_name']}.done.json"
        if marker_path.is_file():
            states.append(json.loads(marker_path.read_text(encoding="utf-8")))
        else:
            states.append({"repo_id": entry["repo_id"], "status": "pending"})
    summary = {"updated_at": datetime.now(timezone.utc).isoformat(), "models": states}
    temp_path = output_dir / "batch-summary.json.part"
    temp_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(output_dir / "batch-summary.json")


def main(argv: list[str] | None = None) -> int:
    """
    purpose: manifest 순서대로 미완료 모델을 분석하고 배치 상태를 지속 갱신한다.
    input: CLI argv 또는 테스트용 argv list.
    processing: manifest를 검증하고 완료 모델 skip, 미완료 모델 실행, summary 갱신을 수행한다.
    return/side effects: 모두 성공하면 0, 실패가 있으면 1을 반환하고 결과 디렉터리를 갱신한다.
    """

    args = build_parser().parse_args(argv)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _load_manifest(manifest_path)
    failed = False
    for entry in entries:
        if not args.force and _is_complete(entry, output_dir):
            _write_summary(entries, output_dir)
            continue
        marker = _run_entry(entry, output_dir, args.engine)
        failed = failed or marker["status"] != "success"
        _write_summary(entries, output_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
