from __future__ import annotations

import argparse
import json
import sys

from filecollector.analysis.weight_statistics_service import WeightStatisticsService
from filecollector.repository.weight_statistics_repository import PostgresWeightStatisticsRepository


def build_parser() -> argparse.ArgumentParser:
    """
    purpose: weight statistics CLI 인자 parser를 구성한다.
    input: 없음.
    processing: repo-id/path/db/dry-run 관련 옵션을 argparse에 등록한다.
    return/side effects: `ArgumentParser`를 반환하며 외부 상태는 변경하지 않는다.
    """

    parser = argparse.ArgumentParser(
        description="Extract tensor-level LLM weight statistics from a safetensors file."
    )
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. Qwen/Qwen3.5-4B")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision or commit hash")
    parser.add_argument("--path", required=True, help="Local .safetensors file path")
    parser.add_argument("--database-url", default=None, help="PostgreSQL DSN. Defaults to DATABASE_URL/POSTGRES_DSN.")
    parser.add_argument("--histogram-bins", type=int, default=8192, help="Bins for approximate abs quantiles")
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024, help="Read chunk size per tensor")
    parser.add_argument(
        "--engine",
        choices=("auto", "python", "numpy"),
        default="auto",
        help="Statistics engine; auto uses NumPy when installed",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print JSON Lines instead of saving to PostgreSQL")
    parser.add_argument("--no-create-table", action="store_true", help="Skip CREATE TABLE IF NOT EXISTS")
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    purpose: safetensors weight 통계 분석 CLI 진입점을 실행한다.
    input: CLI argv 또는 테스트에서 전달한 argv list.
    processing: service로 tensor 통계를 생성하고 dry-run 출력 또는 PostgreSQL 저장을 수행한다.
    return/side effects: process exit code를 반환하고 stdout/DB에 결과를 기록한다.
    """

    args = build_parser().parse_args(argv)
    service = WeightStatisticsService(
        histogram_bins=args.histogram_bins,
        chunk_bytes=args.chunk_bytes,
        engine=args.engine,
    )
    rows = service.analyze_file(args.repo_id, args.path, revision=args.revision)
    if args.dry_run:
        count = 0
        for row in rows:
            print(json.dumps(row.to_dict(), ensure_ascii=False))
            count += 1
        print(f"analyzed={count} saved=0 dry_run=true", file=sys.stderr)
        return 0

    repository = PostgresWeightStatisticsRepository(dsn=args.database_url)
    if not args.no_create_table:
        repository.create_table()
    saved = repository.save_many(rows)
    print(f"analyzed={saved} saved={saved}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
