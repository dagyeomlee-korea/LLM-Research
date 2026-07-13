from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from filecollector.schemas.weight_statistics import TensorStatistics


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS llm_weight_statistics (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT NOT NULL,
    revision TEXT NOT NULL DEFAULT '',
    file_path TEXT NOT NULL,
    tensor_name TEXT NOT NULL,
    layer_idx INT,
    module_type TEXT NOT NULL,
    dtype TEXT NOT NULL,
    shape JSONB NOT NULL,
    num_elements BIGINT NOT NULL,
    mean DOUBLE PRECISION,
    std DOUBLE PRECISION,
    skewness DOUBLE PRECISION,
    kurtosis DOUBLE PRECISION,
    excess_kurtosis DOUBLE PRECISION,
    l2_norm DOUBLE PRECISION,
    max_abs DOUBLE PRECISION,
    q99_abs DOUBLE PRECISION,
    q999_abs DOUBLE PRECISION,
    sparsity DOUBLE PRECISION,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (repo_id, revision, file_path, tensor_name)
);

CREATE INDEX IF NOT EXISTS idx_llm_weight_statistics_repo
    ON llm_weight_statistics (repo_id);

CREATE INDEX IF NOT EXISTS idx_llm_weight_statistics_module_type
    ON llm_weight_statistics (module_type);
"""

UPSERT_SQL = """
INSERT INTO llm_weight_statistics (
    repo_id, revision, file_path, tensor_name, layer_idx, module_type, dtype, shape, num_elements,
    mean, std, skewness, kurtosis, excess_kurtosis, l2_norm, max_abs,
    q99_abs, q999_abs, sparsity, metadata, analyzed_at
) VALUES (
    %(repo_id)s, %(revision)s, %(file_path)s, %(tensor_name)s, %(layer_idx)s,
    %(module_type)s, %(dtype)s, %(shape)s, %(num_elements)s, %(mean)s, %(std)s, %(skewness)s, %(kurtosis)s,
    %(excess_kurtosis)s, %(l2_norm)s, %(max_abs)s, %(q99_abs)s, %(q999_abs)s,
    %(sparsity)s, %(metadata)s, %(analyzed_at)s
)
ON CONFLICT (repo_id, revision, file_path, tensor_name) DO UPDATE SET
    layer_idx = EXCLUDED.layer_idx,
    module_type = EXCLUDED.module_type,
    dtype = EXCLUDED.dtype,
    shape = EXCLUDED.shape,
    num_elements = EXCLUDED.num_elements,
    mean = EXCLUDED.mean,
    std = EXCLUDED.std,
    skewness = EXCLUDED.skewness,
    kurtosis = EXCLUDED.kurtosis,
    excess_kurtosis = EXCLUDED.excess_kurtosis,
    l2_norm = EXCLUDED.l2_norm,
    max_abs = EXCLUDED.max_abs,
    q99_abs = EXCLUDED.q99_abs,
    q999_abs = EXCLUDED.q999_abs,
    sparsity = EXCLUDED.sparsity,
    metadata = EXCLUDED.metadata,
    analyzed_at = EXCLUDED.analyzed_at;
"""


class PostgresWeightStatisticsRepository:
    """
    purpose: tensor 통계 결과를 PostgreSQL `llm_weight_statistics` 테이블에 저장한다.
    input: PostgreSQL DSN 또는 `DATABASE_URL`/`POSTGRES_DSN` 환경변수.
    processing: psycopg3 또는 psycopg2 중 설치된 드라이버로 연결하고 upsert를 수행한다.
    return/side effects: DB에 테이블/행을 생성 또는 갱신한다.
    """

    def __init__(self, dsn: str | None = None) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL") or os.getenv("POSTGRES_DSN")
        if not self.dsn:
            raise ValueError("DATABASE_URL or POSTGRES_DSN is required")
        self._driver = _load_postgres_driver()

    def create_table(self) -> None:
        """
        purpose: `llm_weight_statistics` 테이블과 조회 인덱스를 준비한다.
        input: repository가 보유한 PostgreSQL DSN.
        processing: CREATE TABLE/INDEX SQL을 트랜잭션으로 실행한다.
        return/side effects: DB schema를 생성 또는 유지한다.
        """

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(CREATE_TABLE_SQL)
            conn.commit()

    def save_many(self, rows: Iterable[TensorStatistics]) -> int:
        """
        purpose: 여러 tensor 통계 row를 PostgreSQL에 저장한다.
        input: `TensorStatistics` iterable.
        processing: 각 row를 DB parameter dict로 변환해 UPSERT SQL을 실행한다.
        return/side effects: 저장한 row 수를 반환하고 DB 행을 생성/갱신한다.
        """

        count = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(UPSERT_SQL, _to_db_params(row, self._driver.name))
                    count += 1
            conn.commit()
        return count

    def _connect(self) -> Any:
        return self._driver.connect(self.dsn)


class _Driver:
    """
    purpose: psycopg 버전 차이를 repository 코드에서 숨긴다.
    input: import 가능한 PostgreSQL Python driver module.
    processing: driver 이름과 connect callable을 보관한다.
    return/side effects: 값 객체이며 외부 상태를 변경하지 않는다.
    """

    def __init__(self, name: str, module: Any) -> None:
        self.name = name
        self.module = module

    def connect(self, dsn: str) -> Any:
        """
        purpose: 선택된 PostgreSQL driver로 DB 연결을 연다.
        input: PostgreSQL DSN 문자열.
        processing: psycopg/psycopg2의 connect 함수를 호출한다.
        return/side effects: DB connection을 반환하고 네트워크/소켓 연결을 생성한다.
        """

        return self.module.connect(dsn)


def _load_postgres_driver() -> _Driver:
    """
    purpose: 설치된 PostgreSQL driver를 자동 선택한다.
    input: 현재 Python 환경.
    processing: psycopg3를 먼저 시도하고 실패하면 psycopg2를 시도한다.
    return/side effects: `_Driver`를 반환하거나 ImportError를 발생시킨다.
    """

    try:
        import psycopg

        return _Driver("psycopg", psycopg)
    except ImportError:
        pass
    try:
        import psycopg2

        return _Driver("psycopg2", psycopg2)
    except ImportError as exc:
        raise ImportError("Install psycopg or psycopg2 to save weight statistics") from exc


def _to_db_params(row: TensorStatistics, driver_name: str) -> dict[str, Any]:
    """
    purpose: schema 객체를 PostgreSQL parameter dict로 변환한다.
    input: `TensorStatistics` row와 driver 이름.
    processing: JSONB 컬럼은 psycopg 계열 호환을 위해 JSON wrapper 또는 JSON 문자열로 변환한다.
    return/side effects: DB execute에 전달할 dict를 반환하며 외부 상태는 변경하지 않는다.
    """

    shape: Any = row.shape
    metadata: Any = row.metadata
    if driver_name == "psycopg":
        try:
            from psycopg.types.json import Jsonb

            shape = Jsonb(row.shape)
            metadata = Jsonb(row.metadata)
        except ImportError:
            shape = json.dumps(row.shape)
            metadata = json.dumps(row.metadata)
    else:
        try:
            from psycopg2.extras import Json

            shape = Json(row.shape)
            metadata = Json(row.metadata)
        except ImportError:
            shape = json.dumps(row.shape)
            metadata = json.dumps(row.metadata)
    return {
        "repo_id": row.repo_id,
        "revision": row.revision or "",
        "file_path": row.file_path,
        "tensor_name": row.tensor_name,
        "layer_idx": row.layer_idx,
        "module_type": row.module_type,
        "dtype": row.dtype,
        "shape": shape,
        "num_elements": row.num_elements,
        "mean": row.mean,
        "std": row.std,
        "skewness": row.skewness,
        "kurtosis": row.kurtosis,
        "excess_kurtosis": row.excess_kurtosis,
        "l2_norm": row.l2_norm,
        "max_abs": row.max_abs,
        "q99_abs": row.q99_abs,
        "q999_abs": row.q999_abs,
        "sparsity": row.sparsity,
        "metadata": metadata,
        "analyzed_at": row.analyzed_at,
    }
