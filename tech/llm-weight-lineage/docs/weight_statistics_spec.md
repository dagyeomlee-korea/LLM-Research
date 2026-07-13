# LLM Weight Statistics Specification

## Purpose

Add a weight fingerprint analysis module that reads local `.safetensors` files,
classifies each tensor by model role, computes tensor-level statistics, and
stores the result in PostgreSQL.

## Scope

- Input: one local `.safetensors` file and a Hugging Face `repo_id`.
- Processing unit: one tensor at a time.
- Output: one row per tensor in `llm_weight_statistics`.
- CLI:

```bash
python -m filecollector.analysis.weight_stats --repo-id xxx --path /models/xxx/model-00001-of-00004.safetensors
```

For large batches, explicitly select the NumPy engine:

```bash
python -m filecollector.analysis.weight_stats \
  --repo-id xxx \
  --path /models/xxx/model.safetensors \
  --engine numpy \
  --dry-run
```

## Tensor Categories

Tensor names are classified into these categories:

- `embedding`
- `attn_q`
- `attn_k`
- `attn_v`
- `attn_o`
- `mlp_gate`
- `mlp_up`
- `mlp_down`
- `norm`
- `lm_head`
- `other`

The classifier uses common Hugging Face naming patterns such as
`embed_tokens`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
`up_proj`, `down_proj`, `norm`, and `lm_head`.

## Statistics

For each tensor:

- `mean`
- `std`
- `skewness`
- `kurtosis`
- `excess_kurtosis`
- `l2_norm`
- `max_abs`
- `q99_abs`
- `q999_abs`
- `sparsity`

Moment statistics are computed in streaming fashion. Absolute quantiles are
estimated with a bounded histogram after discovering `max_abs`; this avoids
keeping all tensor values in memory. The default histogram size is 8192 bins.

## Processing Engines

- `python`: original scalar streaming implementation with no external tensor dependency.
- `numpy`: vectorized chunk implementation for production-size BF16/F16/F32/F64 files.
- `auto`: uses NumPy when installed and otherwise falls back to the Python engine.

Both engines preserve the same raw-moment and histogram definitions. BF16 NumPy
processing expands uint16 values to FP32 bit patterns and accumulates statistics
in float64. The NumPy path remains chunk-bounded and does not load the entire
model into RAM.

## Resumable Batch CLI

```bash
python -m filecollector.analysis.weight_stats_batch \
  --manifest /path/to/manifest.json \
  --output-dir /path/to/results \
  --engine numpy
```

The runner writes `.jsonl.part` during analysis, promotes it to `.jsonl` only
after row-count validation, stores one `.done.json` marker per model, and skips
validated models after restart.

## Table Definition

```sql
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
```

## Environment

The PostgreSQL repository reads one of these environment variables:

- `DATABASE_URL`
- `POSTGRES_DSN`

If no database URL is available, the CLI can still print JSON Lines with
`--dry-run`.
