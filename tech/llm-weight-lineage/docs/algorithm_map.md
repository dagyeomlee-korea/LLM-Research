# Algorithm Map

## 1. Goal

Input:

- Weight files from multiple LLMs.

Output:

- Which models belong to the same lineage.
- Which model is a parent candidate.
- Which model is a child candidate.
- Overall Model Tree / Model Graph structure.

## 2. Core Pipeline

1. Collect model weights.
2. Compute model-to-model weight distance.
3. Compute kurtosis-based directional score.
4. Restore lineage with a directed minimum spanning tree.

High-level flow:

```text
Hugging Face metadata + safetensors
-> group comparable models
-> generate pairwise distance matrix
-> generate kurtosis directional matrix
-> restore Model Tree
```

## 3. Matrix Priority

### Priority 1: Q/K Matrix

- AWM uses Q/K matrices as core fingerprints.
- Important for base lineage verification.

### Priority 2: V/O Matrix

- Used to check attention output flow changes.

### Priority 3: MLP Gate/Up/Down

- Likely to show large changes from domain adaptation and instruction tuning.

### Priority 4: Embedding / LM Head

- Used to check tokenizer changes, vocabulary expansion, and domain-token effects.

### Priority 5: Norm Weights

- Used to check scale adjustment and post-training traces.

## 4. Stage 1: Metadata-Based Candidate Generation

Purpose:

- Reduce the comparison search space before expensive weight-level analysis.
- Create candidate groups from Hugging Face metadata and safetensors tensor metadata.

Expected inputs:

- Hugging Face repo metadata.
- Model card claims.
- safetensors file metadata.
- tensor name, shape, dtype metadata.

Expected outputs:

- Candidate model groups.
- Candidate parent-child pairs.
- Architecture compatibility labels.

## 5. Stage 2: Architecture Compatibility Check

Comparison criteria:

- `hidden_size` is identical or similar.
- `num_layers` is identical, or a pruning relationship is possible.
- Attention head structure is identical or similar.
- Q/K weight shapes are comparable.
- Embedding dimension is identical.
- Shared vocabulary exists.

Compatibility classes:

### A. Identical Architecture

- AWM and kurtosis can be compared directly.

### B. Different Layer Count

- Layer matching is required.

### C. Different Vocabulary Size

- Embedding comparison should use shared tokens.

### D. Different Hidden Size

- Direct comparison is difficult.
- Pruning/upcycling possibility should be handled separately.

### E. Tensor Key Mismatch

- An architecture adapter is required.

## 6. Implementation Guidance

- Start with metadata candidate generation and architecture compatibility checks before downloading or comparing many full weight files.
- Use Q/K matrix comparison as the first high-confidence fingerprint path.
- Treat kurtosis as a directionality signal, not as standalone proof of lineage.
- Store intermediate matrices separately: distance matrix, directional matrix, and restored graph edges.
- Keep graph restoration separate from evidence extraction so scoring logic can be revised without recomputing all tensor statistics.

## 7. Stage 4: Kurtosis Profile

For each model, store this profile:

- `model_id` or `repo_id`
- `revision`
- `layer_idx`
- `module_type`
- `tensor_name`
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
- `dtype`
- `shape`

The key signal is `kurtosis`.

Module types:

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

For model A and model B:

```text
delta_kurtosis = kurtosis(B) - kurtosis(A)
relative_delta = delta_kurtosis / abs(kurtosis(A))
```

Use a small epsilon in implementation to avoid division by zero.

## 8. Database Map

Existing or expected metadata tables:

- `hf_model_repositories`
- `hf_model_repository_files`

Analysis result tables:

- `llm_model_architecture_profiles`
- `llm_weight_tensor_index`
- `llm_weight_statistics`
- `llm_pairwise_fingerprint_scores`
- `llm_pairwise_kurtosis_deltas`
- `llm_lineage_candidates`

### 8.1 `llm_weight_statistics`

```sql
CREATE TABLE llm_weight_statistics (
    id BIGSERIAL PRIMARY KEY,
    repo_id TEXT NOT NULL,
    revision TEXT NOT NULL DEFAULT '',
    file_path TEXT,
    tensor_name TEXT NOT NULL,
    layer_idx INT,
    module_type TEXT,
    shape JSONB,
    dtype TEXT,
    num_elements BIGINT,
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
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE (repo_id, revision, file_path, tensor_name)
);
```

### 8.2 `llm_pairwise_fingerprint_scores`

```sql
CREATE TABLE llm_pairwise_fingerprint_scores (
    id BIGSERIAL PRIMARY KEY,
    model_a_repo_id TEXT NOT NULL,
    model_b_repo_id TEXT NOT NULL,
    revision_a TEXT,
    revision_b TEXT,
    method TEXT NOT NULL,
    awm_score DOUBLE PRECISION,
    q_score DOUBLE PRECISION,
    k_score DOUBLE PRECISION,
    embedding_lap_score DOUBLE PRECISION,
    compared_layers INT,
    status TEXT,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### 8.3 `llm_pairwise_kurtosis_deltas`

```sql
CREATE TABLE llm_pairwise_kurtosis_deltas (
    id BIGSERIAL PRIMARY KEY,
    model_a_repo_id TEXT NOT NULL,
    model_b_repo_id TEXT NOT NULL,
    revision_a TEXT,
    revision_b TEXT,
    layer_idx INT,
    module_type TEXT,
    tensor_name_a TEXT,
    tensor_name_b TEXT,
    kurtosis_a DOUBLE PRECISION,
    kurtosis_b DOUBLE PRECISION,
    delta_kurtosis DOUBLE PRECISION,
    relative_delta_kurtosis DOUBLE PRECISION,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

## 9. Python PoC Direction

Recommended packages for full tensor work:

```bash
pip install torch safetensors transformers scipy numpy pandas tqdm
pip install huggingface_hub
```

The first production-safe implementation may avoid loading a whole tensor stack
at once, but the torch-based PoC is acceptable for small downloaded models.

### 9.1 Tensor Classification

Use a classifier that returns:

```python
tuple[int | None, str]  # layer_idx, module_type
```

The local implementation is `filecollector.analysis.tensor_classifier.classify_tensor`.

### 9.2 Tensor Statistics

Compute:

- mean
- std
- skewness
- kurtosis
- excess_kurtosis
- l2_norm
- max_abs
- q99_abs
- q999_abs
- sparsity

For large tensors, prefer tensor-level processing and bounded-memory quantile
estimation.

### 9.3 Kurtosis Delta

Compare common tensors first by exact tensor name. Later versions can add
architecture adapters for mismatched keys or layer matching.

## 10. AWM PoC Direction

Before implementing full AWM with UCKA and LAP:

- Implement simple linear CKA.
- Apply it first to Q/K matrices.
- Use row sampling for large matrices, for example `max_rows=2048`.
- Store Q and K scores separately before averaging.

Q/K similarity rules:

- Compare only common Q/K tensor names.
- Skip tensors with shape mismatch.
- Sample rows deterministically.
- Average per-tensor CKA scores into the pair score.

This is a simplified AWM PoC. Add embedding LAP and UCKA after the basic Q/K
pipeline is validated.

## 11. Do Not Over-Interpret

- Do not infer parent-child direction from model card claims alone.
- Do not compare models with incompatible hidden size as if they were directly aligned.

## 12. Validated Weight Distance Stage

The first Qwen3.5 0.8B/2B batch validated these pairwise metrics:

- parameter-weighted global symmetric Frobenius/L2 distance;
- global cosine distance;
- tensor-balanced median and p95 distance;
- module-level distance;
- tensor-aligned kurtosis median and p95 distance.

Maintain two separate views:

- `all`: vision, embedding, MTP, language, and other compatible tensors;
- `language_core`: language-layer attention Q/K/V/O, MLP gate/up/down, and norm.

The validated results show that full-model aggregation can dilute targeted
language changes. Q/K detects the CloudGoat-style update, but O/down must also
be included to detect the Huihui-style projection-only modification. Use SHA-256
deduplication before pairwise computation.
- Do not use embedding/lm_head differences without accounting for tokenizer or vocab changes.
- Do not treat metadata similarity as weight-level evidence.
- Do not treat one matrix family as sufficient proof when other matrix families contradict it.
