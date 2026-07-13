from __future__ import annotations

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from filecollector.analysis.tensor_classifier import classify_tensor_name
from filecollector.analysis.tensor_classifier import classify_tensor
from filecollector.analysis.kurtosis_delta import compare_kurtosis
from filecollector.analysis.weight_stats_batch import _is_complete, _load_manifest
from filecollector.analysis.weight_stats_summary import _compare_models, _percentile
from filecollector.analysis.weight_distance import compute_pair
from filecollector.analysis.advanced_fingerprint import delta_alignment, linear_cka, randomized_delta_svd
from filecollector.analysis.weight_statistics_service import WeightStatisticsService
from filecollector.schemas.weight_statistics import TensorStatistics


def _write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], list[float]]]) -> None:
    offsets: dict[str, tuple[int, int]] = {}
    data = bytearray()
    for name, (dtype, _shape, values) in tensors.items():
        start = len(data)
        if dtype == "F32":
            data.extend(struct.pack("<" + "f" * len(values), *values))
        elif dtype == "F64":
            data.extend(struct.pack("<" + "d" * len(values), *values))
        elif dtype == "BF16":
            for value in values:
                fp32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
                data.extend(struct.pack("<H", fp32_bits >> 16))
        else:
            raise AssertionError(f"unsupported test dtype: {dtype}")
        offsets[name] = (start, len(data))
    header = {
        name: {"dtype": dtype, "shape": shape, "data_offsets": list(offsets[name])}
        for name, (dtype, shape, _values) in tensors.items()
    }
    header_raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + data)


class TensorClassifierTest(unittest.TestCase):
    def test_classifies_common_llm_tensor_names(self) -> None:
        self.assertEqual(classify_tensor_name("model.embed_tokens.weight"), "embedding")
        self.assertEqual(classify_tensor_name("model.layers.0.self_attn.q_proj.weight"), "attn_q")
        self.assertEqual(classify_tensor_name("model.layers.0.self_attn.k_proj.weight"), "attn_k")
        self.assertEqual(classify_tensor_name("model.layers.0.self_attn.v_proj.weight"), "attn_v")
        self.assertEqual(classify_tensor_name("model.layers.0.self_attn.o_proj.weight"), "attn_o")
        self.assertEqual(classify_tensor_name("model.layers.0.mlp.gate_proj.weight"), "mlp_gate")
        self.assertEqual(classify_tensor_name("model.layers.0.mlp.up_proj.weight"), "mlp_up")
        self.assertEqual(classify_tensor_name("model.layers.0.mlp.down_proj.weight"), "mlp_down")
        self.assertEqual(classify_tensor_name("model.layers.0.input_layernorm.weight"), "norm")
        self.assertEqual(classify_tensor_name("lm_head.weight"), "lm_head")
        self.assertEqual(classify_tensor_name("unmatched.weight"), "other")

    def test_extracts_layer_index(self) -> None:
        self.assertEqual(classify_tensor("model.layers.12.self_attn.q_proj.weight"), (12, "attn_q"))
        self.assertEqual(classify_tensor("model.embed_tokens.weight"), (None, "embedding"))


class WeightStatisticsServiceTest(unittest.TestCase):
    def test_analyzes_safetensors_file_tensor_by_tensor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.safetensors"
            _write_safetensors(
                path,
                {
                    "model.layers.0.self_attn.q_proj.weight": ("F32", [2, 2], [1.0, -1.0, 0.0, 2.0]),
                    "model.layers.0.mlp.down_proj.weight": ("F64", [2], [0.0, 4.0]),
                },
            )
            service = WeightStatisticsService(histogram_bins=16, chunk_bytes=8)
            rows = list(service.analyze_file("test/repo", path, revision="abc123"))

        self.assertEqual(len(rows), 2)
        q_proj = rows[0]
        self.assertEqual(q_proj.repo_id, "test/repo")
        self.assertEqual(q_proj.revision, "abc123")
        self.assertEqual(q_proj.layer_idx, 0)
        self.assertEqual(q_proj.module_type, "attn_q")
        self.assertEqual(q_proj.num_elements, 4)
        self.assertAlmostEqual(q_proj.mean or math.nan, 0.5)
        self.assertAlmostEqual(q_proj.std or math.nan, math.sqrt(1.25))
        self.assertAlmostEqual(q_proj.l2_norm or math.nan, math.sqrt(6.0))
        self.assertEqual(q_proj.max_abs, 2.0)
        self.assertAlmostEqual(q_proj.sparsity or math.nan, 0.25)
        self.assertGreaterEqual(q_proj.q99_abs or 0.0, 1.8)

    def test_empty_statistics_for_unsupported_dtype(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.safetensors"
            header = {
                "model.layers.0.self_attn.q_proj.weight": {
                    "dtype": "F8_E4M3",
                    "shape": [1],
                    "data_offsets": [0, 1],
                }
            }
            header_raw = json.dumps(header).encode("utf-8")
            path.write_bytes(struct.pack("<Q", len(header_raw)) + header_raw + b"\x00")
            rows = list(WeightStatisticsService().analyze_file("test/repo", path))

        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].mean)
        self.assertIn("Unsupported dtype", rows[0].metadata["error"])

    def test_numpy_engine_matches_python_engine(self) -> None:
        """
        purpose: NumPy 벡터 엔진이 기존 Python 엔진과 같은 통계를 계산하는지 검증한다.
        input: F32와 F64 텐서가 들어 있는 임시 safetensors 파일.
        processing: 두 엔진의 결과를 각각 계산해 주요 통계와 분위수를 비교한다.
        return/side effects: 불일치 시 unittest assertion이 실패하며 임시 파일 외 상태는 변경하지 않는다.
        """

        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "model.safetensors"
            _write_safetensors(
                path,
                {
                    "model.layers.0.self_attn.q_proj.weight": (
                        "F32",
                        [2, 3],
                        [1.0, -1.0, 0.0, 2.0, 3.0, -4.0],
                    ),
                    "model.layers.0.mlp.down_proj.weight": ("F64", [3], [0.5, 4.0, -2.0]),
                    "model.layers.0.input_layernorm.weight": ("BF16", [3], [1.0, -0.5, 2.0]),
                },
            )
            python_rows = list(
                WeightStatisticsService(histogram_bins=16, chunk_bytes=8, engine="python").analyze_file(
                    "test/repo", path
                )
            )
            numpy_rows = list(
                WeightStatisticsService(histogram_bins=16, chunk_bytes=8, engine="numpy").analyze_file(
                    "test/repo", path
                )
            )

        self.assertEqual(len(python_rows), len(numpy_rows))
        for python_row, numpy_row in zip(python_rows, numpy_rows, strict=True):
            self.assertEqual(python_row.tensor_name, numpy_row.tensor_name)
            for field in (
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
            ):
                self.assertAlmostEqual(getattr(python_row, field), getattr(numpy_row, field), places=12)


class WeightStatisticsBatchTest(unittest.TestCase):
    def test_manifest_validation_and_completion_marker(self) -> None:
        """
        purpose: 배치 manifest 검증과 완료 marker 판정이 예상 행 수를 강제하는지 확인한다.
        input: 임시 safetensors, manifest, JSONL, done marker 파일.
        processing: manifest를 로드하고 성공 marker의 행 수 일치·불일치 결과를 비교한다.
        return/side effects: 잘못된 완료 판정 시 unittest assertion이 실패하며 임시 디렉터리만 사용한다.
        """

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            tensor_path = root / "model.safetensors"
            _write_safetensors(tensor_path, {"weight": ("F32", [1], [1.0])})
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "repo_id": "test/repo",
                            "family": "test",
                            "classification": "official_base",
                            "path": str(tensor_path),
                            "expected_tensors": 1,
                            "output_name": "test-repo",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            entry = _load_manifest(manifest_path)[0]
            (root / "test-repo.jsonl").write_text("{}\n", encoding="utf-8")
            marker_path = root / "test-repo.done.json"
            marker_path.write_text(json.dumps({"status": "success", "rows": 1}), encoding="utf-8")
            self.assertTrue(_is_complete(entry, root))
            marker_path.write_text(json.dumps({"status": "success", "rows": 0}), encoding="utf-8")
            self.assertFalse(_is_complete(entry, root))


class WeightStatisticsSummaryTest(unittest.TestCase):
    def test_percentile_and_anchor_comparison(self) -> None:
        """
        purpose: summary 분위수와 tensor별 anchor 첨도 차이 계산을 검증한다.
        input: 두 개 tensor row로 구성된 비교 모델과 anchor fixture.
        processing: p95와 absolute/signed delta, 동일 fingerprint 수를 계산한다.
        return/side effects: 계산이 다르면 unittest assertion이 실패하며 외부 상태는 변경하지 않는다.
        """

        self.assertAlmostEqual(_percentile([0.0, 10.0], 0.95) or 0.0, 9.5)
        anchor_rows = [
            {
                "tensor_name": "a",
                "module_type": "attn_q",
                "dtype": "F32",
                "shape": [1],
                "num_elements": 1,
                "kurtosis": 4.0,
            },
            {
                "tensor_name": "b",
                "module_type": "attn_k",
                "dtype": "F32",
                "shape": [1],
                "num_elements": 1,
                "kurtosis": 2.0,
            },
        ]
        model_rows = [dict(anchor_rows[0]), {**anchor_rows[1], "kurtosis": 5.0}]
        entry = {"repo_id": "model", "family": "test"}
        anchor = {"repo_id": "anchor", "family": "test"}
        comparison = _compare_models(entry, model_rows, anchor, anchor_rows)
        self.assertEqual(comparison["common_tensors"], 2)
        self.assertEqual(comparison["identical_fingerprint_tensors"], 1)
        self.assertAlmostEqual(comparison["absolute_kurtosis_delta"]["median"], 1.5)


class WeightDistanceTest(unittest.TestCase):
    def test_computes_expected_frobenius_and_cosine_distance(self) -> None:
        """
        purpose: synthetic tensor 쌍의 normalized Frobenius와 cosine distance를 수작업 값과 비교한다.
        input: 값 `[1, 2]`와 `[1, 4]`를 가진 F32 safetensors 두 개.
        processing: 실제 chunk pair 계산 후 norm/dot 공식의 기대값을 검증한다.
        return/side effects: 계산이 다르면 unittest assertion이 실패하며 임시 파일만 생성한다.
        """

        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            path_a = root / "a.safetensors"
            path_b = root / "b.safetensors"
            tensor_name = "model.language_model.layers.0.self_attn.q_proj.weight"
            _write_safetensors(path_a, {tensor_name: ("F32", [2], [1.0, 2.0])})
            _write_safetensors(path_b, {tensor_name: ("F32", [2], [1.0, 4.0])})
            entry_a = {"repo_id": "a", "family": "test", "path": str(path_a)}
            entry_b = {"repo_id": "b", "family": "test", "path": str(path_b)}
            records, summary = compute_pair(
                entry_a,
                entry_b,
                {tensor_name: 3.0},
                {tensor_name: 5.0},
                chunk_bytes=4,
            )

        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0]["l2"], 2.0)
        self.assertAlmostEqual(records[0]["symmetric_l2"], 2.0 / math.sqrt(22.0))
        self.assertAlmostEqual(records[0]["cosine_distance"], 1.0 - 9.0 / math.sqrt(85.0))
        self.assertEqual(records[0]["kurtosis_abs_delta"], 2.0)
        self.assertEqual(summary["views"]["language_core"]["tensor_count"], 1)

    def test_identical_bfloat16_tensor_has_zero_distance(self) -> None:
        """
        purpose: 같은 BF16 파일 self-comparison이 모든 핵심 거리에서 0인지 검증한다.
        input: BF16 language MLP tensor 하나가 든 임시 safetensors.
        processing: 동일 경로를 두 entry로 비교해 L2와 cosine distance를 확인한다.
        return/side effects: 0이 아니면 unittest assertion이 실패하며 임시 파일만 생성한다.
        """

        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "same.safetensors"
            tensor_name = "model.language_model.layers.0.mlp.down_proj.weight"
            _write_safetensors(path, {tensor_name: ("BF16", [3], [1.0, -0.5, 2.0])})
            entry_a = {"repo_id": "a", "family": "test", "path": str(path)}
            entry_b = {"repo_id": "b", "family": "test", "path": str(path)}
            records, _ = compute_pair(entry_a, entry_b, {tensor_name: 4.0}, {tensor_name: 4.0})

        self.assertEqual(records[0]["l2"], 0.0)
        self.assertEqual(records[0]["symmetric_l2"], 0.0)
        self.assertEqual(records[0]["cosine_distance"], 0.0)


class AdvancedFingerprintTest(unittest.TestCase):
    def test_linear_cka_is_one_for_identical_and_scaled_matrix(self) -> None:
        """
        purpose: centered linear CKA가 동일 matrix와 상수 배율 matrix에서 1인지 검증한다.
        input: 4x3 synthetic NumPy matrix와 3배 matrix.
        processing: 동일/scale pair의 CKA를 계산해 1과 비교한다.
        return/side effects: CKA가 다르면 unittest assertion이 실패하며 외부 상태는 변경하지 않는다.
        """

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        matrix = np.array([[1, 2, 0], [0, 1, 3], [2, -1, 1], [4, 0, -2]], dtype=np.float32)
        self.assertAlmostEqual(linear_cka(matrix, matrix)["cka"], 1.0, places=12)
        self.assertAlmostEqual(linear_cka(matrix, matrix * 3.0)["cka"], 1.0, places=12)

    def test_randomized_svd_detects_rank_one_delta(self) -> None:
        """
        purpose: randomized SVD가 synthetic rank-1 delta의 energy를 top-1에 집중시키는지 검증한다.
        input: 두 vector outer product로 만든 8x6 rank-1 matrix.
        processing: rank-3 randomized SVD를 수행해 top singular energy와 threshold rank를 확인한다.
        return/side effects: low-rank 탐지 실패 시 unittest assertion이 실패하며 외부 상태는 변경하지 않는다.
        """

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        left = np.arange(1, 9, dtype=np.float32)
        right = np.arange(1, 7, dtype=np.float32)
        result = randomized_delta_svd(np.outer(left, right), rank=3, oversample=2, seed=7)
        repeated = randomized_delta_svd(np.outer(left, right), rank=3, oversample=2, seed=7)
        self.assertEqual(result["singular_values"], repeated["singular_values"])
        self.assertAlmostEqual(result["top_1_energy_fraction"], 1.0, places=6)
        self.assertAlmostEqual(result["top_k_energy_fraction"], 1.0, places=6)
        self.assertEqual(result["energy_rank_90"], 1)

    def test_delta_alignment_detects_same_and_opposite_direction(self) -> None:
        """
        purpose: delta alignment가 같은 방향 +1, 반대 방향 -1을 반환하는지 검증한다.
        input: synthetic 2x2 delta matrix와 양·음 배율 matrix.
        processing: cosine alignment를 계산해 기대 방향과 비교한다.
        return/side effects: 방향 계산이 다르면 unittest assertion이 실패하며 외부 상태는 변경하지 않는다.
        """

        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")
        delta = np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)
        self.assertAlmostEqual(delta_alignment(delta, delta * 2.0) or 0.0, 1.0)
        self.assertAlmostEqual(delta_alignment(delta, delta * -1.0) or 0.0, -1.0)


class KurtosisDeltaTest(unittest.TestCase):
    def test_compares_common_tensor_kurtosis(self) -> None:
        row_a = TensorStatistics.empty(
            repo_id="model/a",
            revision="r1",
            file_path="a.safetensors",
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            layer_idx=0,
            module_type="attn_q",
            dtype="F32",
            shape=[2, 2],
        )
        row_b = TensorStatistics.empty(
            repo_id="model/b",
            revision="r2",
            file_path="b.safetensors",
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            layer_idx=0,
            module_type="attn_q",
            dtype="F32",
            shape=[2, 2],
        )
        row_a = TensorStatistics(**{**row_a.__dict__, "kurtosis": 4.0})
        row_b = TensorStatistics(**{**row_b.__dict__, "kurtosis": 5.0})

        rows = compare_kurtosis([row_a], [row_b])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["delta_kurtosis"], 1.0)
        self.assertAlmostEqual(rows[0]["relative_delta_kurtosis"], 0.25)


if __name__ == "__main__":
    unittest.main()
