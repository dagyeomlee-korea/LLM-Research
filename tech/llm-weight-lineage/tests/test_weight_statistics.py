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
