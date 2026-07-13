from __future__ import annotations

import json
import math
import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
}

_STRUCT_FORMATS = {
    "BOOL": "?",
    "U8": "B",
    "I8": "b",
    "U16": "H",
    "I16": "h",
    "U32": "I",
    "I32": "i",
    "U64": "Q",
    "I64": "q",
    "F16": "e",
    "F32": "f",
    "F64": "d",
}

_NUMPY_DTYPES = {
    "BOOL": "?",
    "U8": "u1",
    "I8": "i1",
    "U16": "<u2",
    "I16": "<i2",
    "U32": "<u4",
    "I32": "<i4",
    "U64": "<u8",
    "I64": "<i8",
    "F16": "<f2",
    "F32": "<f4",
    "F64": "<f8",
}


@dataclass(frozen=True)
class TensorInfo:
    """
    purpose: safetensors 파일 안의 텐서 위치와 기본 메타데이터를 표현한다.
    input: 헤더의 tensor name, dtype, shape, data_offsets와 데이터 시작 위치.
    processing: 절대 byte offset과 원소 수를 계산할 수 있는 값 객체로 보관한다.
    return/side effects: 데이터 객체이며 외부 상태를 변경하지 않는다.
    """

    name: str
    dtype: str
    shape: list[int]
    start: int
    end: int

    @property
    def num_elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total


class SafeTensorFile:
    """
    purpose: `.safetensors` 파일을 mmap으로 열고 텐서 단위 streaming 값을 제공한다.
    input: 로컬 safetensors 파일 경로.
    processing: 헤더 JSON을 읽어 텐서 위치를 파악하고, 요청된 텐서 byte 구간만 chunk 단위로 순회한다.
    return/side effects: context manager로 파일/mmap 리소스를 열고 닫는다.
    """

    def __init__(self, path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> None:
        self.path = Path(path)
        self.chunk_bytes = chunk_bytes
        self._fh = None
        self._mmap = None
        self._data_start = 0
        self._header: dict[str, object] = {}

    def __enter__(self) -> "SafeTensorFile":
        self._fh = self.path.open("rb")
        header_len_raw = self._fh.read(8)
        if len(header_len_raw) != 8:
            raise ValueError("Invalid safetensors file: missing header length")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = self._fh.read(header_len)
        self._header = json.loads(header_raw.decode("utf-8"))
        self._data_start = 8 + header_len
        self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._mmap is not None:
            self._mmap.close()
        if self._fh is not None:
            self._fh.close()

    def tensors(self) -> Iterator[TensorInfo]:
        """
        purpose: 파일 헤더에 있는 텐서 목록을 순회한다.
        input: context manager로 열린 safetensors 헤더.
        processing: `__metadata__`를 제외하고 dtype/shape/data_offsets를 `TensorInfo`로 변환한다.
        return/side effects: `TensorInfo` iterator를 반환하며 외부 상태는 변경하지 않는다.
        """

        for name, entry in self._header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                continue
            offsets = entry["data_offsets"]
            yield TensorInfo(
                name=name,
                dtype=str(entry["dtype"]),
                shape=[int(dim) for dim in entry["shape"]],
                start=self._data_start + int(offsets[0]),
                end=self._data_start + int(offsets[1]),
            )

    def iter_values(self, tensor: TensorInfo) -> Iterator[float]:
        """
        purpose: 지정 텐서의 값을 전체 로드 없이 float stream으로 변환한다.
        input: `TensorInfo`와 열린 mmap.
        processing: dtype별 byte 크기에 맞춰 chunk를 자르고 `struct` 또는 BF16 변환으로 값을 생성한다.
        return/side effects: float iterator를 반환하며 mmap/file 상태는 읽기만 한다.
        """

        if self._mmap is None:
            raise RuntimeError("SafeTensorFile must be opened before reading values")
        dtype = tensor.dtype.upper()
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        item_size = _DTYPE_BYTES[dtype]
        chunk_size = max(item_size, self.chunk_bytes - (self.chunk_bytes % item_size))
        for pos in range(tensor.start, tensor.end, chunk_size):
            end = min(pos + chunk_size, tensor.end)
            raw = self._mmap[pos:end]
            if dtype == "BF16":
                yield from _iter_bfloat16(raw)
                continue
            fmt = "<" + _STRUCT_FORMATS[dtype]
            for value in struct.iter_unpack(fmt, raw):
                yield float(value[0])

    def iter_numpy_chunks(self, tensor: TensorInfo) -> Iterator[Any]:
        """
        purpose: 지정 텐서 값을 제한된 크기의 NumPy float64 배열로 순회한다.
        input: `TensorInfo`와 열린 mmap, 생성자에서 지정한 chunk byte 크기.
        processing: dtype별 raw chunk를 NumPy로 해석하고 BF16은 FP32 bit pattern으로 확장한다.
        return/side effects: 독립 float64 배열 iterator를 반환하며 원본 mmap과 파일은 읽기만 한다.
        """

        if self._mmap is None:
            raise RuntimeError("SafeTensorFile must be opened before reading values")
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("NumPy engine requires the numpy package") from exc

        dtype = tensor.dtype.upper()
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"Unsupported dtype: {tensor.dtype}")
        item_size = _DTYPE_BYTES[dtype]
        chunk_size = max(item_size, self.chunk_bytes - (self.chunk_bytes % item_size))
        for pos in range(tensor.start, tensor.end, chunk_size):
            end = min(pos + chunk_size, tensor.end)
            raw = self._mmap[pos:end]
            if dtype == "BF16":
                bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
                yield np.left_shift(bits, 16).view(np.float32).astype(np.float64)
                continue
            yield np.frombuffer(raw, dtype=_NUMPY_DTYPES[dtype]).astype(np.float64)


def _iter_bfloat16(raw: bytes) -> Iterator[float]:
    """
    purpose: BF16 byte stream을 Python float 값으로 변환한다.
    input: little-endian BF16 raw bytes.
    processing: BF16 상위 16비트를 FP32 bit pattern으로 확장해 `struct`로 해석한다.
    return/side effects: float iterator를 반환하며 외부 상태는 변경하지 않는다.
    """

    for (bits,) in struct.iter_unpack("<H", raw):
        fp32_bits = bits << 16
        yield struct.unpack("<f", struct.pack("<I", fp32_bits))[0]


def dtype_item_size(dtype: str) -> int | None:
    """
    purpose: 지원 dtype의 byte 크기를 조회한다.
    input: safetensors dtype 문자열.
    processing: 내부 dtype map에서 대문자 dtype을 찾는다.
    return/side effects: byte 크기 또는 None을 반환하며 외부 상태는 변경하지 않는다.
    """

    return _DTYPE_BYTES.get(dtype.upper())


def validate_tensor_size(tensor: TensorInfo) -> None:
    """
    purpose: 헤더 shape/dtype과 data_offsets 길이가 일치하는지 검증한다.
    input: safetensors 헤더에서 생성한 `TensorInfo`.
    processing: num_elements * dtype byte와 byte 구간 길이를 비교한다.
    return/side effects: 불일치 시 ValueError를 발생시키고 외부 상태는 변경하지 않는다.
    """

    item_size = dtype_item_size(tensor.dtype)
    if item_size is None:
        return
    expected = tensor.num_elements * item_size
    actual = tensor.end - tensor.start
    if expected != actual:
        raise ValueError(
            f"Tensor byte size mismatch for {tensor.name}: expected={expected} actual={actual}"
        )


def finite_or_none(value: float) -> float | None:
    """
    purpose: DB/JSON 저장에 부적절한 NaN/Inf 값을 None으로 정규화한다.
    input: 계산된 float 값.
    processing: `math.isfinite`로 유한성을 검사한다.
    return/side effects: 유한 float 또는 None을 반환하며 외부 상태는 변경하지 않는다.
    """

    return value if math.isfinite(value) else None
