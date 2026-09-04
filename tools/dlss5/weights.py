"""Outer WEIGHTS_HT parser and proven serialized tensor decoders."""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

from .ops import decode_s_e4m3

@dataclass(frozen=True)
class WeightRecord:
    name: str
    record_offset: int
    data_offset: int
    data_size: int
    element_count: int
    dtype_code: int
    device_code: int
    rank: int
    present: int


class DLSS5WeightMap:
    """Read the outer tensor map without guessing the inner fused layout."""

    def __init__(self, data: bytes, records: list[WeightRecord], source: str = ""):
        self.data = data
        self.records = {record.name: record for record in records}
        self.source = source

    @classmethod
    def from_file(cls, path: str | Path) -> "DLSS5WeightMap":
        path = Path(path)
        data = path.read_bytes()
        if len(data) < 8:
            raise ValueError("WEIGHTS_HT is truncated")
        declared_size = struct.unpack_from("<Q", data, 0)[0]
        if declared_size != len(data):
            raise ValueError(f"declared size {declared_size} != file size {len(data)}")

        records: list[WeightRecord] = []
        offset = 8
        while offset < len(data):
            record_offset = offset
            if offset + 8 > len(data):
                raise ValueError("truncated name length")
            name_length = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            end_name = offset + name_length
            if end_name > len(data):
                raise ValueError("truncated weight name")
            name = data[offset:end_name].decode("ascii")
            offset = end_name
            if offset + 28 > len(data):
                raise ValueError(f"truncated record header for {name}")
            outer_size, inner_size, data_size = struct.unpack_from("<QQQ", data, offset)
            offset += 24
            present = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            data_offset = offset
            end_data = data_offset + data_size
            if end_data + 20 > len(data):
                raise ValueError(f"truncated tensor payload for {name}")
            offset = end_data
            device, rank, dtype_code, element_count = struct.unpack_from(
                "<QIII", data, offset
            )
            offset += 20
            if outer_size != inner_size or outer_size != data_size + 40:
                raise ValueError(f"unexpected record sizes for {name}")
            if data_size != element_count * 2:
                raise ValueError(f"{name} is not a two-byte-element tensor")
            records.append(
                WeightRecord(
                    name=name,
                    record_offset=record_offset,
                    data_offset=data_offset,
                    data_size=data_size,
                    element_count=element_count,
                    dtype_code=dtype_code,
                    device_code=device,
                    rank=rank,
                    present=present,
                )
            )
        return cls(data, records, str(path))

    def __contains__(self, name: str) -> bool:
        return name in self.records

    def record(self, name: str) -> WeightRecord:
        try:
            return self.records[name]
        except KeyError as exc:
            raise KeyError(f"weight is not present: {name}") from exc

    def raw(self, name: str) -> bytes:
        record = self.record(name)
        return self.data[record.data_offset : record.data_offset + record.data_size]

    def uint8(self, name: str) -> Tensor:
        # The clone detaches the tensor from the memory-mapped/bytes backing.
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.uint8).clone()

    def float16(self, name: str) -> Tensor:
        return torch.frombuffer(bytearray(self.raw(name)), dtype=torch.float16).clone()

    def fp8(self, name: str, *, byte_offset: int = 0, byte_count: Optional[int] = None) -> Tensor:
        """Return a byte slice from a fused FP8 payload.

        The outer serializer calls every payload element ``float16`` because
        the container stores two bytes per element.  The CUDA kernels load
        the payload as signed E4M3 bytes, so this method intentionally exposes
        the byte view instead of pretending it is a half tensor.
        """

        values = self.uint8(name)
        if byte_offset < 0 or byte_offset > values.numel():
            raise ValueError(f"invalid FP8 byte offset for {name}: {byte_offset}")
        if byte_count is None:
            byte_count = values.numel() - byte_offset
        if byte_count < 0 or byte_offset + byte_count > values.numel():
            raise ValueError(f"invalid FP8 byte range for {name}")
        return values[byte_offset : byte_offset + byte_count]

    def summary(self) -> dict[str, Any]:
        blocks: dict[int, list[dict[str, Any]]] = {}
        for record in self.records.values():
            match = re.fullmatch(r"block(\d+)\.layer(\d+)\.(.+)", record.name)
            if not match:
                continue
            block = int(match.group(1))
            blocks.setdefault(block, []).append(
                {
                    "name": record.name,
                    "elements": record.element_count,
                    "bytes": record.data_size,
                }
            )
        return {
            "source": self.source,
            "record_count": len(self.records),
            "serialized_size": len(self.data),
            "blocks": {str(k): v for k, v in sorted(blocks.items())},
        }

def decode_fp8_matrix(
    values: bytes | Tensor,
    shape: tuple[int, ...],
    *,
    byte_offset: int = 0,
    storage_order: str = "row_major",
) -> Tensor:
    """Decode a byte-packed signed E4M3 matrix into float32.

    ``row_major`` is the logical order used by the loader below.  The cubin
    itself consumes tiled shared-memory fragments, but the serialized
    matrices identified here have the exact byte count of the corresponding
    logical matrix.  Keeping the order explicit makes a future kernel-tile
    permutation easy to add without silently changing the checkpoint.
    """

    if storage_order != "row_major":
        raise ValueError(f"unsupported FP8 storage order: {storage_order}")
    if isinstance(values, Tensor):
        raw = values.to(dtype=torch.uint8, device="cpu").flatten()
    else:
        raw = torch.frombuffer(bytearray(values), dtype=torch.uint8).clone()
    count = math.prod(shape)
    if byte_offset < 0 or byte_offset + count > raw.numel():
        raise ValueError(
            f"FP8 matrix needs {count} bytes at offset {byte_offset}, payload has {raw.numel()}"
        )
    return decode_s_e4m3(raw[byte_offset : byte_offset + count]).reshape(shape)

def _decode_blob_matrix(weights: DLSS5WeightMap, name: str, spec: dict[str, Any]) -> Tensor:
    shape = tuple(spec["fp8_shape"])
    return decode_fp8_matrix(
        weights.fp8(name),
        shape,
        byte_offset=int(spec.get("fp8_offset", 0)),
    )


def _expect_blob_size(weights: DLSS5WeightMap, name: str, expected: int) -> None:
    actual = weights.record(name).data_size
    if actual != expected:
        raise ValueError(f"unexpected fused blob size for {name}: {actual} != {expected}")


def _decode_blob_f16(
    weights: DLSS5WeightMap,
    name: str,
    byte_offset: int,
    shape: tuple[int, ...],
) -> Tensor:
    if byte_offset % 2:
        raise ValueError(f"FP16 slice is not aligned: {name}+{byte_offset}")
    count = math.prod(shape)
    values = weights.float16(name)
    first = byte_offset // 2
    last = first + count
    if last > values.numel():
        raise ValueError(f"FP16 slice exceeds payload: {name}")
    return values[first:last].float().reshape(shape)


def _decode_blob_f32(
    weights: DLSS5WeightMap,
    name: str,
    byte_offset: int,
    shape: tuple[int, ...],
) -> Tensor:
    if byte_offset % 4:
        raise ValueError(f"FP32 slice is not aligned: {name}+{byte_offset}")
    count = math.prod(shape)
    raw = weights.raw(name)
    end = byte_offset + count * 4
    if end > len(raw):
        raise ValueError(f"FP32 slice exceeds payload: {name}")
    values = torch.frombuffer(bytearray(raw[byte_offset:end]), dtype=torch.float32).clone()
    return values.reshape(shape)


def _copy_parameter(destination: Tensor, source: Tensor, label: str) -> None:
    if tuple(destination.shape) != tuple(source.shape):
        raise ValueError(
            f"shape mismatch for {label}: destination={tuple(destination.shape)} "
            f"source={tuple(source.shape)}"
        )
    with torch.no_grad():
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
