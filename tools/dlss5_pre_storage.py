"""PyTorch views for the recovered DLSS5 pre downsample storage.

The physical dimensions below are proven by the launch parameters and the
safe store-slot probes.  ``to_hwc_candidate`` is deliberately separate: it
is the natural channel/pixel interpretation of the physical bytes, but the
native lane permutation has not been promoted to an exact logical contract.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor


PRE_DOWNSAMPLE_BYTES = 0xC8000
PRE_DOWNSAMPLE_SHAPE = (160, 2, 40, 16, 4)
PRE_DOWNSAMPLE_GPU_VA = 0x1BD36C00
PRE_DOWNSAMPLE_ARENA_OFFSET = 0x336C00


def _raw_tensor(raw: bytes | bytearray | Tensor) -> Tensor:
    if isinstance(raw, Tensor):
        value = raw.detach().to(device="cpu", dtype=torch.uint8).flatten().contiguous()
        return value.clone()
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()


def pre_downsample_physical_view(raw: bytes | bytearray | Tensor) -> Tensor:
    """Return exact storage as ``[row, half, x_tile, lane_word, byte]``."""

    value = _raw_tensor(raw)
    if value.numel() != PRE_DOWNSAMPLE_BYTES:
        raise ValueError(
            f"pre downsample storage has {value.numel()} bytes; expected {PRE_DOWNSAMPLE_BYTES}"
        )
    return value.reshape(PRE_DOWNSAMPLE_SHAPE)


def pre_downsample_from_arena(
    arena: bytes | bytearray | Tensor,
    *,
    arena_offset: int = PRE_DOWNSAMPLE_ARENA_OFFSET,
) -> Tensor:
    """Extract the exact physical pre view from a full arena snapshot."""

    value = _raw_tensor(arena)
    end = arena_offset + PRE_DOWNSAMPLE_BYTES
    if arena_offset < 0 or end > value.numel():
        raise ValueError(f"arena does not contain pre view [{arena_offset:#x}, {end:#x})")
    return pre_downsample_physical_view(value[arena_offset:end])


def pre_downsample_from_file(
    path: str | Path,
    *,
    arena_offset: int = PRE_DOWNSAMPLE_ARENA_OFFSET,
) -> Tensor:
    """Load and extract the exact physical pre view from a raw arena file."""

    return pre_downsample_from_arena(Path(path).read_bytes(), arena_offset=arena_offset)


def pre_downsample_to_hwc_candidate(physical: Tensor) -> Tensor:
    """Reinterpret physical bytes as an explicit, non-proven HWC candidate.

    The candidate uses ``x = x_tile*4 + byte`` and
    ``channel = half*16 + lane_word``.  It is reversible and useful for
    experiments, but callers must not use it as an exact native permutation
    until a consumer-side comparison proves the four-byte order.
    """

    if tuple(physical.shape) != PRE_DOWNSAMPLE_SHAPE:
        raise ValueError(f"expected physical shape {PRE_DOWNSAMPLE_SHAPE}, got {tuple(physical.shape)}")
    return physical.permute(0, 2, 4, 1, 3).reshape(160, 160, 32)


def pre_downsample_from_hwc_candidate(hwc: Tensor) -> Tensor:
    """Inverse of :func:`pre_downsample_to_hwc_candidate`."""

    if tuple(hwc.shape) != (160, 160, 32):
        raise ValueError(f"expected candidate HWC shape (160, 160, 32), got {tuple(hwc.shape)}")
    return hwc.reshape(160, 40, 4, 2, 16).permute(0, 3, 1, 4, 2).contiguous()


def decode_pre_downsample_e4m3(physical: Tensor) -> Tensor:
    """Decode the physical bytes to float32 while retaining physical shape."""

    from dlss5_pytorch import decode_s_e4m3

    if tuple(physical.shape) != PRE_DOWNSAMPLE_SHAPE:
        raise ValueError(f"expected physical shape {PRE_DOWNSAMPLE_SHAPE}, got {tuple(physical.shape)}")
    return decode_s_e4m3(physical)

