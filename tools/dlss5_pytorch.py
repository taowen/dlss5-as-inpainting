"""Semantic PyTorch translation of the pinned DLSS-NR compute graph.

This file deliberately translates the *computation graph*, not the NVIDIA
kernel ABI.  ``WEIGHTS_HT.bin`` contains rank-1, fused and quantized blobs;
the blob names do not contain the inner tensor offsets/shapes.  Consequently
the model below is useful as a graph-faithful reference implementation and
accepts an ordinary PyTorch state dict, but it does not claim to be a
bit-exact loader for the proprietary blobs.

Tensor convention inside the network is NHWC.  This matches the ``tinlayout``
kernel family and makes the window/repack operations explicit.  Public input
convenience arguments are NCHW, as usual for PyTorch image models.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Exact parser for the outer WEIGHTS_HT serialization
# ---------------------------------------------------------------------------


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


def decode_s_e4m3(values: Tensor) -> Tensor:
    """Decode raw signed E4M3 bytes to float32.

    The CUDA package contains both E5M3 and signed E4M3 format names.  This
    helper is intentionally explicit and is only a byte decoder; selecting
    the correct scale/format for each fused blob still requires the missing
    inner blob schema.
    """

    x = values.to(torch.int32)
    sign = ((x >> 7) & 1).to(torch.float32)
    exponent = (x >> 3) & 0x0F
    mantissa = (x & 0x07).to(torch.float32)
    normal = torch.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    subnormal = torch.ldexp(mantissa / 8.0, torch.full_like(exponent, -6))
    result = torch.where(exponent == 0, subnormal, normal)
    result = torch.where(sign != 0, -result, result)
    # The cubin's clamp/NaN path identifies the E4M3FN NaN encodings.  All
    # other exponent-15 values remain finite; 0x7e/0xfe are the finite 448
    # endpoints rather than infinities.
    nan = (exponent == 15) & (mantissa == 7)
    return torch.where(nan, torch.full_like(result, float("nan")), result)


def quantize_s_e4m3_satfinite(values: Tensor) -> Tensor:
    """Round-trip an activation through the cubin's saturating E4M3 format.

    The quantized kernels accumulate with ``QMMA.*.F16.E4M3.E4M3`` and emit
    activations with ``F2FP.SATFINITE.E4M3.F16``.  A plain PyTorch graph must
    preserve that boundary: carrying the accumulators forward as unrestricted
    FP32 values overflows in the gated Split-Swin stack.  PyTorch's E4M3FN cast
    has the required rounding, while the explicit clamp supplies SATFINITE
    behavior for values outside the representable range.
    """

    finite = torch.nan_to_num(values, nan=0.0, posinf=448.0, neginf=-448.0)
    finite = finite.clamp(-448.0, 448.0)
    return finite.to(torch.float8_e4m3fn).to(values.dtype)


def _fp8_boundary(module: nn.Module, values: Tensor) -> Tensor:
    if getattr(module, "_emulate_fp8", False):
        return quantize_s_e4m3_satfinite(values)
    return values


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


def decode_post_output_tile_candidate(tile: Tensor, output_channels: int = 3) -> Tensor:
    """Undo the regular 16x32 post-output tile hole pattern as a candidate.

    The serialized post tail is a 16x32-half payload.  In the raw bytes its
    populated locations are four-row groups at rows 0..3 and 8..11, with
    four-column groups at 0..3, 8..11, 16..19 and 24..27.  Compressing the
    two K halves and the four-column groups produces a padded 4x32 logical
    tile.  This is an evidence-based tensor-core hypothesis, not a proven
    ABI permutation, so callers must opt in explicitly.
    """

    if tuple(tile.shape) != (16, 32):
        raise ValueError(f"post output tile must have shape (16, 32), got {tuple(tile.shape)}")
    if output_channels < 1 or output_channels > 4:
        raise ValueError("post output tile candidate supports 1..4 output channels")
    columns = torch.tensor(
        [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27],
        device=tile.device,
    )
    logical = torch.cat((tile[:4].index_select(1, columns), tile[8:12].index_select(1, columns)), dim=1)
    return logical[:output_channels]


def decode_post_output_tile_column_major(tile: Tensor, output_channels: int = 3) -> Tensor:
    """Decode the leading post-output tile as a column-major ``K x N`` tile.

    The post kernel loads the 16x32 FP16 tail as a linear 512-half fragment,
    while its final RGB matrix is consumed as ``K=32, N=3``.  Reinterpreting
    the leading ``32 * N`` values as ``(K, N)`` and transposing produces the
    logical PyTorch ``(N, K)`` weight.  This is the layout selected by the
    native golden-frame regression; the older ``raw`` and hole-compressed
    candidates remain available for forensic comparison.
    """

    if tuple(tile.shape) != (16, 32):
        raise ValueError(f"post output tile must have shape (16, 32), got {tuple(tile.shape)}")
    if output_channels < 1 or output_channels > 16:
        raise ValueError("post output column-major tile supports 1..16 output channels")
    leading = tile.flatten()[: 32 * output_channels]
    return leading.reshape(32, output_channels).transpose(0, 1).contiguous()


def decode_hmma_16816_f16_tile(tile: Tensor) -> Tensor:
    """Decode one physical FP16 tile used by ``HMMA.16816``.

    The pre front-end loads 32 lanes x 8 half values with one 128-bit load
    per lane.  For the ``m16n8k16`` B fragment, each lane contributes rows
    ``2*t, 2*t+1, 8+2*t, 9+2*t`` at column ``lane >> 2``.  The low and high
    four-half groups in a lane are two independent ``16x8`` B fragments.
    The returned matrix is the logical ``K=16, N=16`` tile obtained by
    placing those two fragments side by side.
    """

    if tuple(tile.shape) != (16, 16):
        raise ValueError(f"HMMA front tile must have shape (16, 16), got {tuple(tile.shape)}")
    physical = tile.flatten().view(32, 8)
    logical = tile.new_zeros((16, 16))
    for lane in range(32):
        group = lane >> 2
        thread = lane & 3
        rows = (2 * thread, 2 * thread + 1, 8 + 2 * thread, 9 + 2 * thread)
        for fragment, start in enumerate((0, 4)):
            for offset, row in enumerate(rows):
                logical[row, group + fragment * 8] = physical[lane, start + offset]
    return logical


# The fused kernels use ``MpCubicSiluActivation`` rather than GELU.  These
# are the exact half constants embedded in the sm_86/sm_120 SASS path.  The
# expression is written in the same order as the host op list:
# clamp -> abs -> mul -> rsub -> mul -> add -> mul.
CCT_CUBIC_SILU_ABS_COEFF = 0.055908203125       # half 0x2b28
CCT_CUBIC_SILU_LINEAR_COEFF = 0.447265625       # half 0x3728
CCT_CUBIC_SILU_BIAS = 0.89453125               # half 0x3b28


def cct_cubic_silu(x: Tensor) -> Tensor:
    """Evaluate the fused ``MpCubicSiluActivation`` elementwise.

    The CUDA sequence clamps only the polynomial input.  Its final multiply
    uses the original value, not the clamped value; keeping those two values
    separate matters for large activations.
    """

    t = x.clamp(-4.0, 4.0)
    p = CCT_CUBIC_SILU_LINEAR_COEFF - CCT_CUBIC_SILU_ABS_COEFF * t.abs()
    a = CCT_CUBIC_SILU_BIAS + t * p
    return x * a


class CCTCubicSiLU(nn.Module):
    """Module wrapper used inside the readable fused-block FFNs."""

    def forward(self, x: Tensor) -> Tensor:
        return cct_cubic_silu(x)


# These are the logical shapes implied by the class constructors and the
# packed sizes.  They are reported for analysis, not silently loaded: the
# original records are fused blobs, not standalone PyTorch tensors.
KNOWN_VIT_BLOB_LAYOUT = {
    "layer0": {
        "operator": "FFN expand",
        "fp8_offset": 0,
        "fp8_shape": (4096, 1024),
        "trailing_bytes": 16,
    },
    "layer1": {
        "operator": "FFN contract",
        "fp8_offset": 0,
        "fp8_shape": (1024, 4096),
        "cos_skip_offset": 4194304,
        "cos_skip_shape": (1024,),
    },
    "layer2": {
        "operator": "QKV",
        # This 64-half header is present before the 3072x1024 FP8 matrix in
        # every ViT blob.  Its producer semantics are not yet identified.
        "fp8_offset": 128,
        "fp8_shape": (3072, 1024),
        "header_f16": 64,
    },
    "layer3": {"operator": "attention scalar", "f16_shape": (1,)},
    "layer4": {
        "operator": "projection",
        "fp8_offset": 0,
        "fp8_shape": (1024, 1024),
        "cos_skip_offset": 1048576,
        "cos_skip_shape": (1024,),
    },
}


KNOWN_SPLIT_BLOB_LAYOUT = {
    "layer0": {
        "operator": "FFN gated pair",
        "fp8_offsets": (0, 262144),
        "fp8_shapes": ((512, 512), (512, 512)),
    },
    "layer1": {
        "operator": "FFN contract",
        "fp8_offset": 0,
        "fp8_shape": (512, 512),
        "cos_skip_offset": 262144,
        "cos_skip_shape": (512,),
    },
    "layer2": {
        "operator": "QKV + attention metadata",
        "fp8_offset": 0,
        "fp8_shape": (1536, 512),
        "attn_bias_offset": 786432,
        "attn_bias_shape": (16, 64, 64),
        "attn_scale_offset": 917504,
        "attn_scale_shape": (16,),
        "attn_scale_dtype": "float32",
    },
    "layer3": {
        "operator": "projection",
        "fp8_offset": 0,
        "fp8_shape": (512, 512),
        "cos_skip_offset": 262144,
        "cos_skip_shape": (512,),
    },
}


# Ordinary (non-downsample/non-upsample) Swin blobs use a custom MLP width
# rather than the usual 4*C.  These widths and byte offsets are recovered by
# matching the host-side names (weight1/weight2/ffn_cos_skip/qkv_weight/
# attn_bias/projection_weight/attn_cos_skip) against the record sizes.  The
# gap immediately before projection is a packed FP32 per-head attention
# scale, with alignment padding for the smaller channel families; the other
# gaps remain opaque.
STANDARD_SWIN_FFN_DIMS = {32: 128, 64: 224, 128: 384, 256: 704}

KNOWN_STANDARD_SWIN_LAYOUT = {
    32: {
        "expected_bytes": 20672,
        "weight1": (0, (128, 32)),
        "weight2": (4096, (32, 128)),
        "ffn_cos_skip": (8208, (32,)),
        "qkv": (8272, (96, 32)),
        "attn_bias": (11360, (1, 64, 64)),
        "attn_scale": (19552, (1,)),
        "projection": (19568, (32, 32)),
        "attn_cos_skip": (20592, (32,)),
    },
    64: {
        "expected_bytes": 61760,
        "weight1": (0, (224, 64)),
        "weight2": (14336, (64, 224)),
        "ffn_cos_skip": (28688, (64,)),
        "qkv": (28816, (192, 64)),
        "attn_bias": (41120, (2, 64, 64)),
        "attn_scale": (57504, (2,)),
        "projection": (57520, (64, 64)),
        "attn_cos_skip": (61616, (64,)),
    },
    128: {
        "expected_bytes": 197184,
        "weight1": (0, (384, 128)),
        "weight2": (49152, (128, 384)),
        "ffn_cos_skip": (98320, (128,)),
        "qkv": (98576, (384, 128)),
        "attn_bias": (147744, (4, 64, 64)),
        "attn_scale": (180512, (4,)),
        "projection": (180528, (128, 128)),
        "attn_cos_skip": (196912, (128,)),
    },
    256: {
        "expected_bytes": 689232,
        "weight1": (0, (704, 256)),
        "weight2": (180224, (256, 704)),
        "ffn_cos_skip": (360464, (256,)),
        "qkv": (360976, (768, 256)),
        "attn_bias": (557600, (8, 64, 64)),
        "attn_scale": (623136, (8,)),
        "projection": (623168, (256, 256)),
        "attn_cos_skip": (688704, (256,)),
    },
}

# Upsample blocks prepend a 2*C^2-byte transition region and insert an
# additional C-sized opaque ``sin`` region before QKV.  The Swin body
# matrices/scales/bias below are independently aligned and can therefore be
# loaded without guessing the transition operand.
KNOWN_UPSAMPLE_SWIN_LAYOUT = {
    32: {
        "weight1": (2048, (128, 32)),
        "weight2": (6144, (32, 128)),
        "ffn_cos_skip": (10256, (32,)),
        "qkv": (10400, (96, 32)),
        "attn_bias": (13472, (1, 64, 64)),
        "attn_scale": (21664, (1,)),
        "projection": (21680, (32, 32)),
        "attn_cos_skip": (22704, (32,)),
        "expected_bytes": 22784,
        "prefix_bytes": 2048,
        "opaque_before_qkv": (10320, 80),
    },
    64: {
        "weight1": (8192, (224, 64)),
        "weight2": (22528, (64, 224)),
        "ffn_cos_skip": (36864, (64,)),
        "qkv": (37120, (192, 64)),
        "attn_bias": (49408, (2, 64, 64)),
        "attn_scale": (65792, (2,)),
        "projection": (65808, (64, 64)),
        "attn_cos_skip": (69904, (64,)),
        "expected_bytes": 70048,
        "prefix_bytes": 8192,
        "opaque_before_qkv": (36992, 128),
    },
    128: {
        "weight1": (32768, (384, 128)),
        "weight2": (81920, (128, 384)),
        "ffn_cos_skip": (131072, (128,)),
        "qkv": (131584, (384, 128)),
        "attn_bias": (180736, (4, 64, 64)),
        "attn_scale": (213504, (4,)),
        "projection": (213520, (128, 128)),
        "attn_cos_skip": (229904, (128,)),
        "expected_bytes": 230176,
        "prefix_bytes": 32768,
        "opaque_before_qkv": (131328, 256),
    },
    256: {
        "weight1": (131072, (704, 256)),
        "weight2": (311296, (256, 704)),
        "ffn_cos_skip": (491520, (256,)),
        "qkv": (492544, (768, 256)),
        "attn_bias": (689152, (8, 64, 64)),
        "attn_scale": (754688, (8,)),
        "projection": (754720, (256, 256)),
        "attn_cos_skip": (820256, (256,)),
        "expected_bytes": 820784,
        "prefix_bytes": 131072,
        "opaque_before_qkv": (492032, 512),
    },
}

KNOWN_POST_SWIN_LAYOUT = {
    "expected_bytes": 21808,
    "weight1": (0, (128, 32)),
    "weight2": (4096, (32, 128)),
    "ffn_cos_skip": (8208, (32,)),
    "qkv": (8400, (96, 32)),
    "attn_bias": (11472, (1, 64, 64)),
    "projection": (19680, (32, 32)),
    "attn_cos_skip": (20704, (32,)),
    "attn_scale": (19664, (1,)),
    "opaque_before_qkv": (8272, 128),
    # The post factory's first four operations are convolution -> alias ->
    # mul -> add.  The 128-byte prefix is two 32-channel FP16 vectors:
    # depthwise input projection and the skip/input scale.  ``sin`` is a
    # dynamic interpolation operand and has no serialized payload here.
    "input_dw_weight_f16": (8272, (32,)),
    "input_scale_f16": (8336, (32,)),
    # Weight registration names put ``out_gain`` before ``out_conv_weight``.
    # The 1040-byte tail therefore splits exactly into an 8-half (16-byte)
    # gain slot followed by a 16x32-half output tile.  The latter is still a
    # tensor-core-swizzled tile; its physical rows are not yet proven to be
    # ordinary row-major output channels.
    "out_gain_f16": (20768, (8,)),
    "opaque_output_tail": (20768, 1040),
    "out_conv_weight_f16": (20784, (16, 32)),
}

KNOWN_PRE_SWIN_LAYOUT = {
    "expected_bytes": 21696,
    # The pre kernel's two body matrices start at the record base, like the
    # ordinary C32 body. Its extra texture/front-end payload is inserted
    # after those matrices, immediately before the residual/QKV metadata.
    "weight1": (0, (128, 32)),
    "weight2": (4096, (32, 128)),
    # Two 512-byte FP16 tiles are loaded by the HMMA front-end before the
    # body. Their producer feature assembly is still texture-dependent.
    "front_weight0_f16": (8208, (16, 16)),
    "front_weight1_f16": (8720, (16, 16)),
    "ffn_cos_skip": (9232, (32,)),
    "qkv": (9296, (96, 32)),
    "attn_bias": (12384, (1, 64, 64)),
    "projection": (20592, (32, 32)),
    "attn_cos_skip": (21616, (32,)),
    "attn_scale": (20576, (1,)),
}

KNOWN_DEC_INPUT_LAYOUT = {
    "expected_bytes": 525312,
    "conv_weight": (0, (512, 1024)),
    "dw_weight": (524288, (512,)),
    "opaque_sin": (525312, 0),
}

# Keep the bias-only view available to callers that want a cheap probe.  The
# offsets here are the starts of the actual FP16 bias arrays, not the starts
# of the preceding opaque alignment bytes.
KNOWN_SWIN_ATTENTION_BIAS = {
    32: {"heads": 1, "offset": 11360},
    64: {"heads": 2, "offset": 41120},
    128: {"heads": 4, "offset": 147744},
    256: {"heads": 8, "offset": 557600},
}

STANDARD_SWIN_BLOCKS = {
    **{index: (32, index - 1) for index in range(1, 4)},
    **{index: (64, index - 5) for index in range(5, 8)},
    **{index: (128, index - 9) for index in range(9, 14)},
    **{index: (256, index - 15) for index in range(15, 22)},
    **{index: (256, index - 49) for index in range(49, 56)},
    **{index: (128, index - 57) for index in range(57, 62)},
    **{index: (64, index - 63) for index in range(63, 66)},
    **{index: (32, index - 67) for index in range(67, 70)},
}

# The downsample variants reuse the ordinary Swin body at the same byte
# offsets and append one extra transition operand.  The operand's matrix shape
# is loaded below; its exact spatial packing remains kernel-specific.
DOWNSAMPLE_SWIN_BLOCKS = {
    4: (32, 0),
    8: (64, 0),
    14: (128, 0),
    22: (256, 0),
}
UPSAMPLE_SWIN_BLOCKS = {
    48: (256, 0),
    56: (128, 0),
    62: (64, 0),
    66: (32, 0),
}
SWIN_BODY_BLOCKS = {
    **STANDARD_SWIN_BLOCKS,
    **DOWNSAMPLE_SWIN_BLOCKS,
    **UPSAMPLE_SWIN_BLOCKS,
}


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


# ---------------------------------------------------------------------------
# Small NHWC building blocks
# ---------------------------------------------------------------------------


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    b, h, w, c = x.shape
    return (
        x.view(b, h // window_size, window_size, w // window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size * window_size, c)
    )


def _window_reverse(windows: Tensor, window_size: int, h: int, w: int, c: int) -> Tensor:
    windows_per_image = (h // window_size) * (w // window_size)
    batch = windows.shape[0] // windows_per_image
    return (
        windows.view(batch, h // window_size, w // window_size, window_size, window_size, c)
        .permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(batch, h, w, c)
    )


def _pad_nhwc(x: Tensor, multiple: int) -> tuple[Tensor, int, int]:
    _, h, w, _ = x.shape
    hp = (h + multiple - 1) // multiple * multiple
    wp = (w + multiple - 1) // multiple * multiple
    if hp == h and wp == w:
        return x, h, w
    return F.pad(x, (0, 0, 0, wp - w, 0, hp - h)), hp, wp


def _avg_pool2_nhwc(x: Tensor) -> Tensor:
    """The fixed 2x2/stride-2 pool emitted by the Split ``ProjPool`` kernel."""
    _, h, w, _ = x.shape
    hp = h + (h & 1)
    wp = w + (w & 1)
    if hp != h or wp != w:
        x = F.pad(x, (0, 0, 0, wp - w, 0, hp - h))
    # The cubin's four quarter-scale HMUL2 epilogues use a fixed 0.25,
    # including the padded edge lanes, rather than a variable divisor.
    return F.avg_pool2d(
        x.permute(0, 3, 1, 2), kernel_size=2, stride=2, count_include_pad=True
    ).permute(0, 2, 3, 1)


def _shift_mask(h: int, w: int, window_size: int, shift_size: int, device: torch.device) -> Tensor:
    if shift_size == 0:
        return torch.zeros(
            (1, 1, window_size * window_size, window_size * window_size),
            device=device,
        )
    mask = torch.zeros((1, h, w, 1), device=device)
    slices = (
        slice(0, -window_size),
        slice(-window_size, -shift_size),
        slice(-shift_size, None),
    )
    value = 0
    for hs in slices:
        for ws in slices:
            mask[:, hs, ws, :] = value
            value += 1
    mask_windows = _window_partition(mask, window_size).view(-1, window_size * window_size)
    result = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return result.masked_fill(result != 0, -100.0).masked_fill(result == 0, 0.0).unsqueeze(1)


class WindowCosineAttention(nn.Module):
    """Windowed cosine-QK attention used by the hierarchical Swin family."""

    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int = 8,
        *,
        use_qkv: bool = True,
        use_projection: bool = True,
        shift_size: int = 0,
        fp8_qkv: bool = False,
        fp8_output: bool = False,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.fp8_qkv = fp8_qkv
        self.fp8_output = fp8_output
        self.qkv = nn.Linear(dim, 3 * dim, bias=False) if use_qkv else None
        self.proj = nn.Linear(dim, dim, bias=False) if use_projection else nn.Identity()
        # The host op list has a direct ``mul`` after Q/K normalization.  The
        # serialized gap before projection contains one FP32 value per head;
        # it is not a log scale and there is no preceding exp operation.
        self.attn_scale = nn.Parameter(torch.ones(heads))
        # The Split-Swin class has an explicit attn_bias blob.  A full window
        # bias is a direct PyTorch representation of that logical operand.
        self.attn_bias = nn.Parameter(torch.zeros(heads, window_size * window_size, window_size * window_size))

    def _attend(self, qkv: Tensor, h: int, w: int, mask: Optional[Tensor]) -> Tensor:
        windows = _window_partition(qkv, self.window_size)
        n = self.window_size * self.window_size
        qkv = windows.view(-1, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # This is the explicit form of linalg_vector_norm -> clamp_min -> div
        # found in the host-side operation descriptions.
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = self.attn_scale.view(1, self.heads, 1, 1)
        scores = (q @ k.transpose(-2, -1)) * scale
        scores = scores + self.attn_bias.to(dtype=scores.dtype).unsqueeze(0)
        if mask is not None:
            scores = scores + mask.to(dtype=scores.dtype).repeat(scores.shape[0] // mask.shape[0], 1, 1, 1)
        weights = scores.softmax(dim=-1)
        output = (weights @ v).transpose(1, 2).contiguous().view(-1, n, self.dim)
        return _window_reverse(output, self.window_size, h, w, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.qkv is None:
            raise RuntimeError("this attention module expects a precomputed QKV tensor")
        original_h, original_w = x.shape[1:3]
        x, h, w = _pad_nhwc(x, self.window_size)
        if self.shift_size:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        qkv = self.qkv(x)
        if self.fp8_qkv:
            qkv = _fp8_boundary(self, qkv)
        mask = _shift_mask(h, w, self.window_size, self.shift_size, x.device)
        out = self._attend(qkv, h, w, mask)
        if self.fp8_output:
            out = _fp8_boundary(self, out)
        if self.shift_size:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        out = out[:, :original_h, :original_w, :]
        return self.proj(out)

    def forward_qkv(self, qkv: Tensor) -> Tensor:
        """Attention entry point for ``CCVitQKV -> CCVitAttention``."""
        original_h, original_w = qkv.shape[1:3]
        qkv, h, w = _pad_nhwc(qkv, self.window_size)
        out = self._attend(qkv, h, w, None)
        out = out[:, :original_h, :original_w, :]
        return _fp8_boundary(self, out) if self.fp8_output else out


class CCTVitAttention(nn.Module):
    """The separate ViT QKV/attention pair exposed by the DLL.

    Unlike the hierarchical Swin kernels, the ViT attention factory exposes
    ``bmm -> mul -> clamp -> exp -> sum -> clamp -> reciprocal -> mul ->
    bmm``.  The cubin implements the ``exp`` with a half2 bit-level
    approximation; ``torch.exp`` is the readable PyTorch equivalent of that
    logical op.  The QKV kernel also multiplies normalized Q/K by
    ``sqrt(head_dim)`` (5.65625 for the 32-wide ViT heads).
    """

    def __init__(self, dim: int = 1024, heads: int = 32, window_size: int = 8) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by head count")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.window_size = window_size
        # ``layer3.layer`` is the scalar input of the first ``mul`` in the
        # CCVitAttention op list (immediately after score bmm).  Keep it as a
        # separate parameter instead of folding it into Q/K: the cubin's
        # QKV kernel already contains the independent sqrt(32) factors.
        self.attn_scale = nn.Parameter(torch.ones(1))

    def forward(self, qkv: Tensor) -> Tensor:
        original_h, original_w = qkv.shape[1:3]
        qkv, h, w = _pad_nhwc(qkv, self.window_size)
        windows = _window_partition(qkv, self.window_size)
        n = self.window_size * self.window_size
        qkv = windows.view(-1, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        head_scale = math.sqrt(self.head_dim)
        q = q * head_scale
        k = k * head_scale

        scores = (q @ k.transpose(-2, -1)) * self.attn_scale
        # The SASS subtracts a detached per-row reference before its
        # bit-level exp path.  Keeping the subtraction explicit avoids the
        # half overflow that a literal ``exp(scores)`` would otherwise cause.
        scores = scores - scores.detach().amax(dim=-1, keepdim=True)
        weights = torch.exp(scores)
        # Exact clamp literal printed by the sm_86 cubin.
        denominator = weights.sum(dim=-1, keepdim=True).clamp_min(6.198883056640625e-5)
        weights = weights / denominator
        output = (weights @ v).transpose(1, 2).contiguous().view(-1, n, self.dim)
        output = _window_reverse(output, self.window_size, h, w, self.dim)
        return _fp8_boundary(self, output[:, :original_h, :original_w, :])


class SwinBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        ffn_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = STANDARD_SWIN_FFN_DIMS.get(dim, 4 * dim)
        self.attn = WindowCosineAttention(dim, heads, window_size, shift_size=shift_size)
        self.mlp = nn.Sequential(
            nn.Linear(dim, ffn_dim, bias=False),
            CCTCubicSiLU(),
            nn.Linear(ffn_dim, dim, bias=False),
        )
        self.ffn_cos_skip = nn.Parameter(torch.ones(dim))
        self.attn_cos_skip = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # The CCT host op list begins with the two FFN convolutions and their
        # mul+add residual epilogue, then emits qkv/attention/projection.
        # There is no LayerNorm op in that list; the only explicit norm is
        # the per-vector Q/K normalization inside WindowCosineAttention.
        hidden = _fp8_boundary(self, cct_cubic_silu(self.mlp[0](x)))
        x = _fp8_boundary(self, x + self.mlp[2](hidden) * self.ffn_cos_skip)
        return _fp8_boundary(self, x + self.attn(x) * self.attn_cos_skip)


class PatchMerging(nn.Module):
    """Semantic equivalent of a ``*_ds`` block's 2x spatial reduction."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_dim)
        self.reduction = nn.Linear(4 * in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x, h, w = _pad_nhwc(x, 2)
        x = torch.cat((x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]), dim=-1)
        return self.reduction(self.norm(x))


class PreSwinDownsample(nn.Module):
    """The block0 ``_ds`` pool, which has no serialized transition matrix."""

    def forward(self, x: Tensor) -> Tensor:
        return _fp8_boundary(self, _avg_pool2_nhwc(x))


class PatchExpand(nn.Module):
    """Semantic equivalent of an upsample block plus skip fusion."""

    def __init__(self, in_dim: int, out_dim: int, skip_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.skip_projection = nn.Linear(skip_dim, out_dim, bias=False)
        self.fuse = nn.Linear(2 * out_dim, out_dim, bias=False)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        b, h, w, _ = x.shape
        y = self.expand(self.norm(x)).view(b, h, w, 2, 2, -1)
        y = y.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, 2 * h, 2 * w, -1)
        skip = F.interpolate(
            skip.permute(0, 3, 1, 2), size=(2 * h, 2 * w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        return self.fuse(torch.cat((y, self.skip_projection(skip)), dim=-1))


class SwinTransitionDown(nn.Module):
    """Reference form of the extra ``*_ds`` transition convolution."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x, _, _ = _pad_nhwc(x, 2)
        x = F.avg_pool2d(
            x.permute(0, 3, 1, 2), kernel_size=2, stride=2, count_include_pad=True
        ).permute(0, 2, 3, 1)
        return _fp8_boundary(self, self.conv_weight(x))


class SwinTransitionUp(nn.Module):
    """Reference form of the extra ``*_upsample`` transition convolution."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv_weight = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        x = self.conv_weight(x)
        skip = F.interpolate(
            skip.permute(0, 3, 1, 2), size=x.shape[1:3], mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        return _fp8_boundary(self, x + skip)


class DecInputUpsample(nn.Module):
    """Reference form of the dedicated ``CCDecInputUpsample`` block.

    The host descriptor exposes a 1024-to-512 convolution followed by a
    channel-wise residual operand.  Spatial interpolation is kept explicit;
    the cubin's exact ``sin``/tile interpolation path remains metadata-only.
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 512) -> None:
        super().__init__()
        self.conv_weight = nn.Linear(in_dim, out_dim, bias=False)
        self.dw_weight = nn.Parameter(torch.ones(out_dim))

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(
            x.permute(0, 3, 1, 2), scale_factor=2, mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        x = self.conv_weight(x)
        skip = F.interpolate(
            skip.permute(0, 3, 1, 2), size=x.shape[1:3], mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        return _fp8_boundary(self, x + skip * self.dw_weight)


class SwinDownBlock(nn.Module):
    def __init__(self, dim: int, out_dim: int, heads: int, window_size: int = 8) -> None:
        super().__init__()
        self.body = SwinBlock(dim, heads, window_size, shift_size=window_size // 2)
        self.downsample = SwinTransitionDown(dim, out_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.body(x)
        return self.downsample(skip), skip


class SwinUpBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int, window_size: int = 8) -> None:
        super().__init__()
        self.upsample = SwinTransitionUp(in_dim, out_dim)
        self.body = SwinBlock(out_dim, heads, window_size, shift_size=window_size // 2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return self.body(self.upsample(x, skip))


# ---------------------------------------------------------------------------
# Split-Swin and ViT semantic translations
# ---------------------------------------------------------------------------


class SplitSwinBlock(nn.Module):
    """512-channel Split-Swin block.

    The five child names mirror the binary factory:
    QKVAttn, ProjPool, Ffwd, FfwdProj and the optional FinalHead.  The host
    constructor registers the children in the same order used by the
    fused block: FFN, FFN residual epilogue, QKV attention, and projection
    residual epilogue.  On the encoder's final block, ``ProjPool`` returns
    both the full-resolution residual tensor and its fixed 2x2 pooled view;
    FinalHead consumes the pooled view while the decoder receives the full
    view as output 1.  ``*_cos_skip`` are explicit per-channel residual
    scales; the corresponding epilogues are not affine Linear biases.
    """

    def __init__(self, dim: int = 512, final_head: bool = False, window_size: int = 8) -> None:
        super().__init__()
        if dim != 512:
            raise ValueError("the embedded Split-Swin implementation is specialized for 512 channels")
        self.qkv_attn = WindowCosineAttention(
            dim,
            heads=16,
            window_size=window_size,
            use_projection=False,
            fp8_qkv=True,
            fp8_output=True,
        )
        self.projection = nn.Linear(dim, dim, bias=False)
        self.attn_cos_skip = nn.Parameter(torch.ones(dim))
        # The host constructor keeps both FFN dimensions at 512.  The
        # layer0 blob contains two 512x512 FP8 matrices used on the two sides
        # of the fused gated activation; it is not a conventional 2*dim
        # expansion as in the ViT block.
        self.ffwd = nn.Linear(dim, dim, bias=False)
        self.ffwd_gate = nn.Linear(dim, dim, bias=False)
        self.ffwd_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_cos_skip = nn.Parameter(torch.ones(dim))
        # The FinalHead cubin has one HMMA matrix and its record is exactly
        # 1024*512 FP8 bytes plus padding.  Keep the first host-side
        # convolution slot as an identity and model the learned pointwise
        # 512 -> 1024 operation explicitly.
        self.final_head = nn.Identity() if final_head else None
        # block 30's compound wrapper exposes the pre-FinalHead 512-channel
        # full-resolution tensor as output 1 for the decoder.  Its output 0
        # is FinalHead(pool(full)), with half the spatial dimensions and
        # 1024 channels, for the first ViT block.
        self.final_output = nn.Conv2d(dim, 1024, 1, bias=False) if final_head else None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # ``layer0`` is two parallel 512x512 convolutions.  One branch goes
        # through MpCubicSiLU and is multiplied elementwise by the other;
        # it is not a serial Linear -> activation -> Linear stack.
        ffn_input = x
        activated = cct_cubic_silu(self.ffwd(ffn_input))
        gate = self.ffwd_gate(ffn_input)
        gated = _fp8_boundary(self, activated * gate)
        feed_forward = self.ffwd_proj(gated)
        x = _fp8_boundary(self, x + feed_forward * self.ffn_cos_skip)
        attention = self.qkv_attn(x)
        full = _fp8_boundary(self, x + self.projection(attention) * self.attn_cos_skip)
        if self.final_head is None:
            return full, full
        pooled = _avg_pool2_nhwc(full)
        skip = full
        main = self.final_output(self.final_head(pooled).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return _fp8_boundary(self, main), skip


class ViTBlock(nn.Module):
    """1024-channel ViT block (2-D and 1-D repack have the same math here)."""

    def __init__(self, dim: int = 1024, window_size: int = 8, layout: str = "2d") -> None:
        super().__init__()
        if dim != 1024:
            raise ValueError("the embedded ViT implementation is specialized for 1024 channels")
        if layout not in {"2d", "1d"}:
            raise ValueError("layout must be '2d' or '1d'")
        self.layout = layout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attention = CCTVitAttention(dim, heads=32, window_size=window_size)
        self.projection = nn.Linear(dim, dim, bias=False)
        self.attn_cos_skip = nn.Parameter(torch.ones(dim))
        self.ffn_expand = nn.Linear(dim, 4 * dim, bias=False)
        self.ffn_contract = nn.Linear(4 * dim, dim, bias=False)
        self.ffn_cos_skip = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        # The CCVit block registers and executes its FFN pair before QKV /
        # attention / projection, matching the fused host graph order.
        hidden = _fp8_boundary(self, cct_cubic_silu(self.ffn_expand(x)))
        x = _fp8_boundary(self, x + self.ffn_contract(hidden) * self.ffn_cos_skip)
        qkv = _fp8_boundary(self, self.qkv(x))
        x = x + self.projection(self.attention(qkv)) * self.attn_cos_skip
        return _fp8_boundary(self, x)


# ---------------------------------------------------------------------------
# Complete 71-block graph
# ---------------------------------------------------------------------------


class DLSS5Graph(nn.Module):
    """Graph-faithful translation of blocks 0..70.

    The cubin core contract is one RGB texture input to block0.  The
    ``color/history/motion`` arguments are optional convenience assembly for
    callers that have an outer temporal pipeline; use ``rgb=`` to pass the
    core input directly.  The default therefore stays at the statically
    confirmed 3 input channels.  ``pre_features=`` is an advanced hook for
    the 32-channel tensor produced by the unresolved CUDA texture front-end;
    when supplied, it enters the C32 body through an identity adapter.  The
    more precise ``pre_front_features=`` hook accepts the 15-channel tensor
    before the serialized FP16 front projection (or 16 channels with the
    final alignment lane present) and applies both tiles.
    """

    def __init__(
        self,
        *,
        color_channels: int = 3,
        history_channels: int = 0,
        motion_channels: int = 0,
        output_channels: int = 3,
        window_size: int = 8,
        vit_layout: str = "2d",
        post_output_layout: str = "column_major_prefix",
    ) -> None:
        super().__init__()
        if post_output_layout not in {"raw", "column_major_prefix", "tensor_core_candidate"}:
            raise ValueError(
                "post_output_layout must be 'raw', 'column_major_prefix', or "
                "'tensor_core_candidate'"
            )
        self.color_channels = color_channels
        self.history_channels = history_channels
        self.motion_channels = motion_channels
        self.output_channels = output_channels
        self.input_channels = color_channels + history_channels + motion_channels
        self.window_size = window_size
        self.post_output_layout = post_output_layout

        # The pre kernel has a texture/layout front-end before its C32 body.
        # The two serialized 16x16 FP16 front tiles are kept as buffers by
        # the loader below. Their texture feature producer is not yet
        # resolved, so this module remains an identity hook for callers that
        # can provide the assembled 32-channel body input.
        self.pre_texture_adapter = nn.Conv2d(32, 32, 1, bias=False)
        with torch.no_grad():
            self.pre_texture_adapter.weight.zero_()
            self.pre_texture_adapter.weight[:, :, 0, 0].copy_(torch.eye(32))
        self.register_buffer("pre_front_weight0_f16", torch.zeros(16, 16))
        self.register_buffer("pre_front_weight1_f16", torch.zeros(16, 16))
        # This is only a runnable RGB fallback.  The host method confirms an
        # RGB texture input, but the exact cat/ones_like/detach feature
        # assembly is still fused into the CUDA front-end and is not replaced
        # by this projection.
        if self.input_channels > 32:
            raise ValueError("pre input adapter supports at most 32 logical input channels")
        self.input_adapter = nn.Conv2d(self.input_channels, 32, 1, bias=False)
        with torch.no_grad():
            self.input_adapter.weight.zero_()
        self.pre_body = SwinBlock(32, 1, window_size, shift_size=window_size // 2)
        # Unlike blocks 4/8/14/22, block0 has no appended learned transition
        # matrix.  Its second output is the full-resolution Swin tensor and
        # output0 is the fixed pool branch.
        self.pre_down = PreSwinDownsample()

        self.enc32 = nn.ModuleList([SwinBlock(32, 1, window_size, i % 2 * (window_size // 2)) for i in range(3)])
        self.down32_64 = SwinDownBlock(32, 64, 1, window_size)
        self.enc64 = nn.ModuleList([SwinBlock(64, 2, window_size, i % 2 * (window_size // 2)) for i in range(3)])
        self.down64_128 = SwinDownBlock(64, 128, 2, window_size)
        self.enc128 = nn.ModuleList([SwinBlock(128, 4, window_size, i % 2 * (window_size // 2)) for i in range(5)])
        self.down128_256 = SwinDownBlock(128, 256, 4, window_size)
        self.enc256 = nn.ModuleList([SwinBlock(256, 8, window_size, i % 2 * (window_size // 2)) for i in range(7)])
        self.down256_512 = SwinDownBlock(256, 512, 8, window_size)

        self.split_enc = nn.ModuleList([SplitSwinBlock(final_head=i == 7, window_size=window_size) for i in range(8)])
        self.vit = nn.ModuleList([ViTBlock(layout=vit_layout, window_size=window_size) for _ in range(8)])

        self.dec_input = DecInputUpsample(1024, 512)
        self.split_dec = nn.ModuleList([SplitSwinBlock(window_size=window_size) for _ in range(8)])
        self.up512_256 = SwinUpBlock(512, 256, 8, window_size)
        self.dec256 = nn.ModuleList([SwinBlock(256, 8, window_size, i % 2 * (window_size // 2)) for i in range(7)])
        self.up256_128 = SwinUpBlock(256, 128, 4, window_size)
        self.dec128 = nn.ModuleList([SwinBlock(128, 4, window_size, i % 2 * (window_size // 2)) for i in range(5)])
        self.up128_64 = SwinUpBlock(128, 64, 2, window_size)
        self.dec64 = nn.ModuleList([SwinBlock(64, 2, window_size, i % 2 * (window_size // 2)) for i in range(3)])
        self.up64_32 = SwinUpBlock(64, 32, 1, window_size)
        self.dec32 = nn.ModuleList([SwinBlock(32, 1, window_size, i % 2 * (window_size // 2)) for i in range(3)])

        # Post input fusion is not a concatenation convolution.  The host op
        # sequence is convolution -> alias -> mul -> add, represented here as
        # a depthwise 1x1 input projection plus a scaled skip residual.
        self.post_input_projection = nn.Conv2d(32, 32, 1, groups=32, bias=False)
        self.post_input_scale = nn.Parameter(torch.ones(32))
        self.post_body = SwinBlock(32, 1, window_size)
        self.post_out = nn.Conv2d(32, output_channels, 1, bias=False)
        self.blend_scale = nn.Parameter(torch.tensor(0.73974609375), requires_grad=False)
        self.weight_report: Optional[dict[str, Any]] = None
        self._fp8_emulation_enabled = False

    def enable_fp8_emulation(self) -> "DLSS5Graph":
        """Emulate the activation storage boundaries of the quantized cubins.

        Outputs are rounded at the fused-kernel boundaries established by the
        ``*_fp8`` entry points. In particular, FFN expansion is quantized after
        MpCubicSiLU, matching the SASS ``QMMA -> activation -> F2FP`` order.
        """

        for module in self.modules():
            module._emulate_fp8 = True
        self._fp8_emulation_enabled = True
        return self

    def disable_fp8_emulation(self) -> "DLSS5Graph":
        for module in self.modules():
            module._emulate_fp8 = False
        self._fp8_emulation_enabled = False
        return self

    @staticmethod
    def _fit_channels(x: Tensor, channels: int) -> Tensor:
        if x.shape[1] == channels:
            return x
        if x.shape[1] > channels:
            return x[:, :channels]
        return F.pad(x, (0, 0, 0, 0, 0, channels - x.shape[1]))

    def _assemble_input(
        self,
        color: Optional[Tensor],
        history: Optional[Tensor],
        motion: Optional[Tensor],
        rgb: Optional[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if rgb is not None:
            if color is None:
                color = rgb[:, : self.color_channels]
            return self._fit_channels(rgb, self.input_channels), color
        if color is None:
            raise ValueError("color or rgb must be supplied")
        color = self._fit_channels(color, self.color_channels)
        if history is None:
            history = torch.zeros(
                color.shape[0], self.history_channels, color.shape[2], color.shape[3],
                device=color.device, dtype=color.dtype
            )
        else:
            history = self._fit_channels(history, self.history_channels)
        if motion is None:
            motion = torch.zeros(
                color.shape[0], self.motion_channels, color.shape[2], color.shape[3],
                device=color.device, dtype=color.dtype
            )
        else:
            motion = self._fit_channels(motion, self.motion_channels)
        return torch.cat((color, history, motion), dim=1), color

    def _run(
        self,
        rgb: Tensor,
        base_color: Tensor,
        pre_features: Optional[Tensor] = None,
        pre_front_features: Optional[Tensor] = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # block 0: pre block, retaining the full-resolution skip.
        if pre_features is not None and pre_front_features is not None:
            raise ValueError("pre_features and pre_front_features are mutually exclusive")
        if pre_front_features is not None:
            if pre_front_features.ndim != 4 or pre_front_features.shape[1] not in {15, 16}:
                raise ValueError("pre_front_features must be NCHW with 15 logical channels (or 16 with padding)")
            if pre_front_features.shape[0] != rgb.shape[0] or pre_front_features.shape[2:] != rgb.shape[2:]:
                raise ValueError("pre_front_features must have the same batch and spatial shape as rgb")
            front_tile0 = decode_hmma_16816_f16_tile(self.pre_front_weight0_f16)
            front_tile1 = decode_hmma_16816_f16_tile(self.pre_front_weight1_f16)
            # The last logical K row is all-zero in both decoded tiles and is
            # the HMMA alignment lane. Keep the public hook at the logical
            # K=15 contract while accepting a caller-provided padded lane.
            front_weight = torch.cat((front_tile0, front_tile1), dim=1)[:15]
            front_weight = front_weight.transpose(0, 1).contiguous().to(
                device=pre_front_features.device,
                dtype=pre_front_features.dtype,
            )
            x = _fp8_boundary(
                self,
                F.conv2d(pre_front_features[:, :15], front_weight.view(32, 15, 1, 1)),
            )
        elif pre_features is None:
            x = _fp8_boundary(self, self.input_adapter(rgb))
        else:
            if pre_features.ndim != 4 or pre_features.shape[1] != 32:
                raise ValueError("pre_features must be NCHW with exactly 32 channels")
            if pre_features.shape[0] != rgb.shape[0] or pre_features.shape[2:] != rgb.shape[2:]:
                raise ValueError("pre_features must have the same batch and spatial shape as rgb")
            x = _fp8_boundary(self, self.pre_texture_adapter(pre_features))
        x = x.permute(0, 2, 3, 1)
        skip0 = self.pre_body(x)
        x = self.pre_down(skip0)

        # blocks 1..22: four encoder resolutions.
        for block in self.enc32:
            x = block(x)
        x, skip1 = self.down32_64(x)       # block 4
        for block in self.enc64:
            x = block(x)
        x, skip2 = self.down64_128(x)      # block 8
        for block in self.enc128:
            x = block(x)
        x, skip3 = self.down128_256(x)     # block 14
        for block in self.enc256:
            x = block(x)
        x, skip4 = self.down256_512(x)     # block 22

        # blocks 23..30 and 31..38.
        split_skip = x
        for block in self.split_enc:
            x, split_skip = block(x)
        for block in self.vit:
            x = block(x)

        # blocks 39..69: symmetric decoder.
        x = self.dec_input(x, split_skip)  # block 39
        for block in self.split_dec:
            x, _ = block(x)
        x = self.up512_256(x, skip4)       # block 48
        for block in self.dec256:
            x = block(x)
        x = self.up256_128(x, skip3)       # block 56
        for block in self.dec128:
            x = block(x)
        x = self.up128_64(x, skip2)        # block 62
        for block in self.dec64:
            x = block(x)
        x = self.up64_32(x, skip1)         # block 66
        for block in self.dec32:
            x = block(x)

        # block 70: post block and final blend.  ControlMask is deliberately
        # absent from all previous calls, matching the SASS dataflow.
        main = x.permute(0, 3, 1, 2)
        skip = skip0.permute(0, 3, 1, 2)
        # Fused kernels pad each pyramid level to an internal tile size.  The
        # post block writes the caller's rectangle, so crop/resample the main
        # branch back to the retained full-resolution skip here.
        if main.shape[-2:] != skip.shape[-2:]:
            main = F.interpolate(main, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # The post factory's first four ops are convolution -> alias ->
        # mul -> add.  Its MpCubicSiLU belongs to the following post-body
        # FFN, so do not insert a GELU between the input fuse and that body.
        fused = self.post_input_projection(main) + skip * self.post_input_scale.view(1, 32, 1, 1)
        fused = _fp8_boundary(self, fused)
        fused = self.post_body(fused.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        neural = _fp8_boundary(self, self.post_out(fused))
        return neural, {"skip0": skip0, "skip1": skip1, "skip2": skip2, "skip3": skip3, "skip4": skip4}

    def forward(
        self,
        color: Optional[Tensor] = None,
        history: Optional[Tensor] = None,
        motion: Optional[Tensor] = None,
        control_mask: Optional[Tensor] = None,
        *,
        rgb: Optional[Tensor] = None,
        pre_features: Optional[Tensor] = None,
        pre_front_features: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        assembled, base_color = self._assemble_input(color, history, motion, rgb)
        neural, features = self._run(
            assembled,
            base_color,
            pre_features=pre_features,
            pre_front_features=pre_front_features,
        )
        base = self._fit_channels(base_color, self.output_channels)
        if control_mask is None:
            weight = self.blend_scale.to(dtype=neural.dtype, device=neural.device)
        else:
            mask = F.interpolate(control_mask[:, :1], size=neural.shape[-2:], mode="bilinear", align_corners=False)
            weight = (mask * self.blend_scale.to(dtype=mask.dtype, device=mask.device)).clamp(0.0, 1.0)
        # The simple-blend sm_120 tail computes, per RGB channel,
        #
        #   delta = base - neural
        #   output = neural + blend * delta
        #
        # (FADD at d400..d420 followed by FFMA at d430..d450).  The
        # ControlMask variant has the same final interpolation after
        # saturating the mask-derived blend factor.  Keeping the operand
        # order matters: blend_scale=1 selects the source color, not the
        # neural branch.
        output = neural + weight * (base - neural)
        if return_features:
            features["neural"] = neural
            features["blend_weight"] = weight if isinstance(weight, Tensor) else torch.as_tensor(weight)
            return output, features
        return output

    def load_swin_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Load the proven ordinary-Swin matrices, scales, and bias.

        The ordinary block is one fused ``layer0.layer`` record.  Its inner
        layout is now fixed for all four channel families, including the
        packed FP32 per-head ``attn_scale`` immediately before projection.
        """

        targets: dict[int, SwinBlock] = {}
        for index, module in enumerate(self.enc32):
            targets[1 + index] = module
        targets[4] = self.down32_64.body
        for index, module in enumerate(self.enc64):
            targets[5 + index] = module
        targets[8] = self.down64_128.body
        for index, module in enumerate(self.enc128):
            targets[9 + index] = module
        targets[14] = self.down128_256.body
        for index, module in enumerate(self.enc256):
            targets[15 + index] = module
        targets[22] = self.down256_512.body
        for index, module in enumerate(self.dec256):
            targets[49 + index] = module
        targets[48] = self.up512_256.body
        for index, module in enumerate(self.dec128):
            targets[57 + index] = module
        targets[56] = self.up256_128.body
        for index, module in enumerate(self.dec64):
            targets[63 + index] = module
        targets[62] = self.up128_64.body
        for index, module in enumerate(self.dec32):
            targets[67 + index] = module
        targets[66] = self.up64_32.body
        transition_targets: dict[int, nn.Module] = {
            4: self.down32_64.downsample,
            8: self.down64_128.downsample,
            14: self.down128_256.downsample,
            22: self.down256_512.downsample,
            48: self.up512_256.upsample,
            56: self.up256_128.upsample,
            62: self.up128_64.upsample,
            66: self.up64_32.upsample,
        }

        if blocks is None:
            blocks = sorted(SWIN_BODY_BLOCKS)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        skipped: list[str] = []
        for block_index in blocks:
            if block_index not in SWIN_BODY_BLOCKS:
                skipped.append(f"block{block_index} non-decoded Swin fused layout")
                continue
            channels, _ = SWIN_BODY_BLOCKS[block_index]
            if block_index in UPSAMPLE_SWIN_BLOCKS:
                spec = KNOWN_UPSAMPLE_SWIN_LAYOUT[channels]
            else:
                spec = KNOWN_STANDARD_SWIN_LAYOUT[channels]
            name = f"block{block_index}.layer0.layer"

            if block_index in STANDARD_SWIN_BLOCKS:
                _expect_blob_size(weights, name, int(spec["expected_bytes"]))
            else:
                required_end = int(spec["attn_cos_skip"][0]) + 2 * channels
                if weights.record(name).data_size < required_end:
                    raise ValueError(f"truncated Swin body in {name}")
            raw = weights.fp8(name)
            weight1_offset, weight1_shape = spec["weight1"]
            weight2_offset, weight2_shape = spec["weight2"]
            qkv_offset, qkv_shape = spec["qkv"]
            projection_offset, projection_shape = spec["projection"]
            weight1 = decode_fp8_matrix(raw, tuple(weight1_shape), byte_offset=int(weight1_offset))
            weight2 = decode_fp8_matrix(raw, tuple(weight2_shape), byte_offset=int(weight2_offset))
            qkv = decode_fp8_matrix(raw, tuple(qkv_shape), byte_offset=int(qkv_offset))
            projection = decode_fp8_matrix(
                raw, tuple(projection_shape), byte_offset=int(projection_offset)
            )
            _copy_parameter(targets[block_index].mlp[0].weight, weight1, f"{name}.weight1")
            _copy_parameter(targets[block_index].mlp[2].weight, weight2, f"{name}.weight2")
            _copy_parameter(targets[block_index].attn.qkv.weight, qkv, f"{name}.qkv_weight")
            _copy_parameter(targets[block_index].attn.proj.weight, projection, f"{name}.projection_weight")
            loaded.extend(
                [
                    f"{name}.weight1 -> mlp[0]",
                    f"{name}.weight2 -> mlp[2]",
                    f"{name}.qkv_weight -> attn.qkv",
                    f"{name}.projection_weight -> attn.proj",
                ]
            )

            ffn_cos_skip_offset, ffn_cos_skip_shape = spec["ffn_cos_skip"]
            ffn_cos_skip = _decode_blob_f16(
                weights, name, int(ffn_cos_skip_offset), tuple(ffn_cos_skip_shape)
            )
            _copy_parameter(
                targets[block_index].ffn_cos_skip,
                ffn_cos_skip,
                f"{name}.ffn_cos_skip",
            )
            loaded.append(f"{name}.ffn_cos_skip -> ffn_cos_skip")

            attn_cos_skip_offset, attn_cos_skip_shape = spec["attn_cos_skip"]
            attn_cos_skip = _decode_blob_f16(
                weights, name, int(attn_cos_skip_offset), tuple(attn_cos_skip_shape)
            )
            _copy_parameter(
                targets[block_index].attn_cos_skip,
                attn_cos_skip,
                f"{name}.attn_cos_skip",
            )
            loaded.append(f"{name}.attn_cos_skip -> attn_cos_skip")

            attn_bias_offset, attn_bias_shape = spec["attn_bias"]
            bias = _decode_blob_f16(
                weights,
                name,
                int(attn_bias_offset),
                tuple(attn_bias_shape),
            )
            if not torch.isfinite(bias).all():
                bad = int((~torch.isfinite(bias)).sum())
                skipped.append(f"{name}.attn_bias contains {bad} non-finite FP16 value(s)")
            else:
                _copy_parameter(targets[block_index].attn.attn_bias, bias, f"{name}.attn_bias")
                loaded.append(
                    f"{name}[{attn_bias_offset}:{int(attn_bias_offset) + bias.numel() * 2}] -> attn_bias"
                )

            scale_offset, scale_shape = spec["attn_scale"]
            scale = _decode_blob_f32(weights, name, int(scale_offset), tuple(scale_shape))
            _copy_parameter(
                targets[block_index].attn.attn_scale,
                scale,
                f"{name}.attn_scale",
            )
            loaded.append(
                f"{name}[{int(scale_offset)}:{int(scale_offset) + scale.numel() * 4}] -> attn_scale"
            )
            if block_index in DOWNSAMPLE_SWIN_BLOCKS:
                transition_bytes = 2 * channels * channels
                transition_offset = weights.record(name).data_size - transition_bytes
                transition = decode_fp8_matrix(
                    raw,
                    (2 * channels, channels),
                    byte_offset=transition_offset,
                )
                _copy_parameter(
                    transition_targets[block_index].conv_weight.weight,
                    transition,
                    f"{name}.downsample_conv",
                )
                loaded.append(f"{name}.downsample_conv -> transition")
                metadata.append(
                    {
                        "name": f"block{block_index}.downsample_spatial",
                        "applied": False,
                        "reason": "kernel spatial reduction is represented by average pooling",
                    }
                )
            if block_index in UPSAMPLE_SWIN_BLOCKS:
                transition = decode_fp8_matrix(
                    raw,
                    (channels, 2 * channels),
                    byte_offset=0,
                )
                _copy_parameter(
                    transition_targets[block_index].conv_weight.weight,
                    transition,
                    f"{name}.upsample_conv",
                )
                loaded.append(f"{name}.upsample_conv -> transition")
                metadata.append(
                    {
                        "name": f"block{block_index}.upsample_prefix",
                        "bytes": int(spec["prefix_bytes"]),
                        "applied": True,
                        "reason": "loaded as transition convolution",
                    }
                )
                opaque_offset, opaque_bytes = spec["opaque_before_qkv"]
                metadata.append(
                    {
                        "name": f"block{block_index}.sin_or_opaque",
                        "offset": int(opaque_offset),
                        "bytes": int(opaque_bytes),
                        "applied": False,
                        "reason": "inserted before QKV; operand meaning is not resolved",
                    }
                )
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_swin_attention_biases(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Compatibility alias for :meth:`load_swin_weights`."""

        return self.load_swin_weights(weights, blocks=blocks)

    def load_pre_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 0's C32 body and retain its unresolved front tiles."""

        name = "block0.layer0.layer"
        spec = KNOWN_PRE_SWIN_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = [
            {
                "name": "block0.pre_texture_front_weight0_f16",
                "offset": int(spec["front_weight0_f16"][0]),
                "bytes": math.prod(spec["front_weight0_f16"][1]) * 2,
                "applied": False,
                "shape": list(spec["front_weight0_f16"][1]),
                "reason": "sm_120 pre HMMA front tile; texture feature producer is unresolved",
            },
            {
                "name": "block0.pre_texture_front_weight1_f16",
                "offset": int(spec["front_weight1_f16"][0]),
                "bytes": math.prod(spec["front_weight1_f16"][1]) * 2,
                "applied": False,
                "shape": list(spec["front_weight1_f16"][1]),
                "reason": "sm_120 pre HMMA front tile; texture feature producer is unresolved",
            },
        ]
        skipped: list[str] = []

        for section, destination in (
            ("front_weight0_f16", self.pre_front_weight0_f16),
            ("front_weight1_f16", self.pre_front_weight1_f16),
        ):
            offset, shape = spec[section]
            tile = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, tile, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> audit buffer")

        # Keep the old direct-RGB path runnable for smoke tests and callers
        # without a reconstruction of the texture front-end, but do not
        # report it as a serialized adapter. There is no proven RGB->32
        # matrix in this record, so use a zero fallback rather than slicing
        # body or front payload bytes into a false projection.
        if self.input_channels:
            metadata.append(
                {
                    "name": "block0.rgb_fallback_weight",
                    "offset": None,
                    "bytes": 0,
                    "applied": False,
                    "reason": "zero fallback; no serialized RGB->32 projection is proven",
                }
            )

        for section, destination in (
            ("weight1", self.pre_body.mlp[0].weight),
            ("qkv", self.pre_body.attn.qkv.weight),
            ("projection", self.pre_body.attn.proj.weight),
        ):
            offset, shape = spec[section]
            matrix = decode_fp8_matrix(raw, tuple(shape), byte_offset=int(offset))
            _copy_parameter(destination, matrix, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> pre_body")

        weight2_offset, weight2_shape = spec["weight2"]
        weight2 = decode_fp8_matrix(raw, tuple(weight2_shape), byte_offset=int(weight2_offset))
        if not torch.isfinite(weight2).all():
            bad = int((~torch.isfinite(weight2)).sum())
            # Keep the valid 4092 entries.  The four 0x7f/0xff E4M3FN
            # markers are isolated invalid slots in an otherwise complete
            # matrix; zero is the least surprising runnable fallback, and
            # the report records that it is not a bit-exact interpretation.
            weight2 = torch.nan_to_num(weight2, nan=0.0, posinf=0.0, neginf=0.0)
            metadata.append(
                {
                    "name": f"{name}.weight2.nonfinite",
                    "count": bad,
                    "applied": True,
                    "value": "zero fallback",
                    "reason": "four isolated E4M3FN NaN markers were present in the serialized matrix",
                }
            )
        _copy_parameter(self.pre_body.mlp[2].weight, weight2, f"{name}.weight2")
        loaded.append(f"{name}.weight2 -> pre_body")

        for section, destination in (
            ("ffn_cos_skip", self.pre_body.ffn_cos_skip),
            ("attn_cos_skip", self.pre_body.attn_cos_skip),
        ):
            offset, shape = spec[section]
            scale = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, scale, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> pre_body")

        attn_scale = _decode_blob_f32(
            weights, name, int(spec["attn_scale"][0]), tuple(spec["attn_scale"][1])
        )
        _copy_parameter(self.pre_body.attn.attn_scale, attn_scale, f"{name}.attn_scale")
        loaded.append(f"{name}.attn_scale -> pre_body.attn_scale")

        bias_offset, bias_shape = spec["attn_bias"]
        bias = _decode_blob_f16(weights, name, int(bias_offset), tuple(bias_shape))
        if not torch.isfinite(bias).all():
            bad = int((~torch.isfinite(bias)).sum())
            skipped.append(f"{name}.attn_bias contains {bad} non-finite FP16 value(s)")
        else:
            _copy_parameter(self.pre_body.attn.attn_bias, bias, f"{name}.attn_bias")
            loaded.append(f"{name}.attn_bias -> pre_body")
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_post_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 70's body and its padded FP16 output projection."""

        name = "block70.layer0.layer"
        spec = KNOWN_POST_SWIN_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []

        for section, destination in (
            ("weight1", self.post_body.mlp[0].weight),
            ("weight2", self.post_body.mlp[2].weight),
            ("qkv", self.post_body.attn.qkv.weight),
            ("projection", self.post_body.attn.proj.weight),
        ):
            offset, shape = spec[section]
            matrix = decode_fp8_matrix(raw, tuple(shape), byte_offset=int(offset))
            _copy_parameter(destination, matrix, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> post_body")

        for section, destination in (
            ("ffn_cos_skip", self.post_body.ffn_cos_skip),
            ("attn_cos_skip", self.post_body.attn_cos_skip),
        ):
            offset, shape = spec[section]
            scale = _decode_blob_f16(weights, name, int(offset), tuple(shape))
            _copy_parameter(destination, scale, f"{name}.{section}")
            loaded.append(f"{name}.{section} -> post_body")

        bias_offset, bias_shape = spec["attn_bias"]
        bias = _decode_blob_f16(weights, name, int(bias_offset), tuple(bias_shape))
        if not torch.isfinite(bias).all():
            bad = int((~torch.isfinite(bias)).sum())
            return {
                "loaded": loaded,
                "metadata": metadata,
                "skipped": [f"{name}.attn_bias contains {bad} non-finite FP16 value(s)"],
            }
        _copy_parameter(self.post_body.attn.attn_bias, bias, f"{name}.attn_bias")
        loaded.append(f"{name}.attn_bias -> post_body")

        input_dw = _decode_blob_f16(
            weights,
            name,
            int(spec["input_dw_weight_f16"][0]),
            tuple(spec["input_dw_weight_f16"][1]),
        )
        _copy_parameter(
            self.post_input_projection.weight,
            input_dw.reshape(32, 1, 1, 1),
            f"{name}.dw_weight",
        )
        loaded.append(f"{name}.dw_weight -> post_input_projection")
        input_scale = _decode_blob_f16(
            weights,
            name,
            int(spec["input_scale_f16"][0]),
            tuple(spec["input_scale_f16"][1]),
        )
        _copy_parameter(self.post_input_scale, input_scale, f"{name}.inp_upsample_input_scale")
        loaded.append(f"{name}.inp_upsample_input_scale -> post_input_scale")

        gain_offset, gain_shape = spec["out_gain_f16"]
        out_gain = _decode_blob_f16(weights, name, int(gain_offset), tuple(gain_shape))
        metadata.append(
            {
                "name": "block70.out_gain",
                "offset": int(gain_offset),
                "bytes": int(math.prod(gain_shape) * 2),
                "applied": False,
                "min": float(out_gain.min()),
                "max": float(out_gain.max()),
                "zeros": int((out_gain == 0).sum()),
                "reason": "the registered 8-half out_gain slot is decoded and is all zero; static tracing finds only constructor registration, not an independent post-execute consumer, but index-based binding prevents proving it is padding",
            }
        )

        output_offset, output_shape = spec["out_conv_weight_f16"]
        padded_output = _decode_blob_f16(
            weights, name, int(output_offset), tuple(output_shape)
        )
        if self.output_channels > padded_output.shape[0]:
            raise ValueError(
                f"post output has only {padded_output.shape[0]} padded channels, "
                f"requested {self.output_channels}"
            )
        if self.post_output_layout == "tensor_core_candidate":
            if self.output_channels > 4:
                raise ValueError(
                    "tensor_core_candidate post output layout supports at most 4 channels"
                )
            logical_output = decode_post_output_tile_candidate(
                padded_output,
                output_channels=self.output_channels,
            )
            output_weight = logical_output.reshape(logical_output.shape[0], 32, 1, 1)
        elif self.post_output_layout == "column_major_prefix":
            logical_output = decode_post_output_tile_column_major(
                padded_output,
                output_channels=self.output_channels,
            )
            output_weight = logical_output.reshape(logical_output.shape[0], 32, 1, 1)
        else:
            output_weight = padded_output[: self.output_channels].reshape(
                self.output_channels, 32, 1, 1
            )
        _copy_parameter(self.post_out.weight, output_weight, f"{name}.out_conv_weight")
        loaded.append(
            f"{name}.out_conv_weight -> post_out[1x1] ({self.post_output_layout})"
        )

        scale_offset, scale_shape = spec["attn_scale"]
        scale = _decode_blob_f32(weights, name, int(scale_offset), tuple(scale_shape))
        _copy_parameter(self.post_body.attn.attn_scale, scale, f"{name}.attn_scale")
        loaded.append(
            f"{name}[{int(scale_offset)}:{int(scale_offset) + scale.numel() * 4}] -> post_body.attn_scale"
        )
        metadata.extend(
            [
                {
                    "name": "block70.front_scale_or_activation",
                    "offset": int(spec["opaque_before_qkv"][0]),
                    "bytes": int(spec["opaque_before_qkv"][1]),
                    "applied": False,
                    "reason": "opaque pre-QKV section remains unresolved; post-body attention scale is loaded from the FP32 gap before projection",
                },
                {
                    "name": "block70.out_conv_weight.tile_swizzle",
                    "offset": int(output_offset),
                    "bytes": int(padded_output.numel() * 2),
                    "applied": False,
                    "mode": self.post_output_layout,
                    "reason": "out_conv_weight is loaded from the post-gain 16x32 FP16 tile; tensor-core row/lane swizzle is not resolved even when the evidence-based candidate is selected",
                },
            ]
        )
        return {"loaded": loaded, "metadata": metadata, "skipped": []}

    def load_dec_input_weights(self, weights: DLSS5WeightMap) -> dict[str, Any]:
        """Load block 39's 1024-to-512 projection and residual scale."""

        name = "block39.layer0.layer"
        spec = KNOWN_DEC_INPUT_LAYOUT
        _expect_blob_size(weights, name, int(spec["expected_bytes"]))
        raw = weights.fp8(name)
        conv_offset, conv_shape = spec["conv_weight"]
        conv_weight = decode_fp8_matrix(raw, tuple(conv_shape), byte_offset=int(conv_offset))
        _copy_parameter(self.dec_input.conv_weight.weight, conv_weight, f"{name}.conv_weight")
        dw_offset, dw_shape = spec["dw_weight"]
        dw_weight = _decode_blob_f16(weights, name, int(dw_offset), tuple(dw_shape))
        _copy_parameter(self.dec_input.dw_weight, dw_weight, f"{name}.dw_weight")
        return {
            "loaded": [
                f"{name}.conv_weight -> dec_input",
                f"{name}.dw_weight -> dec_input",
            ],
            "metadata": [
                {
                    "name": "block39.sin",
                    "offset": int(spec["opaque_sin"][0]),
                    "bytes": int(spec["opaque_sin"][1]),
                    "applied": False,
                    "reason": "sin/tile interpolation is represented by explicit bilinear reference code",
                }
            ],
            "skipped": [],
        }

    def load_vit_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        block_start: int = 31,
        block_count: int = 8,
    ) -> dict[str, Any]:
        """Load the strictly identified ViT FP8 matrices and residual scales.

        The matrix byte order is the explicit row-major candidate documented
        by :func:`decode_fp8_matrix`.  The QKV blob has a verified 64-half
        prefix whose producer meaning is still unknown; it is reported but
        not injected into a PyTorch parameter.
        """

        if block_count != len(self.vit):
            raise ValueError("the embedded ViT loader expects eight blocks")
        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        for module_index, block_index in enumerate(range(block_start, block_start + block_count)):
            module = self.vit[module_index]
            prefix = f"block{block_index}"

            def matrix(layer: int) -> Tensor:
                name = f"{prefix}.layer{layer}.layer"
                spec = KNOWN_VIT_BLOB_LAYOUT[f"layer{layer}"]
                if layer == 0:
                    expected = 4096 * 1024 + 16
                elif layer == 1:
                    expected = 1024 * 4096 + 1024 * 2
                elif layer == 2:
                    expected = 128 + 3072 * 1024
                else:
                    expected = 1024 * 1024 + 1024 * 2
                _expect_blob_size(weights, name, expected)
                return _decode_blob_matrix(weights, name, spec)

            ffn_expand = matrix(0)
            _copy_parameter(module.ffn_expand.weight, ffn_expand, f"{prefix}.layer0")
            loaded.append(f"{prefix}.layer0 -> vit[{module_index}].ffn_expand.weight")

            ffn_contract = matrix(1)
            _copy_parameter(module.ffn_contract.weight, ffn_contract, f"{prefix}.layer1")
            ffn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer1.layer",
                KNOWN_VIT_BLOB_LAYOUT["layer1"]["cos_skip_offset"],
                (1024,),
            )
            _copy_parameter(module.ffn_cos_skip, ffn_cos_skip, f"{prefix}.layer1.ffn_cos_skip")
            loaded.append(f"{prefix}.layer1 -> vit[{module_index}].ffn_contract/ffn_cos_skip")

            qkv_spec = KNOWN_VIT_BLOB_LAYOUT["layer2"]
            qkv = _decode_blob_matrix(weights, f"{prefix}.layer2.layer", qkv_spec)
            _copy_parameter(module.qkv.weight, qkv, f"{prefix}.layer2")
            header = _decode_blob_f16(weights, f"{prefix}.layer2.layer", 0, (64,))
            metadata.append(
                {
                    "name": f"{prefix}.layer2.header",
                    "elements": 64,
                    "min": float(header.min()),
                    "max": float(header.max()),
                }
            )
            loaded.append(f"{prefix}.layer2 -> vit[{module_index}].qkv.weight")

            scalar = _decode_blob_f16(weights, f"{prefix}.layer3.layer", 0, (1,))
            _expect_blob_size(weights, f"{prefix}.layer3.layer", 2)
            _copy_parameter(module.attention.attn_scale, scalar, f"{prefix}.layer3.attention_scalar")
            loaded.append(f"{prefix}.layer3 -> vit[{module_index}].attention.attn_scale")

            projection = matrix(4)
            _copy_parameter(module.projection.weight, projection, f"{prefix}.layer4")
            attn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer4.layer",
                KNOWN_VIT_BLOB_LAYOUT["layer4"]["cos_skip_offset"],
                (1024,),
            )
            _copy_parameter(module.attn_cos_skip, attn_cos_skip, f"{prefix}.layer4.attn_cos_skip")
            loaded.append(f"{prefix}.layer4 -> vit[{module_index}].projection/attn_cos_skip")
        return {"loaded": loaded, "metadata": metadata, "skipped": []}

    def load_split_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        blocks: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """Load the four proven matrix/attention sections of Split-Swin.

        ``layer4`` contains the proven 1024x512 pointwise FP8 operand.  The
        host graph has a second convolution slot, but the record has no
        independent depthwise bytes and the final-head cubin has no matching
        depthwise multiply; the reference therefore treats that slot as
        identity.  The 32-value attention-scale section is retained in the
        report rather than guessing how it maps onto the 16 heads.
        """

        if self.window_size != 8:
            raise ValueError("embedded Split-Swin blobs are for 8x8 windows")
        if blocks is None:
            blocks = list(range(23, 31)) + list(range(40, 48))
        modules: dict[int, SplitSwinBlock] = {}
        for index, module in enumerate(self.split_enc):
            modules[23 + index] = module
        for index, module in enumerate(self.split_dec):
            modules[40 + index] = module

        loaded: list[str] = []
        metadata: list[dict[str, Any]] = []
        skipped: list[str] = []
        for block_index in blocks:
            if block_index not in modules:
                raise ValueError(f"unsupported Split-Swin block: {block_index}")
            module = modules[block_index]
            prefix = f"block{block_index}"

            layer0_name = f"{prefix}.layer0.layer"
            layer0_spec = KNOWN_SPLIT_BLOB_LAYOUT["layer0"]
            _expect_blob_size(weights, layer0_name, 2 * 512 * 512)
            ffn_a = decode_fp8_matrix(
                weights.fp8(layer0_name), tuple(layer0_spec["fp8_shapes"][0]), byte_offset=0
            )
            ffn_b = decode_fp8_matrix(
                weights.fp8(layer0_name), tuple(layer0_spec["fp8_shapes"][1]), byte_offset=262144
            )
            _copy_parameter(module.ffwd.weight, ffn_a, f"{prefix}.layer0.weight0")
            _copy_parameter(module.ffwd_gate.weight, ffn_b, f"{prefix}.layer0.weight1")
            loaded.append(f"{prefix}.layer0 -> ffwd/ffwd_gate")

            ffn_contract = _decode_blob_matrix(
                weights, f"{prefix}.layer1.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer1"]
            )
            _expect_blob_size(weights, f"{prefix}.layer1.layer", 512 * 512 + 512 * 2)
            _copy_parameter(module.ffwd_proj.weight, ffn_contract, f"{prefix}.layer1")
            ffn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer1.layer",
                KNOWN_SPLIT_BLOB_LAYOUT["layer1"]["cos_skip_offset"],
                (512,),
            )
            _copy_parameter(module.ffn_cos_skip, ffn_cos_skip, f"{prefix}.layer1.ffn_cos_skip")
            loaded.append(f"{prefix}.layer1 -> ffwd_proj/ffn_cos_skip")

            qkv = _decode_blob_matrix(
                weights, f"{prefix}.layer2.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer2"]
            )
            _expect_blob_size(weights, f"{prefix}.layer2.layer", 1536 * 512 + 65536 * 2 + 32 * 2)
            _copy_parameter(module.qkv_attn.qkv.weight, qkv, f"{prefix}.layer2.qkv")
            attn_bias = _decode_blob_f16(
                weights, f"{prefix}.layer2.layer", 786432, (16, 64, 64)
            )
            _copy_parameter(module.qkv_attn.attn_bias, attn_bias, f"{prefix}.layer2.attn_bias")
            # The 64-byte tail is 16 FP32 per-head scales, not 32 FP16
            # values.  Reading it as half produces implausible alternating
            # magnitudes; FP32 gives the expected positive scale vector.
            attn_scale = _decode_blob_f32(
                weights, f"{prefix}.layer2.layer", 917504, (16,)
            )
            _copy_parameter(
                module.qkv_attn.attn_scale,
                attn_scale,
                f"{prefix}.layer2.attn_scale",
            )
            loaded.append(f"{prefix}.layer2 -> qkv/attn_bias/attn_scale")

            projection = _decode_blob_matrix(
                weights, f"{prefix}.layer3.layer", KNOWN_SPLIT_BLOB_LAYOUT["layer3"]
            )
            _expect_blob_size(weights, f"{prefix}.layer3.layer", 512 * 512 + 512 * 2)
            _copy_parameter(module.projection.weight, projection, f"{prefix}.layer3")
            attn_cos_skip = _decode_blob_f16(
                weights,
                f"{prefix}.layer3.layer",
                KNOWN_SPLIT_BLOB_LAYOUT["layer3"]["cos_skip_offset"],
                (512,),
            )
            _copy_parameter(module.attn_cos_skip, attn_cos_skip, f"{prefix}.layer3.attn_cos_skip")
            loaded.append(f"{prefix}.layer3 -> projection/attn_cos_skip")

            if f"{prefix}.layer4.layer" in weights:
                if block_index != 30:
                    skipped.append(f"{prefix}.layer4 unexpected FinalHead")
                    continue
                final_name = f"{prefix}.layer4.layer"
                _expect_blob_size(weights, final_name, 1024 * 512 + 16)
                final_weight = decode_fp8_matrix(
                    weights.fp8(final_name), (1024, 512), byte_offset=0
                ).reshape(1024, 512, 1, 1)
                _copy_parameter(
                    module.final_output.weight,
                    final_weight,
                    f"{prefix}.layer4.weight",
                )
                loaded.append(f"{prefix}.layer4.weight -> FinalHead.pointwise")
                metadata.append(
                    {
                        "name": f"{prefix}.layer4.dw_weight",
                        "applied": True,
                        "value": "identity (no independent bytes in layer4 record)",
                        "reason": "final-head cubin has one 1024x512 HMMA matrix; the host slot has no separate serialized operand",
                    }
                )
        return {"loaded": loaded, "metadata": metadata, "skipped": skipped}

    def load_known_weights(
        self,
        weights: DLSS5WeightMap,
        *,
        swin: bool = True,
        vit: bool = True,
        split: bool = True,
        post: bool = True,
        pre: bool = True,
        dec_input: bool = True,
    ) -> dict[str, Any]:
        """Load all currently proven sections and return an audit report."""

        report: dict[str, Any] = {"loaded": [], "metadata": [], "skipped": []}
        if "block70.layer0.blend_scale" in weights:
            value = weights.float16("block70.layer0.blend_scale")[0].float()
            with torch.no_grad():
                self.blend_scale.copy_(value.to(device=self.blend_scale.device))
            report["loaded"].append("block70.layer0.blend_scale")
        if vit:
            result = self.load_vit_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if swin:
            result = self.load_swin_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if split:
            result = self.load_split_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if post:
            result = self.load_post_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if pre:
            result = self.load_pre_weights(weights)
            for key in report:
                report[key].extend(result[key])
        if dec_input:
            result = self.load_dec_input_weights(weights)
            for key in report:
                report[key].extend(result[key])
        return report

    @classmethod
    def with_weight_map(
        cls,
        root: str | Path,
        *,
        load_known: bool = False,
        load_swin: bool = True,
        load_vit: bool = True,
        load_split: bool = True,
        load_post: bool = True,
        load_pre: bool = True,
        load_dec_input: bool = True,
        **kwargs: Any,
    ) -> tuple["DLSS5Graph", DLSS5WeightMap]:
        """Construct the graph and attach the parsed outer map.

        By default only ``blend_scale`` is loaded.  ``load_known=True`` also
        loads the proven pre/post, Swin, ViT, and Split-Swin sections whose
        byte counts and operands are established by the cubin.  Remaining
        fused sections are returned in the audit report and are never
        silently reshaped.
        """

        root = Path(root)
        weights = DLSS5WeightMap.from_file(root / "WEIGHTS_HT.bin")
        model = cls(**kwargs)
        if "block70.layer0.blend_scale" in weights:
            model.blend_scale.copy_(weights.float16("block70.layer0.blend_scale")[0].float())
        if load_known:
            model.weight_report = model.load_known_weights(
                weights,
                swin=load_swin,
                vit=load_vit,
                split=load_split,
                post=load_post,
                pre=load_pre,
                dec_input=load_dec_input,
            )
        return model, weights


def _self_test() -> None:
    # A small smoke test for the graph and the final ControlMask dataflow.
    # float16 keeps this CPU-only test reasonably small.
    probe = torch.tensor([-4.0, -2.0, 0.0, 1.0, 4.0])
    expected = torch.tensor([0.0, -0.447265625, 0.0, 1.285888671875, 7.15625])
    assert torch.allclose(cct_cubic_silu(probe), expected, atol=1e-5, rtol=0.0)
    model = DLSS5Graph(color_channels=3, history_channels=3, motion_channels=2, output_channels=3).half()
    color = torch.randn(1, 3, 64, 64).half()
    history = torch.randn_like(color)
    motion = torch.zeros(1, 2, 64, 64).half()
    mask = torch.ones(1, 1, 64, 64).half()
    with torch.no_grad():
        y = model(color, history, motion)
        z = model(color, history, motion, mask)
    assert y.shape == color.shape == z.shape
    assert torch.isfinite(y.float()).all()
    assert torch.isfinite(z.float()).all()
    print("DLSS5Graph smoke test passed", tuple(y.shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if args.weights:
        weight_map = DLSS5WeightMap.from_file(args.weights)
        print(json.dumps(weight_map.summary(), indent=2))


if __name__ == "__main__":
    main()
