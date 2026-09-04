"""Backend-neutral numeric, tensor-layout, and front-end operations."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    # CUDA builds of PyTorch expose a native conversion that is closest to
    # the hardware path.  Some non-NVIDIA backends expose the float8 dtype but
    # do not implement the conversion kernel; use the backend-neutral decoder
    # below in that case rather than making the translated model CUDA-only.
    if values.device.type == "cuda":
        try:
            return finite.to(torch.float8_e4m3fn).to(values.dtype)
        except (AttributeError, NotImplementedError, RuntimeError):
            pass
    return _quantize_s_e4m3_satfinite_portable(finite)


def _quantize_s_e4m3_satfinite_portable(values: Tensor) -> Tensor:
    """Pure tensor E4M3FN round-trip used when a backend lacks float8 casts."""

    source_dtype = values.dtype
    work = values.float()
    magnitude = work.abs()
    min_normal = 2.0**-6
    subnormal_step = 2.0**-9

    # E4M3FN uses exponent bits 1..15 for finite normal values.  The last
    # exponent has mantissas 0..6; mantissa 7 is the NaN code.  Thus the finite
    # maximum is 2**8 * (1 + 6/8) = 448.
    subnormal_quantum = torch.round(magnitude / subnormal_step)
    promoted = (magnitude < min_normal) & (subnormal_quantum >= 8.0)
    subnormal_value = subnormal_quantum.clamp(0.0, 7.0) * subnormal_step

    normal_magnitude = magnitude.clamp(min_normal, 448.0)
    exponent = torch.floor(torch.log2(normal_magnitude)).clamp(-6.0, 8.0)
    base = torch.pow(torch.tensor(2.0, device=work.device, dtype=work.dtype), exponent)
    mantissa = torch.round((normal_magnitude / base - 1.0) * 8.0)
    carry = mantissa >= 8.0
    exponent = torch.where(carry, (exponent + 1.0).clamp(max=8.0), exponent)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    # The exponent-8 row has no finite mantissa-7 encoding.
    mantissa = torch.where(exponent >= 8.0, mantissa.clamp(max=6.0), mantissa.clamp(0.0, 7.0))
    normal_value = torch.pow(
        torch.tensor(2.0, device=work.device, dtype=work.dtype), exponent
    ) * (1.0 + mantissa / 8.0)

    quantized = torch.where(magnitude < min_normal, subnormal_value, normal_value)
    quantized = torch.where(promoted, torch.full_like(quantized, min_normal), quantized)
    quantized = torch.where(magnitude == 0.0, torch.zeros_like(quantized), quantized)
    return torch.copysign(quantized, work).to(source_dtype)


def _fp8_boundary(module: nn.Module, values: Tensor) -> Tensor:
    if getattr(module, "_emulate_fp8", False):
        return quantize_s_e4m3_satfinite(values)
    return values

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


def assemble_pre_front_feature_lanes(
    generated_lanes: Tensor,
    texture_lanes: Tensor,
    constant: Optional[Tensor] = None,
) -> Tensor:
    """Pack the externally reconstructed block0 front lanes into ``K=15``.

    The public reverse-engineering evidence describes the front producer as
    three generated FP16 lanes, one constant ``1.0`` lane, and four sampled
    texture lanes. A CUDA half2 lane carries two scalar FP16 values, so this
    helper exposes that candidate contract explicitly:

    ``generated_lanes: [N, 3, 2, H, W]`` -> 6 channels
    ``constant:       [N, 1, H, W]``     -> 1 channel
    ``texture_lanes:  [N, 4, 2, H, W]`` -> 8 channels

    The result is ``[N, 15, H, W]`` and is accepted by ``pre_front_features``.
    This is only a packing boundary: the hash/coordinate producer and the
    exact lane ordering are still supplied by the caller until SASS parity is
    established.
    """

    if generated_lanes.ndim != 5 or generated_lanes.shape[1:3] != (3, 2):
        raise ValueError("generated_lanes must have shape [N, 3, 2, H, W]")
    if texture_lanes.ndim != 5 or texture_lanes.shape[1:3] != (4, 2):
        raise ValueError("texture_lanes must have shape [N, 4, 2, H, W]")
    if generated_lanes.shape[0] != texture_lanes.shape[0] or generated_lanes.shape[-2:] != texture_lanes.shape[-2:]:
        raise ValueError("generated_lanes and texture_lanes must share batch/spatial shape")
    if constant is None:
        constant = torch.ones(
            generated_lanes.shape[0], 1, *generated_lanes.shape[-2:],
            device=generated_lanes.device,
            dtype=generated_lanes.dtype,
        )
    if constant.ndim != 4 or constant.shape[1] != 1 or constant.shape[0] != generated_lanes.shape[0]:
        raise ValueError("constant must have shape [N, 1, H, W]")
    if constant.shape[-2:] != generated_lanes.shape[-2:]:
        raise ValueError("constant must share the generated lane spatial shape")
    return torch.cat(
        (generated_lanes.flatten(1, 2), constant, texture_lanes.flatten(1, 2)),
        dim=1,
    )


def _front_hash_u32(x: Tensor, y: Tensor, seed: int | Tensor) -> Tensor:
    """Return a deterministic uint32-like hash as an int64 tensor.

    The pre CUBIN uses a tile-coordinate hash followed by four uniform-like
    values and ``LG2 -> *ln(2) -> -2 -> SQRT``.  The exact scalar seed and
    lane permutation are not exposed by the host ABI, so this helper mirrors
    the observed integer/hash shape without claiming to reproduce the hidden
    constant buffer byte-for-byte.
    """

    mask = 0xFFFFFFFF
    seed_value = torch.as_tensor(seed, device=x.device, dtype=torch.int64)
    value = (x * 0x9E3779B1 + y * 0x85EBCA6B + seed_value) & mask
    value = (value ^ (value >> 16)) & mask
    value = (value * 0x7FEB352D) & mask
    value = (value ^ (value >> 15)) & mask
    value = (value * 0x846CA68B) & mask
    return (value ^ (value >> 16)) & mask


def _front_uniform(x: Tensor, y: Tensor, seed: int | Tensor) -> Tensor:
    # Avoid exactly zero: the native path's logarithm input is formed from a
    # positive integer plus one, so a zero uniform is not a valid state.
    value = _front_hash_u32(x, y, seed).to(torch.float32)
    return (value + 1.0) / 4294967297.0


def _front_gaussian_lanes(x: Tensor, y: Tensor, seed: int | Tensor) -> Tensor:
    """Generate three deterministic half2 lanes using the observed math path."""

    values: list[Tensor] = []
    # Six Box-Muller pairs produce the six scalar values carried by three
    # generated half2 lanes.  Keeping this in FP32 until the final cast mirrors
    # the SASS sequence's F32 transcendental work followed by F2F.F16.
    for pair in range(3):
        u = _front_uniform(x, y, seed + pair * 2)
        v = _front_uniform(x, y, seed + pair * 2 + 1)
        radius = torch.sqrt((-2.0 * torch.log(u)).clamp_min(0.0))
        angle = 2.0 * math.pi * v
        values.extend((radius * torch.cos(angle), radius * torch.sin(angle)))
    return torch.stack(values, dim=1).reshape(x.shape[0], 3, 2, *x.shape[-2:])


def _sample_front_texture(texture: Tensor, dx: float, dy: float) -> Tensor:
    """Sample a four-component texture at a pixel offset, like a TEX 2D read."""

    _, _, height, width = texture.shape
    device = texture.device
    dtype = torch.float32
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    # align_corners=False maps pixel centers to (2*(p+0.5)/size - 1).
    grid = torch.stack(
        (
            2.0 * (xx + 0.5 + dx) / width - 1.0,
            2.0 * (yy + 0.5 + dy) / height - 1.0,
        ),
        dim=-1,
    ).unsqueeze(0).expand(texture.shape[0], -1, -1, -1)
    return F.grid_sample(
        texture.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    )


def build_pre_front_sass_candidate(
    rgb: Tensor, *, seed: int = 0x44D9, feature_scale: float = 1.0
) -> Tensor:
    """Build an executable SASS-informed RGB front-end candidate.

    The recovered pre path has a deterministic coordinate/hash prefix, a
    Box--Muller-like generated half2 path, and four sampled texture half2
    lanes before the two serialized FP16 front tiles.  This implementation
    uses two RGBA bilinear reads (each split into RG and BA half2 lanes), plus
    the six generated scalars and a constant one lane, yielding the proven
    ``K=15`` input shape.

    The hash seed, texture offsets, half2 lane order, and feature scale are
    still unresolved in the proprietary ABI.  The function is therefore
    intentionally named ``candidate`` and is opt-in; it is useful for
    experiments and dynamic comparison, while the default graph keeps the
    conservative zero RGB fallback.
    """

    if rgb.ndim != 4 or rgb.shape[1] not in {3, 4}:
        raise ValueError("rgb must be NCHW with 3 or 4 channels")
    if not rgb.is_floating_point():
        raise TypeError("rgb must be a floating-point tensor")
    if not math.isfinite(feature_scale) or feature_scale < 0.0:
        raise ValueError("feature_scale must be finite and non-negative")
    n, _, height, width = rgb.shape
    if min(n, height, width) < 1:
        raise ValueError("rgb must have positive batch and spatial dimensions")

    texture = rgb[:, :4]
    if texture.shape[1] == 3:
        texture = torch.cat(
            (texture, torch.ones(n, 1, height, width, device=rgb.device, dtype=rgb.dtype)),
            dim=1,
        )
    # The analyzed kernel performs multiple 0x7 texture reads around a
    # transformed coordinate.  The center and +x samples are the smallest
    # explicit candidate that preserves all four returned components.
    sample0 = _sample_front_texture(texture, 0.0, 0.0)
    sample1 = _sample_front_texture(texture, 1.0, 0.0)
    texture_lanes = torch.stack(
        (
            sample0[:, 0:2],
            sample0[:, 2:4],
            sample1[:, 0:2],
            sample1[:, 2:4],
        ),
        dim=1,
    )

    yy, xx = torch.meshgrid(
        torch.arange(height, device=rgb.device, dtype=torch.int64),
        torch.arange(width, device=rgb.device, dtype=torch.int64),
        indexing="ij",
    )
    x = xx.unsqueeze(0).expand(n, -1, -1)
    y = yy.unsqueeze(0).expand(n, -1, -1)
    batch = torch.arange(n, device=rgb.device, dtype=torch.int64).view(n, 1, 1)
    generated = _front_gaussian_lanes(x, y, seed + batch * 0x9E3779B9)
    generated = (generated * feature_scale).to(dtype=rgb.dtype)
    texture_lanes = (texture_lanes * feature_scale).to(dtype=rgb.dtype)
    return assemble_pre_front_feature_lanes(generated, texture_lanes)

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
