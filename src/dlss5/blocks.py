"""Reusable NHWC Swin, Split-Swin, and ViT building blocks."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layouts import STANDARD_SWIN_FFN_DIMS
from .ops import CCTCubicSiLU, _fp8_boundary, cct_cubic_silu

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
