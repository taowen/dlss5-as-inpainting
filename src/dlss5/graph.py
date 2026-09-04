"""Complete semantic 71-block DLSS5 graph and serialized weight loaders."""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .loaders import DLSS5WeightLoader
from .blocks import (
    DecInputUpsample, PreSwinDownsample, SplitSwinBlock, SwinBlock,
    SwinDownBlock, SwinUpBlock, ViTBlock,
)
from .ops import (
    assemble_pre_front_feature_lanes, decode_hmma_16816_f16_tile, _fp8_boundary,
)

class DLSS5Graph(DLSS5WeightLoader, nn.Module):
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
        pre_front_generated_lanes: Optional[Tensor] = None,
        pre_front_texture_lanes: Optional[Tensor] = None,
        pre_front_constant: Optional[Tensor] = None,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if pre_front_generated_lanes is not None or pre_front_texture_lanes is not None:
            if pre_front_features is not None:
                raise ValueError("packed and component pre-front inputs are mutually exclusive")
            if pre_front_generated_lanes is None or pre_front_texture_lanes is None:
                raise ValueError("generated and texture pre-front lane tensors are both required")
            pre_front_features = assemble_pre_front_feature_lanes(
                pre_front_generated_lanes,
                pre_front_texture_lanes,
                pre_front_constant,
            )
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
