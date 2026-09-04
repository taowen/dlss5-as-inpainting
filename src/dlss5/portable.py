"""Portable PyTorch inference model for the recovered DLSS 5 contract.

The native DLSS 5 front-end is a private CUDA texture/SASS producer.  It is
not safe to feed a guessed feature layout into the recovered 71-block graph:
the body can amplify a wrong front tensor by several orders of magnitude.  A
portable model therefore has an explicit, image-preserving fallback and a
small residual head that can be distilled from native input/output pairs.

This module deliberately uses ordinary PyTorch operators only.  It does not
use CUDA, FP8, custom extensions, or the native carrier, so a checkpoint made
with it can run on CPU, CUDA, ROCm, Intel, or any other backend supported by
the installed PyTorch build.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


PORTABLE_FORMAT = "dlss5_pytorch_portable_v1"


def _validate_image(name: str, value: Tensor, channels: int | None = None) -> None:
    if value.ndim != 4:
        raise ValueError(f"{name} must be NCHW, got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.shape[0] < 1 or value.shape[2] < 1 or value.shape[3] < 1:
        raise ValueError(f"{name} must have positive batch and spatial dimensions")
    if channels is not None and value.shape[1] != channels:
        raise ValueError(f"{name} must have {channels} channels, got {value.shape[1]}")


def _fit_condition(value: Optional[Tensor], *, channels: int, reference: Tensor) -> Tensor:
    """Return a condition in the reference batch/spatial/device/dtype contract."""

    if value is None:
        return torch.zeros(
            reference.shape[0],
            channels,
            reference.shape[2],
            reference.shape[3],
            device=reference.device,
            dtype=reference.dtype,
        )
    _validate_image("condition", value)
    if value.shape[0] != reference.shape[0]:
        raise ValueError("condition and color batch sizes must match")
    value = value.to(device=reference.device, dtype=reference.dtype)
    if value.shape[-2:] != reference.shape[-2:]:
        value = F.interpolate(value, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    if value.shape[1] > channels:
        return value[:, :channels]
    if value.shape[1] < channels:
        return F.pad(value, (0, 0, 0, 0, 0, channels - value.shape[1]))
    return value


class DLSS5PortableModel(nn.Module):
    """A backend-independent image-to-image approximation.

    ``color`` is RGB in NCHW layout and normally lives in ``[0, 1]``.  Depth,
    history, motion, and a control mask are optional.  Missing conditions are
    zero-filled, so a single RGB image is a valid call.  The last layer is
    initialized to zero, making a freshly exported model exactly preserve its
    input while remaining trainable for native distillation.

    The model is intentionally not advertised as bit exact.  Its guarantee is
    stronger and more useful for a portable fallback: finite RGB output,
    unchanged spatial dimensions, and no hallucinated content before the
    optional native-calibrated residual is loaded.
    """

    def __init__(
        self,
        *,
        color_channels: int = 3,
        depth_channels: int = 1,
        history_channels: int = 3,
        motion_channels: int = 2,
        hidden_channels: int = 32,
        residual_scale: float = 1.0,
        clamp_output: bool = True,
    ) -> None:
        super().__init__()
        if color_channels < 3:
            raise ValueError("color_channels must be at least 3")
        if min(depth_channels, history_channels, motion_channels) < 0 or hidden_channels < 1:
            raise ValueError("condition channel counts must be non-negative and hidden_channels must be positive")
        if residual_scale < 0.0 or not torch.isfinite(torch.tensor(residual_scale)):
            raise ValueError("residual_scale must be finite and non-negative")

        self.color_channels = color_channels
        self.depth_channels = depth_channels
        self.history_channels = history_channels
        self.motion_channels = motion_channels
        self.hidden_channels = hidden_channels
        self.clamp_output = bool(clamp_output)
        self.condition_channels = depth_channels + history_channels + motion_channels
        self.input_channels = color_channels + self.condition_channels
        self.output_channels = 3

        self.in_conv = nn.Conv2d(self.input_channels, hidden_channels, 3, padding=1)
        self.mid_conv = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1)
        self.out_conv = nn.Conv2d(hidden_channels, 3, 3, padding=1)
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Kaiming initialization gives the distillation head useful gradients;
        # zeroing only the output makes the initial model an identity map.
        nn.init.kaiming_normal_(self.in_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.in_conv.bias)
        nn.init.kaiming_normal_(self.mid_conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(self.mid_conv.bias)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        color: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
        history: Optional[Tensor] = None,
        motion: Optional[Tensor] = None,
        control_mask: Optional[Tensor] = None,
        *,
        rgb: Optional[Tensor] = None,
    ) -> Tensor:
        if color is not None and rgb is not None:
            raise ValueError("color and rgb are mutually exclusive")
        if color is None:
            color = rgb
        if color is None:
            raise ValueError("color or rgb must be supplied")
        _validate_image("color", color)
        if color.shape[1] < 3:
            raise ValueError("color must contain at least RGB channels")
        parameter = self.in_conv.weight
        color = color.to(device=parameter.device, dtype=parameter.dtype)
        base = color[:, :3]
        model_color = color[:, : self.color_channels]
        if model_color.shape[1] < self.color_channels:
            model_color = F.pad(model_color, (0, 0, 0, 0, 0, self.color_channels - model_color.shape[1]))
        conditions = torch.cat(
            (
                _fit_condition(depth, channels=self.depth_channels, reference=base),
                _fit_condition(history, channels=self.history_channels, reference=base),
                _fit_condition(motion, channels=self.motion_channels, reference=base),
            ),
            dim=1,
        )
        hidden = F.silu(self.in_conv(torch.cat((model_color, conditions), dim=1)))
        hidden = F.silu(self.mid_conv(hidden))
        residual = self.out_conv(hidden) * self.residual_scale.to(dtype=hidden.dtype, device=hidden.device)
        output = base + residual
        if control_mask is not None:
            mask = _fit_condition(control_mask, channels=1, reference=base).clamp(0.0, 1.0)
            output = base + mask * (output - base)
        if self.clamp_output:
            output = output.clamp(0.0, 1.0)
        return output


def load_portable_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> DLSS5PortableModel:
    """Load and validate a portable checkpoint produced by the exporter."""

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != PORTABLE_FORMAT:
        raise ValueError(f"unsupported portable checkpoint format: {checkpoint.get('format')!r}")
    model = DLSS5PortableModel(**checkpoint["model_kwargs"])
    missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"portable checkpoint/model mismatch: missing={missing}, unexpected={unexpected}")
    return model.eval().to(device)
