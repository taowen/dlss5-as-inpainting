"""Compare a native RGBA16F harness frame with the translated PyTorch graph.

This is the fairer comparison path for the native harness: both sides use the
same tightly packed linear RGBA16F input, instead of comparing a native output
against an independently regenerated 8-bit worker pattern.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph, build_pre_front_sass_candidate


def read_rgba16f(path: Path, width: int, height: int, device: str) -> torch.Tensor:
    payload = path.read_bytes()
    expected = width * height * 4 * 2
    if len(payload) != expected:
        raise ValueError(f"{path} has {len(payload)} bytes; expected {expected}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(
        height, width, 4
    ).to(device=device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="native harness RGBA16F input")
    parser.add_argument("native", type=Path, help="native harness RGBA16F output")
    parser.add_argument("--weights", type=Path, default=Path("DLSS5-extracted"))
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument(
        "--front-source",
        choices=("zero", "sass_candidate"),
        default="zero",
        help="pre-block source; sass_candidate is an opt-in, non-bit-exact hypothesis",
    )
    parser.add_argument("--front-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    source = read_rgba16f(args.input, args.width, args.height, args.device)
    native = read_rgba16f(args.native, args.width, args.height, args.device).float()
    image = source[..., :3].permute(2, 0, 1).unsqueeze(0).to(dtype=dtype)

    model, _ = DLSS5Graph.with_weight_map(args.weights, load_known=True)
    model.enable_fp8_emulation().eval().to(device=args.device, dtype=dtype)
    with torch.inference_mode():
        front = None
        if args.front_source == "sass_candidate":
            front = build_pre_front_sass_candidate(image, feature_scale=args.front_scale)
        predicted = model(rgb=image, pre_front_features=front)[0].permute(1, 2, 0).float()
    clipped = predicted.clamp(0.0, 1.0)

    difference = predicted - native[..., :3]
    clipped_difference = clipped - native[..., :3]
    predicted_flat = predicted.flatten()
    native_flat = native[..., :3].flatten()
    correlation = (
        ((predicted_flat - predicted_flat.mean()) * (native_flat - native_flat.mean())).mean()
        / (predicted_flat.std(unbiased=False) * native_flat.std(unbiased=False)).clamp_min(1e-12)
    )
    clipped_flat = clipped.flatten()
    clipped_correlation = (
        ((clipped_flat - clipped_flat.mean()) * (native_flat - native_flat.mean())).mean()
        / (clipped_flat.std(unbiased=False) * native_flat.std(unbiased=False)).clamp_min(1e-12)
    )
    report = {
        "input": str(args.input.resolve()),
        "native": str(args.native.resolve()),
        "weights": str(args.weights.resolve()),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "dtype": args.dtype,
        "size": [args.width, args.height],
        "front_source": args.front_source,
        "front_scale": args.front_scale,
        "native_rgb_mean": float(native[..., :3].mean().item()),
        "pytorch_rgb_mean": float(predicted.mean().item()),
        "pytorch_clipped_rgb_mean": float(clipped.mean().item()),
        "pytorch_raw_min": float(predicted.min().item()),
        "pytorch_raw_max": float(predicted.max().item()),
        "pytorch_finite": int(torch.isfinite(predicted).sum().item()),
        "rgb_elements": predicted.numel(),
        "rgb_correlation": float(correlation.item()),
        "rgb_mae": float(difference.abs().mean().item()),
        "rgb_rmse": float(difference.square().mean().sqrt().item()),
        "rgb_max_abs": float(difference.abs().max().item()),
        "clipped_rgb_correlation": float(clipped_correlation.item()),
        "clipped_rgb_mae": float(clipped_difference.abs().mean().item()),
        "clipped_rgb_rmse": float(clipped_difference.square().mean().sqrt().item()),
        "clipped_rgb_max_abs": float(clipped_difference.abs().max().item()),
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
