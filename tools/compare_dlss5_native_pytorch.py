"""Compare a native worker golden frame with the executable PyTorch graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph


def worker_pattern(width: int, height: int, variant: int, device: str) -> torch.Tensor:
    """Reproduce dlss5_worker_probe.make_pattern without an image dependency."""

    y = torch.arange(height, device=device, dtype=torch.int64).view(height, 1)
    x = torch.arange(width, device=device, dtype=torch.int64).view(1, width)
    if variant:
        sx = (x - 12) % width
        sy = (y + 5) % height
    else:
        sx, sy = x, y
    red = sx.expand(height, width) * 255 // max(1, width - 1)
    green = sy.expand(height, width) * 255 // max(1, height - 1)
    blue = (37 + sx + sy + sx * 2 + sy * 4) & 0xFF
    return torch.stack((red, green, blue), dim=0).unsqueeze(0).float().div_(255.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path, help="native worker RGBA8 frame")
    parser.add_argument("--weights", type=Path, default=Path("DLSS5-extracted"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--variant", type=int, choices=(0, 1), default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--post-output-layout", choices=("raw", "tensor_core_candidate"), default="raw")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = args.native.read_bytes()
    expected_bytes = args.width * args.height * 4
    if len(payload) != expected_bytes:
        parser.error(f"native frame has {len(payload)} bytes; expected {expected_bytes}")

    dtype = getattr(torch, args.dtype)
    image = worker_pattern(args.width, args.height, args.variant, args.device).to(dtype=dtype)
    native = torch.frombuffer(bytearray(payload), dtype=torch.uint8).reshape(
        args.height, args.width, 4
    )[..., :3].to(device=args.device, dtype=torch.float32).div_(255.0)

    model, _ = DLSS5Graph.with_weight_map(
        args.weights,
        load_known=True,
        post_output_layout=args.post_output_layout,
    )
    model.enable_fp8_emulation().eval().to(device=args.device, dtype=dtype)
    with torch.inference_mode():
        predicted = model(rgb=image)[0].permute(1, 2, 0).float().clamp_(0.0, 1.0)
    difference = predicted - native
    predicted_flat = predicted.flatten()
    native_flat = native.flatten()
    correlation = (
        ((predicted_flat - predicted_flat.mean()) * (native_flat - native_flat.mean())).mean()
        / (predicted_flat.std(unbiased=False) * native_flat.std(unbiased=False)).clamp_min(1e-12)
    )
    report = {
        "native": str(args.native.resolve()),
        "weights": str(args.weights.resolve()),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "dtype": args.dtype,
        "size": [args.width, args.height],
        "variant": args.variant,
        "post_output_layout": args.post_output_layout,
        "native_rgb_mean": float(native.mean().item()),
        "pytorch_rgb_mean": float(predicted.mean().item()),
        "rgb_correlation": float(correlation.item()),
        "rgb_mae": float(difference.abs().mean().item()),
        "rgb_rmse": float(difference.square().mean().sqrt().item()),
        "rgb_max_abs": float(difference.abs().max().item()),
    }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
