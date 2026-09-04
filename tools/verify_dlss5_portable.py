"""Verify portable DLSS 5 inference on CPU or another PyTorch backend."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dlss5_portable import load_portable_checkpoint


def read_rgba16f(path: Path, width: int, height: int, device: str) -> torch.Tensor:
    payload = path.read_bytes()
    expected = width * height * 8
    if len(payload) != expected:
        raise ValueError(f"{path} has {len(payload)} bytes; expected {expected}")
    return torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(height, width, 4).to(
        device=device, dtype=torch.float32
    )


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    difference = prediction - target
    pred_flat = prediction.flatten()
    target_flat = target.flatten()
    correlation = (
        ((pred_flat - pred_flat.mean()) * (target_flat - target_flat.mean())).mean()
        / (pred_flat.std(unbiased=False) * target_flat.std(unbiased=False)).clamp_min(1e-12)
    )
    return {
        "mae": float(difference.abs().mean().cpu()),
        "rmse": float(difference.square().mean().sqrt().cpu()),
        "max_abs": float(difference.abs().max().cpu()),
        "correlation": float(correlation.cpu()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--pair", action="append", help="RGBA16F INPUT=TARGET; repeatable")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = load_portable_checkpoint(args.checkpoint, device=args.device)
    pairs: list[tuple[Path, Path | None]] = []
    for spec in args.pair or []:
        if "=" not in spec:
            parser.error(f"--pair must be INPUT=TARGET, got {spec!r}")
        left, right = spec.split("=", 1)
        pairs.append((Path(left), Path(right)))
    if not pairs:
        torch.manual_seed(0)
        source = torch.rand(1, 3, args.height, args.width, device=args.device)
        with torch.inference_mode():
            output = model(source)
        report: dict[str, object] = {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "finite": bool(torch.isfinite(output).all()),
            "range": [float(output.min().cpu()), float(output.max().cpu())],
            "identity_max_abs": float((output - source).abs().max().cpu()),
        }
    else:
        reports: list[dict[str, object]] = []
        for source_path, target_path in pairs:
            source = read_rgba16f(source_path, args.width, args.height, args.device)
            image = source[..., :3].permute(2, 0, 1).unsqueeze(0)
            target = None
            if target_path is not None:
                target = read_rgba16f(target_path, args.width, args.height, args.device)[..., :3]
            with torch.inference_mode():
                output = model(image)
            item: dict[str, object] = {
                "input": str(source_path.resolve()),
                "target": str(target_path.resolve()) if target_path is not None else None,
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "finite": bool(torch.isfinite(output).all()),
                "range": [float(output.min().cpu()), float(output.max().cpu())],
            }
            if target is not None:
                item.update(metrics(output[0].permute(1, 2, 0), target))
            reports.append(item)
        report = {
            "device": args.device,
            "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
            "pairs": reports,
            "mean_mae": sum(float(item["mae"]) for item in reports if "mae" in item) / max(1, len(reports)),
        }
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
