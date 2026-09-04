"""Scan the 15 logical pre-front K lanes with a one-hot constant feature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph


def summary(value: torch.Tensor) -> dict[str, float | int]:
    value = value.float()
    return {
        "finite": int(torch.isfinite(value).sum().item()),
        "elements": value.numel(),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "abs_mean": float(value.abs().mean().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("DLSS5-extracted"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("runtime_probe_output/pre_front_columns.json"))
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    image = torch.linspace(
        0.0,
        1.0,
        3 * args.size * args.size,
        device=args.device,
        dtype=torch.float32,
    ).to(dtype=dtype).reshape(1, 3, args.size, args.size)
    model, _ = DLSS5Graph.with_weight_map(args.weights, load_known=True)
    model.enable_fp8_emulation().eval().to(device=args.device, dtype=dtype)

    results: list[dict[str, object]] = []
    with torch.inference_mode():
        for column in range(15):
            features = torch.zeros(
                1, 15, args.size, args.size, device=args.device, dtype=dtype
            )
            features[:, column] = 1.0
            output = model(rgb=image, pre_front_features=features)[0]
            results.append({"column": column, "output": summary(output)})

    report = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "dtype": args.dtype,
        "size": [args.size, args.size],
        "columns": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
