"""Load an exported DLSS 5 PyTorch checkpoint and execute one inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "dlss5_pytorch_reference_v1":
        raise ValueError("unsupported DLSS 5 checkpoint format")
    dtype = getattr(torch, checkpoint["dtype"])
    model = DLSS5Graph(**checkpoint["model_kwargs"]).to(dtype=dtype)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if checkpoint.get("fp8_emulation"):
        model.enable_fp8_emulation()
    model = model.eval().to(args.device)

    image = torch.linspace(
        0.0,
        1.0,
        3 * args.size * args.size,
        dtype=torch.float32,
        device=args.device,
    ).to(dtype=dtype).reshape(1, 3, args.size, args.size)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(rgb=image)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    print(f"shape={tuple(output.shape)}")
    print(f"dtype={output.dtype}")
    print(f"finite={bool(torch.isfinite(output).all())}")
    print(f"range={float(output.min())},{float(output.max())}")
    print(f"elapsed_seconds={time.perf_counter() - started}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
