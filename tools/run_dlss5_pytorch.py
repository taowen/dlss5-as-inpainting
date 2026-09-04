"""Load an exported DLSS 5 PyTorch checkpoint and execute one inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph
from dlss5_portable import PORTABLE_FORMAT, load_portable_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    portable = checkpoint.get("format") == PORTABLE_FORMAT
    if portable:
        model = load_portable_checkpoint(args.checkpoint, device=args.device)
        dtype = next(model.parameters()).dtype
    else:
        if checkpoint.get("format") != "dlss5_pytorch_reference_v1":
            raise ValueError("unsupported DLSS 5 checkpoint format")
        dtype = getattr(torch, checkpoint["dtype"])
        model = DLSS5Graph(**checkpoint["model_kwargs"]).to(dtype=dtype)
        missing, unexpected = model.load_state_dict(checkpoint["state_dict"], strict=False)
        # These audit-only buffers were added after the first reference checkpoint
        # was exported. They default to zero, which is exactly the historical
        # unresolved-front behavior; every learned parameter must still match.
        allowed_missing = {"pre_front_weight0_f16", "pre_front_weight1_f16"}
        unexpected_set = set(unexpected)
        unexpected_missing = set(missing) - allowed_missing
        if unexpected_set or unexpected_missing:
            raise RuntimeError(
                "checkpoint/model state mismatch: "
                f"missing={sorted(unexpected_missing)}, unexpected={sorted(unexpected_set)}"
            )
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
        output = model(image) if portable else model(rgb=image)
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
