"""Run a backend-independent DLSS 5 portable checkpoint on an image tensor."""

from __future__ import annotations

import argparse
import time
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("input", type=Path, help="tightly packed RGBA16F input")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, help="optional RGBA16F output path")
    args = parser.parse_args()

    model = load_portable_checkpoint(args.checkpoint, device=args.device)
    source = read_rgba16f(args.input, args.width, args.height, args.device)
    image = source[..., :3].permute(2, 0, 1).unsqueeze(0)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(image)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    if args.output:
        rgba = torch.cat((output[0], torch.ones(1, args.height, args.width, device=output.device)), dim=0)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        rgba.permute(1, 2, 0).to(dtype=torch.float16, device="cpu").contiguous().numpy().tofile(args.output)
    print(f"shape={tuple(output.shape)}")
    print(f"dtype={output.dtype}")
    print(f"finite={bool(torch.isfinite(output).all())}")
    print(f"range={float(output.min())},{float(output.max())}")
    print(f"elapsed_seconds={time.perf_counter() - started}")
    if args.output:
        print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
