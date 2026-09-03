"""Export the translated DLSS 5 graph as a loadable PyTorch checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from dlss5_pytorch import DLSS5Graph


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("DLSS5-extracted"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("DLSS5-extracted/dlss5_pytorch_reference_fp16.pt"),
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    model, weight_map = DLSS5Graph.with_weight_map(args.weights, load_known=True)
    dtype = getattr(torch, args.dtype)
    model = model.eval().to(dtype=dtype)
    checkpoint = {
        "format": "dlss5_pytorch_reference_v1",
        "dtype": args.dtype,
        "fp8_emulation": True,
        "model_kwargs": {
            "color_channels": 3,
            "history_channels": 0,
            "motion_channels": 0,
            "output_channels": 3,
            "window_size": 8,
            "vit_layout": "2d",
            "post_output_layout": "column_major_prefix",
        },
        "weight_source": weight_map.source,
        "weight_records": len(weight_map.records),
        "weight_report": model.weight_report,
        "state_dict": model.state_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"output={args.output.resolve()}")
    print(f"bytes={args.output.stat().st_size}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
