"""Independently verify the PyTorch-facing DLSS5 carrier byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))

from dlss5_bit_exact import DLSS5BitExactModel  # noqa: E402
from dlss5_fp16_harness_probe import run_harness, write_contracts  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True, help="prepared runtime directory")
    parser.add_argument("--harness", type=Path, help="native harness; defaults to runtime/dlss5_eval.exe")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", type=Path, help="optional JSON report")
    args = parser.parse_args()

    harness = (args.harness or args.runtime / "dlss5_eval.exe").resolve()
    with tempfile.TemporaryDirectory(prefix="dlss5-bit-exact-verify-") as temporary:
        contracts = write_contracts(Path(temporary), args.width, args.height)
        native_output = Path(temporary) / "native-golden.rgba16f.bin"
        run_harness(
            harness,
            args.width,
            args.height,
            contracts["depth"],
            contracts["motion_zero"],
            [(contracts["color"], 1), (contracts["checker"], 0)],
            native_output,
        )
        with DLSS5BitExactModel(
            harness,
            width=args.width,
            height=args.height,
            depth=contracts["depth"],
            motion=contracts["motion_zero"],
        ) as carrier:
            color = torch.frombuffer(
                bytearray(contracts["color"].read_bytes()), dtype=torch.float16
            ).reshape(args.height, args.width, 4)[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
            checker = torch.frombuffer(
                bytearray(contracts["checker"].read_bytes()), dtype=torch.float16
            ).reshape(args.height, args.width, 4)[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()
            carrier(color)
            predicted = carrier(checker)
        native = torch.frombuffer(
            bytearray(native_output.read_bytes()), dtype=torch.float16
        ).reshape(args.height, args.width, 4).permute(2, 0, 1).unsqueeze(0).contiguous()
        predicted_bytes = predicted.cpu().contiguous().permute(0, 2, 3, 1).numpy().tobytes()
        native_bytes = native.contiguous().permute(0, 2, 3, 1).numpy().tobytes()

    report = {
        "format": "dlss5_bit_exact_carrier_verification_v1",
        "harness": str(harness),
        "runtime": str(args.runtime.resolve()),
        "size": [args.width, args.height],
        "dtype": str(predicted.dtype),
        "shape": list(predicted.shape),
        "native_sha256": hashlib.sha256(native_bytes).hexdigest(),
        "pytorch_carrier_sha256": hashlib.sha256(predicted_bytes).hexdigest(),
        "bytes": len(native_bytes),
        "byte_equal": native_bytes == predicted_bytes,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["byte_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
