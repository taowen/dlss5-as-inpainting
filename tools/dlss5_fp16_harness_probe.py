"""Drive an external DLSS 5 D3D12 harness and record full-precision A/B data.

The repository deliberately does not redistribute NVIDIA or third-party
runtime binaries.  Pass a locally built ``dlss5_eval.exe`` (or another
compatible line-protocol harness) and this script creates a deterministic
RGBA16F/depth/RG16F test contract, runs history and motion-vector cases, and
writes raw outputs plus JSON metrics.  It uses only Python's standard library.

The harness protocol is:

    READY ...
    FRAME <rgba16f> <jitter_x> <jitter_y> <reset>
    WRITE <output>
    QUIT

The supplied harness must accept ``--width``, ``--height``, ``--depth``,
``--motion`` and ``--frames`` options and be started from its runtime folder so
that its local D3D12/ReShade carrier DLLs can be found.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from pathlib import Path


def write_contracts(directory: Path, width: int, height: int) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    color = bytearray()
    checker = bytearray()
    for y in range(height):
        fy = y / max(1, height - 1)
        for x in range(width):
            fx = x / max(1, width - 1)
            color.extend(struct.pack("<eeee", fx, fy, 0.25, 1.0))
            value = 0.85 if ((x // 16 + y // 16) & 1) else 0.05
            checker.extend(struct.pack("<eeee", value, value * 0.9, value * 0.8, 1.0))

    depth = struct.pack("<f", 1.0) * (width * height)
    motion_zero = struct.pack("<ee", 0.0, 0.0) * (width * height)
    motion_shifted = struct.pack("<ee", 4.0, 0.0) * (width * height)

    paths = {
        "color": directory / "color.rgba16f.bin",
        "checker": directory / "checker.rgba16f.bin",
        "depth": directory / "depth.r32f.bin",
        "motion_zero": directory / "motion_zero.rg16f.bin",
        "motion_shifted": directory / "motion_shifted.rg16f.bin",
    }
    paths["color"].write_bytes(color)
    paths["checker"].write_bytes(checker)
    paths["depth"].write_bytes(depth)
    paths["motion_zero"].write_bytes(motion_zero)
    paths["motion_shifted"].write_bytes(motion_shifted)
    return paths


def run_harness(
    harness: Path,
    width: int,
    height: int,
    depth: Path,
    motion: Path,
    frames: list[tuple[Path, int]],
    output: Path,
) -> None:
    process = subprocess.Popen(
        [
            str(harness),
            "--width",
            str(width),
            "--height",
            str(height),
            "--depth",
            str(depth),
            "--motion",
            str(motion),
            "--frames",
            str(len(frames)),
        ],
        cwd=harness.parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdin is not None and process.stdout is not None

    def read_line() -> str:
        line = process.stdout.readline()
        if not line:
            stderr = process.stderr.read() if process.stderr is not None else ""
            raise RuntimeError(f"DLSS 5 harness stopped unexpectedly: {stderr[-4000:]}")
        return line.strip()

    try:
        ready = read_line()
        if not ready.startswith("READY"):
            raise RuntimeError(f"unexpected harness setup response: {ready}")
        for color, reset in frames:
            process.stdin.write(f"FRAME {color} 0 0 {reset}\n")
            process.stdin.flush()
            response = read_line()
            if not response.startswith("FRAME_OK"):
                raise RuntimeError(f"unexpected frame response: {response}")
        process.stdin.write(f"WRITE {output}\n")
        process.stdin.flush()
        response = read_line()
        if not response.startswith("WRITE_OK"):
            raise RuntimeError(f"unexpected write response: {response}")
        process.stdin.write("QUIT\n")
        process.stdin.flush()
        response = read_line()
        if response != "BYE":
            raise RuntimeError(f"unexpected quit response: {response}")
        process.stdin.close()
        process.wait(timeout=60)
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    expected = width * height * 8
    if output.stat().st_size != expected:
        raise RuntimeError(
            f"harness output is {output.stat().st_size} bytes, expected {expected}"
        )


def half_values(path: Path) -> list[float]:
    payload = path.read_bytes()
    if len(payload) % 2:
        raise ValueError(f"odd-sized FP16 payload: {path}")
    return list(struct.unpack(f"<{len(payload) // 2}e", payload))


def compare(left: Path, right: Path) -> dict[str, float | int]:
    a = half_values(left)
    b = half_values(right)
    if len(a) != len(b) or not a:
        raise ValueError(f"cannot compare payloads {left} and {right}")
    deltas = [x - y for x, y in zip(a, b)]
    return {
        "mean_absolute_difference": sum(abs(value) for value in deltas) / len(deltas),
        "rmse": math.sqrt(sum(value * value for value in deltas) / len(deltas)),
        "max_absolute_difference": max(abs(value) for value in deltas),
        "changed_above_1e-3": sum(abs(value) > 1e-3 for value in deltas),
        "elements": len(deltas),
    }


def channel_summary(path: Path) -> dict[str, list[float]]:
    values = half_values(path)
    channels = [values[index::4] for index in range(4)]
    return {
        "mean": [sum(channel) / len(channel) for channel in channels],
        "min": [min(channel) for channel in channels],
        "max": [max(channel) for channel in channels],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runtime_probe_output/fp16_harness"))
    args = parser.parse_args()

    harness = args.harness.resolve()
    if not harness.is_file():
        parser.error(f"harness not found: {harness}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    import tempfile

    with tempfile.TemporaryDirectory(prefix="dlss5-fp16-contract-") as temporary:
        contracts = write_contracts(Path(temporary), args.width, args.height)
        history_output = output / "history_after_color.rgba16f.bin"
        isolated_output = output / "isolated_checker.rgba16f.bin"
        zero_motion_output = output / "motion_zero.rgba16f.bin"
        shifted_motion_output = output / "motion_shifted.rgba16f.bin"

        run_harness(
            harness,
            args.width,
            args.height,
            contracts["depth"],
            contracts["motion_zero"],
            [(contracts["color"], 1), (contracts["checker"], 0)],
            history_output,
        )
        run_harness(
            harness,
            args.width,
            args.height,
            contracts["depth"],
            contracts["motion_zero"],
            [(contracts["checker"], 1)],
            isolated_output,
        )
        run_harness(
            harness,
            args.width,
            args.height,
            contracts["depth"],
            contracts["motion_zero"],
            [(contracts["color"], 1), (contracts["checker"], 0)],
            zero_motion_output,
        )
        run_harness(
            harness,
            args.width,
            args.height,
            contracts["depth"],
            contracts["motion_shifted"],
            [(contracts["color"], 1), (contracts["checker"], 0)],
            shifted_motion_output,
        )

    manifest = {
        "harness": str(harness),
        "width": args.width,
        "height": args.height,
        "history_effect": compare(history_output, isolated_output),
        "motion_effect": compare(zero_motion_output, shifted_motion_output),
        "history_output": str(history_output),
        "isolated_output": str(isolated_output),
        "zero_motion_output": str(zero_motion_output),
        "shifted_motion_output": str(shifted_motion_output),
        "history_output_channels": channel_summary(history_output),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
