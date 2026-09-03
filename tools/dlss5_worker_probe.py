"""Reproducible Feature 18 runtime probe for the native DLSS 5 worker.

The worker is the small ``nvngx.dll --video`` host shipped by the community
DLSS 5 Visual Enhancer.  This probe deliberately uses only the worker's binary
protocol and Python's standard library: no PyTorch, CUDA toolkit, or image
package is needed.  It records output bytes and A/B metrics for the temporal
history, motion-vector, reset, and intensity paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Iterable


VIDEO_MAGIC = 0x34563544
FRAME_MAGIC = 0x314D5246
SETUP_FORMAT = "<12I"
FRAME_FORMAT = "<4Iq"
RESULT_FORMAT = "<5Iq"
VIDEO_FORMAT = "<14I4f"


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"worker stopped after {size - remaining} of {size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def make_pattern(width: int, height: int, variant: int) -> bytes:
    """Make a deterministic, non-degenerate RGBA8 test image."""
    result = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            if variant:
                sx = (x - 12) % width
                sy = (y + 5) % height
            else:
                sx = x
                sy = y
            index = (y * width + x) * 4
            result[index + 0] = (sx * 255 // max(1, width - 1))
            result[index + 1] = (sy * 255 // max(1, height - 1))
            result[index + 2] = (37 + sx * 3 + sy * 5) & 0xFF
            result[index + 3] = 255
    return bytes(result)


def make_motion(width: int, height: int, dx: float, dy: float) -> bytes:
    value = struct.pack("<ee", dx, dy)
    return value * (width * height)


def mean_abs(a: bytes, b: bytes) -> float:
    if len(a) != len(b) or not a:
        return math.nan
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def payload_stats(payload: bytes) -> dict[str, float | int]:
    return {
        "bytes": len(payload),
        "min": min(payload) if payload else 0,
        "max": max(payload) if payload else 0,
        "mean": sum(payload) / len(payload) if payload else 0.0,
        "nonzero": sum(value != 0 for value in payload),
    }


def gpu_snapshot() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,pstate,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return completed.stdout.strip()


def setup_worker(
    worker: Path,
    width: int,
    height: int,
    frame_count: int,
    intensity: float,
    output_dir: Path,
) -> tuple[subprocess.Popen[bytes], dict[str, int]]:
    host_dir = worker.parent
    process = subprocess.Popen(
        [str(worker), "--video"],
        cwd=host_dir,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    header = struct.pack(
        VIDEO_FORMAT,
        VIDEO_MAGIC,
        width,
        height,
        width,
        height,
        0,  # caller warm-up frames; the host may still discard its carrier warm-up
        frame_count,
        5,  # DLAA/native mode, so the model is tested without an upscaler resize
        0,  # driver-selected DLSS model preset
        0,  # profile
        0,  # NR preset
        0,  # NR style
        0,  # automatic mask off
        0,  # UI correction off
        intensity,
        1.0,  # local tone
        1.0,  # local structure
        -1.0,  # native skin structure default
    )
    process.stdin.write(header)
    process.stdin.flush()
    setup_values = struct.unpack(SETUP_FORMAT, read_exact(process.stdout, struct.calcsize(SETUP_FORMAT)))
    names = (
        "magic",
        "ok",
        "ngx_result",
        "render_width",
        "render_height",
        "output_width",
        "output_height",
        "minimum_width",
        "minimum_height",
        "maximum_width",
        "maximum_height",
        "model_preset",
    )
    setup = dict(zip(names, setup_values))
    if setup["magic"] != 0x34505553 or setup["ok"] != 1:
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        process.kill()
        raise RuntimeError(f"worker setup failed: {setup}; stderr={stderr[-4000:]}")
    if (setup["output_width"], setup["output_height"]) != (width, height):
        raise RuntimeError(f"worker negotiated unexpected size: {setup}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return process, setup


def run_case(
    *,
    worker: Path,
    width: int,
    height: int,
    name: str,
    frames: Iterable[tuple[int, bytes, bytes]],
    intensity: float,
    output_dir: Path,
) -> dict[str, object]:
    frame_list = list(frames)
    process, setup = setup_worker(worker, width, height, len(frame_list), intensity, output_dir)
    assert process.stdin is not None and process.stdout is not None
    outputs: list[bytes] = []
    started = time.perf_counter()
    try:
        for index, (reset, rgba, motion) in enumerate(frame_list):
            process.stdin.write(struct.pack(FRAME_FORMAT, FRAME_MAGIC, index, reset, 0, index))
            process.stdin.write(rgba)
            process.stdin.write(motion)
            process.stdin.flush()
            result = struct.unpack(
                RESULT_FORMAT,
                read_exact(process.stdout, struct.calcsize(RESULT_FORMAT)),
            )
            magic, output_index, ok, byte_count, ngx_result, pts = result
            if magic != 0x3154554F or output_index != index or ok != 1 or ngx_result != 1:
                raise RuntimeError(f"invalid frame {index} result: {result}")
            payload = read_exact(process.stdout, byte_count)
            if byte_count != width * height * 4:
                raise RuntimeError(f"unexpected output byte count: {result}")
            outputs.append(payload)
            (output_dir / f"{name}_frame{index}.rgba8.bin").write_bytes(payload)
    finally:
        process.stdin.close()
    process.wait(timeout=90)
    stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
    if process.returncode != 0:
        raise RuntimeError(f"worker case {name} exited {process.returncode}:\n{stderr[-6000:]}")
    (output_dir / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "intensity": intensity,
        "setup": setup,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": [payload_stats(payload) for payload in outputs],
        "output_files": [f"{name}_frame{i}.rgba8.bin" for i in range(len(outputs))],
        "stderr_file": f"{name}.stderr.log",
        "payloads": outputs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True, help="path to the release's host/nvngx.dll")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runtime_probe_output/worker_probe"))
    args = parser.parse_args()
    worker = args.worker.resolve()
    output_dir = args.output.resolve()
    if not worker.is_file():
        parser.error(f"worker not found: {worker}")
    if args.width < 64 or args.height < 64:
        parser.error("both dimensions must be at least 64")

    base_a = make_pattern(args.width, args.height, 0)
    base_b = make_pattern(args.width, args.height, 1)
    zero_mv = make_motion(args.width, args.height, 0.0, 0.0)
    shifted_mv = make_motion(args.width, args.height, 4.0, 0.0)

    cases: dict[str, dict[str, object]] = {}
    cases["temporal_zero_mv"] = run_case(
        worker=worker,
        width=args.width,
        height=args.height,
        name="temporal_zero_mv",
        frames=((1, base_a, zero_mv), (0, base_b, zero_mv)),
        intensity=1.0,
        output_dir=output_dir,
    )
    cases["temporal_shifted_mv"] = run_case(
        worker=worker,
        width=args.width,
        height=args.height,
        name="temporal_shifted_mv",
        frames=((1, base_a, zero_mv), (0, base_b, shifted_mv)),
        intensity=1.0,
        output_dir=output_dir,
    )
    cases["reset_current"] = run_case(
        worker=worker,
        width=args.width,
        height=args.height,
        name="reset_current",
        frames=((1, base_b, zero_mv),),
        intensity=1.0,
        output_dir=output_dir,
    )
    cases["intensity_zero"] = run_case(
        worker=worker,
        width=args.width,
        height=args.height,
        name="intensity_zero",
        frames=((1, base_b, zero_mv),),
        intensity=0.0,
        output_dir=output_dir,
    )

    def payload(case: str, index: int = 0) -> bytes:
        return cases[case]["payloads"][index]  # type: ignore[index]

    metrics = {
        "temporal_history_effect": mean_abs(payload("temporal_zero_mv", 1), payload("reset_current")),
        "motion_vector_effect": mean_abs(payload("temporal_zero_mv", 1), payload("temporal_shifted_mv", 1)),
        "intensity_effect": mean_abs(payload("reset_current"), payload("intensity_zero")),
        "model_vs_input": mean_abs(payload("reset_current"), base_b),
    }
    manifest = {
        "timestamp_local": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu": gpu_snapshot(),
        "worker": str(worker),
        "dimensions": [args.width, args.height],
        "protocol": "DLSS 5 Visual Enhancer v4 video stream",
        "cases": {
            name: {key: value for key, value in case.items() if key != "payloads"}
            for name, case in cases.items()
        },
        "metrics_mean_absolute_byte_difference": metrics,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "gpu": manifest["gpu"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
