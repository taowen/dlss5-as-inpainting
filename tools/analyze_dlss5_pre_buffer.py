#!/usr/bin/env python3
"""Analyze exact pre-block and inpview snapshots from the driver arena.

This tool reports byte ranges only where the capture proves them.  It keeps
candidate tensor shapes separate from proven byte ranges because the native
``tinlayout`` swizzle is part of the CUBIN ABI and must not be silently
flattened into ordinary NCHW order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonzero_runs(data: bytes, *, value: int = 0) -> list[list[int]]:
    runs: list[list[int]] = []
    start: int | None = None
    for offset, byte in enumerate(data):
        if byte != value and start is None:
            start = offset
        elif byte == value and start is not None:
            runs.append([start, offset])
            start = None
    if start is not None:
        runs.append([start, len(data)])
    return runs


def changed_runs(left: bytes, right: bytes) -> list[list[int]]:
    if len(left) != len(right):
        raise ValueError(f"buffer sizes differ: {len(left)} != {len(right)}")
    runs: list[list[int]] = []
    start: int | None = None
    for offset, (a, b) in enumerate(zip(left, right)):
        if a != b and start is None:
            start = offset
        elif a == b and start is not None:
            runs.append([start, offset])
            start = None
    if start is not None:
        runs.append([start, len(left)])
    return runs


def summarize(path: Path, arena_va: int) -> dict[str, Any]:
    data = path.read_bytes()
    runs = nonzero_runs(data)
    return {
        "path": str(path.resolve()),
        "bytes": len(data),
        "sha256": sha256(path),
        "nonzero_bytes": sum(byte != 0 for byte in data),
        "nonzero_runs": len(runs),
        "first_nonzero": runs[0][0] if runs else None,
        "last_nonzero_exclusive": runs[-1][1] if runs else None,
        "arena_gpu_va": f"0x{arena_va:x}",
    }


def region(name: str, gpu_va: int, start_va: int, end_va: int) -> dict[str, Any]:
    byte_count = end_va - start_va
    return {
        "name": name,
        "gpu_va": f"0x{start_va:x}",
        "arena_offset": f"0x{start_va - gpu_va:x}",
        "bytes": byte_count,
        "end_gpu_va_exclusive": f"0x{end_va:x}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-pre", required=True, type=Path)
    parser.add_argument("--after-inpview", type=Path)
    parser.add_argument("--arena-gpu-va", type=lambda value: int(value, 0), default=0x1BA00000)
    parser.add_argument("--pre-skip-gpu-va", type=lambda value: int(value, 0), default=0x1BA16C00)
    parser.add_argument("--pre-downsample-gpu-va", type=lambda value: int(value, 0), default=0x1BD36C00)
    parser.add_argument("--inpview-output-gpu-va", type=lambda value: int(value, 0), default=0x1BDFEC00)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.after_pre.stat().st_size == 0:
        parser.error("--after-pre is empty")
    arena_end = args.arena_gpu_va + args.after_pre.stat().st_size
    output_bytes = args.inpview_output_gpu_va - args.pre_downsample_gpu_va
    if not args.arena_gpu_va <= args.pre_skip_gpu_va < args.pre_downsample_gpu_va:
        parser.error("pre output addresses must be ordered inside the arena")
    if not args.pre_downsample_gpu_va < args.inpview_output_gpu_va <= arena_end:
        parser.error("inpview output address must follow pre downsample output inside the arena")

    pre = args.after_pre.read_bytes()
    report: dict[str, Any] = {
        "format": "dlss5-pre-buffer-analysis-v1",
        "buffer": summarize(args.after_pre, args.arena_gpu_va),
        "proven_regions": {
            "pre_skip_to_pre_downsample": region(
                "pre_skip_candidate",
                args.arena_gpu_va,
                args.pre_skip_gpu_va,
                args.pre_downsample_gpu_va,
            ),
            "pre_downsample_to_inpview_output": region(
                "pre_downsample_candidate",
                args.arena_gpu_va,
                args.pre_downsample_gpu_va,
                args.inpview_output_gpu_va,
            ),
        },
        "candidate_shapes": [
            {
                "region": "pre_skip_candidate",
                "shape": [320, 320, 32],
                "dtype": "uint8/e4m3",
                "bytes": 320 * 320 * 32,
                "status": "size-factor candidate; native tile swizzle remains unresolved",
            },
            {
                "region": "pre_downsample_candidate",
                "shape": [160, 160, 32],
                "dtype": "uint8/e4m3 candidate",
                "bytes": 160 * 160 * 32,
                "status": "size-factor candidate; native tile swizzle remains unresolved",
            },
        ],
        "nonzero_runs": nonzero_runs(pre),
    }
    if args.after_inpview:
        inpview = args.after_inpview.read_bytes()
        runs = changed_runs(pre, inpview)
        report["after_inpview"] = {
            "buffer": summarize(args.after_inpview, args.arena_gpu_va),
            "changed_bytes_vs_after_pre": sum(end - start for start, end in runs),
            "changed_runs": runs,
            "proven_output_region": region(
                "inpview_output_candidate",
                args.arena_gpu_va,
                args.inpview_output_gpu_va,
                args.inpview_output_gpu_va + output_bytes,
            ),
        }

    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
