#!/usr/bin/env python3
"""Validate the four existing pre-block downsample store slots.

Each input is an isolated CUBIN run in which one legal ``STG.E`` source was
changed to ``RZ``.  Comparing it with the clean raw arena identifies the rows
written by that store without adding a new memory instruction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def changed_word_indices(baseline: bytes, patched: bytes, offset: int, bytes_count: int) -> list[int]:
    left = baseline[offset : offset + bytes_count]
    right = patched[offset : offset + bytes_count]
    if len(left) != bytes_count or len(right) != bytes_count:
        raise ValueError("all buffers must contain the requested range")
    return [
        index // 4
        for index in range(0, bytes_count, 4)
        if left[index : index + 4] != right[index : index + 4]
    ]


def runs(values: list[int]) -> list[list[int]]:
    if not values:
        return []
    output: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            output.append([start, previous + 1])
            start = value
        previous = value
    output.append([start, previous + 1])
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--slot", required=True, action="append", type=Path, metavar="BUFFER")
    parser.add_argument("--arena-offset", type=lambda value: int(value, 0), default=0x336C00)
    parser.add_argument("--bytes", dest="bytes_count", type=lambda value: int(value, 0), default=0xC8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.slot) != 4:
        parser.error("provide exactly four --slot buffers in SASS store order")

    baseline = args.baseline.read_bytes()
    row_words = 1280
    rows = 160
    expected_rows = [
        list(range(0, 80, 2)),
        list(range(1, 80, 2)),
        list(range(80, 160, 2)),
        list(range(81, 160, 2)),
    ]
    slots: list[dict[str, Any]] = []
    for slot_index, path in enumerate(args.slot):
        patched = path.read_bytes()
        changed = changed_word_indices(baseline, patched, args.arena_offset, args.bytes_count)
        changed_rows = sorted({word // row_words for word in changed})
        expected_words = sum(row_words for _ in expected_rows[slot_index])
        slots.append({
            "slot": slot_index,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(patched).hexdigest(),
            "changed_words": len(changed),
            "changed_runs": runs(changed),
            "changed_rows": changed_rows,
            "expected_rows": expected_rows[slot_index],
            "expected_words": expected_words,
            "mismatched_rows": sorted(set(changed_rows) ^ set(expected_rows[slot_index])),
            "row_word_counts": {
                str(row): sum(word // row_words == row for word in changed)
                for row in changed_rows
            },
        })

    report: dict[str, Any] = {
        "format": "dlss5-store-slot-map-v1",
        "baseline": {
            "path": str(args.baseline.resolve()),
            "sha256": hashlib.sha256(baseline).hexdigest(),
        },
        "arena_offset": f"0x{args.arena_offset:x}",
        "bytes": args.bytes_count,
        "physical_storage": {
            "shape_candidate": [rows, 2, 40, 16, 4],
            "dtype": "uint8/e4m3",
            "row_bytes": row_words * 4,
            "cta_x_formula": "(word_index // 16) % 40",
            "lane_formula": "(word_index % 16) + 16 * ((word_index // 640) % 2)",
            "logical_permutation": "unresolved; this is the physical store map",
        },
        "slots": slots,
        "all_slots_exact": all(
            item["changed_words"] == item["expected_words"] and not item["mismatched_rows"]
            for item in slots
        ),
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["all_slots_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
