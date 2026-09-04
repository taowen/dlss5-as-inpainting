#!/usr/bin/env python3
"""Validate physical tinlayout coordinates from two safe SASS probes.

The probes replace only existing pre-block output stores in disposable CUBINs:
one writes a stable lane register and one writes the stable CTA-X register.
This validator checks the resulting raw storage map without pretending that a
physical word is already a logical NCHW element.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


def read_words(path: Path, offset: int, bytes_count: int) -> tuple[bytes, list[int]]:
    payload = path.read_bytes()[offset : offset + bytes_count]
    if len(payload) != bytes_count:
        raise ValueError(f"{path}: requested {bytes_count} bytes at 0x{offset:x}, got {len(payload)}")
    return payload, [struct.unpack_from("<I", payload, index)[0] for index in range(0, bytes_count, 4)]


def validate(name: str, words: list[int], expected) -> dict[str, Any]:
    bad = [(index, value, expected(index)) for index, value in enumerate(words) if value != expected(index)]
    return {
        "name": name,
        "words": len(words),
        "distinct_values": len(set(words)),
        "min": min(words),
        "max": max(words),
        "mismatches": len(bad),
        "first_mismatches": bad[:8],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-buffer", required=True, type=Path)
    parser.add_argument("--cta-x-buffer", required=True, type=Path)
    parser.add_argument("--arena-offset", type=lambda value: int(value, 0), default=0x336C00)
    parser.add_argument("--bytes", dest="bytes_count", type=lambda value: int(value, 0), default=0xC8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    lane_payload, lane_words = read_words(args.lane_buffer, args.arena_offset, args.bytes_count)
    cta_payload, cta_words = read_words(args.cta_x_buffer, args.arena_offset, args.bytes_count)
    if len(lane_words) != len(cta_words):
        raise ValueError("lane and CTA-X probe word counts differ")

    report: dict[str, Any] = {
        "format": "dlss5-tinlayout-physical-map-v1",
        "arena_offset": f"0x{args.arena_offset:x}",
        "bytes": args.bytes_count,
        "word_bytes": 4,
        "lane_probe": {
            "path": str(args.lane_buffer.resolve()),
            "sha256": hashlib.sha256(args.lane_buffer.read_bytes()).hexdigest(),
        },
        "cta_x_probe": {
            "path": str(args.cta_x_buffer.resolve()),
            "sha256": hashlib.sha256(args.cta_x_buffer.read_bytes()).hexdigest(),
        },
        "map": {
            "cta_x": validate("cta_x", cta_words, lambda word: (word // 16) % 40),
            "lane": validate(
                "lane",
                lane_words,
                lambda word: (word % 16) + 16 * ((word // 640) % 2),
            ),
        },
        "storage": {
            "shape_candidate": [160, 160, 32],
            "dtype": "uint8/e4m3",
            "physical_word_count": len(lane_words),
            "cta_grid": [40, 40, 1],
            "words_per_cta": len(lane_words) // (40 * 40),
            "logical_permutation": "unresolved; retain physical map above",
        },
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if all(item["mismatches"] == 0 for item in report["map"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
