"""Map the native GPU model buffer back to WEIGHTS_HT records exactly.

The native carrier allocates a 147,719,680-byte UAV for the model payload. It
is not a byte-for-byte copy of the serialized resource: records are copied
without the outer resource headers and are aligned for the GPU loader. This
tool proves the mapping using full-record byte comparisons, including the two
block-0 FP16 front tiles used by the sm_120 pre kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_records(buffer: bytes, weights: bytes, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cursor = 0
    previous_end = 0
    rows: list[dict[str, Any]] = []
    for record in sorted(manifest["records"], key=lambda item: item["record_offset"]):
        source_offset = int(record["data_offset"])
        size = int(record["data_size"])
        raw = weights[source_offset : source_offset + size]
        if len(raw) != size:
            raise ValueError(f"truncated source record {record['name']}")
        probe = raw[: min(64, len(raw))]
        buffer_offset = buffer.find(probe, cursor)
        if buffer_offset < 0 and len(probe) > 16:
            buffer_offset = buffer.find(raw[:16], cursor)
        if buffer_offset < 0:
            raise ValueError(f"could not locate {record['name']} after 0x{cursor:x}")
        available = min(size, len(buffer) - buffer_offset)
        equal_bytes = 0
        while equal_bytes < available and raw[equal_bytes] == buffer[buffer_offset + equal_bytes]:
            equal_bytes += 1
        row = {
            "name": record["name"],
            "source_data_offset": source_offset,
            "bytes": size,
            "buffer_offset": buffer_offset,
            "padding_before": buffer_offset - previous_end,
            "full_byte_equal": equal_bytes == size,
            "equal_bytes": equal_bytes,
        }
        rows.append(row)
        cursor = buffer_offset + size
        previous_end = cursor
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buffer", type=Path, help="native 147 MiB model-buffer snapshot")
    parser.add_argument(
        "--weights", type=Path, default=Path("DLSS5-extracted/WEIGHTS_HT.bin")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("DLSS5-extracted/weights_manifest.json")
    )
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--strict", action="store_true", help="return failure unless every record is exact"
    )
    args = parser.parse_args()

    buffer = args.buffer.read_bytes()
    weights = args.weights.read_bytes()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = locate_records(buffer, weights, manifest)
    full = sum(bool(row["full_byte_equal"]) for row in records)
    block0 = {row["name"]: row for row in records if row["name"] == "block0.layer0.layer"}
    block0_row = block0.get("block0.layer0.layer")
    front_tiles = {}
    if block0_row is not None:
        for name, relative in (("front_weight0_f16", 0x2010), ("front_weight1_f16", 0x2210)):
            front_tiles[name] = {
                "relative_to_block0_record": relative,
                "buffer_offset": block0_row["buffer_offset"] + relative,
                "source_data_offset": int(block0_row["source_data_offset"]) + relative,
            }
    report = {
        "buffer": str(args.buffer.resolve()),
        "buffer_bytes": len(buffer),
        "buffer_sha256": hashlib.sha256(buffer).hexdigest(),
        "weights": str(args.weights.resolve()),
        "weights_bytes": len(weights),
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "serialized_size_delta": len(buffer) - len(weights),
        "record_count": len(records),
        "records_found": len(records),
        "records_full_byte_equal": full,
        "bit_exact": full == len(records),
        "block0_front_tiles": front_tiles,
        "records": records,
    }
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report["bit_exact"] or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
