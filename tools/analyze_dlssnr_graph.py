#!/usr/bin/env python3
"""Build static DLSS-NR weight, cubin, and block-graph manifests."""

from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
from pathlib import Path


def parse_weights(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    declared_size = struct.unpack_from("<Q", data, 0)[0]
    if declared_size != len(data):
        raise ValueError(f"declared size {declared_size} != file size {len(data)}")

    records: list[dict[str, object]] = []
    offset = 8
    while offset < len(data):
        record_offset = offset
        name_length = struct.unpack_from("<Q", data, offset)[0]
        offset += 8
        name = data[offset : offset + name_length].decode("ascii")
        offset += name_length
        outer_size, inner_size, data_size = struct.unpack_from("<QQQ", data, offset)
        offset += 24
        present = struct.unpack_from("<I", data, offset)[0]
        offset += 4 + data_size
        device, rank, dtype_code, element_count = struct.unpack_from("<QIII", data, offset)
        offset += 20
        if outer_size != inner_size or outer_size != data_size + 40:
            raise ValueError(f"unexpected record sizes for {name}")
        if data_size != element_count * 2:
            raise ValueError(f"{name} is not a two-byte-element tensor")
        records.append(
            {
                "name": name,
                "record_offset": record_offset,
                "data_offset": record_offset + 8 + name_length + 24 + 4,
                "data_size": data_size,
                "element_count": element_count,
                "dtype_code": dtype_code,
                "dtype_inference": "float16",
                "device_code": device,
                "rank": rank,
                "present": present,
            }
        )

    blocks: dict[int, list[dict[str, object]]] = {}
    for record in records:
        match = re.fullmatch(r"block(\d+)\.layer(\d+)\.(.+)", str(record["name"]))
        if not match:
            raise ValueError(f"unexpected weight name {record['name']}")
        block = int(match.group(1))
        blocks.setdefault(block, []).append(record)

    return {
        "source": str(path),
        "serialized_size": len(data),
        "record_count": len(records),
        "block_count": len(blocks),
        "raw_tensor_bytes": sum(int(record["data_size"]) for record in records),
        "element_count": sum(int(record["element_count"]) for record in records),
        "records": records,
    }


def parse_cubins(root: Path, readelf: str) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    for cubin in sorted(root.glob("fatbin_*/*.sm_86.cubin")):
        output = subprocess.run(
            [readelf, "-s", "--wide", str(cubin)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        kernels = []
        for line in output.splitlines():
            if " FUNC " in line and " GLOBAL " in line:
                kernels.append(line.split()[-1])
        groups.append(
            {
                "group": cubin.parent.name,
                "representative": str(cubin),
                "kernel_count": len(kernels),
                "kernels": kernels,
            }
        )
    return {
        "architecture_used_for_symbols": "sm_86",
        "group_count": len(groups),
        "unique_kernel_count": sum(int(group["kernel_count"]) for group in groups),
        "groups": groups,
    }


def graph_manifest() -> dict[str, object]:
    blocks: list[dict[str, object]] = []

    def add(index: int, kind: str, channels: str, inputs: list[object], notes: str = "") -> None:
        blocks.append(
            {"index": index, "kind": kind, "channels": channels, "inputs": inputs, "notes": notes}
        )

    add(0, "CCTinlayoutFusedPreBlockSwin1H (downsample)", "RGB texture→fused 32ch→32", ["network:rgb"],
        "the kernel's texture front-end assembles the 32-channel input tile; output 0 continues down the encoder; output 1 is the full-resolution skip used by block 70")
    for index in range(1, 4):
        add(index, "CCTinlayoutFusedSwin1H", "32", [index - 1])
    add(4, "CCTinlayoutFusedSwin1H (downsample)", "32→64", [3],
        "output 1 is the 32-channel skip used by block 66")
    for index in range(5, 8):
        add(index, "CCTinlayoutFusedSwin2H", "64", [index - 1])
    add(8, "CCTinlayoutFusedSwin2H (downsample)", "64→128", [7],
        "output 1 is the 64-channel skip used by block 62")
    for index in range(9, 14):
        add(index, "CCTinlayoutFusedSwin4H", "128", [index - 1])
    add(14, "CCTinlayoutFusedSwin4H (downsample)", "128→256", [13],
        "output 1 is the 128-channel skip used by block 56")
    for index in range(15, 22):
        add(index, "CCTinlayoutFusedSwin8H", "256", [index - 1])
    add(22, "CCTinlayoutFusedSwin8H (downsample)", "256→512", [21],
        "output 1 is the 256-channel skip used by block 48")
    for index in range(23, 31):
        add(index, "CCSplitSwin16HBlock", "512", [index - 1],
            "block 30 additionally contains CCSplitSwin16HFinalHead")
    for index in range(31, 39):
        add(index, "CCVit1DBlock or CCVitBlock", "1024", [index - 1],
            "factory-selected layout variant; both cubin families are embedded")
    add(39, "CCDecInputUpsample", "1024→512", [
        {"block": 38, "output": 0, "role": "main"},
        {"block": 30, "output": 1, "role": "skip"},
    ])
    for index in range(40, 48):
        add(index, "CCSplitSwin16HBlock", "512", [index - 1])
    add(48, "CCTinlayoutFusedSwin8H (upsample)", "512+256→256", [
        {"block": 47, "output": 0, "role": "main"},
        {"block": 22, "output": 1, "role": "skip"},
    ])
    for index in range(49, 56):
        add(index, "CCTinlayoutFusedSwin8H", "256", [index - 1])
    add(56, "CCTinlayoutFusedSwin4H (upsample)", "256+128→128", [
        {"block": 55, "output": 0, "role": "main"},
        {"block": 14, "output": 1, "role": "skip"},
    ])
    for index in range(57, 62):
        add(index, "CCTinlayoutFusedSwin4H", "128", [index - 1])
    add(62, "CCTinlayoutFusedSwin2H (upsample)", "128+64→64", [
        {"block": 61, "output": 0, "role": "main"},
        {"block": 8, "output": 1, "role": "skip"},
    ])
    for index in range(63, 66):
        add(index, "CCTinlayoutFusedSwin2H", "64", [index - 1])
    add(66, "CCTinlayoutFusedSwin1H (upsample)", "64+32→32", [
        {"block": 65, "output": 0, "role": "main"},
        {"block": 4, "output": 1, "role": "skip"},
    ])
    for index in range(67, 70):
        add(index, "CCTinlayoutFusedSwin1H", "32", [index - 1])
    add(70, "CCTinlayoutFusedPostBlockSwin1H", "32+enc0-skip(32)→RGB", [
        {"block": 69, "output": 0, "role": "main"},
        {"block": 0, "output": 1, "role": "encoder skip"},
    ], "enc0 skip is block0's full-resolution 32-channel output; contains blend_scale and optional ControlMask/simple-blend output variants")

    if [block["index"] for block in blocks] != list(range(71)):
        raise AssertionError("graph must contain blocks 0 through 70")
    return {
        "model_name": "hnet-vigilant-squid",
        "weight_preset": "CC_Control_History_Blend_Quantize_With_Teacher_honest_tench_2026_07_04_22_30_weights",
        "block_count": len(blocks),
        "blocks": blocks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("extracted", type=Path)
    parser.add_argument("--readelf", default="llvm-readelf")
    args = parser.parse_args()
    root = args.extracted.resolve()

    weights = parse_weights(root / "WEIGHTS_HT.bin")
    cubins = parse_cubins(root / "cubins", args.readelf)
    graph = graph_manifest()

    (root / "weights_manifest.json").write_text(json.dumps(weights, indent=2) + "\n")
    (root / "cubin_kernels.json").write_text(json.dumps(cubins, indent=2) + "\n")
    (root / "compute_graph.json").write_text(json.dumps(graph, indent=2) + "\n")
    print(
        f"{weights['record_count']} tensors, {graph['block_count']} blocks, "
        f"{cubins['unique_kernel_count']} SM86 kernel entry points"
    )


if __name__ == "__main__":
    main()
