#!/usr/bin/env python3
"""Turn a DLSS5 NVAPI capture into a replay-oriented launch manifest.

The ReShade capture add-on records the private CUDA-on-D3D12 launch ABI as
ordinary text plus exact parameter payload files.  This tool joins function
handles to names, joins launch records to their parameter bytes, and records
the CUDA texture/surface objects returned by the NVAPI descriptor helpers.
It deliberately does not decode or normalize the payload bytes: the manifest
keeps their exact SHA-256 and little-endian words so a later native replay can
prove that it used the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any


HEX = r"0x[0-9a-fA-F]+"


def as_int(value: str) -> int:
    return int(value, 0)


def classify(name: str) -> str:
    if name.startswith("cc_tinlayout_fused_pre_"):
        return "pre"
    if name.startswith("cc_tinlayout_fused_post_"):
        return "post"
    if name.startswith("cc_tinlayout_"):
        return "tinlayout"
    if name.startswith("cc_split_"):
        return "split_swin"
    if name.startswith("cc_vit_"):
        return "vit"
    if name.startswith("cg2r_"):
        return "cg2r"
    if name.startswith("cc_"):
        return "cc_utility"
    return "other"


def payload_words(data: bytes) -> list[int]:
    words = []
    for offset in range(0, len(data) - len(data) % 8, 8):
        words.append(struct.unpack_from("<Q", data, offset)[0])
    return words


def resource_fields(line: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("resource", "tex_resource", "smp_resource"):
        match = re.search(rf"{key}=({HEX})", line)
        if match:
            result[key] = as_int(match.group(1))
    for key in ("gpu_va",):
        match = re.search(rf"{key}=({HEX})", line)
        if match:
            result[key] = as_int(match.group(1))
    for key in ("width", "height", "depth_or_layers", "mips", "format", "dimension"):
        match = re.search(rf"{key}=([0-9]+)", line)
        if match:
            result[key] = int(match.group(1))
    return result


def parse_capture(runtime: Path) -> dict[str, Any]:
    log_path = runtime / "dlss5_reshade_capture.log"
    if not log_path.is_file():
        raise SystemExit(f"capture log not found: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")

    functions: dict[int, dict[str, Any]] = {}
    function_pattern = re.compile(
        rf"^nvapi_create_cu_function device=({HEX}) module=({HEX}) "
        rf"name=(\S+) function=({HEX}) status=([-0-9]+)$", re.MULTILINE
    )
    for match in function_pattern.finditer(text):
        device, module, name, function, status = match.groups()
        handle = as_int(function)
        functions[handle] = {
            "handle": handle,
            "handle_hex": function.lower(),
            "device": as_int(device),
            "module": as_int(module),
            "name": name,
            "class": classify(name),
            "status": int(status),
        }

    objects: list[dict[str, Any]] = []
    merged_pattern = re.compile(
        rf"^nvapi_get_cuda_merged_texture_sampler_object status=([-0-9]+) "
        rf"struct_in=([0-9]+) struct_out=([0-9]+) device=({HEX}) "
        rf"tex_desc=({HEX}) smp_desc=({HEX}) texture_handle=({HEX}) "
        rf"tex_resource=({HEX})(?P<rest>.*)$", re.MULTILINE
    )
    for match in merged_pattern.finditer(text):
        status, struct_in, struct_out, device, tex_desc, smp_desc, handle, tex_resource, rest = match.groups()
        item: dict[str, Any] = {
            "kind": "merged_texture_sampler",
            "status": int(status),
            "struct_size_in": int(struct_in),
            "struct_size_out": int(struct_out),
            "device": as_int(device),
            "texture_descriptor": as_int(tex_desc),
            "sampler_descriptor": as_int(smp_desc),
            "handle": as_int(handle),
            "handle_hex": handle.lower(),
            "texture_resource": as_int(tex_resource),
        }
        item.update({f"texture_{k}": v for k, v in resource_fields(rest).items()})
        objects.append(item)

    independent_pattern = re.compile(
        rf"^nvapi_get_cuda_independent_descriptor_object status=([-0-9]+) "
        rf"struct_in=([0-9]+) struct_out=([0-9]+) device=({HEX}) type=([0-9]+) "
        rf"descriptor=({HEX}) handle=({HEX}) resource=({HEX})(?P<rest>.*)$", re.MULTILINE
    )
    for match in independent_pattern.finditer(text):
        status, struct_in, struct_out, device, object_type, descriptor, handle, resource, rest = match.groups()
        item = {
            "kind": "independent_descriptor",
            "status": int(status),
            "struct_size_in": int(struct_in),
            "struct_size_out": int(struct_out),
            "device": as_int(device),
            "type": int(object_type),
            "type_name": {0: "surface", 1: "texture", 2: "sampler"}.get(int(object_type), "unknown"),
            "descriptor": as_int(descriptor),
            "handle": as_int(handle),
            "handle_hex": handle.lower(),
            "resource": as_int(resource),
        }
        item.update(resource_fields(rest))
        objects.append(item)

    raw_descriptors: list[dict[str, Any]] = []
    descriptor_pattern = re.compile(
        rf"^nvapi_descriptor_written kind=(\S+) descriptor=({HEX}) "
        rf"file=(\S+) bytes=([0-9]+) written=([01])$", re.MULTILINE
    )
    for match in descriptor_pattern.finditer(text):
        kind, descriptor, file_name, byte_count, written = match.groups()
        path = runtime / file_name
        item = {
            "kind": kind,
            "descriptor": as_int(descriptor),
            "descriptor_hex": descriptor.lower(),
            "file": file_name,
            "bytes": int(byte_count),
            "written": bool(int(written)),
        }
        if path.is_file():
            payload = path.read_bytes()
            item["actual_bytes"] = len(payload)
            item["sha256"] = hashlib.sha256(payload).hexdigest()
            item["hex"] = payload.hex()
        else:
            item["file_missing"] = True
        raw_descriptors.append(item)

    launches: list[dict[str, Any]] = []
    launch_pattern = re.compile(
        rf"^nvapi_launch_chain command_list=({HEX}) kernels=({HEX}) count=([0-9]+) "
        rf"kernel_file=(\S+) kernel_written=([01]) (?P<items>.*)$", re.MULTILINE
    )
    item_pattern = re.compile(
        rf"item([0-9]+)=\{{function=({HEX}),grid=([0-9]+),([0-9]+),([0-9]+),"
        rf"block=([0-9]+),([0-9]+),([0-9]+),shared=([0-9]+),params=({HEX}),param_size=([0-9]+)\}}"
    )
    for call_index, match in enumerate(launch_pattern.finditer(text)):
        command_list, kernels, count, kernel_file, kernel_written, item_text = match.groups()
        items = []
        for item_match in item_pattern.finditer(item_text):
            (
                item_index, function, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                shared, params, param_size,
            ) = item_match.groups()
            function_handle = as_int(function)
            item: dict[str, Any] = {
                "index": int(item_index),
                "function": function_handle,
                "function_hex": function.lower(),
                "function_info": functions.get(function_handle),
                "grid": [int(grid_x), int(grid_y), int(grid_z)],
                "block": [int(block_x), int(block_y), int(block_z)],
                "dynamic_shared_bytes": int(shared),
                "params_address": as_int(params),
                "param_size": int(param_size),
            }
            sequence_match = re.search(r"dlss5_nvapi_launch_([0-9]+)_kernels\.bin$", kernel_file)
            if sequence_match:
                sequence = int(sequence_match.group(1))
                param_path = runtime / f"dlss5_nvapi_launch_{sequence}_item{item['index']}_params.bin"
                item["capture_sequence"] = sequence
                item["param_file"] = param_path.name
                if param_path.is_file():
                    payload = param_path.read_bytes()
                    item["param_bytes"] = len(payload)
                    item["param_sha256"] = hashlib.sha256(payload).hexdigest()
                    item["param_words_u64"] = [f"0x{word:016x}" for word in payload_words(payload)]
                    item["param_hex"] = payload.hex()
                else:
                    item["param_file_missing"] = True
            items.append(item)
        launches.append({
            "call_index": call_index,
            "command_list": as_int(command_list),
            "kernels_address": as_int(kernels),
            "count": int(count),
            "kernel_file": kernel_file,
            "kernel_written": bool(int(kernel_written)),
            "items": items,
        })

    launch_names = [
        item.get("function_info", {}).get("name", "<unknown>")
        for launch in launches for item in launch["items"]
    ]
    class_counts = Counter(classify(name) for name in launch_names if name != "<unknown>")
    name_counts = Counter(launch_names)
    return {
        "schema": "dlss5-nvapi-launch-manifest-v1",
        "runtime": str(runtime),
        "log": str(log_path),
        "function_count": len(functions),
        "functions": sorted(functions.values(), key=lambda item: item["handle"]),
        "descriptor_object_count": len(objects),
        "descriptor_objects": objects,
        "raw_descriptor_count": len(raw_descriptors),
        "raw_descriptors": raw_descriptors,
        "launch_count": len(launches),
        "launch_item_count": sum(len(launch["items"]) for launch in launches),
        "launch_class_counts": dict(sorted(class_counts.items())),
        "launch_name_counts": dict(sorted(name_counts.items())),
        "launches": launches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path, help="runtime directory containing the capture log")
    parser.add_argument("--output", type=Path, help="JSON output path; defaults to <runtime>/nvapi_launch_manifest.json")
    args = parser.parse_args()
    manifest = parse_capture(args.runtime)
    output = args.output or args.runtime / "nvapi_launch_manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "functions": manifest["function_count"],
        "descriptor_objects": manifest["descriptor_object_count"],
        "launches": manifest["launch_count"],
        "launch_items": manifest["launch_item_count"],
        "classes": manifest["launch_class_counts"],
    }, indent=2))


if __name__ == "__main__":
    main()
