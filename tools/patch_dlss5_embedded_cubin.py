"""Patch one decompressed CUBIN byte inside the DLL's Zstandard bundle.

This tool is deliberately conservative. It patches a temporary/output DLL
only, accepts one byte mutation in one selected architecture frame, and aborts
unless CUDA Zstandard level 5 reproduces the original compressed-frame length.
No PE signature or unrelated bytes are rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import zstandard

from extract_dlss5_embedded_cubins import GPU_SUFFIXES, OUTER_MAGIC, ZSTD_MAGIC, offsets


def locate_frame(data: bytes, bundle_index: int, gpu: str) -> tuple[int, int, bytes]:
    bundles = offsets(data, OUTER_MAGIC)
    if bundle_index < 0 or bundle_index >= len(bundles):
        raise ValueError(f"bundle index {bundle_index} is outside 0..{len(bundles) - 1}")
    bundle_offset = bundles[bundle_index]
    bundle_end = bundles[bundle_index + 1] if bundle_index + 1 < len(bundles) else len(data)
    frame_start = bundle_offset + 0x50
    cubin_index = 0
    while frame_start < bundle_end:
        frame_start = data.find(ZSTD_MAGIC, frame_start, bundle_end)
        if frame_start < 0:
            break
        decoder = zstandard.ZstdDecompressor().decompressobj()
        compressed = data[frame_start:bundle_end]
        decoded = decoder.decompress(compressed)
        consumed = len(compressed) - len(decoder.unused_data)
        if not decoder.eof or consumed <= 0:
            raise ValueError(f"invalid Zstandard frame at 0x{frame_start:x}")
        if decoded[:4] == b"\x7fELF":
            frame_gpu = GPU_SUFFIXES[cubin_index] if cubin_index < len(GPU_SUFFIXES) else None
            if frame_gpu == gpu:
                return frame_start, consumed, decoded
            cubin_index += 1
        frame_start += consumed
    raise ValueError(f"could not find {gpu} CUBIN in bundle {bundle_index}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--bundle", type=int, required=True)
    parser.add_argument("--gpu", choices=GPU_SUFFIXES, required=True)
    parser.add_argument("--cubin-offset", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--byte-value", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if not 0 <= args.byte_value <= 0xFF:
        parser.error("--byte-value must be in 0..255")

    source = args.dll.resolve()
    output = args.output.resolve()
    data = source.read_bytes()
    frame_start, compressed_size, original_cubin = locate_frame(data, args.bundle, args.gpu)
    if not 0 <= args.cubin_offset < len(original_cubin):
        parser.error(f"--cubin-offset must be in 0..{len(original_cubin) - 1:#x}")

    original_byte = original_cubin[args.cubin_offset]
    patched_cubin = bytearray(original_cubin)
    patched_cubin[args.cubin_offset] = args.byte_value

    compressor = zstandard.ZstdCompressor(
        level=5, write_content_size=True, write_checksum=False, write_dict_id=False
    )
    original_recompressed = compressor.compress(original_cubin)
    if len(original_recompressed) != compressed_size:
        raise RuntimeError(
            "Zstandard level-5 parameters did not reproduce the original frame length; "
            "refusing to patch the container"
        )
    patched_compressed = compressor.compress(patched_cubin)
    if len(patched_compressed) != compressed_size:
        raise RuntimeError(
            f"patched frame length changed from {compressed_size} to {len(patched_compressed)}; "
            "choose a different mutation or update the container format first"
        )

    result = bytearray(data)
    result[frame_start : frame_start + compressed_size] = patched_compressed
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)

    check_decoder = zstandard.ZstdDecompressor().decompressobj()
    check = check_decoder.decompress(bytes(result[frame_start : frame_start + compressed_size]))
    if check != bytes(patched_cubin):
        raise RuntimeError("post-write decompression did not reproduce the patched CUBIN")

    report = {
        "source": str(source),
        "output": str(output),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "output_sha256": hashlib.sha256(result).hexdigest(),
        "bundle": args.bundle,
        "gpu": args.gpu,
        "frame_offset": frame_start,
        "compressed_bytes": compressed_size,
        "cubin_bytes": len(original_cubin),
        "cubin_offset": args.cubin_offset,
        "original_byte": original_byte,
        "patched_byte": args.byte_value,
        "original_cubin_sha256": hashlib.sha256(original_cubin).hexdigest(),
        "patched_cubin_sha256": hashlib.sha256(patched_cubin).hexdigest(),
        "container_size_unchanged": len(data) == len(result),
    }
    manifest = args.manifest.resolve() if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
