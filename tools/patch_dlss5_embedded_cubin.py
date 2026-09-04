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
import struct
from pathlib import Path

import zstandard

from extract_dlss5_embedded_cubins import GPU_SUFFIXES, OUTER_MAGIC, ZSTD_MAGIC, offsets


def locate_frame(
    data: bytes, bundle_index: int, gpu: str
) -> tuple[int, int, int, int, bytes]:
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
                return bundle_offset, bundle_end, frame_start, consumed, decoded
            cubin_index += 1
        frame_start += consumed
    raise ValueError(f"could not find {gpu} CUBIN in bundle {bundle_index}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--bundle", type=int, required=True)
    parser.add_argument("--gpu", choices=GPU_SUFFIXES, required=True)
    mutation = parser.add_mutually_exclusive_group(required=True)
    mutation.add_argument("--cubin-offset", type=lambda value: int(value, 0))
    mutation.add_argument("--replacement-cubin", type=Path)
    parser.add_argument("--byte-value", type=lambda value: int(value, 0))
    parser.add_argument(
        "--allow-length-change",
        action="store_true",
        help="allow a compressed frame size change by consuming bundle-end padding",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if args.cubin_offset is not None and args.byte_value is None:
        parser.error("--byte-value is required with --cubin-offset")
    if args.cubin_offset is None and args.byte_value is not None:
        parser.error("--byte-value requires --cubin-offset")
    if args.byte_value is not None and not 0 <= args.byte_value <= 0xFF:
        parser.error("--byte-value must be in 0..255")

    source = args.dll.resolve()
    output = args.output.resolve()
    data = source.read_bytes()
    bundle_offset, bundle_end, frame_start, compressed_size, original_cubin = locate_frame(
        data, args.bundle, args.gpu
    )
    original_byte = None
    if args.replacement_cubin is not None:
        replacement = args.replacement_cubin.resolve().read_bytes()
        if len(replacement) != len(original_cubin):
            raise RuntimeError(
                f"replacement CUBIN is {len(replacement)} bytes; expected {len(original_cubin)}"
            )
        patched_cubin = bytearray(replacement)
    else:
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
    delta = len(patched_compressed) - compressed_size
    if delta != 0 and not args.allow_length_change:
        raise RuntimeError(
            f"patched frame length changed from {compressed_size} to {len(patched_compressed)}; "
            "choose a different mutation or update the container format first"
        )

    if delta == 0:
        result = bytearray(data)
        result[frame_start : frame_start + compressed_size] = patched_compressed
    else:
        # The bundle stores a descriptor between each Zstandard frame. For a
        # length-changing patch, shift the remainder of this bundle and reclaim
        # the same number of bytes from the bundle-end zero padding. The target
        # sm_120 pre frame is the fourth frame, so its descriptor is in the
        # preceding gap and the bundle header's +0x08 footer offset moves too.
        tail = bytearray(data[frame_start + compressed_size : bundle_end])
        if delta > 0:
            if len(tail) < delta or any(tail[-delta:]):
                raise RuntimeError(
                    "bundle has no zero padding available for this length-changing patch"
                )
            tail = tail[:-delta]
        else:
            tail.extend(b"\x00" * (-delta))
        result = bytearray(data[:frame_start]) + patched_compressed + tail + bytearray(data[bundle_end:])

        metadata_start = bundle_offset + 0x50
        previous_end = data.rfind(ZSTD_MAGIC, metadata_start, frame_start)
        if previous_end < 0:
            raise RuntimeError("could not locate the preceding frame metadata")
        previous_decoder = zstandard.ZstdDecompressor().decompressobj()
        previous_decoder.decompress(data[previous_end:frame_start])
        previous_frame_end = previous_end + (
            len(data[previous_end:frame_start]) - len(previous_decoder.unused_data)
        )
        gap_start = previous_frame_end
        gap_end = frame_start
        # The container keeps both the exact frame length and a small
        # framing-inclusive length in the descriptor. The latter is not a
        # universal +N: it depends on the descriptor variant, so update every
        # matching old-length form present in this immediate metadata gap.
        metadata_updates = 0
        for addend in (0, 3, 4, 7):
            old_size = struct.pack("<I", compressed_size + addend)
            new_size = struct.pack("<I", len(patched_compressed) + addend)
            cursor = gap_start
            while True:
                old_size_position = data.find(old_size, cursor, gap_end)
                if old_size_position < 0:
                    break
                result[old_size_position : old_size_position + 4] = new_size
                metadata_updates += 1
                cursor = old_size_position + 1
        if metadata_updates < 2:
            raise RuntimeError(
                "could not locate all compressed-size metadata for the target frame"
            )
        footer_offset = struct.unpack_from("<Q", data, bundle_offset + 8)[0]
        if footer_offset < bundle_end - bundle_offset:
            struct.pack_into("<Q", result, bundle_offset + 8, footer_offset + delta)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(result)

    check_decoder = zstandard.ZstdDecompressor().decompressobj()
    check = check_decoder.decompress(bytes(result[frame_start : frame_start + len(patched_compressed)]))
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
        "compressed_bytes": len(patched_compressed),
        "compressed_delta": delta,
        "cubin_bytes": len(original_cubin),
        "cubin_offset": args.cubin_offset,
        "original_byte": original_byte,
        "patched_byte": args.byte_value,
        "replacement_cubin": str(args.replacement_cubin.resolve()) if args.replacement_cubin else None,
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
