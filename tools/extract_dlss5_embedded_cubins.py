"""Extract the Zstandard-packed DLSS5 CUBIN frames from nvngx_dlssnr.dll.

The DLL does not contain plain ELF files. Each kernel bundle starts with the
private ``50 ED 55 BA`` container header at the offsets used by the repository
names, followed by four concatenated Zstandard frames for sm_75/sm_86/sm_89/
sm_120. The tool keeps the original DLL untouched and writes loose CUBINs plus
a manifest for hash comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

try:
    import zstandard
except ImportError as exc:  # pragma: no cover - exercised by the CLI only
    raise SystemExit("Install the extractor dependency with: python -m pip install zstandard") from exc


OUTER_MAGIC = bytes.fromhex("50ed55ba")
ZSTD_MAGIC = bytes.fromhex("28b52ffd")
GPU_SUFFIXES = ("sm_75", "sm_86", "sm_89", "sm_120")


def offsets(data: bytes, needle: bytes) -> list[int]:
    result: list[int] = []
    start = 0
    while True:
        found = data.find(needle, start)
        if found < 0:
            return result
        result.append(found)
        start = found + 1


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract(dll: Path, output: Path | None) -> dict[str, object]:
    data = dll.read_bytes()
    bundles = offsets(data, OUTER_MAGIC)
    if not bundles:
        raise ValueError(f"no DLSS5 bundle headers found in {dll}")

    if output is not None:
        output.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for bundle_index, bundle_offset in enumerate(bundles):
        bundle_end = bundles[bundle_index + 1] if bundle_index + 1 < len(bundles) else len(data)
        frame_start = bundle_offset + 0x50
        frame_index = 0
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
                raise ValueError(
                    f"invalid Zstandard frame at 0x{frame_start:x} in bundle {bundle_index}"
                )
            is_cubin = decoded[:4] == b"\x7fELF"
            gpu = GPU_SUFFIXES[cubin_index] if cubin_index < len(GPU_SUFFIXES) else None
            name = (
                f"fatbin_{bundle_index:02d}_0x{bundle_offset:x}.{gpu}.cubin"
                if is_cubin and gpu
                else None
            )
            record: dict[str, object] = {
                "bundle": bundle_index,
                "bundle_offset": bundle_offset,
                "frame": frame_index,
                "frame_offset": frame_start,
                "frame_offset_in_bundle": frame_start - bundle_offset,
                "compressed_bytes": consumed,
                "uncompressed_bytes": len(decoded),
                "format": "cubin" if is_cubin else decoded[:4].hex(),
                "gpu": gpu if is_cubin else None,
                "sha256": sha256(decoded),
                "file": name if output is not None else None,
            }
            if output is not None and name is not None:
                (output / name).write_bytes(decoded)
            records.append(record)
            frame_index += 1
            if is_cubin:
                cubin_index += 1
            frame_start += consumed

    return {
        "dll": str(dll.resolve()),
        "dll_sha256": sha256(data),
        "bundle_count": len(bundles),
        "frame_count": len(records),
        "bundles": records,
    }


def compare_existing(manifest: dict[str, object], existing: Path) -> list[dict[str, object]]:
    files = sorted(existing.glob("fatbin_*/*.cubin"))
    by_size: dict[int, list[Path]] = {}
    for path in files:
        by_size.setdefault(path.stat().st_size, []).append(path)
    comparisons: list[dict[str, object]] = []
    for record in manifest["bundles"]:  # type: ignore[index]
        size = int(record["uncompressed_bytes"])
        candidates = by_size.get(size, [])
        match = None
        for candidate in candidates:
            if hashlib.sha256(candidate.read_bytes()).hexdigest() == record["sha256"]:
                match = candidate
                break
        comparisons.append(
            {
                "file": record["file"],
                "gpu": record["gpu"],
                "sha256": record["sha256"],
                "matching_existing": str(match.resolve()) if match else None,
            }
        )
    return comparisons


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runtime_probe_output/embedded_cubins"))
    parser.add_argument("--existing-cubins", type=Path, default=Path("cubins"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    manifest = extract(args.dll.resolve(), args.output.resolve() if args.output else None)
    manifest["existing_comparison"] = compare_existing(manifest, args.existing_cubins.resolve())
    manifest_path = args.manifest or (args.output / "manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
