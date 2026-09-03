"""Extract the embedded DLSS-NR weight package from nvngx_dlssnr.dll."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pefile


def resource_name(entry: object) -> str:
    name = getattr(entry, "name", None)
    if name is not None:
        return str(name)
    return str(entry.struct.Id)


def extract_named_resource(dll: Path, name: str) -> bytes:
    pe = pefile.PE(str(dll), fast_load=False)
    if not hasattr(pe, "DIRECTORY_ENTRY_RESOURCE"):
        raise ValueError(f"{dll} has no PE resource directory")

    for type_entry in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        if not hasattr(type_entry, "directory"):
            continue
        for name_entry in type_entry.directory.entries:
            if resource_name(name_entry) != name or not hasattr(name_entry, "directory"):
                continue
            languages = name_entry.directory.entries
            if len(languages) != 1:
                raise ValueError(f"resource {name!r} has {len(languages)} language variants")
            data = languages[0].data.struct
            return pe.get_memory_mapped_image()[data.OffsetToData : data.OffsetToData + data.Size]
    raise KeyError(f"resource {name!r} was not found in {dll}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    parser.add_argument("--output", type=Path, default=Path("DLSS5-extracted/WEIGHTS_HT.bin"))
    parser.add_argument("--name", default="WEIGHTS_HT")
    args = parser.parse_args()

    dll = args.dll.resolve()
    output = args.output.resolve()
    payload = extract_named_resource(dll, args.name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    print(f"resource={args.name}")
    print(f"bytes={len(payload)}")
    print(f"sha256={digest}")
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
