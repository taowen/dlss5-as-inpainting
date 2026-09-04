"""Download the license-cleared image fixtures used by the native cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


SOURCES = {
    "blue_marble.jpg": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Nasa_blue_marble.jpg?width=1200",
        "source_page": "https://commons.wikimedia.org/wiki/File:Nasa_blue_marble.jpg",
        "license": "Public domain; NASA/USGS/NOAA",
    },
    "scenic_landscape.jpg": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/Vintage_scenic_landscape_photo.jpg?width=1200",
        "source_page": "https://commons.wikimedia.org/wiki/File:Vintage_scenic_landscape_photo.jpg",
        "license": "Public domain; U.S. Fish and Wildlife Service",
    },
    "stone_texture.jpg": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/STONE_TEXTURE.jpg?width=1200",
        "source_page": "https://commons.wikimedia.org/wiki/File:STONE_TEXTURE.jpg",
        "license": "CC0",
    },
    "portrait_cc0.jpg": {
        "url": "https://commons.wikimedia.org/wiki/Special:FilePath/-Portrait_of_a_Man-_MET_DP326401.jpg?width=1200",
        "source_page": "https://commons.wikimedia.org/wiki/File:-Portrait_of_a_Man-_MET_DP326401.jpg",
        "license": "CC0; The Metropolitan Museum of Art Open Access",
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("examples/assets/input"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"schema": "public_image_sources_v1", "files": {}}
    for name, metadata in SOURCES.items():
        destination = args.output_dir / name
        if args.force or not destination.is_file():
            request = urllib.request.Request(
                metadata["url"], headers={"User-Agent": "dlss5-as-inpainting-example-fetcher/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                destination.write_bytes(response.read())
        manifest["files"][name] = {
            **metadata,
            "bytes": destination.stat().st_size,
            "sha256": file_sha256(destination),
        }
    manifest_path = args.output_dir.parent / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "files": manifest["files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
