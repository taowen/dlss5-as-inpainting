"""Patch one decoded SM120 SASS instruction and preserve the CUBIN container.

This is the instruction-level companion to ``patch_dlss5_embedded_cubin.py``.
It intentionally operates on a loose/extracted CUBIN and requires an exact
old-instruction match.  The frozen control words are retained by cubit, which
makes it suitable for one-instruction telemetry experiments on a disposable
copy of the embedded pre kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cubin", type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cubit", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument(
        "--replace", nargs=2, action="append", metavar=("OLD", "NEW"), required=True,
        help="exact old SASS text and replacement text; repeat for multiple slots",
    )
    parser.add_argument("--sass-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    cubin = args.cubin.resolve()
    cubit = args.cubit.resolve()
    table = args.table.resolve()
    sass = args.sass_output.resolve()
    output = args.output.resolve()

    disassemble = [
        str(cubit), "disassemble", str(cubin), "--table", str(table),
        "--kernel", args.kernel, "--frozen", "--output", str(sass),
    ]
    run(disassemble)
    source = sass.read_text(encoding="utf-8")
    replacements = []
    for old, new in args.replace:
        old_line = old.strip().rstrip(";") + " ;"
        new_line = new.strip().rstrip(";") + " ;"
        matches = source.count(old_line)
        if matches != 1:
            raise RuntimeError(f"expected one exact SASS match for {old_line!r}, found {matches}")
        source = source.replace(old_line, new_line, 1)
        replacements.append({"old": old_line, "new": new_line})
    sass.write_text(source, encoding="utf-8")

    run([
        str(cubit), "asm", str(sass), "--table", str(table),
        "--template", str(cubin), "--kernel", args.kernel,
        "--output", str(output),
    ])
    report = {
        "input": str(cubin),
        "output": str(output),
        "kernel": args.kernel,
        "replacements": replacements,
        "sass": str(sass),
        "input_bytes": cubin.stat().st_size,
        "output_bytes": output.stat().st_size,
        "input_sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    manifest = args.manifest.resolve() if args.manifest else output.with_suffix(output.suffix + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
