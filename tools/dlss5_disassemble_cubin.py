"""Extract one DLSS CUBIN function and summarize its SASS front-end."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def function_block(sass: str, function: str) -> str:
    marker = f".text.{function}"
    starts = [match.start() for match in re.finditer(re.escape(marker), sass)]
    if not starts:
        raise ValueError(f"function section not found: {function}")
    start = starts[0]
    next_section = re.search(r"\n//-+ \.text\.", sass[start + len(marker) :])
    end = start + len(marker) + next_section.start() if next_section else len(sass)
    return sass[start:end]


def summarize(block: str, function: str) -> dict[str, object]:
    instructions = [line.strip() for line in block.splitlines() if "*/" in line]
    tex = [line for line in instructions if re.search(r"\bTEX", line)]
    hmma = [line for line in instructions if "HMMA." in line]
    shared_stores = [line for line in instructions if "STS.128" in line]
    front_loads = [
        line
        for line in instructions
        if "LDG" in line and ("+0x2010" in line or "+0x2210" in line)
    ]
    return {
        "function": function,
        "tex_instruction_count": len(tex),
        "tex_instructions": tex,
        "shared_128_store_count": len(shared_stores),
        "shared_128_stores": shared_stores,
        "front_weight_loads": front_loads,
        "hmma_instruction_count": len(hmma),
        "first_hmma_instructions": hmma[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvdisasm", type=Path, required=True)
    parser.add_argument("--cubin", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--sass-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()

    result = subprocess.run(
        [
            str(args.nvdisasm),
            "--print-code",
            "--no-dataflow",
            "--separate-functions",
            str(args.cubin),
        ],
        check=True,
        capture_output=True,
    )
    sass = result.stdout.decode("utf-8", errors="replace")
    block = function_block(sass, args.function)
    summary = summarize(block, args.function)
    if args.sass_output:
        args.sass_output.parent.mkdir(parents=True, exist_ok=True)
        args.sass_output.write_text(block, encoding="utf-8")
    if args.summary_output:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
