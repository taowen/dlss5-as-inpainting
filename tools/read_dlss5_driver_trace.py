"""Read the fixed-layout private CUDA export-table trace emitted by the ReShade probe."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


MAGIC = 0x4452434535444C53
HEADER = struct.Struct("<QIIIIQQ24x")
RECORD = struct.Struct("<Q4Q4QQQQ8Q2QQ4Q4Q8Q2QQ")
SAMPLE = struct.Struct("<4Q8Q2QQ")


def fmt_words(words: tuple[int, ...]) -> str:
    return ",".join(f"0x{value:x}" for value in words)


def read_trace(path: Path, include_samples: bool = False) -> int:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"trace is truncated: {len(data)} bytes")
    magic, version, header_bytes, pid, max_records, record_bytes, record_count = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ValueError(f"unexpected magic 0x{magic:x}")
    if header_bytes != HEADER.size:
        raise ValueError(f"unexpected header size: {header_bytes}")
    if record_bytes == RECORD.size:
        sample_count = 0
    elif record_bytes >= RECORD.size and (record_bytes - RECORD.size) % SAMPLE.size == 0:
        sample_count = (record_bytes - RECORD.size) // SAMPLE.size
    else:
        raise ValueError(
            f"layout mismatch: header={header_bytes}, record={record_bytes}; "
            f"expected legacy {HEADER.size}, {RECORD.size} or an extended sample record"
        )

    available = (len(data) - header_bytes) // record_bytes
    count = min(record_count, max_records, available)
    print(
        f"trace={path} version={version} pid={pid} "
        f"records={count}/{record_count} record_bytes={record_bytes} samples={sample_count}"
    )
    emitted = 0
    for ordinal in range(count):
        values = RECORD.unpack_from(data, header_bytes + ordinal * record_bytes)
        calls = values[0]
        if calls == 0:
            continue
        regs = values[1:5]
        stack = values[5:9]
        table, index, original = values[9:12]
        nonvolatile = values[12:20]
        extra = values[20:22]
        return_address = values[22]
        last_regs = values[23:27]
        last_stack = values[27:31]
        last_nonvolatile = values[31:39]
        last_extra = values[39:41]
        last_return_address = values[41]
        print(
            f"record={ordinal} index={index} count={calls} "
            f"table=0x{table:x} original=0x{original:x} "
            f"regs=[{fmt_words(regs)}] stack=[{fmt_words(stack)}] "
            f"nonvolatile=[{fmt_words(nonvolatile)}] "
            f"extra=[{fmt_words(extra)}] return=0x{return_address:x}"
        )
        print(
            f"  latest regs=[{fmt_words(last_regs)}] stack=[{fmt_words(last_stack)}] "
            f"nonvolatile=[{fmt_words(last_nonvolatile)}] "
            f"extra=[{fmt_words(last_extra)}] return=0x{last_return_address:x}"
        )
        if include_samples and sample_count:
            sample_base = header_bytes + ordinal * record_bytes + RECORD.size
            for sample_index in range(sample_count):
                sample = SAMPLE.unpack_from(data, sample_base + sample_index * SAMPLE.size)
                if sample[-1] == 0:
                    continue
                sample_regs = sample[0:4]
                sample_stack = sample[4:12]
                sample_extra = sample[12:14]
                sample_return = sample[14]
                print(
                    f"  sample={sample_index} regs=[{fmt_words(sample_regs)}] "
                    f"stack=[{fmt_words(sample_stack)}] extra=[{fmt_words(sample_extra)}] "
                    f"return=0x{sample_return:x}"
                )
        emitted += 1
    print(f"called_records={emitted}")
    return emitted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--samples",
        action="store_true",
        help="print the extended per-call sample ring when present",
    )
    args = parser.parse_args()
    read_trace(args.trace, include_samples=args.samples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
