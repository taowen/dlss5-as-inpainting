"""Run safe, one-instruction DLSS5 pre-front mutations and compare live images.

Each case is isolated in a copied runtime directory. The tool patches the
decompressed SM120 pre CUBIN with ``cubit``, repacks the embedded frame into a
disposable ``nvngx_dlssnr.dll``, runs the validated two-frame temporal sequence
on the local carrier, and compares both the final output and the optional Neural
texture readback.

The readback is enabled by ``DLSS5_D3D12_CAPTURE_NEURAL=1``. This tool never
inserts a new GPU store into the CUBIN; it only changes an existing instruction
and uses the already-visible composition path as the observation point.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_ROOT))

from dlss5_fp16_harness_probe import run_harness, write_contracts  # noqa: E402


KERNEL = "cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8"
CAPTURE_RE = re.compile(r"dlss5_d3d12_capture_(\d+)_(\d+)\.rgba16f\.bin$")
DISPATCH_RE = re.compile(r"d3d12_dispatch .* neural=(\d+)")
CAPTURE_SCHEDULE_RE = re.compile(
    r"d3d12_capture_scheduled label=([^ ]+) .* index=(\d+)"
)
MUTATIONS: tuple[dict[str, str], ...] = (
    {
        "name": "coord_fadd_half_to_zero",
        "old": "FADD.FTZ R5, R5, 0.5",
        "new": "FADD.FTZ R5, R5, 0.0",
        "reason": "remove the first half-pixel coordinate offset",
    },
    {
        "name": "coord_second_fadd_half_to_zero",
        "old": "FADD.FTZ R6, R5, 0.5",
        "new": "FADD.FTZ R6, R5, 0.0",
        "reason": "remove the second half-pixel coordinate offset",
    },
    {
        "name": "coord_post_x_fadd_half_to_zero",
        "old": "FADD.FTZ R48, R43, 0.5",
        "new": "FADD.FTZ R48, R43, 0.0",
        "reason": "remove the post-transform half-pixel x offset",
    },
    {
        "name": "tex_initial_mask_7_to_1",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the initial active texture read mask",
    },
    {
        "name": "tex_initial_mask_7_to_2",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x02",
        "reason": "isolate the second initial texture component",
    },
    {
        "name": "tex_initial_mask_7_to_4",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x04",
        "reason": "isolate the third initial texture component",
    },
    {
        "name": "tex_initial_mask_7_to_3",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x03",
        "reason": "isolate the first two initial texture components",
    },
    {
        "name": "tex_initial_mask_7_to_5",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x05",
        "reason": "isolate the first and third initial texture components",
    },
    {
        "name": "tex_initial_mask_7_to_6",
        "sass_pc": "0x0620",
        "cubin_offset": "0x1a1829",
        "old_byte": "0x07",
        "new_byte": "0x06",
        "reason": "isolate the second and third initial texture components",
    },
    {
        "name": "tex_1590_mask_7_to_1",
        "sass_pc": "0x1590",
        "cubin_offset": "0x1a2799",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the first late texture read mask",
    },
    {
        "name": "tex_15c0_mask_7_to_1",
        "sass_pc": "0x15c0",
        "cubin_offset": "0x1a27c9",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the second late texture read mask",
    },
    {
        "name": "tex_15d0_mask_7_to_1",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the active late texture feature read mask",
    },
    {
        "name": "tex_15d0_mask_7_to_2",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x02",
        "reason": "isolate the second active late texture component",
    },
    {
        "name": "tex_15d0_mask_7_to_4",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x04",
        "reason": "isolate the third active late texture component",
    },
    {
        "name": "tex_15d0_mask_7_to_3",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x03",
        "reason": "isolate the first two active late texture components",
    },
    {
        "name": "tex_15d0_mask_7_to_5",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x05",
        "reason": "isolate the first and third active late texture components",
    },
    {
        "name": "tex_15d0_mask_7_to_6",
        "sass_pc": "0x15d0",
        "cubin_offset": "0x1a27d9",
        "old_byte": "0x07",
        "new_byte": "0x06",
        "reason": "isolate the second and third active late texture components",
    },
    {
        "name": "tex_15e0_mask_7_to_1",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the second active late texture feature read mask",
    },
    {
        "name": "tex_15e0_mask_7_to_2",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x02",
        "reason": "isolate the second component of the second active late texture read",
    },
    {
        "name": "tex_15e0_mask_7_to_4",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x04",
        "reason": "isolate the third component of the second active late texture read",
    },
    {
        "name": "tex_15e0_mask_7_to_3",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x03",
        "reason": "isolate the first two components of the second active late texture read",
    },
    {
        "name": "tex_15e0_mask_7_to_5",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x05",
        "reason": "isolate the first and third components of the second active late texture read",
    },
    {
        "name": "tex_15e0_mask_7_to_6",
        "sass_pc": "0x15e0",
        "cubin_offset": "0x1a27e9",
        "old_byte": "0x07",
        "new_byte": "0x06",
        "reason": "isolate the second and third components of the second active late texture read",
    },
    {
        "name": "tex_15f0_mask_7_to_1",
        "sass_pc": "0x15f0",
        "cubin_offset": "0x1a27f9",
        "old_byte": "0x07",
        "new_byte": "0x01",
        "reason": "change the fourth late texture read mask",
    },
)


def command(args: list[str], cwd: Path) -> None:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(args)}\n"
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )


def read_rgba16f(path: Path, width: int, height: int) -> list[float]:
    payload = path.read_bytes()
    expected = width * height * 4 * 2
    if len(payload) != expected:
        raise ValueError(f"{path} has {len(payload)} bytes; expected {expected}")
    return list(struct.unpack(f"<{len(payload) // 2}e", payload))


def summary(values: list[float]) -> dict[str, Any]:
    count = len(values) // 4
    return {
        "mean": [sum(values[i::4]) / count for i in range(4)],
        "min": [min(values[i::4]) for i in range(4)],
        "max": [max(values[i::4]) for i in range(4)],
    }


def difference(left: list[float], right: list[float]) -> dict[str, float | int]:
    if len(left) != len(right):
        raise ValueError("cannot compare buffers with different lengths")
    delta = [a - b for a, b in zip(left, right)]
    return {
        "mae": sum(abs(value) for value in delta) / len(delta),
        "rmse": math.sqrt(sum(value * value for value in delta) / len(delta)),
        "max_abs": max(abs(value) for value in delta),
        "changed_above_1e-3": sum(abs(value) > 1e-3 for value in delta),
        "elements": len(delta),
    }


def copy_runtime(template: Path, destination: Path, harness: Path, addon: Path, dll: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite existing case runtime: {destination}")
    patterns = (
        "*.rgba16f.bin", "*.r32f.bin", "*.rg16f.bin", "*.log", "*.cubin", "*.dxil"
    )

    def ignore_runtime_artifacts(_path: str, names: list[str]) -> list[str]:
        return [
            name for name in names
            if name == "dlss5_reshade_capture.addon64"
            or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
        ]

    shutil.copytree(template, destination, ignore=ignore_runtime_artifacts)
    shutil.copy2(harness, destination / "dlss5_eval.exe")
    shutil.copy2(addon, destination / "dlss5_reshade_capture.addon64")
    shutil.copy2(dll, destination / "nvngx_dlssnr.dll")


def capture_pair(
    runtime: Path, width: int, height: int, capture_all: bool
) -> tuple[Path, Path, list[Path], list[str], Path | None, Path | None]:
    groups: dict[str, list[tuple[int, Path]]] = {}
    for path in runtime.glob("dlss5_d3d12_capture_*.rgba16f.bin"):
        match = CAPTURE_RE.fullmatch(path.name)
        if match:
            groups.setdefault(match.group(1), []).append((int(match.group(2)), path))
    if not groups:
        raise RuntimeError("Neural readback produced no capture files")
    captures = max(groups.values(), key=lambda group: max(index for index, _ in group))
    captures.sort(key=lambda item: item[0])
    if len(captures) < 2:
        raise RuntimeError("Neural readback produced fewer than two SRV captures")
    layout_by_index: dict[int, str] = {}
    current_kind = "unknown"
    log_path = runtime / "dlss5_reshade_capture.log"
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            dispatch = DISPATCH_RE.search(line)
            if dispatch:
                current_kind = "neural" if dispatch.group(1) == "1" else "original"
            scheduled = CAPTURE_SCHEDULE_RE.search(line)
            if scheduled:
                label, index = scheduled.groups()
                layout_by_index[int(index)] = f"{current_kind}:{label}"
    capture_layout = [layout_by_index.get(index, "legacy_capture") for index, _ in captures]
    hidden_neural = None
    final_texture = None
    if capture_all:
        # Group by the actual dispatch labels from the add-on log. The carrier
        # can issue a trailing Original dispatch while writing the harness
        # output, so the latest files are not necessarily the latest Neural
        # resources.
        neural_batches: list[list[tuple[int, Path]]] = []
        current_batch: list[tuple[int, Path]] = []
        for item in captures:
            label = layout_by_index.get(item[0], "")
            if label == "neural:root0_descriptor0":
                if current_batch:
                    neural_batches.append(current_batch)
                current_batch = [item]
            elif current_batch and label.startswith("neural:"):
                current_batch.append(item)
        if current_batch:
            neural_batches.append(current_batch)
        if not neural_batches:
            raise RuntimeError("all-resource capture found no complete Neural dispatch")
        latest = neural_batches[-1]
        latest_by_label = {
            layout_by_index.get(index, ""): path for index, path in latest
        }
        original = latest_by_label.get("neural:root0_descriptor0")
        neural = latest_by_label.get("neural:root0_descriptor1")
        hidden_neural = latest_by_label.get("neural:root0_descriptor2")
        final_texture = latest_by_label.get("neural:root0_descriptor5")
        if not original or not neural or not hidden_neural or not final_texture:
            raise RuntimeError("latest Neural dispatch did not expose descriptors 0, 1, 2, and 5")
    else:
        original = captures[-2][1]
        neural = captures[-1][1]
    all_paths = [path for _, path in captures]
    read_rgba16f(original, width, height)
    read_rgba16f(neural, width, height)
    for path in all_paths:
        read_rgba16f(path, width, height)
    return original, neural, all_paths, capture_layout, hidden_neural, final_texture


def run_case(
    *,
    name: str,
    mutation: dict[str, str] | None,
    args: argparse.Namespace,
    run_root: Path,
    contracts: dict[str, Path],
) -> dict[str, Any]:
    case_root = run_root / name
    runtime = case_root / "runtime"
    case_root.mkdir(parents=True, exist_ok=True)
    patched_dll = case_root / "nvngx_dlssnr.patched.dll"
    patched_cubin = case_root / "pre.patched.cubin"
    patch_sass = case_root / "pre.patched.sass"
    patch_manifest = case_root / "patch.json"
    dll = args.dll.resolve()

    result: dict[str, Any] = {
        "name": name,
        "mutation": mutation,
        "runtime": str(runtime),
        "status": "failed",
    }
    try:
        if mutation is not None:
            if "cubin_offset" in mutation:
                command(
                    [
                        sys.executable,
                        str(TOOLS_ROOT / "patch_dlss5_embedded_cubin.py"),
                        str(dll),
                        "--bundle",
                        "0",
                        "--gpu",
                        "sm_120",
                        "--cubin-offset",
                        mutation["cubin_offset"],
                        "--byte-value",
                        mutation["new_byte"],
                        "--allow-length-change",
                        "--output",
                        str(patched_dll),
                        "--manifest",
                        str(patch_manifest),
                    ],
                    REPO_ROOT,
                )
                patch_info = json.loads(patch_manifest.read_text(encoding="utf-8"))
                if patch_info.get("original_byte") != int(mutation["old_byte"], 0):
                    raise RuntimeError(
                        f"unexpected byte at {mutation['cubin_offset']}: "
                        f"{patch_info.get('original_byte')!r} != {mutation['old_byte']}"
                    )
            else:
                command(
                    [
                        sys.executable,
                        str(TOOLS_ROOT / "patch_dlss5_sass_instruction.py"),
                        str(args.cubin.resolve()),
                        "--kernel",
                        KERNEL,
                        "--cubit",
                        str(args.cubit.resolve()),
                        "--table",
                        str(args.table.resolve()),
                        "--replace",
                        mutation["old"],
                        mutation["new"],
                        "--sass-output",
                        str(patch_sass),
                        "--output",
                        str(patched_cubin),
                        "--manifest",
                        str(patch_manifest),
                    ],
                    REPO_ROOT,
                )
                command(
                    [
                        sys.executable,
                        str(TOOLS_ROOT / "patch_dlss5_embedded_cubin.py"),
                        str(dll),
                        "--bundle",
                        "0",
                        "--gpu",
                        "sm_120",
                        "--replacement-cubin",
                        str(patched_cubin),
                        "--allow-length-change",
                        "--output",
                        str(patched_dll),
                        "--manifest",
                        str(patch_manifest),
                    ],
                    REPO_ROOT,
                )
            runtime_dll = patched_dll
        else:
            runtime_dll = dll
        copy_runtime(args.runtime_template.resolve(), runtime, args.harness.resolve(), args.addon.resolve(), runtime_dll)

        output = (case_root / "final.rgba16f.bin").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["DLSS5_D3D12_CAPTURE_NEURAL"] = "1"
        if args.capture_all_neural:
            env["DLSS5_D3D12_CAPTURE_ALL_NEURAL"] = "1"
        else:
            env.pop("DLSS5_D3D12_CAPTURE_ALL_NEURAL", None)
        if args.capture_all_dispatches:
            env["DLSS5_D3D12_CAPTURE_ALL_DISPATCHES"] = "1"
        else:
            env.pop("DLSS5_D3D12_CAPTURE_ALL_DISPATCHES", None)
        env["DLSS5_DARK_NO_PRIVATE_HOOK"] = "1"
        for variable in ("DLSS5_DARK_SCAN", "DLSS5_DARK_SCAN_ALL", "DLSS5_DARK_DUMP_STRUCTS", "DLSS5_DARK_NOOP"):
            env.pop(variable, None)
        old_env = os.environ.copy()
        os.environ.update(env)
        try:
            second_input = "checker" if args.input == "color" else "color"
            run_harness(
                runtime / "dlss5_eval.exe",
                args.width,
                args.height,
                contracts["depth"],
                contracts["motion_zero"],
                [(contracts[args.input], 1), (contracts[second_input], 0)],
                output,
            )
        finally:
            os.environ.clear()
            os.environ.update(old_env)

        (
            original_path,
            neural_path,
            all_captures,
            capture_layout,
            hidden_neural,
            final_texture,
        ) = capture_pair(
            runtime, args.width, args.height, args.capture_all_neural
        )
        result.update(
            {
                "status": "ok",
                "final": str(output),
                "original_capture": str(original_path),
                "neural_capture": str(neural_path),
                "all_captures": [str(path) for path in all_captures],
                "capture_layout": capture_layout,
                "hidden_neural_capture": str(hidden_neural) if hidden_neural else None,
                "final_texture_capture": str(final_texture) if final_texture else None,
                "final_summary": summary(read_rgba16f(output, args.width, args.height)),
                "original_summary": summary(read_rgba16f(original_path, args.width, args.height)),
                "neural_summary": summary(read_rgba16f(neural_path, args.width, args.height)),
                "sha256": {
                    "final": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "original_capture": hashlib.sha256(original_path.read_bytes()).hexdigest(),
                    "neural_capture": hashlib.sha256(neural_path.read_bytes()).hexdigest(),
                },
            }
        )
    except Exception as exc:  # noqa: BLE001 - one bad GPU mutation must not stop the matrix
        result["error"] = str(exc)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-template", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--addon", type=Path, required=True)
    parser.add_argument("--dll", type=Path, default=REPO_ROOT / "bin" / "nvngx_dlssnr.dll")
    parser.add_argument("--cubin", type=Path, default=REPO_ROOT / "cubins/fatbin_00/fatbin_00_0xdf0e0.4.sm_120.cubin")
    parser.add_argument("--cubit", type=Path, default=REPO_ROOT / "third_party/cubit/target/release/cubit.exe")
    parser.add_argument("--table", type=Path, default=REPO_ROOT / "third_party/cubit/tables/sm120.json")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--input", choices=("color", "checker"), default="color")
    parser.add_argument("--only", action="append", help="run only the named mutation; repeatable")
    parser.add_argument(
        "--capture-all-neural",
        action="store_true",
        help="capture every resolved 2D descriptor around each Neural dispatch",
    )
    parser.add_argument(
        "--capture-all-dispatches",
        action="store_true",
        help="capture the same descriptor set after Original and Neural dispatches",
    )
    parser.add_argument("--workdir", type=Path, default=REPO_ROOT / ".native-build/front-mutations")
    args = parser.parse_args()

    for path in (args.runtime_template, args.harness, args.addon, args.dll, args.cubin, args.cubit, args.table):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")
    if args.width != 256 or args.height != 256:
        parser.error("the current capture carrier is validated only at 256x256")

    run_root = args.workdir.resolve() / time.strftime("run-%Y%m%d-%H%M%S")
    suffix = 0
    while run_root.exists():
        suffix += 1
        run_root = args.workdir.resolve() / (time.strftime("run-%Y%m%d-%H%M%S") + f"-{suffix}")
    run_root.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="dlss5-front-mutations-") as temporary:
        contracts = write_contracts(Path(temporary), args.width, args.height)
        selected = [mutation for mutation in MUTATIONS if not args.only or mutation["name"] in args.only]
        unknown = set(args.only or ()) - {mutation["name"] for mutation in MUTATIONS}
        if unknown:
            parser.error(f"unknown mutation(s): {', '.join(sorted(unknown))}")
        cases = [{"name": "baseline", "mutation": None}]
        cases.extend({"name": mutation["name"], "mutation": mutation} for mutation in selected)
        reports = [
            run_case(name=case["name"], mutation=case["mutation"], args=args, run_root=run_root, contracts=contracts)
            for case in cases
        ]

    baseline = next(report for report in reports if report["name"] == "baseline")
    if baseline["status"] == "ok":
        baseline_final = read_rgba16f(Path(baseline["final"]), args.width, args.height)
        baseline_original = read_rgba16f(Path(baseline["original_capture"]), args.width, args.height)
        baseline_neural = read_rgba16f(Path(baseline["neural_capture"]), args.width, args.height)
        baseline_captures = [
            read_rgba16f(Path(path), args.width, args.height)
            for path in baseline["all_captures"]
        ]
        baseline_hidden = (
            read_rgba16f(Path(baseline["hidden_neural_capture"]), args.width, args.height)
            if baseline["hidden_neural_capture"] else None
        )
        baseline_final_texture = (
            read_rgba16f(Path(baseline["final_texture_capture"]), args.width, args.height)
            if baseline["final_texture_capture"] else None
        )
        for report in reports:
            if report["status"] != "ok":
                continue
            capture_values = [
                read_rgba16f(Path(path), args.width, args.height)
                for path in report["all_captures"]
            ]
            if len(capture_values) != len(baseline_captures):
                raise RuntimeError(
                    f"capture count changed for {report['name']}: "
                    f"{len(capture_values)} != {len(baseline_captures)}"
                )
            report["diff_vs_baseline"] = {
                "final": difference(read_rgba16f(Path(report["final"]), args.width, args.height), baseline_final),
                "original_capture": difference(read_rgba16f(Path(report["original_capture"]), args.width, args.height), baseline_original),
                "neural_capture": difference(read_rgba16f(Path(report["neural_capture"]), args.width, args.height), baseline_neural),
                "all_captures": [
                    difference(values, baseline_values)
                    for values, baseline_values in zip(capture_values, baseline_captures)
                ],
            }
            if baseline_hidden is not None and report["hidden_neural_capture"]:
                report["diff_vs_baseline"]["hidden_neural_capture"] = difference(
                    read_rgba16f(Path(report["hidden_neural_capture"]), args.width, args.height),
                    baseline_hidden,
                )
            if baseline_final_texture is not None and report["final_texture_capture"]:
                report["diff_vs_baseline"]["final_texture_capture"] = difference(
                    read_rgba16f(Path(report["final_texture_capture"]), args.width, args.height),
                    baseline_final_texture,
                )

    output = run_root / "report.json"
    output.write_text(
        json.dumps(
            {
                "width": args.width,
                "height": args.height,
                "input": args.input,
                "sequence": [args.input, "checker" if args.input == "color" else "color"],
                "capture_all_neural": args.capture_all_neural,
                "capture_all_dispatches": args.capture_all_dispatches,
                "kernel": KERNEL,
                "runtime_template": str(args.runtime_template.resolve()),
                "dll": str(args.dll.resolve()),
                "cases": reports,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(output), "cases": reports}, indent=2))
    return 0 if baseline.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
