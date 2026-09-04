"""Run reproducible DLSS5 image, depth, history, and motion-vector cases.

This is a development tool. It converts public image files into the native
RGBA16F/R32F/RG16F contract, runs a locally supplied native harness, converts
the outputs to reviewable PNGs, and writes a relative-path JSON manifest.

The generated ``examples/cases/native`` directory is intentionally small and
human-readable; temporary binary contracts live under ``examples/.work``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from dlss5_fp16_harness_probe import run_harness  # noqa: E402


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    fallback = REPO_ROOT / "ffmpeg-7.1.1" / "bin" / "ffmpeg.exe"
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError("ffmpeg is required to decode the downloaded images")


def run_ffmpeg(ffmpeg: str, arguments: list[str]) -> None:
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", *arguments], check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_rgb8(ffmpeg: str, source: Path, destination: Path, width: int, height: int) -> np.ndarray:
    destination.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ffmpeg,
        [
            "-y",
            "-i",
            str(source),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            str(destination),
        ],
    )
    payload = np.fromfile(destination, dtype=np.uint8)
    expected = width * height * 3
    if payload.size != expected:
        raise RuntimeError(f"ffmpeg produced {payload.size} bytes for {source}; expected {expected}")
    return payload.reshape(height, width, 3)


def encode_rgb8(ffmpeg: str, image: np.ndarray, destination: Path) -> None:
    raw = destination.with_suffix(".rgb8.tmp")
    image.astype(np.uint8, copy=False).tofile(raw)
    try:
        run_ffmpeg(
            ffmpeg,
            [
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{image.shape[1]}x{image.shape[0]}",
                "-i",
                str(raw),
                "-frames:v",
                "1",
                str(destination),
            ],
        )
    finally:
        raw.unlink(missing_ok=True)


def srgb_to_linear(rgb8: np.ndarray) -> np.ndarray:
    encoded = rgb8.astype(np.float32) / 255.0
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    clipped = np.clip(linear.astype(np.float32), 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        clipped * 12.92,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def linear_luma(linear: np.ndarray) -> np.ndarray:
    return (
        0.2126 * linear[..., 0]
        + 0.7152 * linear[..., 1]
        + 0.0722 * linear[..., 2]
    ).astype(np.float32)


def write_contracts(
    work: Path,
    name: str,
    rgb8: np.ndarray,
    linear: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Path]:
    contract_dir = work / "contracts"
    contract_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.concatenate((linear, np.ones((height, width, 1), dtype=np.float32)), axis=2)
    color = contract_dir / f"{name}.rgba16f.bin"
    rgba.astype(np.float16).tofile(color)

    luma = np.clip(linear_luma(linear), 0.0, 1.0)
    depth_luma = np.clip(1.0 - luma, 0.0, 1.0).astype(np.float32)
    depth_flat = np.ones((height, width), dtype=np.float32)
    depth_luma_path = contract_dir / f"{name}.depth_luma.r32f.bin"
    depth_flat_path = contract_dir / f"{name}.depth_flat.r32f.bin"
    depth_luma.tofile(depth_luma_path)
    depth_flat.tofile(depth_flat_path)

    motion_zero = np.zeros((height, width, 2), dtype=np.float16)
    motion_vectors: dict[int, np.ndarray] = {}
    for magnitude in (1, 4, 8, 16):
        vector = np.zeros((height, width, 2), dtype=np.float16)
        vector[..., 0] = magnitude
        motion_vectors[magnitude] = vector
    motion_zero_path = contract_dir / f"{name}.motion_zero.rg16f.bin"
    motion_zero.tofile(motion_zero_path)
    motion_paths: dict[int, Path] = {}
    for magnitude, vector in motion_vectors.items():
        path = contract_dir / f"{name}.motion_right{magnitude}.rg16f.bin"
        vector.tofile(path)
        motion_paths[magnitude] = path

    shifted = np.zeros_like(rgb8)
    shifted[:, 8:] = rgb8[:, :-8]
    shifted_linear = srgb_to_linear(shifted)
    shifted_rgba = np.concatenate(
        (shifted_linear, np.ones((height, width, 1), dtype=np.float32)), axis=2
    )
    shifted_path = contract_dir / f"{name}.shifted_right8.rgba16f.bin"
    shifted_rgba.astype(np.float16).tofile(shifted_path)
    return {
        "color": color,
        "depth_luma": depth_luma_path,
        "depth_flat": depth_flat_path,
        "motion_zero": motion_zero_path,
        **{f"motion_right{magnitude}": path for magnitude, path in motion_paths.items()},
        "shifted_color": shifted_path,
    }


def read_rgba16f(path: Path, width: int, height: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float16)
    expected = width * height * 4
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} FP16 values; expected {expected}")
    return values.astype(np.float32).reshape(height, width, 4)


def write_output_png(ffmpeg: str, raw: Path, destination: Path, width: int, height: int) -> np.ndarray:
    image = np.clip(read_rgba16f(raw, width, height)[..., :3], 0.0, 1.0)
    # Native outputs are linear-light RGBA16F. PNG viewers expect an sRGB
    # transfer curve; keep the returned array linear for numerical metrics,
    # and only encode the display preview here.
    encoded = np.rint(linear_to_srgb(image) * 255.0).astype(np.uint8)
    encode_rgb8(ffmpeg, encoded, destination)
    return image


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = left.astype(np.float32) - right.astype(np.float32)
    flat_left = left.reshape(-1).astype(np.float32)
    flat_right = right.reshape(-1).astype(np.float32)
    denominator = float(flat_left.std() * flat_right.std())
    correlation = float(((flat_left - flat_left.mean()) * (flat_right - flat_right.mean())).mean() / max(denominator, 1e-12))
    return {
        "mae": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.square(difference).mean())),
        "max_abs": float(np.abs(difference).max()),
        "changed_fraction_gt_1e-3": float((np.abs(difference) > 1e-3).mean()),
        "correlation": correlation,
        "mean_rgb_left": [float(value) for value in left.mean(axis=(0, 1))],
        "mean_rgb_right": [float(value) for value in right.mean(axis=(0, 1))],
    }


def run_one(
    harness: Path,
    color_frames: list[tuple[Path, int]],
    depth: Path,
    motion: Path,
    raw_output: Path,
    width: int,
    height: int,
) -> None:
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    run_harness(
        harness.resolve(),
        width,
        height,
        depth.resolve(),
        motion.resolve(),
        [(path.resolve(), reset) for path, reset in color_frames],
        raw_output.resolve(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True, help="local native dlss5_eval.exe")
    parser.add_argument("--input-dir", type=Path, default=Path("examples/assets/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/cases/native"))
    parser.add_argument("--work-dir", type=Path, default=Path("examples/.work"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    if not args.harness.is_file():
        parser.error(f"harness not found: {args.harness}")
    if args.width < 64 or args.height < 64:
        parser.error("native harness dimensions must be at least 64")

    ffmpeg = find_ffmpeg()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    normalized_dir = Path("examples/assets/normalized")
    condition_dir = Path("examples/assets/conditions")
    normalized_dir.mkdir(parents=True, exist_ok=True)
    condition_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not inputs:
        parser.error(f"no supported images found in {args.input_dir}")

    images: dict[str, dict[str, object]] = {}
    contracts: dict[str, dict[str, Path]] = {}
    arrays: dict[str, np.ndarray] = {}
    for source in inputs:
        name = source.stem.lower().replace(" ", "_")
        normalized_raw = args.work_dir / "normalized" / f"{name}.rgb8.bin"
        rgb8 = decode_rgb8(ffmpeg, source, normalized_raw, args.width, args.height)
        normalized_png = normalized_dir / f"{name}.png"
        encode_rgb8(ffmpeg, rgb8, normalized_png)
        linear = srgb_to_linear(rgb8)
        contracts[name] = write_contracts(args.work_dir, name, rgb8, linear, args.width, args.height)
        arrays[name] = linear
        depth_preview = np.rint(np.clip(1.0 - linear_luma(linear), 0.0, 1.0) * 255).astype(np.uint8)
        encode_rgb8(ffmpeg, np.repeat(depth_preview[..., None], 3, axis=2), condition_dir / f"{name}_depth_luma.png")
        images[name] = {
            "source": str(source.as_posix()),
            "source_sha256": sha256(source),
            "normalized": str(normalized_png.as_posix()),
            "shape": [args.height, args.width, 3],
            "color_contract": "sRGB source decoded to linear RGB, packed as RGBA16F with alpha=1",
        }

    flat_depth = np.ones((args.height, args.width), dtype=np.uint8) * 255
    encode_rgb8(ffmpeg, np.repeat(flat_depth[..., None], 3, axis=2), condition_dir / "depth_flat.png")

    results: dict[str, object] = {}
    def execute(name: str, frames: list[tuple[Path, int]], depth: Path, motion: Path) -> np.ndarray:
        raw = args.work_dir / "outputs" / f"{name}.rgba16f.bin"
        png = args.output_dir / f"{name}.png"
        run_one(args.harness, frames, depth, motion, raw, args.width, args.height)
        image = write_output_png(ffmpeg, raw, png, args.width, args.height)
        results[name] = {
            "output": str(png.as_posix()),
            "raw_contract": "examples/.work/" + raw.relative_to(args.work_dir).as_posix(),
            "shape": [args.height, args.width, 4],
            "range_rgb": [float(image.min()), float(image.max())],
        }
        return image

    generated: dict[str, np.ndarray] = {}
    for name, item in contracts.items():
        generated[f"{name}_static_luma"] = execute(
            f"{name}_static_luma",
            [(item["color"], 1), (item["color"], 0)],
            item["depth_luma"],
            item["motion_zero"],
        )

    for name in [key for key in ("blue_marble", "stone_texture", "portrait_cc0") if key in contracts]:
        item = contracts[name]
        generated[f"{name}_depth_flat"] = execute(
            f"{name}_depth_flat",
            [(item["color"], 1), (item["color"], 0)],
            item["depth_flat"],
            item["motion_zero"],
        )

    if "blue_marble" in contracts:
        item = contracts["blue_marble"]
        generated["blue_marble_motion_zero"] = execute(
            "blue_marble_motion_zero",
            [(item["color"], 1), (item["shifted_color"], 0)],
            item["depth_luma"],
            item["motion_zero"],
        )
        generated["blue_marble_motion_right8"] = execute(
            "blue_marble_motion_right8",
            [(item["color"], 1), (item["shifted_color"], 0)],
            item["depth_luma"],
            item["motion_right8"],
        )
        for magnitude in (1, 4, 16):
            generated[f"blue_marble_motion_right{magnitude}"] = execute(
                f"blue_marble_motion_right{magnitude}",
                [(item["color"], 1), (item["shifted_color"], 0)],
                item["depth_luma"],
                item[f"motion_right{magnitude}"],
            )

    if "blue_marble" in contracts and "scenic_landscape" in contracts:
        previous = contracts["blue_marble"]["color"]
        current = contracts["scenic_landscape"]["color"]
        current_item = contracts["scenic_landscape"]
        generated["history_blue_to_scenic"] = execute(
            "history_blue_to_scenic",
            [(previous, 1), (current, 0)],
            current_item["depth_luma"],
            current_item["motion_zero"],
        )
        generated["history_scenic_isolated"] = execute(
            "history_scenic_isolated",
            [(current, 1)],
            current_item["depth_luma"],
            current_item["motion_zero"],
        )

    def relative_metric(left_name: str, right_name: str) -> dict[str, object]:
        return {
            "left": left_name,
            "right": right_name,
            **compare(generated[left_name], generated[right_name]),
        }

    def difference_preview(left_name: str, right_name: str, output_name: str, amplify: float = 8.0) -> None:
        difference = np.abs(generated[left_name] - generated[right_name])
        preview = np.repeat(
            np.clip(difference.mean(axis=2) * amplify, 0.0, 1.0)[..., None],
            3,
            axis=2,
        )
        destination = args.output_dir / f"{output_name}.png"
        encode_rgb8(ffmpeg, np.rint(linear_to_srgb(preview) * 255.0).astype(np.uint8), destination)
        results[output_name] = {
            "output": str(destination.as_posix()),
            "preview": "absolute RGB difference, amplified before sRGB display encoding",
            "left": left_name,
            "right": right_name,
            "amplify": amplify,
        }

    metrics: dict[str, object] = {}
    for name, image in generated.items():
        source_name = name.split("_static_luma")[0] if "_static_luma" in name else None
        if source_name in arrays:
            metrics[f"{name}_versus_linear_input"] = compare(image, arrays[source_name])
    if "blue_marble_motion_zero" in generated and "blue_marble_motion_right8" in generated:
        metrics["motion_vector_effect"] = relative_metric("blue_marble_motion_zero", "blue_marble_motion_right8")
        difference_preview("blue_marble_motion_zero", "blue_marble_motion_right8", "blue_marble_motion_difference_x8")
        metrics["motion_vector_sweep"] = {
            str(magnitude): relative_metric("blue_marble_motion_zero", f"blue_marble_motion_right{magnitude}")
            for magnitude in (1, 4, 8, 16)
        }
    if "history_blue_to_scenic" in generated and "history_scenic_isolated" in generated:
        metrics["history_effect"] = relative_metric("history_blue_to_scenic", "history_scenic_isolated")
        difference_preview("history_blue_to_scenic", "history_scenic_isolated", "history_difference_x8")
    for name in ("blue_marble", "stone_texture", "portrait_cc0"):
        luma = f"{name}_static_luma"
        flat = f"{name}_depth_flat"
        if luma in generated and flat in generated:
            metrics[f"{name}_depth_effect"] = relative_metric(luma, flat)
    if "blue_marble_static_luma" in generated and "blue_marble_depth_flat" in generated:
        difference_preview("blue_marble_static_luma", "blue_marble_depth_flat", "blue_marble_depth_difference_x8")

    manifest = {
        "schema": "dlss5_image_cases_v1",
        "harness": args.harness.name,
        "dimensions": [args.width, args.height],
        "ffmpeg": Path(ffmpeg).name,
        "contract": {
            "color": "RGBA16F little-endian, linear RGB plus alpha=1",
            "png_preview": "native linear RGB converted to sRGB for display",
            "depth": "R32F reversed-Z proxy; depth_luma is synthetic 1-luma, depth_flat is 1.0",
            "motion": "RG16F; motion_rightN supplies +N.0 in the horizontal component",
            "temporal": "reset=1 seeds history; reset=0 evaluates with prior state",
        },
        "inputs": images,
        "outputs": results,
        "metrics": metrics,
        "interpretation": {
            "depth": "These depth maps are controlled proxies, not a depth-estimator result.",
            "motion": "The experiment measures sensitivity to the supplied vector; it does not assume the sign convention is correct for every host.",
            "history": "The blue-to-scenic case intentionally changes content to expose temporal carry-over.",
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
