"""Compare geometric stereo hole filling with a native DLSS5 pass.

The experiment follows ``docs/PLAN.md``:

* estimate a relative depth map with Distill-Any-Depth small;
* forward-splat the left image into a synthetic right eye with a Z-buffer;
* build ``RightWarpedColor``, ``HoleMask``, ``RightMVec`` and a simple nearest
  pixel prefill;
* run the native harness as ``Left(reset=1) -> Right(reset=0)``;
* compare the simple prefill and DLSS output, including a planar oracle case
  whose complete right-eye image is known.

The stock native harness accepts only a constant ``--mask`` value, so the
default DLSS run uses ``mask=0`` everywhere. A locally patched harness can be
selected with ``--spatial-mask`` to pass ``valid=255/hole=0`` as an R8 file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = REPO_ROOT / "tools"
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from dlss5_fp16_harness_probe import run_harness  # noqa: E402
from experiments.run_dlss5_image_cases import encode_rgb8, linear_to_srgb, srgb_to_linear  # noqa: E402


def read_image(path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    image = Image.open(path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    rgb8 = np.asarray(image, dtype=np.uint8)
    return rgb8, srgb_to_linear(rgb8)


def read_depth(path: Path, width: int, height: int) -> np.ndarray:
    depth = np.fromfile(path, dtype=np.float32)
    if depth.size != width * height:
        raise ValueError(f"{path} contains {depth.size} depth values; expected {width * height}")
    return np.clip(depth.reshape(height, width), 0.0, 1.0)


def write_rgba16f(path: Path, rgb: np.ndarray) -> None:
    rgba = np.concatenate((rgb.astype(np.float32), np.ones((*rgb.shape[:2], 1), dtype=np.float32)), axis=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgba.astype(np.float16).tofile(path)


def write_r32f(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.astype(np.float32).tofile(path)


def write_rg16f(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value.astype(np.float16).tofile(path)


def write_control_mask(path: Path, hole_mask: np.ndarray) -> None:
    """Write the PLAN convention: valid pixels=255, holes=0."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.where(hole_mask, 0, 255).astype(np.uint8).tofile(path)


def read_native_output(path: Path, width: int, height: int) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float16)
    expected = width * height * 4
    if values.size != expected:
        raise ValueError(f"{path} contains {values.size} FP16 values; expected {expected}")
    return values.astype(np.float32).reshape(height, width, 4)[..., :3]


def forward_splat(left: np.ndarray, depth: np.ndarray, max_disparity: float) -> dict[str, np.ndarray]:
    height, width, _ = left.shape
    disparity = 0.5 + max_disparity * depth
    right = np.zeros_like(left)
    right_depth = np.full((height, width), -np.inf, dtype=np.float32)
    source_map = np.full((height, width, 2), -1, dtype=np.int32)
    for y in range(height):
        for x in range(width):
            target_x = int(np.rint(x - disparity[y, x]))
            if target_x < 0 or target_x >= width:
                continue
            # Larger normalized inverse depth is nearer. Keep the front-most
            # source when multiple left pixels land on one right pixel.
            if depth[y, x] >= right_depth[y, target_x]:
                right[y, target_x] = left[y, x]
                right_depth[y, target_x] = depth[y, x]
                source_map[y, target_x] = (y, x)
    valid = source_map[..., 0] >= 0
    hole_mask = ~valid

    # Baseline: nearest valid pixel in the same scanline. This is deliberately
    # simple and exposes the stretched-edge artifact that DLSS might improve.
    prefilled = right.copy()
    for y in range(height):
        valid_x = np.flatnonzero(valid[y])
        if valid_x.size == 0:
            prefilled[y] = left[y]
            continue
        holes_x = np.flatnonzero(~valid[y])
        nearest = valid_x[np.abs(valid_x[None, :] - holes_x[:, None]).argmin(axis=1)]
        prefilled[y, holes_x] = right[y, nearest]

    motion = np.zeros((height, width, 2), dtype=np.float32)
    target_x = np.arange(width, dtype=np.float32)[None, :]
    motion[..., 0] = np.where(valid, source_map[..., 1].astype(np.float32) - target_x, 0.0)
    return {
        "disparity": disparity,
        "warped": right,
        "prefilled": prefilled,
        "right_depth": np.where(valid, right_depth, 0.0),
        "source_map": source_map,
        "hole_mask": hole_mask,
        "motion": motion,
        "valid": valid,
    }


def metric(left: np.ndarray, right: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    difference = left.astype(np.float32) - right.astype(np.float32)
    if mask is not None:
        difference = difference[mask]
    return {
        "mae": float(np.abs(difference).mean()) if difference.size else 0.0,
        "rmse": float(np.sqrt(np.square(difference).mean())) if difference.size else 0.0,
        "max_abs": float(np.abs(difference).max()) if difference.size else 0.0,
    }


def encode_preview(encode_fn, image: np.ndarray, path: Path) -> None:
    encoded = np.rint(linear_to_srgb(np.clip(image, 0.0, 1.0)) * 255.0).astype(np.uint8)
    encode_fn(encoded, path)


def save_gray(encode_fn, image: np.ndarray, path: Path, amplify: float = 1.0) -> None:
    value = np.clip(image.astype(np.float32) * amplify, 0.0, 1.0)
    encode_fn(np.repeat(np.rint(value * 255.0).astype(np.uint8)[..., None], 3, axis=2), path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("examples/assets/normalized"))
    parser.add_argument("--depth-dir", type=Path, default=Path("examples/assets/depth/distill_any_depth_small"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/cases/stereo_inpainting"))
    parser.add_argument("--work-dir", type=Path, default=Path("examples/.work/stereo_inpainting"))
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--max-disparity", type=float, default=16.0)
    parser.add_argument("--plane-shift", type=int, default=16)
    parser.add_argument(
        "--spatial-mask",
        action="store_true",
        help="pass --mask-file to a locally patched harness instead of constant mask=0",
    )
    args = parser.parse_args()
    if not args.harness.is_file():
        parser.error(f"harness not found: {args.harness}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    # Reuse the image-case encoder without importing the native runner as a
    # command-line side effect.
    from experiments.run_dlss5_image_cases import find_ffmpeg

    ffmpeg = find_ffmpeg()

    def encode(image: np.ndarray, path: Path) -> None:
        encode_rgb8(ffmpeg, image, path)

    records: dict[str, object] = {}
    for name in ("blue_marble", "portrait_cc0"):
        image_path = args.input_dir / f"{name}.png"
        depth_path = args.depth_dir / f"{name}.r32f.bin"
        if not image_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"missing image/depth pair: {image_path} / {depth_path}")
        _, left = read_image(image_path, args.width, args.height)
        depth = read_depth(depth_path, args.width, args.height)
        geometry = forward_splat(left, depth, args.max_disparity)
        prefix = f"{name}_predicted_depth"
        left_path = args.work_dir / f"{prefix}.left.rgba16f.bin"
        right_path = args.work_dir / f"{prefix}.right_prefilled.rgba16f.bin"
        depth_right_path = args.work_dir / f"{prefix}.right_depth.r32f.bin"
        motion_path = args.work_dir / f"{prefix}.motion.rg16f.bin"
        mask_path = args.work_dir / f"{prefix}.control_mask.r8.bin"
        write_rgba16f(left_path, left)
        write_rgba16f(right_path, geometry["prefilled"])
        write_r32f(depth_right_path, geometry["right_depth"])
        write_rg16f(motion_path, geometry["motion"])
        write_control_mask(mask_path, geometry["hole_mask"])
        native_raw = args.work_dir / f"{prefix}.dlss5.rgba16f.bin"
        mask_args = ["--mask-file", str(mask_path.resolve())] if args.spatial_mask else ["--mask", "0"]
        run_harness(
            args.harness.resolve(),
            args.width,
            args.height,
            depth_right_path.resolve(),
            motion_path.resolve(),
            [(left_path.resolve(), 1), (right_path.resolve(), 0)],
            native_raw.resolve(),
            extra_args=mask_args,
        )
        dlss = read_native_output(native_raw, args.width, args.height)
        encode_preview(encode, left, args.output_dir / f"{prefix}_left.png")
        encode_preview(encode, geometry["warped"], args.output_dir / f"{prefix}_right_warped.png")
        encode_preview(encode, geometry["prefilled"], args.output_dir / f"{prefix}_right_simple.png")
        encode_preview(encode, dlss, args.output_dir / f"{prefix}_right_dlss5.png")
        save_gray(encode, geometry["hole_mask"].astype(np.float32), args.output_dir / f"{prefix}_hole_mask.png")
        save_gray(encode, np.abs(dlss - geometry["prefilled"]).mean(axis=2), args.output_dir / f"{prefix}_dlss5_minus_simple_x8.png", 8.0)
        records[prefix] = {
            "input": str(image_path.as_posix()),
            "depth": str(depth_path.as_posix()),
            "hole_fraction": float(geometry["hole_mask"].mean()),
            "valid_fraction": float(geometry["valid"].mean()),
            "disparity_range": [float(geometry["disparity"].min()), float(geometry["disparity"].max())],
            "simple_prefill_vs_warped": metric(geometry["prefilled"], geometry["warped"], geometry["hole_mask"]),
            "dlss5_vs_simple": metric(dlss, geometry["prefilled"]),
            "dlss5_vs_simple_holes": metric(dlss, geometry["prefilled"], geometry["hole_mask"]),
            "dlss5_vs_simple_valid": metric(dlss, geometry["prefilled"], geometry["valid"]),
            "output_range": [float(dlss.min()), float(dlss.max())],
            "mask_mode": "spatial valid=255/hole=0" if args.spatial_mask else "constant 0 (spatial HoleMask not bound)",
        }

    # A controlled planar case supplies a known complete right-eye reference.
    # It tests whether DLSS can undo the edge hole introduced by a pure shift;
    # it does not claim to reconstruct genuinely unseen stereo background.
    plane_name = "stone_texture_plane_oracle"
    image_path = args.input_dir / "stone_texture.png"
    _, left = read_image(image_path, args.width, args.height)
    plane_depth = np.full((args.height, args.width), 0.5, dtype=np.float32)
    geometry = forward_splat(left, plane_depth, max(0.0, float(args.plane_shift - 0.5)) / 0.5)
    ground_truth = np.roll(left, -args.plane_shift, axis=1)
    left_path = args.work_dir / f"{plane_name}.left.rgba16f.bin"
    right_path = args.work_dir / f"{plane_name}.right_prefilled.rgba16f.bin"
    depth_path = args.work_dir / f"{plane_name}.right_depth.r32f.bin"
    motion_path = args.work_dir / f"{plane_name}.motion.rg16f.bin"
    mask_path = args.work_dir / f"{plane_name}.control_mask.r8.bin"
    write_rgba16f(left_path, left)
    write_rgba16f(right_path, geometry["prefilled"])
    write_r32f(depth_path, geometry["right_depth"])
    write_rg16f(motion_path, geometry["motion"])
    write_control_mask(mask_path, geometry["hole_mask"])
    native_raw = args.work_dir / f"{plane_name}.dlss5.rgba16f.bin"
    mask_args = ["--mask-file", str(mask_path.resolve())] if args.spatial_mask else ["--mask", "0"]
    run_harness(
        args.harness.resolve(), args.width, args.height, depth_path.resolve(), motion_path.resolve(),
        [(left_path.resolve(), 1), (right_path.resolve(), 0)], native_raw.resolve(),
        extra_args=mask_args,
    )
    dlss = read_native_output(native_raw, args.width, args.height)
    native_mask255_raw = args.work_dir / f"{plane_name}.dlss5_mask255.rgba16f.bin"
    run_harness(
        args.harness.resolve(), args.width, args.height, depth_path.resolve(), motion_path.resolve(),
        [(left_path.resolve(), 1), (right_path.resolve(), 0)], native_mask255_raw.resolve(),
        extra_args=["--mask", "255"],
    )
    dlss_mask255 = read_native_output(native_mask255_raw, args.width, args.height)
    encode_preview(encode, left, args.output_dir / f"{plane_name}_left.png")
    encode_preview(encode, geometry["warped"], args.output_dir / f"{plane_name}_right_warped.png")
    encode_preview(encode, geometry["prefilled"], args.output_dir / f"{plane_name}_right_simple.png")
    encode_preview(encode, ground_truth, args.output_dir / f"{plane_name}_right_ground_truth.png")
    encode_preview(encode, dlss, args.output_dir / f"{plane_name}_right_dlss5.png")
    encode_preview(encode, dlss_mask255, args.output_dir / f"{plane_name}_right_dlss5_mask255.png")
    save_gray(encode, geometry["hole_mask"].astype(np.float32), args.output_dir / f"{plane_name}_hole_mask.png")
    save_gray(encode, np.abs(geometry["prefilled"] - ground_truth).mean(axis=2), args.output_dir / f"{plane_name}_simple_error_x8.png", 8.0)
    save_gray(encode, np.abs(dlss - ground_truth).mean(axis=2), args.output_dir / f"{plane_name}_dlss5_error_x8.png", 8.0)
    records[plane_name] = {
        "input": str(image_path.as_posix()),
        "ground_truth": "np.roll(left, -plane_shift) for a horizontally periodic planar texture",
        "plane_shift": args.plane_shift,
        "hole_fraction": float(geometry["hole_mask"].mean()),
        "simple_error_all": metric(geometry["prefilled"], ground_truth),
        "simple_error_holes": metric(geometry["prefilled"], ground_truth, geometry["hole_mask"]),
        "dlss5_error_all": metric(dlss, ground_truth),
        "dlss5_error_holes": metric(dlss, ground_truth, geometry["hole_mask"]),
        "selected_mask_vs_constant255": metric(dlss, dlss_mask255),
        "hole_mae_improvement": metric(geometry["prefilled"], ground_truth, geometry["hole_mask"])["mae"] - metric(dlss, ground_truth, geometry["hole_mask"])["mae"],
        "mask_mode": "spatial valid=255/hole=0" if args.spatial_mask else "constant 0 (spatial HoleMask not bound)",
        "interpretation": "known planar translation proxy, not genuinely unseen disocclusion ground truth",
    }

    manifest = {
        "schema": "dlss5_stereo_inpainting_v1",
        "harness": args.harness.name,
        "dimensions": [args.width, args.height],
        "depth_model": "Distill-Any-Depth-Small-hf for predicted_depth cases",
        "geometry": {
            "target_x": "round(source_x - (0.5 + max_disparity * normalized_depth))",
            "z_buffer": "larger normalized inverse-depth value wins",
            "motion": "source_x - target_x in right-eye pixel coordinates; holes use zero",
            "simple_fill": "nearest valid source pixel on each scanline",
        },
        "records": records,
        "limitations": [
            "relative monocular depth is a disparity proxy, not metric stereo depth",
            "the default harness accepts constant --mask only; --spatial-mask requires the locally patched mask-file harness",
            "unknown disoccluded background has no image-only ground truth; only the planar oracle has a known reference",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
