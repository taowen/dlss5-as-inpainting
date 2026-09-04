"""Run the official Distill-Any-Depth small checkpoint on example images.

The model produces relative monocular depth, not metric depth. This tool
normalizes each image independently to ``[0, 1]`` and records that convention
so the stereo experiment can use it as a controlled disparity proxy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_ID = "xingyang1/Distill-Any-Depth-Small-hf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--input-dir", type=Path, default=Path("examples/assets/normalized"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/assets/depth/distill_any_depth_small"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).eval().to(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        parser.error(f"no images found in {args.input_dir}")

    records: dict[str, object] = {}
    for image_path in images:
        image = Image.open(image_path).convert("RGB").resize((args.width, args.height), Image.Resampling.LANCZOS)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        post_processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(args.height, args.width)],
        )
        depth = post_processed[0]["predicted_depth"].float().squeeze()
        low = depth.amin()
        high = depth.amax()
        normalized = ((depth - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
        normalized_np = normalized.detach().cpu().numpy().astype(np.float32)
        name = image_path.stem.lower().replace(" ", "_")
        raw_path = args.output_dir / f"{name}.r32f.bin"
        png_path = args.output_dir / f"{name}.png"
        normalized_np.tofile(raw_path)
        Image.fromarray(np.rint(normalized_np * 255.0).astype(np.uint8), mode="L").save(png_path)
        records[name] = {
            "input": str(image_path.as_posix()),
            "input_sha256": sha256(image_path),
            "raw": str(raw_path.as_posix()),
            "preview": str(png_path.as_posix()),
            "shape": [args.height, args.width],
            "normalization": "per-image min/max of predicted_depth; larger value is treated as nearer for disparity proxy",
            "predicted_depth_min": float(low.cpu()),
            "predicted_depth_max": float(high.cpu()),
        }

    manifest = {
        "schema": "distill_any_depth_small_v1",
        "model": args.model,
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dimensions": [args.width, args.height],
        "records": records,
        "warning": "relative monocular depth; not metric depth and not a proof of native DLSS depth semantics",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path.resolve()), "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
