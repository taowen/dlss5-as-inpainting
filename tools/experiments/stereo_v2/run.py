"""Layered occlusion oracle; hidden background is used only by the evaluator."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))
from dlss5.stereo.geometry import splat, fill_scanlines
from dlss5_fp16_harness_probe import run_harness
from experiments.run_dlss5_image_cases import linear_to_srgb, srgb_to_linear
from experiments.run_stereo_inpainting_cases import write_rgba16f, read_native_output


def scene(kind, gap, size=256):
    y, x = np.mgrid[:size, :size + 32]
    if kind == "smooth":
        bg = np.stack((0.15 + x / 1000, 0.2 + y / 900, 0.3 + (x+y)/1600), -1).astype(np.float32)
    elif kind == "stripes":
        a = ((x//7 + y//19) % 2).astype(np.float32)
        bg = np.stack((0.15 + a*.4, .2 + a*.15, .1 + a*.25), -1)
    else:
        im = Image.open(ROOT / "examples/assets/input/stone_texture.jpg").convert("RGB").resize((size+32,size))
        bg = srgb_to_linear(np.asarray(im))
    left = bg[:, :size].copy()
    left[48:208, 96:152] = [0.65, 0.06, 0.1]
    disparity = np.full((size, size), 4, np.float32)
    disparity[48:208,96:152] = 4+gap
    right = bg[:,4:4+size].copy()
    right[48:208,96-4-gap:152-4-gap] = [0.65, .06, .1]
    internal = np.zeros((size,size), bool)
    internal[48:208,152-4-gap:152-4] = True
    return left, disparity, right, internal


def save(path, data):
    if data.ndim == 2:
        Image.fromarray((data*255).astype(np.uint8)).save(path)
    else:
        Image.fromarray(np.rint(linear_to_srgb(data)*255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/cases/stereo_v2/layered")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    work = ROOT / "examples/.work/stereo_v2"
    work.mkdir(parents=True, exist_ok=True)
    zero = work / "zero.rg16f.bin"
    np.zeros((256,256,2),np.float16).tofile(zero)
    depth = work / "depth.r32f.bin"
    np.ones((256,256),np.float32).tofile(depth)
    records = []
    for kind in ("smooth", "stripes", "stone"):
        for gap in (8,16):
            name = f"{kind}_{gap}"
            target = args.output / name
            target.mkdir(exist_ok=True)
            left, disp, truth, internal = scene(kind,gap)
            warped, valid, z, sx, mv = splat(left, disp)
            assert not valid[internal].any()
            assert np.array_equal(warped[valid], truth[valid])
            assert np.all((np.arange(256)[None,:]+mv[...,0])[valid] == sx[valid])
            save(target / "left.png", left)
            save(target / "truth.png", truth)
            save(target / "internal_holes.png", internal)
            save(target / "warped.png", warped)
            left_path = work / f"{name}_left.bin"
            write_rgba16f(left_path,left)
            motion = work / f"{name}_mv.bin"
            mv.astype(np.float16).tofile(motion)
            baselines = {"nearest":fill_scanlines(warped,valid,z), "background":fill_scanlines(warped,valid,z,True)}
            images = dict(baselines)
            for method, prefill in baselines.items():
                current = work / f"{name}_{method}.bin"
                write_rgba16f(current,prefill)
                # Reset control separates single-frame carrier effects from cross-eye history.
                for mode in ("history", "reset"):
                    output = work / f"{name}_{method}_{mode}.bin"
                    frames = [(left_path,1),(current,0)] if mode == "history" else [(current,1)]
                    motions = [zero,motion] if mode == "history" else [zero]
                    run_harness(args.harness.resolve(),256,256,depth,zero,frames,output,frame_motion=motions)
                    candidate = read_native_output(output,256,256)
                    if not np.isfinite(candidate).all():
                        raise RuntimeError("nonfinite native output")
                    images[f"{method}_{mode}_raw"] = candidate
                    images[f"{method}_{mode}_local"] = np.where(internal[...,None],candidate,prefill)
            scores = {}
            for method, image in images.items():
                save(target / f"{method}.png",image)
                scores[method] = {}
                for region, mask in (("internal", internal),("visible",valid),("boundary",~valid & ~internal)):
                    delta = image[mask]-truth[mask]
                    scores[method][region] = {"mae":float(abs(delta).mean()),"rmse":float(np.sqrt((delta**2).mean()))}
            records.append({"case":name,"internal_pixels":int(internal.sum()),"scores":scores})
            print(name,json.dumps({m:round(s["internal"]["mae"],6) for m,s in scores.items()}),flush=True)
    report = {"schema":"layered_stereo_v2", "harness_sha256":hashlib.sha256(args.harness.read_bytes()).hexdigest(),
              "geometry":"background disparity 4, foreground disparity 4+gap; independent two-layer right render",
              "protocol":"left reset with zero motion, right non-reset with per-frame geometric motion; reset control uses current only",
              "limitations":["whole carrier includes DLAA and Feature18; no isolated Feature18 attribution", "synthetic opaque rectangles; no generalization claim"],"records":records}
    (args.output / "metrics.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
