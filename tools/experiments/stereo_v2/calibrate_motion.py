"""Known plane translation: compare zero/positive/negative MV on observed overlap."""

import argparse
import json
from pathlib import Path

import numpy as np
from run import ROOT, read_native_output, run_harness, scene, write_rgba16f


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness",type=Path,required=True)
    args = p.parse_args()
    work = ROOT / "examples/.work/stereo_v2/calibration"
    work.mkdir(parents=True,exist_ok=True)
    left, _, _, _ = scene("stone",8)
    current = left.copy()
    current[:,:-8] = left[:,8:]
    current[:,-8:] = left[:,-1:]
    lp, cp = work/"left.bin", work/"current.bin"
    write_rgba16f(lp,left)
    write_rgba16f(cp,current)
    dp,zp = work/"depth.bin",work/"zero.bin"
    np.ones((256,256),np.float32).tofile(dp)
    np.zeros((256,256,2),np.float16).tofile(zp)
    scores = {}
    for dx in (0,8,-8):
        mv = np.zeros((256,256,2),np.float16)
        mv[...,0] = dx
        mp,op = work/f"mv{dx}.bin", work/f"out{dx}.bin"
        mv.tofile(mp)
        run_harness(args.harness.resolve(),256,256,dp,zp,[(lp,1),(cp,0)],op,frame_motion=[zp,mp])
        prediction = read_native_output(op,256,256)
        delta = prediction[8:-8,8:-16]-current[8:-8,8:-16]
        scores[str(dx)] = {"overlap_mae":float(abs(delta).mean()),"overlap_rmse":float(np.sqrt((delta**2).mean()))}
    report = {"shift_left_pixels":8,"geometric_current_to_previous_x":8,"scores":scores,
              "caveat":"photometric score includes neural appearance changes; ranking alone cannot prove native MV convention"}
    output = ROOT/"examples/cases/stereo_v2/layered/motion_calibration.json"
    output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
