"""Self-test and inspection CLI for the split semantic graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .graph import DLSS5Graph
from .ops import (assemble_pre_front_feature_lanes, build_pre_front_sass_candidate, cct_cubic_silu)
from .weights import DLSS5WeightMap

def _self_test() -> None:
    # A small smoke test for the graph and the final ControlMask dataflow.
    # float16 keeps this CPU-only test reasonably small.
    probe = torch.tensor([-4.0, -2.0, 0.0, 1.0, 4.0])
    expected = torch.tensor([0.0, -0.447265625, 0.0, 1.285888671875, 7.15625])
    assert torch.allclose(cct_cubic_silu(probe), expected, atol=1e-5, rtol=0.0)
    model = DLSS5Graph(color_channels=3, history_channels=3, motion_channels=2, output_channels=3).half()
    color = torch.randn(1, 3, 64, 64).half()
    history = torch.randn_like(color)
    motion = torch.zeros(1, 2, 64, 64).half()
    mask = torch.ones(1, 1, 64, 64).half()
    generated = torch.zeros(1, 3, 2, 8, 8).half()
    texture = torch.zeros(1, 4, 2, 8, 8).half()
    packed = assemble_pre_front_feature_lanes(generated, texture)
    assert packed.shape == (1, 15, 8, 8)
    assert torch.equal(packed[:, 6], torch.ones(1, 8, 8).half())
    candidate = build_pre_front_sass_candidate(color)
    assert candidate.shape == (1, 15, 64, 64)
    assert torch.isfinite(candidate.float()).all()
    with torch.no_grad():
        y = model(color, history, motion)
        z = model(color, history, motion, mask)
        component_front = model(
            color,
            history,
            motion,
            pre_front_generated_lanes=torch.zeros(1, 3, 2, 64, 64).half(),
            pre_front_texture_lanes=torch.zeros(1, 4, 2, 64, 64).half(),
        )
    assert y.shape == color.shape == z.shape
    assert component_front.shape == color.shape
    assert torch.isfinite(y.float()).all()
    assert torch.isfinite(z.float()).all()
    assert torch.isfinite(component_front.float()).all()
    print("DLSS5Graph smoke test passed", tuple(y.shape))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if args.weights:
        weight_map = DLSS5WeightMap.from_file(args.weights)
        print(json.dumps(weight_map.summary(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if args.weights:
        weight_path = args.weights / "WEIGHTS_HT.bin" if args.weights.is_dir() else args.weights
        weight_map = DLSS5WeightMap.from_file(weight_path)
        print(json.dumps(weight_map.summary(), indent=2))

if __name__ == "__main__":
    main()
