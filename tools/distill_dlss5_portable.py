"""Distill native DLSS 5 input/output pairs into a portable PyTorch model.

Example (native RGBA16F files are local evidence, not redistributed assets)::

    python tools/distill_dlss5_portable.py \
      --pair native/full_checker.rgba16f.bin=native/full_checker.out.rgba16f.bin \
      --pair native/full_red_ramp.rgba16f.bin=native/full_red_ramp.out.rgba16f.bin \
      --output DLSS5-extracted/dlss5_pytorch_portable_v1.pt \
      --device cuda --steps 400

The resulting checkpoint contains only ordinary convolutional weights.  It is
an approximation/distillation artifact, not a replacement for the native
CUBIN and not a bit-exact reconstruction of its private front-end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from torch import Tensor

from dlss5_portable import DLSS5PortableModel, PORTABLE_FORMAT


def read_rgba16f(path: Path, width: int, height: int) -> Tensor:
    payload = path.read_bytes()
    expected = width * height * 8
    if len(payload) != expected:
        raise ValueError(f"{path} has {len(payload)} bytes; expected {expected}")
    values = torch.frombuffer(bytearray(payload), dtype=torch.float16).clone()
    return values.reshape(height, width, 4).permute(2, 0, 1).contiguous()


def parse_pair(spec: str) -> tuple[Path, Path]:
    if "=" not in spec:
        raise ValueError(f"--pair must be INPUT=TARGET, got {spec!r}")
    left, right = (Path(value).expanduser() for value in spec.split("=", 1))
    if not left.is_file() or not right.is_file():
        raise FileNotFoundError(f"missing pair: {left} / {right}")
    return left, right


def crop_pair(source: Tensor, target: Tensor, crop_size: int, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    if crop_size <= 0 or crop_size >= source.shape[-1]:
        return source, target
    height, width = source.shape[-2:]
    top = int(torch.randint(0, height - crop_size + 1, (), generator=generator).item())
    left = int(torch.randint(0, width - crop_size + 1, (), generator=generator).item())
    return (
        source[..., top : top + crop_size, left : left + crop_size],
        target[..., top : top + crop_size, left : left + crop_size],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", action="append", required=True, help="RGBA16F INPUT=TARGET; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--identity-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=5080)
    args = parser.parse_args()
    if args.steps < 0 or args.lr <= 0 or args.identity_weight < 0:
        parser.error("steps must be non-negative, lr must be positive, and identity-weight non-negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    pairs = [parse_pair(spec) for spec in args.pair]
    sources: list[Tensor] = []
    targets: list[Tensor] = []
    for source_path, target_path in pairs:
        source = read_rgba16f(source_path, args.width, args.height).float()
        target = read_rgba16f(target_path, args.width, args.height).float()
        if source.shape != target.shape:
            raise ValueError(f"pair shape mismatch: {source_path} {source.shape} vs {target_path} {target.shape}")
        if source.shape[0] < 3 or target.shape[0] < 3:
            raise ValueError("RGBA16F pair must contain RGB")
        sources.append(source[:3].clamp(0.0, 1.0))
        targets.append(target[:3].clamp(0.0, 1.0))

    model = DLSS5PortableModel(hidden_channels=args.hidden_channels).to(device=device, dtype=torch.float32)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    loss_history: list[float] = []

    for _step in range(args.steps):
        batch_source: list[Tensor] = []
        batch_target: list[Tensor] = []
        for source, target in zip(sources, targets):
            cropped_source, cropped_target = crop_pair(source, target, args.crop_size, generator)
            batch_source.append(cropped_source)
            batch_target.append(cropped_target)
        source_batch = torch.stack(batch_source).to(device=device)
        target_batch = torch.stack(batch_target).to(device=device)
        prediction = model(source_batch)
        loss = (prediction - target_batch).abs().mean()
        if args.identity_weight:
            loss = loss + args.identity_weight * (prediction - source_batch).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite distillation loss at step {_step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.inference_mode():
        source_batch = torch.stack(sources).to(device=device)
        target_batch = torch.stack(targets).to(device=device)
        prediction = model(source_batch)
        distilled_mae = float((prediction - target_batch).abs().mean().cpu())
        identity_mae = float((source_batch - target_batch).abs().mean().cpu())

    state_dict = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    checkpoint = {
        "format": PORTABLE_FORMAT,
        "model_kwargs": {
            "color_channels": model.color_channels,
            "depth_channels": model.depth_channels,
            "history_channels": model.history_channels,
            "motion_channels": model.motion_channels,
            "hidden_channels": model.hidden_channels,
            "residual_scale": float(model.residual_scale),
            "clamp_output": model.clamp_output,
        },
        # Keep the checkpoint relocatable and free of machine-specific paths.
        "source_pairs": [[left.name, right.name] for left, right in pairs],
        "steps": args.steps,
        "learning_rate": args.lr,
        "identity_weight": args.identity_weight,
        "identity_mae": identity_mae,
        "distilled_mae": distilled_mae,
        "loss_history": loss_history,
        "warning": "portable native distillation; not bit-exact CUBIN recovery",
        "state_dict": state_dict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "output": str(args.output.resolve()),
        "bytes": args.output.stat().st_size,
        "sha256": digest,
        "pairs": len(pairs),
        "steps": args.steps,
        "identity_mae": identity_mae,
        "distilled_mae": distilled_mae,
        "last_loss": loss_history[-1] if loss_history else None,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
