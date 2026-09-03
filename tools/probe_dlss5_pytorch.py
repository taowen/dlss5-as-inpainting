"""Run the translated DLSS 5 graph and report where values become non-finite."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from dlss5_pytorch import DLSS5Graph


def tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": finite_count,
        "elements": value.numel(),
    }
    if finite_count:
        selected = value.detach()[finite]
        result.update(
            min=float(selected.min().item()),
            max=float(selected.max().item()),
            mean=float(selected.float().mean().item()),
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("DLSS5-extracted"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--no-fp8-emulation", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runtime_probe_output/pytorch_probe.json"))
    args = parser.parse_args()

    model, weight_map = DLSS5Graph.with_weight_map(args.weights, load_known=True)
    if not args.no_fp8_emulation:
        model.enable_fp8_emulation()
    dtype = getattr(torch, args.dtype)
    model = model.eval().to(device=args.device, dtype=dtype)
    trace: list[dict[str, Any]] = []
    first_nonfinite: dict[str, Any] | None = None

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            nonlocal first_nonfinite
            summaries = [tensor_summary(item) for item in tensors(output)]
            entry = {"module": name, "outputs": summaries}
            trace.append(entry)
            if first_nonfinite is None and any(item["finite"] != item["elements"] for item in summaries):
                first_nonfinite = entry

        return hook

    handles = []
    for name, module in model.named_modules():
        if name and not any(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))

    image = torch.linspace(
        0.0,
        1.0,
        3 * args.size * args.size,
        device=args.device,
        dtype=torch.float32,
    ).to(dtype=dtype).reshape(1, 3, args.size, args.size)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(rgb=image)
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    report = {
        "torch": torch.__version__,
        "device": str(args.device),
        "gpu": torch.cuda.get_device_name() if args.device.startswith("cuda") else None,
        "compute_dtype": args.dtype,
        "weights": str(args.weights.resolve()),
        "weight_records": len(weight_map.records),
        "loaded_entries": len(model.weight_report["loaded"]),
        "metadata_entries": len(model.weight_report["metadata"]),
        "skipped_entries": len(model.weight_report["skipped"]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "fp8_emulation": not args.no_fp8_emulation,
        "input_shape": list(image.shape),
        "output": tensor_summary(output),
        "elapsed_seconds": elapsed,
        "peak_memory_mib": (
            torch.cuda.max_memory_allocated() / 1048576 if args.device.startswith("cuda") else None
        ),
        "first_nonfinite": first_nonfinite,
        "trace": trace,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "trace"}, indent=2))

    for handle in handles:
        handle.remove()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
