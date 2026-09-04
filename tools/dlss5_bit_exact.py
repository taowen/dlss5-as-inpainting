"""PyTorch-facing carrier for bit-exact DLSS5 inference.

This module intentionally keeps the native CUBIN carrier in the loop.  A
re-written ``nn.Module`` made from ordinary PyTorch operators is useful for a
portable approximation, but it cannot promise equality with proprietary
FP8/QMMA/SASS execution.  ``DLSS5BitExactCarrier`` exposes the native carrier
as an inference-only ``nn.Module`` so callers can still use normal tensor
inputs and compare raw FP16 output bytes.

The carrier is stateful: the native DLSS feature owns temporal history, so a
sequence must use one instance and call it once per frame.  Inputs are NCHW
RGB FP16 tensors with batch size one.  The native contract uses RGBA16F, so an
exact half-one alpha channel is appended.  Output is NCHW RGBA FP16.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
from pathlib import Path

import torch
from torch import Tensor, nn


class DLSS5BitExactCarrier(nn.Module):
    """Run the pinned native DLSS5 carrier while presenting a PyTorch API.

    Parameters
    ----------
    harness:
        Path to ``dlss5_eval.exe``.  Its parent directory must contain the
        same NGX/ReShade/DLSS5 runtime used for the native golden run.
    width, height:
        Fixed render/output dimensions for the native feature.
    depth, motion:
        Optional raw contract files.  Defaults are constant depth 1.0 FP32
        and zero motion RG16F, matching the repository's carrier probe.
    reversed_depth:
        Must match the native golden run.  The front-mutation evidence uses
        ``False``; the GUI converter's production path may use ``True``.
    workdir:
        Optional directory for the temporary raw contract files.
    """

    def __init__(
        self,
        harness: str | Path,
        *,
        width: int,
        height: int,
        depth: str | Path | None = None,
        motion: str | Path | None = None,
        reversed_depth: bool = False,
        workdir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.harness = Path(harness).resolve()
        if not self.harness.is_file():
            raise FileNotFoundError(self.harness)
        self.width = int(width)
        self.height = int(height)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        self.reversed_depth = bool(reversed_depth)
        self._owned_workdir = tempfile.TemporaryDirectory(
            prefix="dlss5-pytorch-", dir=str(workdir) if workdir else None
        )
        self._workdir = Path(self._owned_workdir.name)
        self._depth = self._copy_or_create_contract(depth, "depth.r32f.bin", self._depth_bytes())
        self._motion = self._copy_or_create_contract(motion, "motion.rg16f.bin", self._motion_bytes())
        self._process: subprocess.Popen[str] | None = None
        self._frame = 0
        self._lock = threading.RLock()

    def _depth_bytes(self) -> bytes:
        return torch.ones((self.width * self.height,), dtype=torch.float32).numpy().tobytes()

    def _motion_bytes(self) -> bytes:
        return torch.zeros((self.width * self.height * 2,), dtype=torch.float16).numpy().tobytes()

    def _copy_or_create_contract(self, source: str | Path | None, name: str, default: bytes) -> Path:
        destination = self._workdir / name
        if source is None:
            destination.write_bytes(default)
        else:
            source_path = Path(source).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            destination.write_bytes(source_path.read_bytes())
        return destination

    def _start(self) -> None:
        command = [
            str(self.harness),
            "--width", str(self.width),
            "--height", str(self.height),
            "--frames", "1",
            "--depth", str(self._depth),
            "--motion", str(self._motion),
        ]
        if self.reversed_depth:
            command.append("--reversed-depth")
        environment = os.environ.copy()
        self._process = subprocess.Popen(
            command,
            cwd=str(self.harness.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=environment,
        )
        line = self._read_line()
        if not line.startswith("READY"):
            self.close()
            raise RuntimeError(f"native DLSS5 carrier did not start: {line}")

    def _read_line(self) -> str:
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("native DLSS5 carrier is not running")
        line = self._process.stdout.readline()
        if not line:
            error = ""
            if self._process.stderr is not None:
                error = self._process.stderr.read()[-4000:]
            code = self._process.poll()
            raise RuntimeError(f"native DLSS5 carrier stopped ({code}): {error}")
        line = line.strip()
        if line.startswith("ERROR"):
            raise RuntimeError(line)
        return line

    def _send(self, command: str, expected: str) -> str:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("native DLSS5 carrier is not running")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        response = self._read_line()
        if not response.startswith(expected):
            raise RuntimeError(f"native DLSS5 carrier replied {response!r}, expected {expected!r}")
        return response

    def _input_bytes(self, rgb: Tensor) -> bytes:
        if rgb.ndim != 4 or tuple(rgb.shape[:2]) != (1, 3):
            raise ValueError(f"expected [1, 3, H, W], got {tuple(rgb.shape)}")
        if tuple(rgb.shape[2:]) != (self.height, self.width):
            raise ValueError(
                f"expected spatial shape {(self.height, self.width)}, got {tuple(rgb.shape[2:])}"
            )
        if rgb.dtype != torch.float16:
            raise TypeError("bit-exact carrier requires torch.float16 input")
        # The conversion below only repacks existing half bits.  Callers that
        # start with FP32 must perform and audit that conversion explicitly.
        image = rgb.detach().to(device="cpu").contiguous()[0].permute(1, 2, 0).contiguous()
        rgba = torch.ones((self.height, self.width, 4), dtype=torch.float16)
        rgba[..., :3].copy_(image)
        return rgba.numpy().tobytes()

    def _output_tensor(self, path: Path, device: torch.device) -> Tensor:
        expected = self.width * self.height * 4 * 2
        payload = path.read_bytes()
        if len(payload) != expected:
            raise RuntimeError(f"native output has {len(payload)} bytes, expected {expected}")
        # bytearray makes the tensor independent of the temporary file before
        # the next frame and before the carrier is closed.
        output = torch.frombuffer(bytearray(payload), dtype=torch.float16).reshape(
            self.height, self.width, 4
        )
        return output.permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device)

    def forward(self, rgb: Tensor) -> Tensor:
        """Evaluate one frame and return exact native RGBA16F as NCHW."""

        with self._lock:
            if self._process is None:
                self._start()
            frame_path = self._workdir / f"frame_{self._frame:06d}.rgba16f.bin"
            output_path = self._workdir / f"output_{self._frame:06d}.rgba16f.bin"
            frame_path.write_bytes(self._input_bytes(rgb))
            reset = 1 if self._frame == 0 else 0
            self._send(f"FRAME {frame_path} 0 0 {reset}", "FRAME_OK")
            self._send(f"WRITE {output_path}", "WRITE_OK")
            self._frame += 1
            return self._output_tensor(output_path, rgb.device)

    def reset(self) -> None:
        """Close the feature so the next call starts a fresh temporal session."""

        with self._lock:
            self._stop_process()
            self._frame = 0

    def _stop_process(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.poll() is None and process.stdin is not None:
                    process.stdin.write("QUIT\n")
                    process.stdin.flush()
                    self._read_line()
                    process.wait(timeout=10)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
            finally:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
                self._process = None

    def close(self) -> None:
        with self._lock:
            self._stop_process()
            if getattr(self, "_owned_workdir", None) is not None:
                self._owned_workdir.cleanup()
                self._owned_workdir = None

    def __enter__(self) -> "DLSS5BitExactCarrier":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructors must not raise
            pass


class DLSS5BitExactModel(DLSS5BitExactCarrier):
    """Public model name for the exact native-CUBIN PyTorch interface.

    This is an alias-level subclass rather than a second implementation: the
    native feature session, temporal state, input contract, and byte-level
    guarantee are identical to :class:`DLSS5BitExactCarrier`.
    """
